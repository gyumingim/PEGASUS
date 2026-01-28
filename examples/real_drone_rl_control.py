#!/usr/bin/env python
"""
실제 PX4 드론 강화학습 제어기

Pegasus 시뮬레이터 없이 실제 드론을 RL로 제어
- MAVSDK로 드론 연결 (UDP)
- Telemetry에서 상태 읽기
- RL output → Attitude 명령 변환 후 Offboard 제어
"""

import asyncio
import numpy as np
from scipy.spatial.transform import Rotation
import signal
import sys

# MAVSDK
from mavsdk import System
from mavsdk.offboard import OffboardError, Attitude

# Stable-Baselines3 (강화학습)
try:
    from stable_baselines3 import PPO
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    print("[WARN] stable-baselines3 not available. Install: pip install stable-baselines3")

try:
    import torch
except ImportError:
    torch = None


class IsaacLabToPX4Converter:
    """
    Isaac Lab 토크 명령 → PX4 Attitude 명령 변환 레이어
    (a.py에서 가져옴 - 동일한 변환 로직)
    """

    def __init__(self):
        # Isaac Lab 학습 파라미터
        self.TRAIN_MASS = 0.033
        self.TRAIN_THRUST_TO_WEIGHT = 1.9
        self.TRAIN_MOMENT_SCALE = 0.002
        self.TRAIN_GRAVITY = 9.81
        self.TRAIN_Ixx = 1.4e-5
        self.TRAIN_Iyy = 1.4e-5
        self.TRAIN_Izz = 2.17e-5

        # 실제 드론 파라미터 (Iris 기준, 실제 드론에 맞게 조정 필요)
        self.DRONE_MASS = 1.5
        self.DRONE_HOVER_THRUST = 0.65  # 캘리브레이션 필요!

        # 변환 게인
        self.TORQUE_TO_ANGLE_GAIN_ROLL = 7.0
        self.TORQUE_TO_ANGLE_GAIN_PITCH = 7.0
        self.TORQUE_TO_ANGLE_GAIN_YAW = 15.0

        self.MAX_ROLL = 35.0
        self.MAX_PITCH = 35.0

        # 상태
        self.integrated_yaw = 0.0
        self.prev_action = np.zeros(4)
        self.dt = 0.02
        self.debug_count = 0

    def convert(self, isaac_action: np.ndarray, current_attitude: np.ndarray,
                current_angular_vel: np.ndarray) -> Attitude:
        """Isaac Lab 액션을 PX4 Attitude 명령으로 변환"""

        # 필터링
        alpha = 0.3
        filtered_action = alpha * isaac_action + (1 - alpha) * self.prev_action
        self.prev_action = filtered_action.copy()
        filtered_action = np.clip(filtered_action, -1.0, 1.0)

        # 액션 분리
        thrust_action = filtered_action[0]
        roll_action = filtered_action[1]
        pitch_action = filtered_action[2]
        yaw_action = filtered_action[3]

        # 추력 변환
        isaac_thrust_ratio = self.TRAIN_THRUST_TO_WEIGHT * (thrust_action + 1.0) / 2.0
        thrust_scale = 0.5
        px4_thrust = self.DRONE_HOVER_THRUST + (isaac_thrust_ratio - 1.0) * thrust_scale
        px4_thrust = np.clip(px4_thrust, 0.0, 1.0)

        # 토크 → 각도
        target_roll = roll_action * self.TORQUE_TO_ANGLE_GAIN_ROLL
        target_pitch = pitch_action * self.TORQUE_TO_ANGLE_GAIN_PITCH

        # Yaw 적분
        yaw_rate = yaw_action * self.TORQUE_TO_ANGLE_GAIN_YAW
        self.integrated_yaw += yaw_rate * self.dt
        self.integrated_yaw = (self.integrated_yaw + 180) % 360 - 180

        # 제한
        target_roll = np.clip(target_roll, -self.MAX_ROLL, self.MAX_ROLL)
        target_pitch = np.clip(target_pitch, -self.MAX_PITCH, self.MAX_PITCH)

        final_roll = target_roll
        final_pitch = -target_pitch  # 부호 조정
        final_yaw = self.integrated_yaw

        # 디버그
        if self.debug_count % 50 == 0:
            print(f"[Converter] action=[{thrust_action:+.2f}, {roll_action:+.2f}, "
                  f"{pitch_action:+.2f}, {yaw_action:+.2f}] → "
                  f"roll={final_roll:+.1f}°, pitch={final_pitch:+.1f}°, "
                  f"yaw={final_yaw:+.1f}°, thrust={px4_thrust:.2f}")
        self.debug_count += 1

        return Attitude(float(final_roll), float(final_pitch),
                       float(final_yaw), float(px4_thrust))

    def reset(self):
        self.integrated_yaw = 0.0
        self.prev_action = np.zeros(4)
        self.debug_count = 0


class DroneState:
    """드론 상태 저장용 클래스"""
    def __init__(self):
        self.position = np.zeros(3)           # NED [m]
        self.velocity = np.zeros(3)           # NED [m/s]
        self.attitude_quat = np.array([0, 0, 0, 1])  # [x, y, z, w]
        self.angular_velocity = np.zeros(3)   # body [rad/s]
        self.armed = False
        self.connected = False


class RealDroneRLController:
    """
    실제 드론 강화학습 제어기

    시뮬레이터 없이 MAVSDK telemetry에서 상태를 읽고
    RL 모델로 Attitude 명령 생성
    """

    def __init__(self, model_path: str, target_position: np.ndarray, device: str = "cuda"):
        """
        Args:
            model_path: RL 모델 경로 (.zip)
            target_position: 목표 위치 NED [x, y, z] (착륙 지점)
            device: cuda 또는 cpu
        """
        self.device = device
        self.target_pos = np.array(target_position, dtype=np.float32)

        # RL 모델 로드
        if not RL_AVAILABLE:
            raise ImportError("stable-baselines3 필요: pip install stable-baselines3")

        print(f"[RL] 모델 로딩: {model_path}")
        self.model = PPO.load(model_path, device=device)
        print(f"[RL] 모델 로드 완료 (device: {device})")

        # 변환기
        self.converter = IsaacLabToPX4Converter()

        # 드론 상태
        self.state = DroneState()

        # 스케일 (a.py와 동일)
        self.VEL_SCALE = 1.0

        # 디버그
        self._debug_count = 0

    def update_state(self, position: np.ndarray, velocity: np.ndarray,
                     attitude_quat: np.ndarray, angular_velocity: np.ndarray):
        """텔레메트리에서 상태 업데이트"""
        self.state.position = np.array(position, dtype=np.float32)
        self.state.velocity = np.array(velocity, dtype=np.float32)
        self.state.attitude_quat = np.array(attitude_quat, dtype=np.float32)
        self.state.angular_velocity = np.array(angular_velocity, dtype=np.float32)

    def get_attitude_command(self) -> Attitude:
        """RL 모델로 Attitude 명령 생성"""

        # Observation 구성 (Isaac Lab 16차원과 동일)
        obs = self._construct_observation()

        # RL 추론
        action, _ = self.model.predict(obs, deterministic=True)

        if torch is not None and isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        action = action.flatten()

        # 현재 자세 (degrees)
        r = Rotation.from_quat(self.state.attitude_quat)
        euler = r.as_euler('XYZ', degrees=True)

        # 변환
        attitude_cmd = self.converter.convert(
            action, euler, self.state.angular_velocity
        )

        return attitude_cmd

    def _construct_observation(self) -> np.ndarray:
        """Isaac Lab 환경과 동일한 16차원 observation"""

        pos = self.state.position
        lin_vel = self.state.velocity
        ang_vel = self.state.angular_velocity
        quat = self.state.attitude_quat

        R = Rotation.from_quat(quat)

        # 1. 드론 속도 (body frame)
        lin_vel_b = R.inv().apply(lin_vel) * self.VEL_SCALE

        # 2. 각속도 (body frame)
        ang_vel_b = R.inv().apply(ang_vel)

        # 3. 중력 방향 (body frame)
        gravity_world = np.array([0, 0, -1.0])
        gravity_b = R.inv().apply(gravity_world)

        # 4. 목표 위치 (body frame)
        goal_rel_world = self.target_pos - pos
        desired_pos_b = R.inv().apply(goal_rel_world)

        # 5. 상대 속도 (body frame) - 타겟이 정지해 있다고 가정
        rel_vel_b = lin_vel_b

        # 6. Yaw
        quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])
        yaw = np.arctan2(
            2.0 * (quat_wxyz[0]*quat_wxyz[3] + quat_wxyz[1]*quat_wxyz[2]),
            1.0 - 2.0 * (quat_wxyz[2]**2 + quat_wxyz[3]**2)
        )

        # 디버그
        if self._debug_count % 50 == 0:
            dist = np.linalg.norm(goal_rel_world)
            print(f"[Obs] pos=[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}] "
                  f"target=[{self.target_pos[0]:.1f}, {self.target_pos[1]:.1f}, {self.target_pos[2]:.1f}] "
                  f"dist={dist:.2f}m")
        self._debug_count += 1

        # 16차원 observation
        obs = np.concatenate([
            lin_vel_b,        # 3
            ang_vel_b,        # 3
            gravity_b,        # 3
            desired_pos_b,    # 3
            rel_vel_b,        # 3
            [yaw]             # 1
        ])

        return obs.astype(np.float32)

    def set_target(self, target_pos: np.ndarray):
        """목표 위치 변경"""
        self.target_pos = np.array(target_pos, dtype=np.float32)
        print(f"[RL] 새 목표 위치: {self.target_pos}")

    def reset(self):
        self.converter.reset()
        self._debug_count = 0


class RealDroneController:
    """
    실제 PX4 드론 제어 메인 클래스

    MAVSDK로 연결하고 텔레메트리 수신 + Offboard 제어
    """

    def __init__(self,
                 model_path: str,
                 target_position: list = [0, 0, -2],  # NED 좌표
                 connection_string: str = "udp://:14540",
                 device: str = "cuda"):
        """
        Args:
            model_path: RL 모델 경로
            target_position: 목표 위치 [x, y, z] NED 좌표 (z는 음수가 위)
            connection_string: MAVSDK 연결 문자열
            device: RL 추론 디바이스
        """
        self.connection_string = connection_string
        self.drone = System()

        # RL 제어기
        self.rl_controller = RealDroneRLController(
            model_path=model_path,
            target_position=target_position,
            device=device
        )

        # 상태
        self.running = False
        self.state = DroneState()

        # 제어 주기
        self.control_rate = 50  # Hz
        self.dt = 1.0 / self.control_rate

        # 콜백 설정
        self._position_task = None
        self._attitude_task = None

    async def connect(self):
        """드론 연결"""
        print(f"[MAVSDK] 연결 중: {self.connection_string}")
        await self.drone.connect(system_address=self.connection_string)

        print("[MAVSDK] 연결 대기...")
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("[MAVSDK] 연결 완료!")
                self.state.connected = True
                break

    async def wait_for_gps(self):
        """GPS 위치 추정 대기"""
        print("[MAVSDK] GPS 위치 추정 대기...")
        async for health in self.drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                print("[MAVSDK] GPS 준비 완료")
                return True
        return False

    async def arm(self):
        """드론 Arm"""
        print("[MAVSDK] Arming...")
        await self.drone.action.arm()
        self.state.armed = True
        print("[MAVSDK] Armed!")

    async def disarm(self):
        """드론 Disarm"""
        print("[MAVSDK] Disarming...")
        await self.drone.action.disarm()
        self.state.armed = False

    async def start_offboard(self):
        """Offboard 모드 시작"""
        # 초기 setpoint 필요
        await self.drone.offboard.set_attitude(Attitude(0.0, 0.0, 0.0, 0.0))

        print("[MAVSDK] Offboard 모드 시작...")
        try:
            await self.drone.offboard.start()
            print("[MAVSDK] Offboard 시작됨!")
            return True
        except OffboardError as e:
            print(f"[MAVSDK] Offboard 실패: {e}")
            return False

    async def stop_offboard(self):
        """Offboard 모드 중지"""
        try:
            await self.drone.offboard.stop()
            print("[MAVSDK] Offboard 중지됨")
        except OffboardError as e:
            print(f"[MAVSDK] Offboard 중지 실패: {e}")

    async def land(self):
        """착륙"""
        print("[MAVSDK] 착륙 중...")
        await self.drone.action.land()

    async def _telemetry_position_loop(self):
        """위치/속도 텔레메트리 수신"""
        async for pos_vel in self.drone.telemetry.position_velocity_ned():
            # NED 좌표
            self.state.position = np.array([
                pos_vel.position.north_m,
                pos_vel.position.east_m,
                pos_vel.position.down_m
            ])
            self.state.velocity = np.array([
                pos_vel.velocity.north_m_s,
                pos_vel.velocity.east_m_s,
                pos_vel.velocity.down_m_s
            ])

            if not self.running:
                break

    async def _telemetry_attitude_loop(self):
        """자세 텔레메트리 수신"""
        async for attitude in self.drone.telemetry.attitude_quaternion():
            # MAVSDK: w, x, y, z → 우리: x, y, z, w
            self.state.attitude_quat = np.array([
                attitude.x,
                attitude.y,
                attitude.z,
                attitude.w
            ])

            if not self.running:
                break

    async def _telemetry_angular_velocity_loop(self):
        """각속도 텔레메트리 수신"""
        async for body_rates in self.drone.telemetry.attitude_angular_velocity_body():
            self.state.angular_velocity = np.array([
                body_rates.roll_rad_s,
                body_rates.pitch_rad_s,
                body_rates.yaw_rad_s
            ])

            if not self.running:
                break

    async def control_loop(self, duration: float = 60.0):
        """
        메인 제어 루프

        Args:
            duration: 최대 실행 시간 (초)
        """
        self.running = True
        self.rl_controller.converter.dt = self.dt

        # 텔레메트리 태스크 시작
        position_task = asyncio.create_task(self._telemetry_position_loop())
        attitude_task = asyncio.create_task(self._telemetry_attitude_loop())
        angular_task = asyncio.create_task(self._telemetry_angular_velocity_loop())

        # 텔레메트리가 시작될 때까지 잠시 대기
        await asyncio.sleep(0.5)

        print(f"\n{'='*60}")
        print(f"RL 제어 루프 시작 (최대 {duration}초)")
        print(f"{'='*60}\n")

        start_time = asyncio.get_event_loop().time()

        try:
            while self.running:
                loop_start = asyncio.get_event_loop().time()

                # 시간 초과 체크
                elapsed = loop_start - start_time
                if elapsed > duration:
                    print(f"[Control] 시간 초과 ({duration}초)")
                    break

                # RL 제어기에 상태 전달
                self.rl_controller.update_state(
                    self.state.position,
                    self.state.velocity,
                    self.state.attitude_quat,
                    self.state.angular_velocity
                )

                # RL로 Attitude 명령 생성
                attitude_cmd = self.rl_controller.get_attitude_command()

                # PX4에 명령 전송
                await self.drone.offboard.set_attitude(attitude_cmd)

                # 주기 맞추기
                loop_time = asyncio.get_event_loop().time() - loop_start
                sleep_time = max(0, self.dt - loop_time)
                await asyncio.sleep(sleep_time)

        finally:
            self.running = False
            # 텔레메트리 태스크 정리
            position_task.cancel()
            attitude_task.cancel()
            angular_task.cancel()

    async def run_mission(self, duration: float = 60.0):
        """
        전체 미션 실행

        1. 연결
        2. GPS 대기
        3. Arm
        4. Offboard 시작
        5. RL 제어 루프
        6. Offboard 종료
        7. 착륙
        """
        try:
            await self.connect()
            await self.wait_for_gps()
            await self.arm()

            if not await self.start_offboard():
                await self.disarm()
                return

            await self.control_loop(duration=duration)

        except Exception as e:
            print(f"[ERROR] 미션 중 오류: {e}")
            import traceback
            traceback.print_exc()

        finally:
            await self.stop_offboard()
            await self.land()
            await asyncio.sleep(5)  # 착륙 대기
            print("[MAVSDK] 미션 종료")

    def stop(self):
        """제어 중지"""
        self.running = False


async def main():
    """메인 함수"""

    # ===== 설정 =====
    MODEL_PATH = "/home/rtx5080/s/ISAAC_LAB_DRONE/logs/sb3/Template-DroneLanding-v0/2026-01-27_09-06-42/model.zip"

    # 목표 위치 (NED 좌표: x=north, y=east, z=down)
    # z=-2는 지면에서 2m 위
    TARGET_POSITION = [0, 0, -2]

    # 연결 설정
    # 실제 드론: "serial:///dev/ttyUSB0:57600" 또는 "udp://:14540"
    # SITL 테스트: "udp://:14540"
    CONNECTION_STRING = "udp://:14540"

    # 제어 시간 (초)
    DURATION = 60.0

    # CUDA 사용 여부
    DEVICE = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"

    print("\n" + "="*60)
    print("실제 드론 RL 제어기")
    print("="*60)
    print(f"  모델: {MODEL_PATH}")
    print(f"  목표: {TARGET_POSITION} (NED)")
    print(f"  연결: {CONNECTION_STRING}")
    print(f"  디바이스: {DEVICE}")
    print(f"  제어 시간: {DURATION}초")
    print("="*60 + "\n")

    if not RL_AVAILABLE:
        print("[ERROR] stable-baselines3가 설치되어 있지 않습니다!")
        return

    # 종료 시그널 핸들러
    controller = None

    def signal_handler(signum, frame):
        print("\n[Signal] 종료 요청...")
        if controller:
            controller.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 제어기 생성 및 실행
    controller = RealDroneController(
        model_path=MODEL_PATH,
        target_position=TARGET_POSITION,
        connection_string=CONNECTION_STRING,
        device=DEVICE
    )

    await controller.run_mission(duration=DURATION)


if __name__ == "__main__":
    asyncio.run(main())
