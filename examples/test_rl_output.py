#!/usr/bin/env python3
"""
RL 출력 테스트 스크립트

실제 드론 연결 없이:
- 카메라(b.py)로 AprilTag 상대위치 인식
- 드론 상태는 임의 값 사용 또는 실제 telemetry
- RL 추론 후 PX4 Attitude 명령 로그 출력
- (옵션) 실제 PX4에 명령 전송
- PX4 응답 확인 (telemetry 비교)
"""

import numpy as np
import cv2
import time
import sys
import asyncio
import threading
from typing import Optional
from collections import deque
from threading import Lock

# b.py에서 AprilTagLanding 가져오기
from b import AprilTagLanding

# MAVSDK
try:
    from mavsdk import System
    from mavsdk.offboard import OffboardError, Attitude, VelocityNedYaw
    MAVSDK_AVAILABLE = True
except ImportError:
    MAVSDK_AVAILABLE = False
    print("[WARN] MAVSDK 필요: pip install mavsdk")


def quat_to_rotation_matrix(quat):
    """Quaternion [x,y,z,w] to 3x3 rotation matrix"""
    x, y, z, w = quat
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])


def euler_to_quat(roll, pitch, yaw, degrees=True):
    """Euler angles (XYZ) to quaternion [x,y,z,w]"""
    if degrees:
        roll, pitch, yaw = np.radians([roll, pitch, yaw])

    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)

    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy

    return np.array([x, y, z, w])

# RL 관련
try:
    from stable_baselines3 import PPO
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    print("[WARN] stable-baselines3 필요: pip install stable-baselines3")

try:
    import torch
except ImportError:
    torch = None


class IsaacLabToPX4Converter:
    """Isaac Lab 토크 → PX4 Attitude 변환"""

    def __init__(self):
        self.TRAIN_THRUST_TO_WEIGHT = 1.9
        self.DRONE_HOVER_THRUST = 0.75  # ★ hover thrust 증가

        self.TORQUE_TO_ANGLE_GAIN_ROLL = 7.0
        self.TORQUE_TO_ANGLE_GAIN_PITCH = 7.0
        self.TORQUE_TO_ANGLE_GAIN_YAW = 15.0

        self.MAX_ROLL = 35.0
        self.MAX_PITCH = 35.0

        self.integrated_yaw = 0.0
        self.prev_action = np.zeros(4)
        self.dt = 0.02

    def convert(self, isaac_action: np.ndarray) -> dict:
        """변환 후 dict로 반환 (로깅용)"""
        alpha = 0.3
        filtered_action = alpha * isaac_action + (1 - alpha) * self.prev_action
        self.prev_action = filtered_action.copy()
        filtered_action = np.clip(filtered_action, -1.0, 1.0)

        thrust_action = filtered_action[0]
        roll_action = filtered_action[1]
        pitch_action = filtered_action[2]
        yaw_action = filtered_action[3]

        isaac_thrust_ratio = self.TRAIN_THRUST_TO_WEIGHT * (thrust_action + 1.0) / 2.0
        thrust_scale = 0.5
        px4_thrust = self.DRONE_HOVER_THRUST + (isaac_thrust_ratio - 1.0) * thrust_scale
        px4_thrust = np.clip(px4_thrust, 0.0, 1.0)

        target_roll = roll_action * self.TORQUE_TO_ANGLE_GAIN_ROLL
        target_pitch = pitch_action * self.TORQUE_TO_ANGLE_GAIN_PITCH

        yaw_rate = yaw_action * self.TORQUE_TO_ANGLE_GAIN_YAW
        self.integrated_yaw += yaw_rate * self.dt
        self.integrated_yaw = (self.integrated_yaw + 180) % 360 - 180

        target_roll = np.clip(target_roll, -self.MAX_ROLL, self.MAX_ROLL)
        target_pitch = np.clip(target_pitch, -self.MAX_PITCH, self.MAX_PITCH)

        return {
            'roll_deg': float(target_roll),
            'pitch_deg': float(-target_pitch),
            'yaw_deg': float(self.integrated_yaw),
            'thrust': float(px4_thrust),
            'raw_action': filtered_action.tolist()
        }

    def reset(self):
        self.integrated_yaw = 0.0
        self.prev_action = np.zeros(4)


class PX4Controller:
    """PX4 연결 및 제어"""

    def __init__(self, connection_string: str = "udp://:14540", hover_thrust: float = 0.75):
        if not MAVSDK_AVAILABLE:
            raise ImportError("MAVSDK 필요!")

        self.connection_string = connection_string
        self.hover_thrust = hover_thrust  # ★ 호버링 추력
        self.drone = System()
        self.connected = False
        self.armed = False
        self.offboard_active = False
        
        # 최신 telemetry 데이터 (Thread-safe)
        self._telemetry_lock = Lock()
        self.latest_position = None
        self.latest_velocity = None
        self.latest_attitude_euler = None  # [roll, pitch, yaw] in degrees
        self.latest_angular_velocity = None
        self.latest_flight_mode = None
        self.latest_relative_altitude = 0.0  # 상대 고도 (m)
        
        # 명령 vs 실제 비교용
        self.last_command = None  # 마지막 전송한 명령
        self.command_history = deque(maxlen=50)  # 최근 50개 명령 기록
        
        # 제어 활성화 플래그
        self.control_enabled = False
        
        # Asyncio 루프
        self.loop = None
        self.thread = None
        self.running = False

    def start(self):
        """별도 스레드에서 asyncio 루프 시작"""
        self.running = True
        self.thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.thread.start()
        print(f"[PX4] 연결 시작: {self.connection_string}")
        
        # ✅ 연결 대기 (최대 5초)
        for _ in range(50):
            if self.connected:
                print("[PX4] ✓ 연결 완료")
                break
            time.sleep(0.1)
        else:
            print("[PX4] ⚠ 연결 대기 타임아웃 (5초)")

    def stop(self):
        """종료"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        print("[PX4] 연결 종료")

    def _run_async_loop(self):
        """Asyncio 루프 실행"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._async_main())
        except Exception as e:
            print(f"[PX4] ✗ 비동기 루프 예외: {e}")
        finally:
            self.loop.close()

    async def _async_main(self):
        """메인 비동기 루프"""
        try:
            # 연결
            await self.drone.connect(system_address=self.connection_string)
            
            # 연결 대기
            print("[PX4] 드론 연결 대기 중...")
            async for state in self.drone.core.connection_state():
                if state.is_connected:
                    self.connected = True
                    print("[PX4] ✓ 드론 연결 완료")
                    break
            
            # Telemetry 구독 시작
            asyncio.create_task(self._telemetry_position())
            asyncio.create_task(self._telemetry_velocity())
            asyncio.create_task(self._telemetry_attitude())
            asyncio.create_task(self._telemetry_angular_velocity())
            asyncio.create_task(self._telemetry_flight_mode())
            
            # 메인 루프
            while self.running:
                await asyncio.sleep(0.02)  # 50Hz
                
        except Exception as e:
            print(f"[PX4] ✗ 비동기 메인 루프 예외: {e}")
            self.running = False

    async def _telemetry_position(self):
        """위치 telemetry"""
        try:
            async for position in self.drone.telemetry.position():
                with self._telemetry_lock:
                    self.latest_position = np.array([
                        position.latitude_deg,
                        position.longitude_deg,
                        position.absolute_altitude_m
                    ])
                    self.latest_relative_altitude = position.relative_altitude_m
        except Exception as e:
            print(f"[PX4] Position telemetry 예외: {e}")

    async def _telemetry_velocity(self):
        """속도 telemetry (NED)"""
        try:
            async for velocity in self.drone.telemetry.velocity_ned():
                with self._telemetry_lock:
                    self.latest_velocity = np.array([
                        velocity.north_m_s,
                        velocity.east_m_s,
                        velocity.down_m_s
                    ])
        except Exception as e:
            print(f"[PX4] Velocity telemetry 예외: {e}")

    async def _telemetry_attitude(self):
        """자세 telemetry (Euler angles)"""
        try:
            async for attitude in self.drone.telemetry.attitude_euler():
                with self._telemetry_lock:
                    self.latest_attitude_euler = np.array([
                        attitude.roll_deg,
                        attitude.pitch_deg,
                        attitude.yaw_deg
                    ])
        except Exception as e:
            print(f"[PX4] Attitude telemetry 예외: {e}")

    async def _telemetry_angular_velocity(self):
        """각속도 telemetry"""
        try:
            async for angular_vel in self.drone.telemetry.attitude_angular_velocity_body():
                with self._telemetry_lock:
                    self.latest_angular_velocity = np.array([
                        angular_vel.roll_rad_s,
                        angular_vel.pitch_rad_s,
                        angular_vel.yaw_rad_s
                    ])
        except Exception as e:
            print(f"[PX4] Angular velocity telemetry 예외: {e}")

    async def _telemetry_flight_mode(self):
        """비행 모드 telemetry"""
        try:
            async for flight_mode in self.drone.telemetry.flight_mode():
                with self._telemetry_lock:
                    self.latest_flight_mode = str(flight_mode)
        except Exception as e:
            print(f"[PX4] Flight mode telemetry 예외: {e}")

    def get_state(self) -> Optional[dict]:
        """최신 드론 상태 반환 (Thread-safe)"""
        if not self.connected:
            return None
        
        with self._telemetry_lock:
            # Euler angles to quaternion
            attitude_quat = np.array([0, 0, 0, 1])
            if self.latest_attitude_euler is not None:
                attitude_quat = euler_to_quat(
                    self.latest_attitude_euler[0],
                    self.latest_attitude_euler[1],
                    self.latest_attitude_euler[2],
                    degrees=True
                )
            
            return {
                'position': self.latest_position,
                'velocity': self.latest_velocity if self.latest_velocity is not None else np.zeros(3),
                'attitude_quat': attitude_quat,
                'angular_velocity': self.latest_angular_velocity if self.latest_angular_velocity is not None else np.zeros(3)
            }

    def get_attitude_error(self) -> Optional[dict]:
        """
        명령 vs 실제 자세 오차 계산
        
        Returns:
            dict with error info or None
        """
        if self.last_command is None:
            return None
        
        with self._telemetry_lock:
            if self.latest_attitude_euler is None:
                return None
            
            actual = self.latest_attitude_euler.copy()
        
        cmd = self.last_command
        
        # 각도 차이 계산
        roll_error = actual[0] - cmd['roll_deg']
        pitch_error = actual[1] - cmd['pitch_deg']
        
        # Yaw는 -180~180 범위 고려
        yaw_error = actual[2] - cmd['yaw_deg']
        if yaw_error > 180:
            yaw_error -= 360
        elif yaw_error < -180:
            yaw_error += 360
        
        return {
            'roll_error': roll_error,
            'pitch_error': pitch_error,
            'yaw_error': yaw_error,
            'total_error': np.sqrt(roll_error**2 + pitch_error**2 + yaw_error**2),
            'commanded': {
                'roll': cmd['roll_deg'],
                'pitch': cmd['pitch_deg'],
                'yaw': cmd['yaw_deg'],
                'thrust': cmd['thrust']
            },
            'actual': {
                'roll': actual[0],
                'pitch': actual[1],
                'yaw': actual[2]
            }
        }

    def enable_control(self):
        """제어 활성화 (Arm + Offboard)"""
        if not self.connected:
            print("[PX4] 연결되지 않음!")
            return False
        
        asyncio.run_coroutine_threadsafe(self._enable_control_async(), self.loop)
        return True

    async def _enable_control_async(self):
        """비동기 제어 활성화"""
        try:
            # GPS 대기
            print("[PX4] GPS 위치 추정 대기 중...")
            async for health in self.drone.telemetry.health():
                if health.is_global_position_ok and health.is_home_position_ok:
                    print("[PX4] ✓ GPS 위치 추정 완료")
                    break

            # ★ RC Failsafe 비활성화 (SITL에서 RC 없이 비행)
            print("[PX4] RC Failsafe 비활성화...")
            await self.drone.param.set_param_int("COM_RCL_EXCEPT", 4)
            print("[PX4] ✓ COM_RCL_EXCEPT = 4 (Offboard)")

            # Arming
            print("[PX4] Arming...")
            await self.drone.action.arm()
            self.armed = True
            print("[PX4] ✓ Armed")

            # action.takeoff() 사용 (6m)
            print("[PX4] Takeoff (6m)...")
            await self.drone.action.set_takeoff_altitude(6.0)
            await self.drone.action.takeoff()
            print("[PX4] ✓ Takeoff 명령 전송됨")

            # ★ 이륙 완료 대기 (고도 체크)
            print("[PX4] 이륙 대기 중 (목표: 6m)...")
            takeoff_start = asyncio.get_event_loop().time()
            while True:
                await asyncio.sleep(0.5)
                elapsed = asyncio.get_event_loop().time() - takeoff_start
                
                with self._telemetry_lock:
                    alt = self.latest_relative_altitude
                
                print(f"[PX4] 상승 중... {alt:.2f}m / 6.0m ({elapsed:.1f}s)")

                # 목표 고도 도달
                if alt >= 2.8:
                    print(f"[PX4] ✓ 목표 고도 도달: {alt:.2f}m")
                    break

                # 타임아웃 (최대 15초)
                if elapsed > 15:
                    print(f"[PX4] ⚠ 타임아웃! 현재 고도: {alt:.2f}m")
                    break

            # Offboard 모드 전환 준비 (hover thrust로!)
            print(f"[PX4] Offboard 모드 준비... (hover_thrust={self.hover_thrust})")
            for _ in range(20):
                await self.drone.offboard.set_attitude(Attitude(0.0, 0.0, 0.0, self.hover_thrust))
                await asyncio.sleep(0.05)

            # Offboard 모드 시작
            print("[PX4] Offboard 모드 시작...")
            await self.drone.offboard.start()
            self.offboard_active = True
            self.control_enabled = True
            print("[PX4] ✓ Offboard 활성화 - RL 제어 시작!")

        except OffboardError as error:
            print(f"[PX4] ✗ Offboard 실패: {error._result.result}")
            self.control_enabled = False
        except Exception as e:
            print(f"[PX4] ✗ 제어 활성화 실패: {e}")
            self.control_enabled = False

    def disable_control(self):
        """제어 비활성화 (Offboard 중지 + Disarm)"""
        if not self.connected:
            return
        
        asyncio.run_coroutine_threadsafe(self._disable_control_async(), self.loop)

    async def _disable_control_async(self):
        """비동기 제어 비활성화"""
        try:
            if self.offboard_active:
                print("[PX4] Offboard 모드 중지...")
                await self.drone.offboard.stop()
                self.offboard_active = False
            
            if self.armed:
                print("[PX4] Landing...")
                await self.drone.action.land()
                await asyncio.sleep(3)
                
                print("[PX4] Disarming...")
                await self.drone.action.disarm()
                self.armed = False
            
            self.control_enabled = False
            print("[PX4] ✓ 제어 비활성화")
            
        except Exception as e:
            print(f"[PX4] ✗ 제어 비활성화 실패: {e}")

    def send_attitude(self, cmd: dict):
        """Attitude 명령 전송"""
        if not self.control_enabled:
            return False

        attitude = Attitude(
            cmd['roll_deg'],
            cmd['pitch_deg'],
            cmd['yaw_deg'],
            cmd['thrust']
        )

        asyncio.run_coroutine_threadsafe(
            self.drone.offboard.set_attitude(attitude),
            self.loop
        )

        # 명령 저장
        self.last_command = cmd.copy()
        self.command_history.append({
            'timestamp': time.time(),
            'command': cmd.copy()
        })

        return True

    def send_velocity_ned(self, north: float, east: float, down: float, yaw_deg: float):
        """속도 명령 전송 (NED frame) - 위치 유지용"""
        if not self.control_enabled:
            return False

        velocity = VelocityNedYaw(north, east, down, yaw_deg)

        asyncio.run_coroutine_threadsafe(
            self.drone.offboard.set_velocity_ned(velocity),
            self.loop
        )

        # ✅ velocity 모드에서는 last_command를 무효화
        # (attitude error 계산 불가)
        self.last_command = None

        return True


class RLOutputTester:
    """RL 출력 테스터"""

    def __init__(self, model_path: str, tag_size: float = 0.165, 
                 use_px4: bool = False, px4_connection: str = "udp://:14540"):
        """
        Args:
            model_path: RL 모델 경로
            tag_size: AprilTag 실제 크기 (미터)
            use_px4: PX4 연결 여부
            px4_connection: PX4 연결 문자열
        """
        if not RL_AVAILABLE:
            raise ImportError("stable-baselines3 필요!")

        # RL 모델 로드
        device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
        print(f"[RL] 모델 로딩: {model_path} (device: {device})")
        self.model = PPO.load(model_path, device=device)
        print("[RL] 로드 완료")

        # 변환기
        self.converter = IsaacLabToPX4Converter()

        # ===== 비행 파라미터 (여기서 한번에 조정) =====
        self.HOVER_THRUST = 0.71      # 호버링 추력
        self.TAKEOFF_THRUST = 0.85    # 이륙 추력
        self.TARGET_ALTITUDE = 6.0    # 목표 상승 고도 (m)
        self.MARKER_LOST_TIMEOUT = 2.0  # ✅ 마커 미감지 타임아웃 (초)

        # Converter에도 hover thrust 동기화
        self.converter.DRONE_HOVER_THRUST = self.HOVER_THRUST

        # 카메라 (AprilTag 인식)
        print(f"[Camera] AprilTag 인식기 초기화 (tag_size={tag_size}m)")
        self.tag_detector = AprilTagLanding(tag_size=tag_size)

        # PX4 컨트롤러 (hover_thrust 전달)
        self.use_px4 = use_px4 and MAVSDK_AVAILABLE
        self.px4 = None
        if self.use_px4:
            self.px4 = PX4Controller(px4_connection, hover_thrust=self.HOVER_THRUST)
            self.px4.start()

        # ===== 가상 드론 상태 (임의 값 또는 실제 telemetry) =====
        self.fake_drone_state = {
            'position': np.array([0.0, 0.0, 2.0]),       # NED: 2m 높이
            'velocity': np.array([0.0, 0.0, 0.0]),       # 정지
            'attitude_quat': np.array([0, 0, 0, 1]),     # 수평
            'angular_velocity': np.array([0.0, 0.0, 0.0])  # 회전 없음
        }

        self.VEL_SCALE = 1.0
        self.frame_count = 0

        # PX4 제어 상태
        self.px4_control_active = False

        # 비행 단계: "IDLE" -> "TAKEOFF" -> "LANDING"
        self.flight_phase = "IDLE"

        # ✅ 마커 감지 상태 개선
        self.marker_ever_detected = False
        self.last_valid_target_body = None  # 마지막으로 감지된 목표 위치
        self.last_marker_detection_time = None  # 마지막 마커 감지 시간

    def get_drone_state(self) -> dict:
        """드론 상태 가져오기 (PX4 telemetry 또는 가상)"""
        if self.use_px4 and self.px4.connected:
            state = self.px4.get_state()
            if state and all(v is not None for v in state.values()):
                return state
        
        # PX4 미사용 또는 데이터 없으면 가상 상태
        return self.fake_drone_state

    def _construct_observation(self, target_pos_body: np.ndarray) -> np.ndarray:
        """16차원 observation 구성 (Isaac Lab FLU 좌표계로 변환)"""
        state = self.get_drone_state()

        lin_vel = state['velocity']           # NED world frame
        ang_vel = state['angular_velocity']   # FRD body frame (PX4)
        quat = state['attitude_quat']

        R = quat_to_rotation_matrix(quat)
        R_inv = R.T

        # ===== PX4 (NED/FRD) → Isaac Lab (ENU/FLU) 변환 =====
        # NED velocity → body frame (FRD)
        lin_vel_frd = R_inv @ lin_vel

        # FRD → FLU 변환: [Forward, Right, Down] → [Forward, Left, Up]
        # x_flu = x_frd, y_flu = -y_frd, z_flu = -z_frd
        lin_vel_b = np.array([lin_vel_frd[0], -lin_vel_frd[1], -lin_vel_frd[2]]) * self.VEL_SCALE

        # 각속도: FRD → FLU 변환
        ang_vel_b = np.array([ang_vel[0], -ang_vel[1], -ang_vel[2]])

        # 중력 방향: PX4는 NED이므로 gravity_ned = [0, 0, +1]
        gravity_ned = np.array([0, 0, 1.0])  # NED: +Z = 아래
        gravity_frd = R_inv @ gravity_ned
        # FRD → FLU 변환
        gravity_b = np.array([gravity_frd[0], -gravity_frd[1], -gravity_frd[2]])

        desired_pos_b = target_pos_body
        rel_vel_b = lin_vel_b

        quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])
        yaw = np.arctan2(
            2.0 * (quat_wxyz[0]*quat_wxyz[3] + quat_wxyz[1]*quat_wxyz[2]),
            1.0 - 2.0 * (quat_wxyz[2]**2 + quat_wxyz[3]**2)
        )

        obs = np.concatenate([
            lin_vel_b, ang_vel_b, gravity_b, desired_pos_b, rel_vel_b, [yaw]
        ])

        return obs.astype(np.float32)

    def _create_takeoff_command(self) -> dict:
        """이륙 명령 생성 (수평 자세 + 상승 추력)"""
        return {
            'roll_deg': 0.0,
            'pitch_deg': 0.0,
            'yaw_deg': 0.0,
            'thrust': self.TAKEOFF_THRUST,
            'raw_action': [0, 0, 0, 0]
        }

    def process_frame(self) -> dict:
        """한 프레임 처리"""
        self.frame_count += 1
        result = {
            'frame': self.frame_count,
            'detected': False,
            'px4_sent': False,
            'px4_connected': self.px4.connected if self.px4 else False,
            'px4_control_active': self.px4_control_active,
            'px4_flight_mode': self.px4.latest_flight_mode if self.px4 else None,
            'flight_phase': self.flight_phase,
            'attitude_error': None
        }

        # 1. AprilTag 인식
        tag_result = self.tag_detector.get_relative_pose()
        result['image'] = tag_result.get('image')

        # ===== TAKEOFF 단계: 3m까지 상승 (태그 인식 무관) =====
        if self.flight_phase == "TAKEOFF":
            # PX4 텔레메트리에서 상대 고도 사용
            current_altitude = 0.0
            if self.px4:
                with self.px4._telemetry_lock:
                    current_altitude = self.px4.latest_relative_altitude

            result['current_altitude'] = current_altitude

            # 3m 도달 체크
            if current_altitude >= self.TARGET_ALTITUDE:
                print(f"\n[PHASE] 목표 고도 {self.TARGET_ALTITUDE}m 도달! -> LANDING 단계로 전환")
                self.flight_phase = "LANDING"
                # LANDING 단계로 전환 후 아래 로직 계속 실행
            else:
                # 상승 명령 전송 (태그 인식 여부와 무관하게 계속 상승)
                px4_cmd = self._create_takeoff_command()
                result['px4_cmd'] = px4_cmd
                result['tag'] = None

                if self.px4_control_active and self.px4:
                    if self.px4.send_attitude(px4_cmd):
                        result['px4_sent'] = True

                if self.px4:
                    result['attitude_error'] = self.px4.get_attitude_error()

                return result

        # ===== LANDING 단계: RL 출력으로 이동 =====
        if not tag_result['detected']:
            result['tag'] = None
            current_time = time.time()

            # ★ 마커 한번도 감지 안됨 → 속도 제어로 제자리 호버링
            if not self.marker_ever_detected:
                # 현재 yaw 유지
                current_yaw = 0.0
                if self.px4:
                    with self.px4._telemetry_lock:
                        if self.px4.latest_attitude_euler is not None:
                            current_yaw = self.px4.latest_attitude_euler[2]

                # 속도 0으로 제자리 유지 (VelocityNedYaw)
                result['px4_cmd'] = {
                    'type': 'velocity',
                    'north': 0.0,
                    'east': 0.0,
                    'down': 0.0,
                    'yaw_deg': current_yaw
                }
                if self.frame_count % 25 == 0:
                    print(f"  [HOVER] 마커 미감지 - 속도제어 호버링 (vel=0, yaw={current_yaw:.1f}°)")

                if self.px4_control_active and self.px4:
                    self.px4.send_velocity_ned(0.0, 0.0, 0.0, current_yaw)
                    result['px4_sent'] = True
                
                # ✅ velocity 모드에서는 attitude_error가 None
                result['attitude_error'] = None
                return result

            # ✅ 마커 이전에 감지된 적 있음 → 타임아웃 체크
            if self.last_valid_target_body is not None and self.last_marker_detection_time is not None:
                time_since_last_detection = current_time - self.last_marker_detection_time
                
                # 타임아웃 초과 → velocity hold로 전환
                if time_since_last_detection > self.MARKER_LOST_TIMEOUT:
                    current_yaw = 0.0
                    if self.px4:
                        with self.px4._telemetry_lock:
                            if self.px4.latest_attitude_euler is not None:
                                current_yaw = self.px4.latest_attitude_euler[2]
                    
                    result['px4_cmd'] = {
                        'type': 'velocity',
                        'north': 0.0,
                        'east': 0.0,
                        'down': 0.0,
                        'yaw_deg': current_yaw
                    }
                    
                    if self.frame_count % 25 == 0:
                        print(f"  [TIMEOUT] 마커 미감지 {time_since_last_detection:.1f}s - 호버링 모드")
                    
                    if self.px4_control_active and self.px4:
                        self.px4.send_velocity_ned(0.0, 0.0, 0.0, current_yaw)
                        result['px4_sent'] = True
                    
                    result['attitude_error'] = None
                    return result
                
                # 타임아웃 이내 → 마지막 위치로 RL 계속 사용
                result['detected'] = True
                result['tag'] = {'id': 0, 'x': 0, 'y': 0, 'z': 0, 'distance': 0,
                                'roll': 0, 'pitch': 0, 'yaw': 0, 'using_last': True}
                target_body = self.last_valid_target_body

                if self.frame_count % 25 == 0:
                    print(f"  [RL] 마커 미감지 ({time_since_last_detection:.1f}s) - 마지막 위치로 RL 계속")

                obs = self._construct_observation(target_body)
                result['observation'] = obs.tolist()

                action, _ = self.model.predict(obs, deterministic=True)
                if torch and isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                action = action.flatten()
                result['rl_action'] = action.tolist()

                px4_cmd = self.converter.convert(action)
                result['px4_cmd'] = px4_cmd

                if self.px4_control_active and self.px4:
                    if self.px4.send_attitude(px4_cmd):
                        result['px4_sent'] = True

                if self.px4:
                    result['attitude_error'] = self.px4.get_attitude_error()

                return result

            # fallback: 속도 제어로 제자리 호버링
            current_yaw = 0.0
            if self.px4:
                with self.px4._telemetry_lock:
                    if self.px4.latest_attitude_euler is not None:
                        current_yaw = self.px4.latest_attitude_euler[2]
            
            result['px4_cmd'] = {
                'type': 'velocity',
                'north': 0.0,
                'east': 0.0,
                'down': 0.0,
                'yaw_deg': current_yaw
            }
            if self.px4_control_active and self.px4:
                self.px4.send_velocity_ned(0.0, 0.0, 0.0, current_yaw)
                result['px4_sent'] = True
            
            result['attitude_error'] = None
            return result

        # ✅ 태그 감지됨 → 마커 감지 플래그 및 시간 업데이트
        self.marker_ever_detected = True
        self.last_marker_detection_time = time.time()
        
        result['detected'] = True
        result['tag'] = {
            'id': tag_result['tag_id'],
            'x': tag_result['x'],
            'y': tag_result['y'],
            'z': tag_result['z'],
            'distance': tag_result['distance'],
            'roll': tag_result['roll'],
            'pitch': tag_result['pitch'],
            'yaw': tag_result['yaw']
        }
        result['corners'] = tag_result.get('corners')

        # 2. 카메라 좌표 → Body frame 변환
        target_body = np.array([
            tag_result['y'],      # body_x = cam_y (앞쪽으로)
            -tag_result['x'],     # body_y = -cam_x (왼쪽으로)
            -tag_result['z']      # body_z = -cam_z (아래쪽 = 음수)
        ])
        result['target_body'] = target_body.tolist()

        # ★ 마지막 유효 타겟 저장
        self.last_valid_target_body = target_body.copy()

        # 3. Observation 구성
        obs = self._construct_observation(target_body)
        result['observation'] = obs.tolist()

        # ★ 디버그: 중요 observation 값 출력
        if self.frame_count % 50 == 1:
            print(f"\n  [OBS DEBUG] desired_pos_b: [{target_body[0]:+.3f}, {target_body[1]:+.3f}, {target_body[2]:+.3f}]")
            print(f"              gravity_b: [{obs[6]:+.3f}, {obs[7]:+.3f}, {obs[8]:+.3f}]")

        # 4. RL 추론
        action, _ = self.model.predict(obs, deterministic=True)
        if torch and isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        action = action.flatten()
        result['rl_action'] = action.tolist()

        # 5. PX4 명령 변환
        px4_cmd = self.converter.convert(action)
        result['px4_cmd'] = px4_cmd

        # 6. PX4에 명령 전송
        if self.px4_control_active and self.px4:
            if self.px4.send_attitude(px4_cmd):
                result['px4_sent'] = True

        # 7. PX4 오차 정보
        if self.px4:
            result['attitude_error'] = self.px4.get_attitude_error()

        return result

    def print_result(self, result: dict):
        """결과 출력"""
        if result['frame'] % 25 == 0:
            print(f"\n{'='*70}")
            phase_str = result.get('flight_phase', 'IDLE')
            print(f"Frame {result['frame']} | Phase: {phase_str} | PX4: {'Connected' if result['px4_connected'] else 'Disconnected'} | "
                f"Mode: {result['px4_flight_mode']} | Control: {'ACTIVE' if result['px4_control_active'] else 'INACTIVE'}")
            print(f"{'='*70}")

            # TAKEOFF 단계에서 고도 정보 출력
            if phase_str == "TAKEOFF":
                alt = result.get('current_altitude', 0)
                print(f"  [TAKEOFF] 현재 고도: {alt:.2f}m / 목표: {self.TARGET_ALTITUDE}m")

            if not result['detected']:
                print("  [TAG] Not detected")
                
                # PX4 오차 출력 (태그 없어도)
                if result['attitude_error']:
                    err = result['attitude_error']
                    print(f"  [PX4 ERROR] Roll:{err['roll_error']:+.1f}° Pitch:{err['pitch_error']:+.1f}° "
                        f"Yaw:{err['yaw_error']:+.1f}° Total:{err['total_error']:.1f}°")
                
                return

            tag = result['tag']
            print(f"  [TAG] ID={tag['id']} | dist={tag['distance']:.3f}m")
            print(f"        pos: x={tag['x']:+.3f}m, y={tag['y']:+.3f}m, z={tag['z']:+.3f}m")

            act = result['rl_action']
            print(f"  [RL ACTION] thrust={act[0]:+.3f}, roll={act[1]:+.3f}, pitch={act[2]:+.3f}, yaw={act[3]:+.3f}")

            cmd = result['px4_cmd']
            print(f"  [PX4 CMD] roll={cmd['roll_deg']:+.1f}°, pitch={cmd['pitch_deg']:+.1f}°, "
                f"yaw={cmd['yaw_deg']:+.1f}°, thrust={cmd['thrust']:.3f}")
            
            if result['px4_sent']:
                print(f"  [PX4] ✓ 명령 전송됨")
                
                # 오차 정보 출력
                if result['attitude_error']:
                    err = result['attitude_error']
                    print(f"  [PX4 ACTUAL] roll={err['actual']['roll']:+.1f}°, pitch={err['actual']['pitch']:+.1f}°, "
                        f"yaw={err['actual']['yaw']:+.1f}°")
                    print(f"  [PX4 ERROR] roll:{err['roll_error']:+.1f}° pitch:{err['pitch_error']:+.1f}° "
                        f"yaw:{err['yaw_error']:+.1f}° total:{err['total_error']:.1f}°")

    def visualize(self, result: dict) -> np.ndarray:
        """시각화 이미지 생성"""
        if result['image'] is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        img = result['image'].copy()

        # PX4 상태 표시
        status_y = 30

        # 비행 단계 표시
        phase_str = result.get('flight_phase', 'IDLE')
        if phase_str == "TAKEOFF":
            phase_color = (0, 165, 255)  # Orange
            alt = result.get('current_altitude', 0)
            phase_text = f"PHASE: TAKEOFF ({alt:.1f}m / {self.TARGET_ALTITUDE}m)"
        elif phase_str == "LANDING":
            phase_color = (0, 255, 0)  # Green
            phase_text = "PHASE: LANDING (RL Control)"
        else:
            phase_color = (128, 128, 128)
            phase_text = f"PHASE: {phase_str}"

        cv2.putText(img, phase_text, (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, phase_color, 2)
        status_y += 25

        if result['px4_connected']:
            px4_status = f"PX4: CONNECTED | {result['px4_flight_mode']}"
            status_color = (0, 255, 0)
        else:
            px4_status = "PX4: DISCONNECTED"
            status_color = (0, 0, 255)

        cv2.putText(img, px4_status, (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        status_y += 25

        if result['px4_control_active']:
            control_status = "CONTROL: ACTIVE"
            control_color = (0, 255, 0)
        else:
            control_status = "CONTROL: INACTIVE"
            control_color = (128, 128, 128)

        cv2.putText(img, control_status, (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, control_color, 2)
        status_y += 35

        if result['detected']:
            if result.get('corners') is not None:
                corners = result['corners'].astype(int)
                cv2.polylines(img, [corners], True, (0, 255, 0), 2)
                center = corners.mean(axis=0).astype(int)
                cv2.circle(img, tuple(center), 5, (0, 0, 255), -1)

            y = status_y
            tag = result['tag']
            cmd = result['px4_cmd']

            cv2.putText(img, f"Tag ID: {tag['id']} | Dist: {tag['distance']:.2f}m",
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y += 20

            cv2.putText(img, f"Pos: X={tag['x']:+.2f} Y={tag['y']:+.2f} Z={tag['z']:+.2f}",
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y += 25

            # PX4 명령
            cv2.putText(img, "--- PX4 COMMAND ---",
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y += 20

            if cmd.get('type') == 'velocity':
                cv2.putText(img, f"VEL: N={cmd['north']:.1f} E={cmd['east']:.1f} D={cmd['down']:.1f}",
                           (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                cv2.putText(img, f"R:{cmd['roll_deg']:+.1f} P:{cmd['pitch_deg']:+.1f} "
                           f"Y:{cmd['yaw_deg']:+.1f} T:{cmd['thrust']:.3f}",
                           (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y += 25
            
            # PX4 실제 자세 & 오차
            if result['attitude_error']:
                err = result['attitude_error']
                cv2.putText(img, "--- PX4 ACTUAL ---",
                           (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y += 20
                
                cv2.putText(img, f"R:{err['actual']['roll']:+.1f} P:{err['actual']['pitch']:+.1f} "
                           f"Y:{err['actual']['yaw']:+.1f}",
                           (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                y += 20
                
                # 오차 (빨간색)
                error_color = (0, 0, 255) if err['total_error'] > 5 else (0, 255, 0)
                cv2.putText(img, f"ERROR: {err['total_error']:.1f} deg "
                           f"(R:{err['roll_error']:+.1f} P:{err['pitch_error']:+.1f} Y:{err['yaw_error']:+.1f})",
                           (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, error_color, 1)
            
            if result['px4_sent']:
                y += 20
                cv2.putText(img, "SENT TO PX4", (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        else:
            cv2.putText(img, "No AprilTag detected",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 키 안내
        help_y = img.shape[0] - 20
        cv2.putText(img, "[ESC]=Quit | Auto: TAKEOFF->3m->LANDING(RL)",
                   (10, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return img

    def visualize_control(self, result: dict) -> np.ndarray:
        """RL 제어 명령 시각화 (별도 창)"""
        canvas = np.ones((600, 800, 3), dtype=np.uint8) * 40
        center_x, center_y = 400, 300
        
        # 제목 (PX4 응답 상태에 따라 색상 변경)
        if result.get('attitude_error'):
            err = result['attitude_error']
            if err['total_error'] < 5:
                title_color = (0, 255, 0)  # 녹색: 오차 작음
            elif err['total_error'] < 15:
                title_color = (0, 255, 255)  # 노랑: 오차 보통
            else:
                title_color = (0, 0, 255)  # 빨강: 오차 큼
        else:
            title_color = (255, 255, 255)
        
        cv2.putText(canvas, "RL CONTROL OUTPUT", (250, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, title_color, 2)
        
        # 비행 단계 표시
        phase_str = result.get('flight_phase', 'IDLE')
        if phase_str == "TAKEOFF":
            phase_color = (0, 165, 255)  # Orange
            alt = result.get('current_altitude', 0)
            phase_text = f"TAKEOFF: {alt:.1f}m / {self.TARGET_ALTITUDE}m"
        elif phase_str == "LANDING":
            phase_color = (0, 255, 0)  # Green
            phase_text = "LANDING (RL Control)"
        else:
            phase_color = (128, 128, 128)
            phase_text = phase_str

        cv2.putText(canvas, phase_text, (300, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, phase_color, 2)

        # PX4 상태
        status_text = ""
        if result.get('px4_connected', False):
            status_text = f"PX4: {result['px4_flight_mode']}"
            if result.get('px4_control_active', False):
                status_text += " | ACTIVE"
                if result.get('px4_sent', False):
                    status_text += " | SENT"
        else:
            status_text = "PX4: DISCONNECTED"

        cv2.putText(canvas, status_text, (200, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        if not result['detected'] or result['px4_cmd'] is None:
            cv2.putText(canvas, "No data", (center_x - 50, center_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2)
            return canvas

        cmd = result['px4_cmd']

        # ★ velocity 타입이면 호버링 모드 표시
        if cmd.get('type') == 'velocity':
            cv2.putText(canvas, "VELOCITY HOLD MODE", (250, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(canvas, f"N:{cmd['north']:.1f} E:{cmd['east']:.1f} D:{cmd['down']:.1f}",
                       (280, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(canvas, f"Yaw: {cmd['yaw_deg']:.1f} deg",
                       (300, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            return canvas

        # 드론 중심
        cv2.circle(canvas, (center_x, center_y), 40, (200, 200, 200), 2)
        cv2.circle(canvas, (center_x, center_y), 3, (255, 255, 255), -1)

        # ========== THRUST ==========
        thrust_val = cmd.get('thrust', self.HOVER_THRUST)
        hover_thrust = self.HOVER_THRUST
        thrust_scale = 150
        thrust_offset = (thrust_val - hover_thrust) * thrust_scale
        arrow_end_y = center_y - int(thrust_offset)
        arrow_color = (0, 255, 0) if thrust_offset > 0 else (0, 100, 255)
        cv2.arrowedLine(canvas, (center_x, center_y), (center_x, arrow_end_y),
                       arrow_color, 3, tipLength=0.3)
        thrust_text_y = arrow_end_y - 20 if thrust_offset > 0 else arrow_end_y + 30
        cv2.putText(canvas, f"T:{thrust_val:.3f}", (center_x + 10, thrust_text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, arrow_color, 2)
        
        # ========== ROLL ==========
        roll_deg = cmd['roll_deg']
        roll_scale = 3
        roll_offset = int(roll_deg * roll_scale)
        arrow_end_x = center_x + roll_offset
        roll_color = (255, 128, 0)
        cv2.arrowedLine(canvas, (center_x, center_y), (arrow_end_x, center_y),
                       roll_color, 3, tipLength=0.3)
        roll_text_x = arrow_end_x + 10 if roll_offset > 0 else arrow_end_x - 50
        cv2.putText(canvas, f"R:{roll_deg:+.1f}", (roll_text_x, center_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, roll_color, 2)
        
        # ========== PITCH ==========
        pitch_deg = cmd['pitch_deg']
        pitch_scale = 3
        pitch_offset = int(pitch_deg * pitch_scale)
        pitch_end_x = center_x + int(pitch_offset * 0.707)
        pitch_end_y = center_y - int(pitch_offset * 0.707)
        pitch_color = (128, 0, 255)
        cv2.arrowedLine(canvas, (center_x, center_y), (pitch_end_x, pitch_end_y),
                       pitch_color, 3, tipLength=0.3)
        pitch_text_x = pitch_end_x + 10
        pitch_text_y = pitch_end_y - 10 if pitch_offset > 0 else pitch_end_y + 20
        cv2.putText(canvas, f"P:{pitch_deg:+.1f}", (pitch_text_x, pitch_text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, pitch_color, 2)
        
        # ========== YAW ==========
        yaw_deg = cmd['yaw_deg']
        yaw_color = (0, 255, 255)
        if abs(yaw_deg) > 1:
            start_angle = 0
            end_angle = int(yaw_deg)
            cv2.ellipse(canvas, (center_x, center_y), (80, 80),
                       0, start_angle, end_angle, yaw_color, 2)
            yaw_rad = np.radians(yaw_deg)
            arrow_x = center_x + int(80 * np.cos(yaw_rad))
            arrow_y = center_y + int(80 * np.sin(yaw_rad))
            cv2.arrowedLine(canvas, (center_x + 80, center_y), (arrow_x, arrow_y),
                           yaw_color, 2, tipLength=0.5)
        cv2.putText(canvas, f"Y:{yaw_deg:+.1f}", (center_x + 100, center_y - 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, yaw_color, 2)
        
        # ========== PX4 응답 (오차 표시) ==========
        if result.get('attitude_error'):
            err = result['attitude_error']
            
            # 우측에 오차 그래프
            graph_x = 550
            graph_y = 150
            
            cv2.putText(canvas, "PX4 RESPONSE:", (graph_x, graph_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            graph_y += 30
            
            # 오차 막대그래프
            max_bar_len = 100
            
            # Roll error
            roll_err_bar = int(abs(err['roll_error']) / 35.0 * max_bar_len)
            roll_err_bar = min(roll_err_bar, max_bar_len)
            roll_err_color = (0, 255, 0) if abs(err['roll_error']) < 5 else (0, 0, 255)
            cv2.rectangle(canvas, (graph_x, graph_y), (graph_x + roll_err_bar, graph_y + 15),
                         roll_err_color, -1)
            cv2.putText(canvas, f"Roll err: {err['roll_error']:+.1f}", (graph_x + max_bar_len + 10, graph_y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            graph_y += 25
            
            # Pitch error
            pitch_err_bar = int(abs(err['pitch_error']) / 35.0 * max_bar_len)
            pitch_err_bar = min(pitch_err_bar, max_bar_len)
            pitch_err_color = (0, 255, 0) if abs(err['pitch_error']) < 5 else (0, 0, 255)
            cv2.rectangle(canvas, (graph_x, graph_y), (graph_x + pitch_err_bar, graph_y + 15),
                         pitch_err_color, -1)
            cv2.putText(canvas, f"Pitch err: {err['pitch_error']:+.1f}", (graph_x + max_bar_len + 10, graph_y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            graph_y += 25
            
            # Yaw error
            yaw_err_bar = int(abs(err['yaw_error']) / 180.0 * max_bar_len)
            yaw_err_bar = min(yaw_err_bar, max_bar_len)
            yaw_err_color = (0, 255, 0) if abs(err['yaw_error']) < 10 else (0, 0, 255)
            cv2.rectangle(canvas, (graph_x, graph_y), (graph_x + yaw_err_bar, graph_y + 15),
                         yaw_err_color, -1)
            cv2.putText(canvas, f"Yaw err: {err['yaw_error']:+.1f}", (graph_x + max_bar_len + 10, graph_y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            graph_y += 35
            
            # Total error
            total_err_color = (0, 255, 0) if err['total_error'] < 5 else \
                             (0, 255, 255) if err['total_error'] < 15 else (0, 0, 255)
            cv2.putText(canvas, f"Total Error: {err['total_error']:.1f} deg", (graph_x, graph_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, total_err_color, 2)
            graph_y += 30
            
            # 실제 자세
            cv2.putText(canvas, "Actual Attitude:", (graph_x, graph_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            graph_y += 20
            cv2.putText(canvas, f"Roll:  {err['actual']['roll']:+.1f}", (graph_x, graph_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            graph_y += 18
            cv2.putText(canvas, f"Pitch: {err['actual']['pitch']:+.1f}", (graph_x, graph_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            graph_y += 18
            cv2.putText(canvas, f"Yaw:   {err['actual']['yaw']:+.1f}", (graph_x, graph_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # ========== Raw RL Action ==========
        if 'rl_action' in result:
            act = result['rl_action']
            legend_x = 50
            legend_y = 480
            cv2.putText(canvas, "Raw RL Action:", (legend_x, legend_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            legend_y += 20
            cv2.putText(canvas, f"[{act[0]:+.2f}, {act[1]:+.2f}, {act[2]:+.2f}, {act[3]:+.2f}]",
                       (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return canvas

    def run(self):
        """메인 루프"""
        print("\n" + "="*70)
        print("RL 출력 테스트 시작")
        print("="*70)
        print("  - 카메라로 AprilTag 인식")
        print("  - RL 모델로 추론")
        print("  - PX4 Attitude 명령 출력")
        if self.use_px4:
            print(f"  - PX4 연결: {self.px4.connection_string}")
            print("  - 자동 시작: TAKEOFF -> 3m 상승 -> LANDING (RL 제어)")
            print("  - Telemetry로 명령 vs 실제 자세 비교")
        print("  - [ESC] 키로 종료")
        print("="*70 + "\n")

        # ===== 시작 시 바로 PX4 제어 활성화 =====
        if self.use_px4 and self.px4:
            print("\n[자동 시작] PX4 제어 활성화 중 (Arm -> Takeoff 3m -> Offboard)...")
            
            # ✅ TAKEOFF 단계 설정
            self.flight_phase = "TAKEOFF"
            
            self.px4.enable_control()

            # Offboard 활성화 대기 (화면 업데이트하면서)
            print("[자동 시작] Takeoff + Offboard 활성화 대기 중...")
            wait_start = time.time()
            while time.time() - wait_start < 25:  # 25초 대기
                # 화면 업데이트
                tag_result = self.tag_detector.get_relative_pose()
                if tag_result.get('image') is not None:
                    img = tag_result['image'].copy()
                    elapsed = time.time() - wait_start
                    
                    with self.px4._telemetry_lock:
                        alt = self.px4.latest_relative_altitude
                    
                    cv2.putText(img, f"TAKEOFF... {elapsed:.1f}s", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    cv2.putText(img, f"Altitude: {alt:.2f}m / 6.0m", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if alt >= 2.5 else (255, 255, 255), 2)
                    cv2.imshow('Camera + AprilTag', img)
                    cv2.waitKey(1)

                # Offboard 활성화 체크
                if self.px4.control_enabled:
                    with self.px4._telemetry_lock:
                        current_alt = self.px4.latest_relative_altitude
                    print(f"[자동 시작] ✓ Offboard 활성화 완료! (고도: {current_alt:.2f}m)")
                    break

                time.sleep(0.1)

            if self.px4.control_enabled:
                self.px4_control_active = True
                print("[자동 시작] -> LANDING 단계 시작 (RL 제어)")
                self.flight_phase = "LANDING"
            else:
                print("[자동 시작] PX4 제어 활성화 실패!")
                return

        try:
            while True:
                result = self.process_frame()
                self.print_result(result)

                vis_img = self.visualize(result)
                cv2.imshow('Camera + AprilTag', vis_img)

                control_img = self.visualize_control(result)
                cv2.imshow('RL Control Visualization', control_img)

                key = cv2.waitKey(1) & 0xFF

                if key == 27:  # ESC
                    break

                time.sleep(0.033)

        except KeyboardInterrupt:
            print("\n[종료] Ctrl+C")

        finally:
            if self.px4_control_active and self.px4:
                print("\n[PX4 제어 비활성화 중...]")
                self.px4.disable_control()
                time.sleep(2)
            
            if self.px4:
                self.px4.stop()
            
            self.tag_detector.stop()
            cv2.destroyAllWindows()
            print(f"\n총 {self.frame_count} 프레임 처리됨")


def main():
    MODEL_PATH = "/home/rtx5080/s/ISAAC_LAB_DRONE/logs/sb3/Template-DroneLanding-v0/2026-01-27_09-06-42/model.zip"
    TAG_SIZE = 0.165
    USE_PX4 = True
    PX4_CONNECTION = "udp://:14540"

    if len(sys.argv) > 1:
        MODEL_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        TAG_SIZE = float(sys.argv[2])
    if len(sys.argv) > 3:
        USE_PX4 = sys.argv[3].lower() in ['true', '1', 'yes']

    print(f"\n모델: {MODEL_PATH}")
    print(f"태그 크기: {TAG_SIZE}m")
    print(f"PX4 사용: {USE_PX4}")
    if USE_PX4:
        print(f"PX4 연결: {PX4_CONNECTION}")

    if not RL_AVAILABLE:
        print("\n[ERROR] stable-baselines3 필요!")
        return
    
    if USE_PX4 and not MAVSDK_AVAILABLE:
        print("\n[ERROR] MAVSDK 필요: pip install mavsdk")
        return

    try:
        tester = RLOutputTester(
            model_path=MODEL_PATH, 
            tag_size=TAG_SIZE,
            use_px4=USE_PX4,
            px4_connection=PX4_CONNECTION
        )
        tester.run()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()