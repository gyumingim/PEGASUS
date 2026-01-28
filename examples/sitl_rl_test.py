#!/usr/bin/env python3
"""
PX4 SITL + 실제 카메라 RL 제어 테스트

사용법:
1. PX4 SITL 실행 (터미널 1):
   cd ~/PX4-Autopilot
   make px4_sitl jmavsim

2. 이 스크립트 실행 (터미널 2):
   python sitl_rl_test.py

실제 드론 사용 시:
   python sitl_rl_test.py --real
"""

import asyncio
import numpy as np
import cv2
import time
import sys
import argparse
from collections import deque

# 카메라 AprilTag 인식
from b import AprilTagLanding

# MAVSDK
from mavsdk import System
from mavsdk.offboard import OffboardError, Attitude

# RL
try:
    from stable_baselines3 import PPO
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    print("[WARN] stable-baselines3 필요")

try:
    import torch
except ImportError:
    torch = None


def quat_to_rotation_matrix(quat):
    """Quaternion [x,y,z,w] to 3x3 rotation matrix"""
    x, y, z, w = quat
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])


class IsaacLabToPX4Converter:
    """RL action → PX4 Attitude 변환"""

    def __init__(self):
        self.TRAIN_THRUST_TO_WEIGHT = 1.9
        self.DRONE_HOVER_THRUST = 0.5  # SITL/실제 드론에 맞게 조정

        self.TORQUE_TO_ANGLE_GAIN_ROLL = 7.0
        self.TORQUE_TO_ANGLE_GAIN_PITCH = 7.0
        self.TORQUE_TO_ANGLE_GAIN_YAW = 15.0

        self.MAX_ROLL = 25.0
        self.MAX_PITCH = 25.0

        self.integrated_yaw = 0.0
        self.prev_action = np.zeros(4)
        self.dt = 0.02

    def convert(self, action: np.ndarray) -> Attitude:
        alpha = 0.3
        filtered = alpha * action + (1 - alpha) * self.prev_action
        self.prev_action = filtered.copy()
        filtered = np.clip(filtered, -1.0, 1.0)

        thrust_action, roll_action, pitch_action, yaw_action = filtered

        # Thrust 변환
        isaac_ratio = self.TRAIN_THRUST_TO_WEIGHT * (thrust_action + 1.0) / 2.0
        px4_thrust = self.DRONE_HOVER_THRUST + (isaac_ratio - 1.0) * 0.3
        px4_thrust = np.clip(px4_thrust, 0.0, 1.0)

        # Angle 변환
        roll = np.clip(roll_action * self.TORQUE_TO_ANGLE_GAIN_ROLL, -self.MAX_ROLL, self.MAX_ROLL)
        pitch = np.clip(-pitch_action * self.TORQUE_TO_ANGLE_GAIN_PITCH, -self.MAX_PITCH, self.MAX_PITCH)

        # Yaw 적분
        self.integrated_yaw += yaw_action * self.TORQUE_TO_ANGLE_GAIN_YAW * self.dt
        self.integrated_yaw = (self.integrated_yaw + 180) % 360 - 180

        return Attitude(float(roll), float(pitch), float(self.integrated_yaw), float(px4_thrust))

    def reset(self):
        self.integrated_yaw = 0.0
        self.prev_action = np.zeros(4)


class DroneVisualizer:
    """간단한 2D 시각화"""

    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.trajectory = deque(maxlen=500)
        self.tag_history = deque(maxlen=100)

    def draw(self, drone_pos, drone_att, tag_pos_body, attitude_cmd, tag_detected):
        """
        drone_pos: [x, y, z] NED
        drone_att: [roll, pitch, yaw] degrees
        tag_pos_body: [x, y, z] body frame (카메라에서)
        attitude_cmd: Attitude 명령
        """
        img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 40

        # 좌표 변환 (NED → 화면)
        cx, cy = self.width // 2, self.height // 2
        scale = 50  # 1m = 50px

        # 드론 위치 (XY 평면, 위에서 본 뷰)
        dx = int(cx + drone_pos[1] * scale)  # East → 오른쪽
        dy = int(cy - drone_pos[0] * scale)  # North → 위

        self.trajectory.append((dx, dy))

        # 궤적 그리기
        for i in range(1, len(self.trajectory)):
            cv2.line(img, self.trajectory[i-1], self.trajectory[i], (100, 100, 100), 1)

        # 그리드
        for i in range(-10, 11):
            x = cx + i * scale
            y = cy + i * scale
            cv2.line(img, (x, 0), (x, self.height), (60, 60, 60), 1)
            cv2.line(img, (0, y), (self.width, y), (60, 60, 60), 1)

        # 원점 표시
        cv2.circle(img, (cx, cy), 5, (0, 255, 255), -1)
        cv2.putText(img, "Origin", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # 드론 그리기
        drone_color = (0, 255, 0) if tag_detected else (0, 0, 255)
        cv2.circle(img, (dx, dy), 15, drone_color, 2)

        # 드론 방향 (yaw)
        yaw_rad = np.radians(drone_att[2])
        arrow_len = 30
        ax = int(dx + arrow_len * np.sin(yaw_rad))
        ay = int(dy - arrow_len * np.cos(yaw_rad))
        cv2.arrowedLine(img, (dx, dy), (ax, ay), drone_color, 2, tipLength=0.3)

        # 타겟 위치 (body frame → world frame 추정)
        if tag_detected and tag_pos_body is not None:
            # 간단히 드론 기준 상대위치로 표시
            R = quat_to_rotation_matrix(euler_to_quat(drone_att[0], drone_att[1], drone_att[2]))
            tag_world = drone_pos + R @ tag_pos_body

            tx = int(cx + tag_world[1] * scale)
            ty = int(cy - tag_world[0] * scale)
            self.tag_history.append((tx, ty))

            # 타겟 마커
            cv2.drawMarker(img, (tx, ty), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(img, "TAG", (tx + 10, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 정보 표시
        y = 30
        cv2.putText(img, f"Drone NED: [{drone_pos[0]:.2f}, {drone_pos[1]:.2f}, {drone_pos[2]:.2f}]",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 25
        cv2.putText(img, f"Attitude: R={drone_att[0]:.1f} P={drone_att[1]:.1f} Y={drone_att[2]:.1f}",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 25
        cv2.putText(img, f"Height: {-drone_pos[2]:.2f}m (AGL)",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        y += 35
        cv2.putText(img, "--- PX4 CMD ---", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y += 20
        cv2.putText(img, f"Roll: {attitude_cmd.roll_deg:.1f} deg",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y += 20
        cv2.putText(img, f"Pitch: {attitude_cmd.pitch_deg:.1f} deg",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y += 20
        cv2.putText(img, f"Thrust: {attitude_cmd.thrust_value:.3f}",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 고도 바 (우측)
        bar_x = self.width - 50
        bar_top = 50
        bar_bottom = self.height - 50
        bar_height = bar_bottom - bar_top

        cv2.rectangle(img, (bar_x - 10, bar_top), (bar_x + 10, bar_bottom), (100, 100, 100), 2)

        # 현재 고도 (0~10m 범위)
        alt = -drone_pos[2]  # NED에서 z가 음수가 위
        alt_ratio = np.clip(alt / 10.0, 0, 1)
        alt_y = int(bar_bottom - alt_ratio * bar_height)
        cv2.rectangle(img, (bar_x - 8, alt_y), (bar_x + 8, bar_bottom), (0, 255, 0), -1)
        cv2.putText(img, f"{alt:.1f}m", (bar_x - 25, alt_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 타겟 감지 상태
        status = "TAG DETECTED" if tag_detected else "NO TAG"
        color = (0, 255, 0) if tag_detected else (0, 0, 255)
        cv2.putText(img, status, (self.width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return img


def euler_to_quat(roll, pitch, yaw, degrees=True):
    if degrees:
        roll, pitch, yaw = np.radians([roll, pitch, yaw])
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    return np.array([sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy, cr*cp*cy + sr*sp*sy])


class SITLController:
    """PX4 SITL/실제 드론 제어기"""

    def __init__(self, model_path: str, connection: str = "udp://:14540",
                 tag_size: float = 0.165, is_real: bool = False):
        self.connection = connection
        self.is_real = is_real
        self.drone = System()

        # RL 모델
        device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
        print(f"[RL] 모델 로딩: {model_path}")
        self.model = PPO.load(model_path, device=device)

        # 변환기
        self.converter = IsaacLabToPX4Converter()

        # 카메라
        print(f"[Camera] AprilTag 인식기 초기화 (size={tag_size}m)")
        self.camera = AprilTagLanding(tag_size=tag_size)

        # 시각화
        self.visualizer = DroneVisualizer()

        # 상태
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.attitude_euler = np.zeros(3)
        self.attitude_quat = np.array([0, 0, 0, 1])
        self.angular_velocity = np.zeros(3)

        self.running = False
        self.control_rate = 50
        self.dt = 1.0 / self.control_rate

    async def connect(self):
        print(f"[MAVSDK] 연결 중: {self.connection}")
        await self.drone.connect(system_address=self.connection)

        print("[MAVSDK] 연결 대기...")
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("[MAVSDK] 연결 완료!")
                break

    async def wait_ready(self):
        print("[MAVSDK] GPS/위치 추정 대기...")
        async for health in self.drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                print("[MAVSDK] 준비 완료!")
                return True
        return False

    async def _telemetry_loop(self):
        """텔레메트리 수신"""
        async def pos_loop():
            async for pv in self.drone.telemetry.position_velocity_ned():
                self.position = np.array([pv.position.north_m, pv.position.east_m, pv.position.down_m])
                self.velocity = np.array([pv.velocity.north_m_s, pv.velocity.east_m_s, pv.velocity.down_m_s])
                if not self.running:
                    break

        async def att_loop():
            async for att in self.drone.telemetry.attitude_euler():
                self.attitude_euler = np.array([att.roll_deg, att.pitch_deg, att.yaw_deg])
                if not self.running:
                    break

        async def quat_loop():
            async for q in self.drone.telemetry.attitude_quaternion():
                self.attitude_quat = np.array([q.x, q.y, q.z, q.w])
                if not self.running:
                    break

        async def rate_loop():
            async for r in self.drone.telemetry.attitude_angular_velocity_body():
                self.angular_velocity = np.array([r.roll_rad_s, r.pitch_rad_s, r.yaw_rad_s])
                if not self.running:
                    break

        await asyncio.gather(pos_loop(), att_loop(), quat_loop(), rate_loop())

    def _construct_observation(self, target_body: np.ndarray) -> np.ndarray:
        """16차원 observation"""
        R = quat_to_rotation_matrix(self.attitude_quat)
        R_inv = R.T

        lin_vel_b = R_inv @ self.velocity
        ang_vel_b = R_inv @ self.angular_velocity
        gravity_b = R_inv @ np.array([0, 0, -1.0])

        yaw = np.arctan2(2*(self.attitude_quat[3]*self.attitude_quat[2] +
                            self.attitude_quat[0]*self.attitude_quat[1]),
                         1 - 2*(self.attitude_quat[1]**2 + self.attitude_quat[2]**2))

        return np.concatenate([lin_vel_b, ang_vel_b, gravity_b, target_body, lin_vel_b, [yaw]]).astype(np.float32)

    async def run(self, duration: float = 120.0):
        """메인 제어 루프"""
        await self.connect()
        await self.wait_ready()

        # Arm & Takeoff
        print("[MAVSDK] Arming...")
        await self.drone.action.arm()

        print("[MAVSDK] Takeoff...")
        await self.drone.action.takeoff()
        await asyncio.sleep(5)

        # Offboard 시작
        await self.drone.offboard.set_attitude(Attitude(0, 0, 0, 0.5))
        print("[MAVSDK] Offboard 시작...")
        try:
            await self.drone.offboard.start()
        except OffboardError as e:
            print(f"[ERROR] Offboard 실패: {e}")
            return

        self.running = True
        self.converter.dt = self.dt

        # 텔레메트리 시작
        telemetry_task = asyncio.create_task(self._telemetry_loop())
        await asyncio.sleep(0.5)

        print(f"\n{'='*60}")
        print("RL 제어 시작! (ESC로 종료)")
        print(f"{'='*60}\n")

        start_time = time.time()
        frame_count = 0
        last_attitude_cmd = Attitude(0, 0, 0, 0.5)

        try:
            while self.running:
                frame_count += 1
                elapsed = time.time() - start_time

                if elapsed > duration:
                    print(f"[INFO] 시간 초과 ({duration}s)")
                    break

                # 카메라에서 AprilTag 인식
                tag_result = self.camera.get_relative_pose()
                tag_detected = tag_result['detected']
                tag_pos_body = None

                if tag_detected:
                    # 카메라 좌표 → Body frame
                    # 카메라: X=오른쪽, Y=아래, Z=앞
                    # Body (FLU): X=앞, Y=왼쪽, Z=위
                    # 카메라가 아래를 향한다고 가정
                    tag_pos_body = np.array([
                        tag_result['y'],    # body_x = cam_y
                        -tag_result['x'],   # body_y = -cam_x
                        -tag_result['z']    # body_z = -cam_z
                    ])

                    # Observation & RL
                    obs = self._construct_observation(tag_pos_body)
                    action, _ = self.model.predict(obs, deterministic=True)
                    if torch and isinstance(action, torch.Tensor):
                        action = action.cpu().numpy()
                    action = action.flatten()

                    # 변환
                    last_attitude_cmd = self.converter.convert(action)
                else:
                    # 태그 없으면 호버링
                    last_attitude_cmd = Attitude(0, 0, self.attitude_euler[2], 0.5)

                # PX4에 명령
                await self.drone.offboard.set_attitude(last_attitude_cmd)

                # 시각화
                vis_img = self.visualizer.draw(
                    self.position, self.attitude_euler,
                    tag_pos_body, last_attitude_cmd, tag_detected
                )

                # 카메라 뷰와 합치기
                cam_img = tag_result.get('image')
                if cam_img is not None:
                    cam_small = cv2.resize(cam_img, (320, 240))
                    vis_img[10:250, self.visualizer.width-330:self.visualizer.width-10] = cam_small

                cv2.imshow('SITL RL Control', vis_img)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break

                # 로그 (매 50프레임)
                if frame_count % 50 == 0:
                    print(f"[{elapsed:.1f}s] pos=[{self.position[0]:.2f}, {self.position[1]:.2f}, {-self.position[2]:.2f}m] "
                          f"tag={'OK' if tag_detected else 'NO'} "
                          f"cmd=[R:{last_attitude_cmd.roll_deg:.1f}, P:{last_attitude_cmd.pitch_deg:.1f}, T:{last_attitude_cmd.thrust_value:.2f}]")

                await asyncio.sleep(self.dt)

        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.running = False
            telemetry_task.cancel()

            print("[MAVSDK] Offboard 중지...")
            try:
                await self.drone.offboard.stop()
            except:
                pass

            print("[MAVSDK] 착륙...")
            await self.drone.action.land()
            await asyncio.sleep(5)

            self.camera.stop()
            cv2.destroyAllWindows()
            print("[완료]")


async def main():
    parser = argparse.ArgumentParser(description='PX4 SITL/Real Drone RL Test')
    parser.add_argument('--real', action='store_true', help='실제 드론 모드')
    parser.add_argument('--connection', type=str, default='udp://:14540', help='연결 주소')
    parser.add_argument('--model', type=str,
                       default='/home/rtx5080/s/ISAAC_LAB_DRONE/logs/sb3/Template-DroneLanding-v0/2026-01-27_09-06-42/model.zip',
                       help='RL 모델 경로')
    parser.add_argument('--tag-size', type=float, default=0.165, help='AprilTag 크기 (m)')
    parser.add_argument('--duration', type=float, default=120, help='최대 실행 시간 (s)')
    args = parser.parse_args()

    if args.real:
        print("\n" + "!"*60)
        print("!!! 실제 드론 모드 !!!")
        print("!"*60 + "\n")
        confirm = input("정말 실제 드론으로 비행하시겠습니까? (yes/no): ")
        if confirm.lower() != 'yes':
            print("취소됨")
            return

    print("\n" + "="*60)
    print("PX4 SITL RL 제어 테스트" if not args.real else "실제 드론 RL 제어")
    print("="*60)
    print(f"  연결: {args.connection}")
    print(f"  모델: {args.model}")
    print(f"  태그 크기: {args.tag_size}m")
    print(f"  제어 시간: {args.duration}s")
    print("="*60)
    print("\n먼저 PX4 SITL을 실행하세요:")
    print("  cd ~/PX4-Autopilot && make px4_sitl jmavsim")
    print("\n준비되면 Enter를 누르세요...")
    input()

    if not RL_AVAILABLE:
        print("[ERROR] stable-baselines3 필요!")
        return

    controller = SITLController(
        model_path=args.model,
        connection=args.connection,
        tag_size=args.tag_size,
        is_real=args.real
    )

    await controller.run(duration=args.duration)


if __name__ == "__main__":
    asyncio.run(main())
