#!/usr/bin/env python
"""
Pegasus 드론 착륙 시뮬레이션 (강화학습 모델 사용) - 변환 레이어 적용 버전

핵심 변경사항:
- Isaac Lab (토크 제어) → PX4 (Attitude 제어) 변환 레이어 추가
- Action 인덱스 수정: [0]=thrust, [1]=roll, [2]=pitch, [3]=yaw
- 토크 → 각도 변환 로직 구현
"""

import carb
from isaacsim import SimulationApp

# Isaac Sim 시작
simulation_app = SimulationApp({"headless": False})

import omni.timeline
import omni
from omni.isaac.core.world import World
import torch
import numpy as np
from scipy.spatial.transform import Rotation
import asyncio
import threading

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig

from pxr import Sdf, UsdShade, UsdGeom, Gf, UsdLux

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

# OpenCV (ArUco 감지)
try:
    import cv2
    import cv2.aruco as aruco
    ARUCO_AVAILABLE = True
except ImportError:
    ARUCO_AVAILABLE = False
    print("[WARN] OpenCV not available")



class IsaacLabToPX4Converter:
    """
    Isaac Lab 토크 명령 → PX4 Attitude 명령 변환 레이어

    Isaac Lab 학습 환경:
    - action[0]: thrust 비율 [-1, 1] → [0, 1.9] × 무게
    - action[1]: roll 토크 [-1, 1] → [-0.002, +0.002] N·m
    - action[2]: pitch 토크 [-1, 1] → [-0.002, +0.002] N·m
    - action[3]: yaw 토크 [-1, 1] → [-0.002, +0.002] N·m

    PX4 Attitude 명령:
    - roll: 목표 각도 (degrees)
    - pitch: 목표 각도 (degrees)
    - yaw: 목표 각도 (degrees)
    - thrust: 정규화된 추력 [0, 1]
    """

    def __init__(self):
        # ========== Isaac Lab 학습 파라미터 ==========
        self.TRAIN_MASS = 0.033  # Crazyflie 질량 (kg)
        self.TRAIN_THRUST_TO_WEIGHT = 1.9  # 학습 시 thrust-to-weight
        self.TRAIN_MOMENT_SCALE = 0.002  # 학습 시 moment scale (N·m)
        self.TRAIN_GRAVITY = 9.81

        # Crazyflie 관성모멘트 (kg·m²)
        self.TRAIN_Ixx = 1.4e-5
        self.TRAIN_Iyy = 1.4e-5
        self.TRAIN_Izz = 2.17e-5

        # ========== Iris (Pegasus) 물리 파라미터 ==========
        self.IRIS_MASS = 1.5  # Iris 질량 (kg)
        self.IRIS_GRAVITY = 9.81
        self.IRIS_HOVER_THRUST = 0.65  # 캘리브레이션으로 측정된 호버 추력

        # Iris 관성모멘트 (kg·m²) - 대략적 추정값
        self.IRIS_Ixx = 0.029
        self.IRIS_Iyy = 0.029
        self.IRIS_Izz = 0.055

        # ========== 변환 비율 계산 ==========
        # 관성모멘트 비율: Iris가 ~2000배 더 큼
        self.INERTIA_RATIO_ROLL = self.IRIS_Ixx / self.TRAIN_Ixx
        self.INERTIA_RATIO_PITCH = self.IRIS_Iyy / self.TRAIN_Iyy
        self.INERTIA_RATIO_YAW = self.IRIS_Izz / self.TRAIN_Izz

        # 질량 비율
        self.MASS_RATIO = self.IRIS_MASS / self.TRAIN_MASS

        # ========== 튜닝 파라미터 ==========
        # 토크 → 각도 변환 게인 (실험적으로 조정 필요)
        # 기본 공식: angle = k * torque / I * dt^2 / 2
        # 단순화: angle = TORQUE_TO_ANGLE_GAIN * normalized_torque

        self.TORQUE_TO_ANGLE_GAIN_ROLL = 7.0   # degrees per normalized action
        self.TORQUE_TO_ANGLE_GAIN_PITCH = 7.0  # degrees per normalized action
        self.TORQUE_TO_ANGLE_GAIN_YAW = 15.0    # degrees per normalized action (yaw는 더 크게)

        # 최대 각도 제한
        self.MAX_ROLL = 35.0   # degrees
        self.MAX_PITCH = 35.0  # degrees

        # ========== 상태 변수 (적분용) ==========
        self.integrated_yaw = 0.0

        # 이전 값 (필터링용)
        self.prev_action = np.zeros(4)

        # 시뮬레이션 dt
        self.dt = 0.02  # 50Hz

        # 디버그
        self.debug_count = 0

        print("\n" + "="*70)
        print("Isaac Lab -> PX4 변환 레이어 초기화")
        print("="*70)
        print(f"  학습 환경 (Crazyflie):")
        print(f"    Mass: {self.TRAIN_MASS} kg")
        print(f"    Ixx/Iyy: {self.TRAIN_Ixx} kg*m^2")
        print(f"    Moment scale: {self.TRAIN_MOMENT_SCALE} N*m")
        print(f"  실행 환경 (Iris):")
        print(f"    Mass: {self.IRIS_MASS} kg")
        print(f"    Ixx/Iyy: {self.IRIS_Ixx} kg*m^2")
        print(f"    Hover thrust: {self.IRIS_HOVER_THRUST} (calibrated)")
        print(f"  변환 비율:")
        print(f"    Mass ratio: {self.MASS_RATIO:.1f}x")
        print(f"    Inertia ratio (roll/pitch): {self.INERTIA_RATIO_ROLL:.1f}x")
        print(f"  튜닝 게인:")
        print(f"    Roll/Pitch angle gain: {self.TORQUE_TO_ANGLE_GAIN_ROLL} deg")
        print(f"    Yaw angle gain: {self.TORQUE_TO_ANGLE_GAIN_YAW} deg")
        print("="*70 + "\n")

    def convert(self, isaac_action: np.ndarray, current_attitude: np.ndarray,
                current_angular_vel: np.ndarray) -> Attitude:
        """
        Isaac Lab 액션을 PX4 Attitude 명령으로 변환

        Args:
            isaac_action: [thrust, roll_torque, pitch_torque, yaw_torque] 범위 [-1, 1]
            current_attitude: 현재 오일러 각도 [roll, pitch, yaw] (degrees)
            current_angular_vel: 현재 각속도 [p, q, r] (rad/s)

        Returns:
            Attitude(roll_deg, pitch_deg, yaw_deg, thrust)
        """

        # ========== 1. 액션 필터링 (노이즈 제거) ==========
        alpha = 0.3  # 필터 계수 (0.3~0.5 권장)
        filtered_action = alpha * isaac_action + (1 - alpha) * self.prev_action
        self.prev_action = filtered_action.copy()

        # 클리핑
        filtered_action = np.clip(filtered_action, -1.0, 1.0)

        # ========== 2. 액션 분리 (올바른 인덱스!) ==========
        thrust_action = filtered_action[0]   # 추력
        roll_action = filtered_action[1]     # roll 토크 (인덱스 1!)
        pitch_action = filtered_action[2]    # pitch 토크 (인덱스 2!)
        yaw_action = filtered_action[3]      # yaw 토크

        # ========== 3. 추력 변환 (캘리브레이션 적용) ==========
        # Isaac Lab: thrust_ratio = 1.9 * (action + 1) / 2
        # action = 0 → ratio = 0.95 (거의 호버링)
        # action = 0.053 → ratio ≈ 1.0 (정확한 호버링)

        # Isaac Lab 추력 비율 계산
        isaac_thrust_ratio = self.TRAIN_THRUST_TO_WEIGHT * (thrust_action + 1.0) / 2.0

        # PX4 추력으로 변환 (캘리브레이션된 호버 추력 기준)
        # isaac_ratio = 1.0 (호버링) → px4 = IRIS_HOVER_THRUST
        # 스케일: (isaac_ratio - 1.0) 당 추력 변화량
        thrust_scale = 0.5  # 튜닝 가능: isaac_ratio 1단위 변화당 px4 thrust 변화
        px4_thrust = self.IRIS_HOVER_THRUST + (isaac_thrust_ratio - 1.0) * thrust_scale
        px4_thrust = np.clip(px4_thrust, 0.0, 1.0)

        # ========== 4. 토크 → 목표 각도 변환 ==========
        # 방법 1: 직접 매핑 (토크 크기를 목표 각도로 해석)
        # 토크가 클수록 더 큰 각도를 목표로 함

        # 정규화된 토크 [-1, 1]을 각도로 변환
        target_roll = roll_action * self.TORQUE_TO_ANGLE_GAIN_ROLL
        target_pitch = pitch_action * self.TORQUE_TO_ANGLE_GAIN_PITCH

        # ========== 5. Yaw 처리 (적분 방식) ==========
        # Yaw는 절대 각도가 아닌 변화율로 해석
        yaw_rate = yaw_action * self.TORQUE_TO_ANGLE_GAIN_YAW  # deg/s
        self.integrated_yaw += yaw_rate * self.dt

        # Yaw 정규화 [-180, 180]
        self.integrated_yaw = (self.integrated_yaw + 180) % 360 - 180

        # ========== 6. 각도 제한 ==========
        target_roll = np.clip(target_roll, -self.MAX_ROLL, self.MAX_ROLL)
        target_pitch = np.clip(target_pitch, -self.MAX_PITCH, self.MAX_PITCH)

        # ========== 7. 부호 조정 (좌표계 맞춤) ==========
        # Isaac Lab과 PX4의 좌표계 차이 보정
        final_roll = target_roll  # 부호 조정 (실험적으로 확인 필요)
        final_pitch = -target_pitch
        final_yaw = self.integrated_yaw

        # ========== 8. 디버깅 ==========
        if self.debug_count % 50 == 0:
            print(f"\n{'='*70}")
            print(f"Converter Debug (step {self.debug_count})")
            print(f"{'='*70}")
            print(f"  Isaac action: [{thrust_action:+.3f}, {roll_action:+.3f}, "
                  f"{pitch_action:+.3f}, {yaw_action:+.3f}]")
            print(f"  Isaac thrust ratio: {isaac_thrust_ratio:.3f} (hover=1.0)")
            print(f"  PX4 thrust: {px4_thrust:.3f} (hover={self.IRIS_HOVER_THRUST:.2f}, calibrated)")
            print(f"  Target angle: roll={target_roll:+.1f} deg, pitch={target_pitch:+.1f} deg")
            print(f"  Final cmd: roll={final_roll:+.1f} deg, pitch={final_pitch:+.1f} deg, "
                  f"yaw={final_yaw:+.1f} deg, thrust={px4_thrust:.3f}")
            print(f"  Current attitude: roll={current_attitude[0]:+.1f} deg, "
                  f"pitch={current_attitude[1]:+.1f} deg, yaw={current_attitude[2]:+.1f} deg")
        self.debug_count += 1

        return Attitude(
            float(final_roll),
            float(final_pitch),
            float(final_yaw),
            float(px4_thrust)
        )

    def reset(self):
        """상태 초기화"""
        self.integrated_yaw = 0.0
        self.prev_action = np.zeros(4)
        self.debug_count = 0
        print("[Converter] Reset")


class RLDroneLandingController:
    """강화학습 기반 드론 착륙 제어기 (PX4 Offboard용) - 변환 레이어 적용"""

    # ============================================================
    # 튜닝 파라미터
    # ============================================================

    DEBUG_MODE = True

    # 감지 모드 설정
    # True: AprilTag 인식으로 로버 위치 추정
    # False: Ground Truth (시뮬레이터에서 로버 위치 직접 사용)
    USE_APRILTAG = True

    # Observation 스케일
    VEL_SCALE = 1.0
    ANG_VEL_SCALE = 1.0

    # 목표 위치 오프셋 (튜닝용)
    TARGET_X_OFFSET = 0.9  # X 방향 오프셋 (m)
    TARGET_Y_OFFSET = 0  # Y 방향 오프셋 (m)

    # ============================================================
    # 바람은 PegasusRLLandingApp에서 실제 물리로 적용됨
    # ============================================================

    def __init__(self, rover_initial_pos, rover_velocity, model_path, device="cuda", detection_callback=None):
        self.rl_device = device
        self.vehicle = None

        # 디버그 카운터
        self._obs_debug_count = 0
        self._action_debug_count = 0

        # 변환 레이어 초기화
        self.converter = IsaacLabToPX4Converter()

        # 로버 설정
        self.rover_pos = np.array(rover_initial_pos, dtype=np.float32)
        self.rover_vel = np.array(rover_velocity, dtype=np.float32)

        # RL 모델 로드
        if RL_AVAILABLE:
            print(f"[RL] Loading model from: {model_path}")
            self.model = PPO.load(model_path, device=device)
            print(f"[RL] Model loaded successfully on {device}")
        else:
            raise ImportError("stable-baselines3 not installed!")

        # 물리 파라미터
        self.gravity = 9.81

        # 상태
        self.dt = 0.02
        self.time = 0.0
        self.estimated_rover_pos = None
        self.detection_callback = detection_callback
        self._state = None

        # 착륙 상태
        # 로버 표면 높이 = 로버 중심(-0.4) + 큐브 반높이(0.5) = 0.1m
        # 약간 위에서 착륙 (0.15m 정도 여유)
        self.landing_height = rover_initial_pos[2] + 0.5 + 0.05  # 로버 표면 + 5cm
        print(f"[Landing] Target height: {self.landing_height:.2f}m (rover surface + 5cm)")

        self.takeoff_mode = False
        self.takeoff_target_pos = None

        # 목표 위치 (world frame)
        if self.USE_APRILTAG:
            self.desired_pos_w = None  # AprilTag 감지 후 설정됨
        else:  # ground_truth
            self.desired_pos_w = np.array(rover_initial_pos, dtype=np.float32)
            self.desired_pos_w[2] = self.landing_height

        print("\n" + "="*60)
        print("RL Controller 초기화 완료")
        print("="*60)
        print(f"  DEBUG_MODE: {self.DEBUG_MODE}")
        print(f"  USE_APRILTAG: {self.USE_APRILTAG}")
        if self.USE_APRILTAG:
            print(f"    → AprilTag 인식으로 로버 위치 추정")
        else:
            print(f"    → Ground Truth (시뮬레이터 직접 위치)")
        print(f"  변환 레이어: IsaacLabToPX4Converter 사용")
        print("="*60 + "\n")

    def update(self, dt: float):
        self.dt = dt
        self.time += dt
        self.converter.dt = dt  # 변환 레이어에도 dt 전달

        if self.USE_APRILTAG:
            # AprilTag 인식 모드: 감지된 위치만 사용 (Ground Truth 사용 안함)
            if self.estimated_rover_pos is not None:
                if self.desired_pos_w is None:
                    self.desired_pos_w = self.estimated_rover_pos.copy()
                    self.desired_pos_w[2] = self.landing_height
                else:
                    self.desired_pos_w[:2] = self.estimated_rover_pos[:2]
                    self.desired_pos_w[2] = self.landing_height
            # 태그 미감지 시 desired_pos_w는 None 유지 → 호버링
        else:
            # Ground Truth 모드: 시뮬레이터 로버 위치 직접 사용
            if self.desired_pos_w is None:
                self.desired_pos_w = np.array(self.rover_pos, dtype=np.float32)
                self.desired_pos_w[2] = self.landing_height
            else:
                self.desired_pos_w[:2] = self.rover_pos[:2]
                self.desired_pos_w[2] = self.landing_height

        # XY 오프셋 적용 (목표 위치 보정)
        if self.desired_pos_w is not None:
            self.desired_pos_w[0] += self.TARGET_X_OFFSET
            self.desired_pos_w[1] += self.TARGET_Y_OFFSET

            

    def set_rover_pos(self, pos):
        """App에서 로버 위치를 직접 설정 (sync용)"""
        self.rover_pos[:] = pos

    def get_attitude_rate(self):
        """RL 모델로 액션 결정 후 Attitude 반환"""
        state = self._get_vehicle_state()

        # Takeoff 모드
        if self.takeoff_mode:
            if self.takeoff_target_pos is not None:
                current_height = np.array(self._state.position)[2]
                target_height = self.takeoff_target_pos[2]

                if current_height >= target_height - 0.2:
                    print(f"[Takeoff] 목표 높이 도달!")
                    self.takeoff_mode = False
                    return Attitude(0.0, 0.0, 0.0, 0.55)

                if self._action_debug_count % 50 == 0:
                    print(f"[Takeoff] 상승 중... {current_height:.2f}m / {target_height:.2f}m")

                self._action_debug_count += 1
                return Attitude(0.0, 0.0, 0.0, 0.7)
            else:
                return Attitude(0.0, 0.0, 0.0, 0.6)

        # AprilTag 모드에서 아직 마커 감지 못했으면 호버링
        if self.USE_APRILTAG and self.desired_pos_w is None:
            if self._action_debug_count % 50 == 0:
                print("[RL] AprilTag 미감지 - 호버링 유지")
            return Attitude(0.0, 0.0, 0.0, 0.6)

        # Observation 구성
        obs = self._construct_observation(state)

        # RL 모델로 액션 예측
        action, _states = self.model.predict(obs, deterministic=True)

        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        action = action.flatten()

        # 변환 레이어를 통해 PX4 명령으로 변환
        current_attitude = self._get_euler_from_state(state)
        current_angular_vel = np.array(state.angular_velocity, dtype=np.float32)

        attitude_cmd = self.converter.convert(action, current_attitude, current_angular_vel)

        self._action_debug_count += 1
        return attitude_cmd

    def _get_euler_from_state(self, state):
        """상태에서 오일러 각도 추출 (degrees)"""
        quat_xyzw = np.array(state.attitude, dtype=np.float32)
        r = Rotation.from_quat(quat_xyzw)
        euler = r.as_euler('XYZ', degrees=True)
        return euler  # [roll, pitch, yaw]

    def _construct_observation(self, state):
        """Isaac Lab 환경과 동일한 16차원 observation 구성"""

        pos = np.array(state.position, dtype=np.float32)
        lin_vel = np.array(state.linear_velocity, dtype=np.float32)
        ang_vel = np.array(state.angular_velocity, dtype=np.float32)

        quat_xyzw = np.array(state.attitude, dtype=np.float32)
        R = Rotation.from_quat(quat_xyzw)

        # 1. 드론 속도 (body frame)
        # 실제 물리 바람이 적용되므로 lin_vel에 이미 바람 효과가 반영됨
        lin_vel_b = R.inv().apply(lin_vel)
        lin_vel_b = lin_vel_b * self.VEL_SCALE

        # 2. 각속도 (body frame)
        ang_vel_b = R.inv().apply(ang_vel)

        # 3. 중력 방향 (body frame)
        gravity_world = np.array([0, 0, -1.0], dtype=np.float32)
        gravity_b = R.inv().apply(gravity_world)

        # 4. 목표 위치 (body frame)
        if self.desired_pos_w is not None:
            goal_rel_world = self.desired_pos_w - pos
            desired_pos_b = R.inv().apply(goal_rel_world)
        else:
            desired_pos_b = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # 5. 상대 속도 (body frame)
        # 실제 물리 바람이 적용되므로 lin_vel에 이미 바람 효과가 반영됨
        if self.USE_APRILTAG:
            # AprilTag 모드: 로버 속도를 0으로 가정 (GT 사용 안함)
            rel_vel_world = lin_vel
        else:
            # Ground Truth 모드: 실제 로버 속도 사용
            rel_vel_world = lin_vel - self.rover_vel
        rel_vel_b = R.inv().apply(rel_vel_world)
        rel_vel_b = rel_vel_b * self.VEL_SCALE

        # 6. Yaw 각도
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        current_yaw = np.arctan2(
            2.0 * (quat_wxyz[0]*quat_wxyz[3] + quat_wxyz[1]*quat_wxyz[2]),
            1.0 - 2.0 * (quat_wxyz[2]**2 + quat_wxyz[3]**2)
        )

        # 디버깅 출력
        mode_str = "AprilTag" if self.USE_APRILTAG else "GroundTruth"
        if (self.DEBUG_MODE or self._obs_debug_count < 5) and self._obs_debug_count % 50 == 1:
            print(f"\n{'='*70}")
            print(f"Observation Debug (step {self._obs_debug_count}) | Mode: {mode_str}")
            print(f"{'='*70}")
            print(f"  Drone pos (world):    [{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}]")
            if not self.USE_APRILTAG:
                print(f"  Rover pos (GT):       [{self.rover_pos[0]:6.2f}, {self.rover_pos[1]:6.2f}, {self.rover_pos[2]:6.2f}]")
            if self.USE_APRILTAG and self.estimated_rover_pos is not None:
                print(f"  Estimated pos (tag):  [{self.estimated_rover_pos[0]:6.2f}, {self.estimated_rover_pos[1]:6.2f}, {self.estimated_rover_pos[2]:6.2f}]")
            if self.desired_pos_w is not None:
                goal_rel_world = self.desired_pos_w - pos
                print(f"  Desired pos (world):  [{self.desired_pos_w[0]:6.2f}, {self.desired_pos_w[1]:6.2f}, {self.desired_pos_w[2]:6.2f}]")
                print(f"  Goal rel (world):     [{goal_rel_world[0]:6.2f}, {goal_rel_world[1]:6.2f}, {goal_rel_world[2]:6.2f}] (norm: {np.linalg.norm(goal_rel_world):.2f}m)")
            else:
                print(f"  Desired pos (world):  None (waiting for AprilTag)")

            print(f"  Goal rel (body):      [{desired_pos_b[0]:6.2f}, {desired_pos_b[1]:6.2f}, {desired_pos_b[2]:6.2f}]")
            print(f"  Lin vel (body):       [{lin_vel_b[0]:6.2f}, {lin_vel_b[1]:6.2f}, {lin_vel_b[2]:6.2f}]")
            print(f"  Ang vel (body):       [{ang_vel_b[0]:6.2f}, {ang_vel_b[1]:6.2f}, {ang_vel_b[2]:6.2f}]")
            print(f"  Gravity (body):       [{gravity_b[0]:6.2f}, {gravity_b[1]:6.2f}, {gravity_b[2]:6.2f}]")
            print(f"  Yaw: {np.degrees(current_yaw):6.1f} deg")
        self._obs_debug_count += 1

        # 16차원 observation
        obs = np.concatenate([
            lin_vel_b,        # 3
            ang_vel_b,        # 3
            gravity_b,        # 3
            desired_pos_b,    # 3
            rel_vel_b,        # 3
            [current_yaw]     # 1
        ])

        return obs.astype(np.float32)

    def update_estimator(self, marker_pos_world):
        """태그 감지 결과 업데이트"""
        self.estimated_rover_pos = marker_pos_world

    def update_sensor(self, sensor_type: str, sensor_data: dict):
        pass

    def update_state(self, state: dict):
        self._state = state

    def start(self):
        print("[RL Controller] Started")

    def stop(self):
        print(f"[RL Controller] Stopped")

    def reset(self):
        self.time = 0.0
        self.estimated_rover_pos = None
        self._obs_debug_count = 0
        self._action_debug_count = 0
        self.converter.reset()
        print("[RL Controller] Reset")

    def update_graphical_sensor(self, sensor_data: dict):
        pass

    def _get_vehicle_state(self):
        if hasattr(self, '_state') and self._state is not None:
            return self._state

        class DummyState:
            def __init__(self):
                self.position = np.zeros(3, dtype=np.float32)
                self.linear_velocity = np.zeros(3, dtype=np.float32)
                self.attitude = np.array([0, 0, 0, 1], dtype=np.float32)
                self.angular_velocity = np.zeros(3, dtype=np.float32)

        return DummyState()

    def set_takeoff_target(self, target_pos):
        self.takeoff_mode = True
        self.takeoff_target_pos = np.array(target_pos, dtype=np.float32)
        print(f"[Takeoff] 목표 설정: {target_pos}")


class PegasusRLLandingApp:
    """Pegasus RL 착륙 시뮬레이션 앱"""

    def __init__(self, model_path):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        self.last_detection_time = 0.0

        self.flight_phase = "TAKEOFF"
        self.takeoff_target_height = 2.0
        self.takeoff_complete = False

        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])

        # 로버 설정
        self.rover_pos = np.array([0.0, 0.0, -0.4], dtype=np.float32)
        self.rover_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # RL 제어기 생성
        self.controller = RLDroneLandingController(
            self.rover_pos.copy(),
            self.rover_vel.copy(),
            model_path=model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            detection_callback=self._on_detection
        )

        # 드론 생성
        config = MultirotorConfig()
        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": 0,
            "px4_autolaunch": True,
            "px4_dir": self.pg.px4_path,
            "px4_vehicle_model": self.pg.px4_default_airframe
        })
        config.backends = [PX4MavlinkBackend(mavlink_config)]

        initial_pos = [-2.5, -2.5, 0.5]

        print(f"[Init] Drone starting at: {initial_pos}")
        print(f"[Init] Rover at: {self.rover_pos}")

        self.drone = Multirotor(
            "/World/Drone",
            ROBOTS['Iris'],
            0,
            initial_pos,
            Rotation.from_euler("XYZ", [0, 0, 0], degrees=True).as_quat(),
            config=config
        )

        self.stop_sim = False

        # ========== 바람 물리 설정 (Isaac Lab과 동기화) ==========
        self.wind_enabled = True
        self.wind_max_speed = 0.0  # 최대 바람 속도 (m/s)
        # 랜덤 바람 생성: XY는 -10~10, Z는 항상 0
        random_wind_xy = (np.random.rand(2) * 2 - 1) * self.wind_max_speed
        self.wind_velocity = np.array([random_wind_xy[0], random_wind_xy[1], 0.0], dtype=np.float32)

        # 공기역학 파라미터 (Isaac Lab과 동기화)
        self.air_density = 1.225  # kg/m³ (해수면 기준)
        self.drag_coefficient = 1.0  # 항력 계수
        self.drone_cross_section = 0.1  # 단면적 m²

        print(f"[Wind] Enabled: {self.wind_enabled}")
        print(f"[Wind] Random velocity: ({self.wind_velocity[0]:.2f}, {self.wind_velocity[1]:.2f}, 0.0) m/s")
        print(f"[Wind] Speed: {np.linalg.norm(self.wind_velocity):.2f} m/s")

        self._add_lighting()
        self._create_rover()
        self._setup_camera()

        if ARUCO_AVAILABLE:
            self._init_apriltag()

        # 카메라 뷰 저장 경로
        self.camera_view_path = "/tmp/drone_camera_view.png"
        print(f"[Camera] View saved to: {self.camera_view_path}")

        self.world.reset()

        self.step_count = 0
        self.detection_count = 0
        self.last_saved_frame = -1
        self.last_detection_time = 0.0

        # 마지막 감지 데이터 저장 (태그 안 보일 때도 표시용)
        self.last_marker_in_world = None
        self.last_marker_in_body = None
        self.last_tvec = None
        self.last_tag_center_px = None  # 마지막 태그 중심 픽셀 좌표

    def _add_lighting(self):
        stage = omni.usd.get_context().get_stage()

        distant_light_path = "/World/DistantLight"
        distant_light = UsdLux.DistantLight.Define(stage, distant_light_path)
        distant_light.CreateIntensityAttr(5000.0)
        distant_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 0.95))
        distant_light.CreateAngleAttr(0.53)

        xform = UsdGeom.Xformable(distant_light)
        xform.ClearXformOpOrder()
        rotate_op = xform.AddRotateXYZOp()
        rotate_op.Set(Gf.Vec3f(-45, 45, 0))

        dome_light_path = "/World/DomeLight"
        dome_light = UsdLux.DomeLight.Define(stage, dome_light_path)
        dome_light.CreateIntensityAttr(1000.0)
        dome_light.CreateColorAttr(Gf.Vec3f(0.9, 0.95, 1.0))

        print("[Lighting] Added: DistantLight + DomeLight")

    def _create_rover(self):
        stage = omni.usd.get_context().get_stage()
        from pxr import UsdGeom, UsdPhysics

        rover_path = "/World/Rover"
        xform = UsdGeom.Xform.Define(stage, rover_path)

        cube_path = rover_path + "/Cube"
        cube = UsdGeom.Cube.Define(stage, cube_path)
        cube.GetSizeAttr().Set(1)

        cube_mtl_path = Sdf.Path(cube_path + "_Material")
        cube_mtl = UsdShade.Material.Define(stage, cube_mtl_path)
        cube_shader = UsdShade.Shader.Define(stage, cube_mtl_path.AppendPath("Shader"))
        cube_shader.CreateIdAttr("UsdPreviewSurface")
        cube_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.5, 0.5, 0.5))
        cube_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        cube_mtl.CreateSurfaceOutput().ConnectToSource(cube_shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(cube_mtl)

        rigid_api = UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
        rigid_api.CreateKinematicEnabledAttr(True)
        collision_api = UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

        xform_ops = xform.AddTranslateOp()
        xform_ops.Set(Gf.Vec3d(float(self.rover_pos[0]), float(self.rover_pos[1]), float(self.rover_pos[2])))

        self._add_apriltag_texture()

        light_path = rover_path + "/SpotLight"
        spot_light = UsdLux.SphereLight.Define(stage, light_path)
        spot_light.CreateIntensityAttr(2000.0)
        spot_light.CreateRadiusAttr(0.05)
        spot_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

        light_xform = UsdGeom.Xformable(spot_light)
        light_translate = light_xform.AddTranslateOp()
        light_translate.Set(Gf.Vec3d(0, 0, 0.5))

        print(f"[Rover] Created at {self.rover_pos}")

    def _add_apriltag_texture(self):
        stage = omni.usd.get_context().get_stage()

        mesh_path = "/World/Rover/TagMesh"
        mesh = UsdGeom.Mesh.Define(stage, mesh_path)

        half = 0.5
        mesh.GetPointsAttr().Set([
            Gf.Vec3f(-half, -half, 0),
            Gf.Vec3f(half, -half, 0),
            Gf.Vec3f(half, half, 0),
            Gf.Vec3f(-half, half, 0)
        ])
        mesh.GetFaceVertexCountsAttr().Set([4])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
        mesh.GetNormalsAttr().Set([Gf.Vec3f(0, 0, 1)] * 4)
        mesh.SetNormalsInterpolation("vertex")

        texcoords = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        )
        texcoords.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])

        xform = UsdGeom.Xformable(mesh)
        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(0, 0, 0.51))

        tag_image_path = self._generate_apriltag_image()

        mtl_path = Sdf.Path(mesh_path + "_Material")
        mtl = UsdShade.Material.Define(stage, mtl_path)

        shader = UsdShade.Shader.Define(stage, mtl_path.AppendPath("Shader"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.1)

        st_reader = UsdShade.Shader.Define(stage, mtl_path.AppendPath("stReader"))
        st_reader.CreateIdAttr("UsdPrimvarReader_float2")
        st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

        diffuse_tex = UsdShade.Shader.Define(stage, mtl_path.AppendPath("DiffuseTexture"))
        diffuse_tex.CreateIdAttr("UsdUVTexture")
        diffuse_tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tag_image_path)
        diffuse_tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            st_reader.ConnectableAPI(), "result"
        )

        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            diffuse_tex.ConnectableAPI(), "rgb"
        )
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            diffuse_tex.ConnectableAPI(), "rgb"
        )

        mtl.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(mtl)

        print(f"[Rover] AprilTag texture added")

    def _generate_apriltag_image(self):
        if not ARUCO_AVAILABLE:
            return "/tmp/dummy_tag.png"

        apriltag_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
        tag_size = 512
        border_bits = 1

        tag_image = np.zeros((tag_size, tag_size), dtype=np.uint8)
        tag_image = aruco.generateImageMarker(apriltag_dict, 0, tag_size, tag_image, border_bits)

        full_size = 600
        full_image = np.ones((full_size, full_size), dtype=np.uint8) * 255
        offset = (full_size - tag_size) // 2
        full_image[offset:offset+tag_size, offset:offset+tag_size] = tag_image

        output_path = "/tmp/apriltag_36h11_id0.png"
        cv2.imwrite(output_path, full_image)

        return output_path

    def _setup_camera(self):
        stage = omni.usd.get_context().get_stage()

        camera_path = "/World/Drone/body/Camera"
        camera_prim = UsdGeom.Camera.Define(stage, camera_path)

        # 일반 카메라 FOV (~70도)
        camera_prim.GetFocalLengthAttr().Set(24.0)
        camera_prim.GetHorizontalApertureAttr().Set(33.6)
        camera_prim.GetVerticalApertureAttr().Set(18.9)
        camera_prim.GetFocusDistanceAttr().Set(1000.0)
        camera_prim.GetFStopAttr().Set(0.0)
        camera_prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10000.0))

        xform = UsdGeom.Xformable(camera_prim)
        xform.ClearXformOpOrder()
        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(0, 0, -0.11))

        if ARUCO_AVAILABLE:
            try:
                import omni.replicator.core as rep
                self.render_product = rep.create.render_product(camera_path, (1280, 720))
                self.annotator = rep.AnnotatorRegistry.get_annotator("rgb")
                self.annotator.attach([self.render_product])
                print("[Camera] 1280x720 @ 70 deg FOV")
            except Exception as e:
                print(f"[WARN] Camera setup failed: {e}")
                self.annotator = None

    def _init_apriltag(self):
        """AprilTag 36h11 감지기 초기화"""
        img_w, img_h = 1280, 720
        fov_deg = 130.0  # 일반 카메라 FOV
        self.fx = img_w / (2 * np.tan(np.radians(fov_deg / 2)))
        self.fy = self.fx
        self.cx = img_w / 2
        self.cy = img_h / 2

        self.camera_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        # AprilTag 36h11 사전만 사용
        self.apriltag_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
        self.detector_params = aruco.DetectorParameters()
        self.apriltag_detector = aruco.ArucoDetector(self.apriltag_dict, self.detector_params)

        print(f"[AprilTag] Initialized (36h11)")

    def _detect_apriltag(self):
        try:
            image_data = self.annotator.get_data()

            if image_data is None:
                return

            if not isinstance(image_data, np.ndarray) or image_data.size == 0:
                return

            if len(image_data.shape) == 3:
                gray = cv2.cvtColor(image_data[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
                color_image = image_data[:, :, :3].astype(np.uint8).copy()
            else:
                gray = image_data.astype(np.uint8)
                color_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # AprilTag 36h11 감지
            corners, ids, _ = self.apriltag_detector.detectMarkers(gray)

            vis_img = color_image.copy()
            img_h, img_w = vis_img.shape[:2]

            # 드론 상태 가져오기
            drone_state = self.drone.state
            drone_pos = np.array(drone_state.position)
            drone_quat = np.array(drone_state.attitude)
            r = Rotation.from_quat(drone_quat)

            # 디버그 변수 초기화
            marker_in_world = None
            marker_in_body = None
            tvec_debug = None

            if ids is not None and len(ids) > 0:
                aruco.drawDetectedMarkers(vis_img, corners, ids)

                # 마커 크기: 메쉬 1m × (태그픽셀 512 / 전체픽셀 600) = 0.853m
                MARKER_SIZE = 1.0 * (512.0 / 600.0)  # = 0.8533m
                rvecs, tvecs = self._estimate_pose(corners, MARKER_SIZE)

                if tvecs is not None and len(tvecs) > 0:
                    tvec = tvecs[0][0]
                    tvec_debug = tvec.copy()

                    # 카메라 프레임 → 바디 프레임 변환
                    # OpenCV 카메라: X-right, Y-down, Z-forward (depth)
                    # 바디 (FLU): X-forward, Y-left, Z-up
                    # 카메라가 드론 아래(-Z body)를 향함:
                    #   이미지 아래(+cam_y) = 드론 앞(+body_x)
                    #   이미지 오른쪽(+cam_x) = 드론 오른쪽(-body_y)
                    #   깊이(+cam_z) = 드론 아래(-body_z)
                    marker_in_body = np.array([
                        tvec[0],    # body_x = +cam_y (이미지 아래 → 드론 앞)
                        -tvec[1],   # body_y = -cam_x (이미지 오른쪽 → 드론 오른쪽 = -Y)
                        -tvec[2]    # body_z = -cam_z (깊이 → 드론 아래)
                    ])
                    # 카메라 오프셋 보정 (카메라가 바디 중심에서 z=-0.11 위치)
                    marker_in_body[2] -= 0.11
                    marker_in_world = drone_pos + r.apply(marker_in_body)

                    self._on_detection(marker_in_world[:2])

                    self.detection_count += 1
                    self.last_detection_time = self.step_count * 0.01

                    # 마지막 감지 데이터 저장 (태그 안 보일 때도 표시용)
                    self.last_marker_in_world = marker_in_world.copy()
                    self.last_marker_in_body = marker_in_body.copy()
                    self.last_tvec = tvec_debug.copy()
                    tag_center = corners[0][0].mean(axis=0).astype(int)
                    self.last_tag_center_px = tuple(tag_center)

                    cv2.drawFrameAxes(vis_img, self.camera_matrix, self.dist_coeffs,
                                     rvecs[0].reshape(3,1), tvecs[0].reshape(3,1), 0.3)

                    # 태그 중심점 표시 (녹색 원 - 현재 감지)
                    cv2.circle(vis_img, self.last_tag_center_px, 15, (0, 255, 0), 3)
                    cv2.putText(vis_img, "TAG", (tag_center[0]+20, tag_center[1]),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # ===== 화면 중심 십자선 (파란색) =====
            cv2.line(vis_img, (int(self.cx)-30, int(self.cy)), (int(self.cx)+30, int(self.cy)), (255, 0, 0), 2)
            cv2.line(vis_img, (int(self.cx), int(self.cy)-30), (int(self.cx), int(self.cy)+30), (255, 0, 0), 2)
            cv2.putText(vis_img, "CENTER", (int(self.cx)+5, int(self.cy)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            # ===== Ground Truth 로버 위치를 화면에 투영 (빨간색) =====
            rover_rel_world = self.rover_pos - drone_pos
            rover_rel_body = r.inv().apply(rover_rel_world)
            # Body to Camera (카메라→바디 변환의 역)
            # body_x=cam_y, body_y=-cam_x, body_z=-cam_z 의 역
            # → cam_x=-body_y, cam_y=body_x, cam_z=-body_z
            rover_in_cam = np.array([-rover_rel_body[1], rover_rel_body[0], -rover_rel_body[2]])
            if rover_in_cam[2] > 0.1:  # 카메라 앞에 있을 때만
                gt_px = int(self.fx * rover_in_cam[0] / rover_in_cam[2] + self.cx)
                gt_py = int(self.fy * rover_in_cam[1] / rover_in_cam[2] + self.cy)
                if 0 <= gt_px < img_w and 0 <= gt_py < img_h:
                    cv2.drawMarker(vis_img, (gt_px, gt_py), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
                    cv2.putText(vis_img, "GT", (gt_px+5, gt_py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # ===== 디버그 텍스트 (우측 상단) =====
            y_offset = 30
            line_height = 25

            # 드론 위치
            cv2.putText(vis_img, f"Drone: ({drone_pos[0]:.2f}, {drone_pos[1]:.2f}, {drone_pos[2]:.2f})",
                       (img_w - 400, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height

            # Ground Truth 로버 위치
            cv2.putText(vis_img, f"GT Rover: ({self.rover_pos[0]:.2f}, {self.rover_pos[1]:.2f}, {self.rover_pos[2]:.2f})",
                       (img_w - 400, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            y_offset += line_height

            # 현재 또는 마지막 감지 데이터 사용
            display_marker_world = marker_in_world if marker_in_world is not None else self.last_marker_in_world
            display_marker_body = marker_in_body if marker_in_body is not None else self.last_marker_in_body
            display_tvec = tvec_debug if tvec_debug is not None else self.last_tvec
            is_current = marker_in_world is not None

            if display_marker_world is not None:
                # 추정된 태그 위치 (world)
                status_suffix = "" if is_current else " (last)"
                color = (0, 255, 0) if is_current else (0, 180, 180)  # 현재: 녹색, 마지막: 노란색
                cv2.putText(vis_img, f"Est Tag: ({display_marker_world[0]:.2f}, {display_marker_world[1]:.2f}, {display_marker_world[2]:.2f}){status_suffix}",
                           (img_w - 400, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += line_height

                # 오차 계산 (XY만)
                error_xy = np.linalg.norm(display_marker_world[:2] - self.rover_pos[:2])
                error_vec = display_marker_world[:2] - self.rover_pos[:2]
                cv2.putText(vis_img, f"Error: {error_xy:.2f}m (dx:{error_vec[0]:.2f}, dy:{error_vec[1]:.2f})",
                           (img_w - 400, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                y_offset += line_height

                # tvec (camera frame)
                if display_tvec is not None:
                    cv2.putText(vis_img, f"tvec(cam): ({display_tvec[0]:.2f}, {display_tvec[1]:.2f}, {display_tvec[2]:.2f})",
                               (img_w - 400, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    y_offset += line_height

                # marker_in_body
                if display_marker_body is not None:
                    cv2.putText(vis_img, f"body: ({display_marker_body[0]:.2f}, {display_marker_body[1]:.2f}, {display_marker_body[2]:.2f})",
                               (img_w - 400, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    y_offset += line_height

            # 마지막 태그 위치 표시 (태그 안 보일 때 - 노란색 원)
            if not is_current and self.last_tag_center_px is not None:
                cv2.circle(vis_img, self.last_tag_center_px, 15, (0, 180, 180), 2)
                cv2.putText(vis_img, "LAST", (self.last_tag_center_px[0]+20, self.last_tag_center_px[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 180), 2)

            # 데이터 없음
            if display_marker_world is None:
                cv2.putText(vis_img, "Est Tag: N/A (no detection yet)", (img_w - 400, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

            # ===== 좌측 상단 상태 표시 =====
            num_markers = 0 if ids is None else len(ids)
            if num_markers > 0:
                status = f"Markers: {num_markers}"
                color = (0, 255, 0)
            else:
                status = "No markers"
                color = (0, 0, 255)

            cv2.putText(vis_img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # 바람 속도 표시
            wind_speed = np.linalg.norm(self.wind_velocity)
            wind_text = f"Wind: ({self.wind_velocity[0]:.1f}, {self.wind_velocity[1]:.1f}) m/s | Speed: {wind_speed:.1f} m/s"
            cv2.putText(vis_img, wind_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # 마커 인식 여부 상세 표시
            if ids is not None and len(ids) > 0:
                marker_info = f"Marker ID: {ids[0][0]} | Dict: AprilTag 36h11"
                cv2.putText(vis_img, marker_info, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(vis_img, "Marker: NOT DETECTED", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 카메라 뷰를 파일로 저장 (3프레임마다 - 높은 프레임레이트)
            if self.step_count % 3 == 0:
                vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(self.camera_view_path, vis_img_bgr)

        except Exception as e:
            if self.step_count % 100 == 0:
                print(f"[WARN] Detection error: {e}")

    def _estimate_pose(self, corners, marker_size):
        marker_points = np.array([
            [-marker_size/2, marker_size/2, 0],
            [marker_size/2, marker_size/2, 0],
            [marker_size/2, -marker_size/2, 0],
            [-marker_size/2, -marker_size/2, 0]
        ], dtype=np.float32)

        rvecs, tvecs = [], []
        for corner in corners:
            retval, rvec, tvec = cv2.solvePnP(
                marker_points, corner, self.camera_matrix, self.dist_coeffs,
                None, None, False, cv2.SOLVEPNP_IPPE_SQUARE
            )
            if retval:
                rvecs.append(rvec.reshape(1, 3))
                tvecs.append(tvec.reshape(1, 3))

        if len(rvecs) == 0:
            return None, None
        return np.array(rvecs), np.array(tvecs)

    def _on_detection(self, marker_pos_xy):
        # Z좌표는 고정값 사용 (Ground Truth 사용 안함)
        # 로버 높이는 약 -0.4m (큐브 중심) + 0.5 (큐브 반높이) = 0.1m 정도
        ROVER_Z_FIXED = 0.1  # 로버 상단 높이 (고정값)
        full_pos = np.array([marker_pos_xy[0], marker_pos_xy[1], ROVER_Z_FIXED])
        self.controller.update_estimator(full_pos)

    def _update_rover(self, dt):
        stage = omni.usd.get_context().get_stage()
        rover_prim = stage.GetPrimAtPath("/World/Rover")

        if not rover_prim.IsValid():
            return

        self.rover_pos += self.rover_vel * dt

        # Ground Truth 모드에서만 컨트롤러에 로버 위치 전달
        if not self.controller.USE_APRILTAG:
            self.controller.set_rover_pos(self.rover_pos)

        xformable = UsdGeom.Xformable(rover_prim)
        translate_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break

        if translate_op:
            translate_op.Set(Gf.Vec3d(float(self.rover_pos[0]), float(self.rover_pos[1]), float(self.rover_pos[2])))

    def _apply_wind_force(self):
        """실제 물리 바람 힘을 드론에 적용"""
        if not self.wind_enabled:
            return

        # 드론 상태 가져오기
        drone_state = self.drone.state
        drone_vel = np.array(drone_state.linear_velocity, dtype=np.float32)
        drone_quat = np.array(drone_state.attitude, dtype=np.float32)  # [x, y, z, w]

        # 상대 속도 (airspeed) = 드론 속도 - 바람 속도
        # 바람이 +X로 불면, 드론이 정지해 있어도 -X 방향 상대 속도를 느낌
        relative_velocity = drone_vel - self.wind_velocity

        # 상대 속도 크기
        rel_speed = np.linalg.norm(relative_velocity)

        if rel_speed < 0.01:  # 거의 정지 상태면 스킵
            return

        # 항력 (Drag) 계산: F_drag = 0.5 * rho * Cd * A * v^2 * (-v_hat)
        # 항력은 상대 속도 반대 방향으로 작용
        drag_magnitude = 0.5 * self.air_density * self.drag_coefficient * self.drone_cross_section * rel_speed**2

        # 항력 방향 (상대 속도 반대)
        drag_direction = -relative_velocity / rel_speed

        # World frame에서의 항력
        drag_force_world = drag_magnitude * drag_direction

        # Body frame으로 변환 (apply_force는 body frame 기준)
        R = Rotation.from_quat(drone_quat)
        drag_force_body = R.inv().apply(drag_force_world)

        # FLU 좌표계로 변환 (Pegasus 규약)
        # World: X-forward, Y-left, Z-up
        # Body FLU: X-forward, Y-left, Z-up (동일하지만 회전된 상태)
        force_flu = [
            float(drag_force_body[0]),  # Forward
            float(drag_force_body[1]),  # Left
            float(drag_force_body[2])   # Up
        ]

        # 드론에 힘 적용
        self.drone.apply_force(force_flu, body_part="/body")

        # 디버깅 (가끔 출력)
        if self.step_count % 200 == 0:
            print(f"[Wind] vel={self.wind_velocity}, drag_world=[{drag_force_world[0]:.2f}, "
                  f"{drag_force_world[1]:.2f}, {drag_force_world[2]:.2f}] N")

    async def control_drone(self):
        drone = System()
        await drone.connect(system_address="udp://:14540")

        print("[MAVSDK] 드론 연결 대기 중...")
        async for state in drone.core.connection_state():
            if state.is_connected:
                print("[MAVSDK] -- 드론 연결 완료!")
                break

        print("[MAVSDK] GPS 위치 추정 대기 중...")
        async for health in drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                print("[MAVSDK] -- GPS 위치 추정 완료")
                break

        print("[MAVSDK] -- Arming")
        await drone.action.arm()

        print("[MAVSDK] -- 초기 setpoint 설정")
        await drone.offboard.set_attitude(Attitude(0.0, 0.0, 0.0, 0.0))

        print("[MAVSDK] -- Offboard 모드 시작")
        try:
            await drone.offboard.start()
        except OffboardError as error:
            print(f"[MAVSDK] Offboard 모드 시작 실패: {error._result.result}")
            await drone.action.disarm()
            return

        # 1단계: 상승
        print(f"[MAVSDK] -- 1단계: 상승 시작 (목표: {self.takeoff_target_height}m)")
        drone_state = self.drone.state
        current_pos = np.array(drone_state.position)
        takeoff_target = current_pos.copy()
        takeoff_target[2] = self.takeoff_target_height

        self.controller.set_takeoff_target(takeoff_target)

        while self.controller.takeoff_mode and not self.stop_sim and simulation_app.is_running():
            drone_state = self.drone.state
            self.controller.update_state(drone_state)
            self.controller.update(0.02)

            attitude_cmd = self.controller.get_attitude_rate()

            if isinstance(attitude_cmd, Attitude):
                await drone.offboard.set_attitude(attitude_cmd)
            else:
                await drone.offboard.set_attitude_rate(attitude_cmd)

            await asyncio.sleep(0.02)

        print("[MAVSDK] -- 1단계 완료: 상승 완료")
        await asyncio.sleep(1.0)

        # 2단계: RL 착륙 제어
        print("[MAVSDK] -- 2단계: RL 착륙 제어 시작 (변환 레이어 적용)")
        self.flight_phase = "LANDING"

        while not self.stop_sim and simulation_app.is_running():
            drone_state = self.drone.state
            self.controller.update_state(drone_state)
            self.controller.update(0.02)

            attitude_cmd = self.controller.get_attitude_rate()

            if isinstance(attitude_cmd, Attitude):
                await drone.offboard.set_attitude(attitude_cmd)
            else:
                await drone.offboard.set_attitude_rate(attitude_cmd)

            await asyncio.sleep(0.02)

        print("[MAVSDK] -- Offboard 모드 중지")
        try:
            await drone.offboard.stop()
        except OffboardError as error:
            print(f"[MAVSDK] Offboard 모드 중지 실패: {error._result.result}")

        print("[MAVSDK] -- 착륙")
        await drone.action.land()
        await asyncio.sleep(3)

    def run_control_thread(self):
        print("[MAVSDK] 시뮬레이션 초기화 중...")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self.control_drone())
        finally:
            loop.close()

    def run(self):
        control_thread = threading.Thread(target=self.run_control_thread, daemon=True)
        control_thread.start()

        self.timeline.play()

        print("[Camera] Waiting for initialization (3 seconds)...")
        for _ in range(300):
            self.world.step(render=True)
            self.step_count += 1
        print("[Camera] Ready!")

        while simulation_app.is_running() and not self.stop_sim:
            self._detect_apriltag()
            self._update_rover(self.world.get_physics_dt())
            self._apply_wind_force()  # 실제 바람 물리 적용
            self.world.step(render=True)
            self.step_count += 1

            if self.step_count % 100 == 0:
                drone_state = self.drone.state
                drone_pos = np.array([drone_state.position[0], drone_state.position[1], drone_state.position[2]])
                rover_xy_error = np.linalg.norm(drone_pos[:2] - self.rover_pos[:2])

                if self.controller.estimated_rover_pos is not None:
                    detection_status = "Tracking"
                else:
                    detection_status = "No tag"

                phase_str = f"[{self.flight_phase}]"

                print(f"[{self.step_count*0.01:.1f}s] {phase_str} {detection_status} | "
                    f"XY err: {rover_xy_error:.2f}m | "
                    f"Height: {drone_pos[2]:.2f}m")

        print(f"\n{'='*70}")
        print(f"[Summary] 시뮬레이션 종료")
        print(f"{'='*70}")


        try:
            for backend in self.drone._backends:
                if hasattr(backend, 'stop'):
                    backend.stop()
        except Exception as e:
            print(f"[Cleanup] Backend stop error: {e}")

        self.timeline.stop()
        simulation_app.close()


def main():
    import sys
    import signal

    app = None

    def cleanup_handler(signum, frame):
        print("\n[Signal] 종료 신호 수신...")
        if app is not None:
            app.stop_sim = True

    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "/home/rtx5080/s/ISAAC_LAB_DRONE/logs/sb3/Template-DroneLanding-v0/2026-01-27_09-06-42/model.zip"
    
    print(f"\n{'='*70}")
    print(f"RL 드론 착륙 시뮬레이션 (변환 레이어 적용)")
    print(f"{'='*70}")
    print(f"[Main] Model: {model_path}")
    print(f"[Main] USE_APRILTAG: {RLDroneLandingController.USE_APRILTAG}")
    if RLDroneLandingController.USE_APRILTAG:
        print(f"  → AprilTag 인식으로 로버 위치 추정 (Ground Truth 사용 안함)")
    else:
        print(f"  → Ground Truth (시뮬레이터 직접 위치)")
    print(f"\n변환 레이어:")
    print(f"  - Isaac Lab 토크 명령 -> PX4 Attitude 명령")
    print(f"  - Action 인덱스 수정: [0]=thrust, [1]=roll, [2]=pitch, [3]=yaw")
    print(f"  - 추력/관성모멘트 스케일 보정")
    print(f"{'='*70}\n")
    
    if not RL_AVAILABLE:
        print("[ERROR] stable-baselines3 not installed!")
        return

    try:
        app = PegasusRLLandingApp(model_path)
        app.run()
    except Exception as e:
        print(f"[ERROR] 예외 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        import subprocess
        try:
            subprocess.run(["pkill", "-f", "px4"], capture_output=True, timeout=5)
        except:
            pass


if __name__ == "__main__":
    main()
