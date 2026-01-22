#!/usr/bin/env python
"""
Pegasus 드론 착륙 시뮬레이션 (강화학습 모델 사용) - 개선 버전
주요 개선사항:
1. 마커 인식 시 위치 출력 강화
2. 드론 치우침 문제 해결 (좌표계 변환 수정)
3. 코드 품질 개선 및 안정성 향상
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

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.backends.backend import Backend

from pxr import Sdf, UsdShade, UsdGeom, Gf, UsdLux

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


class RLDroneLandingController(Backend):
    """강화학습 기반 드론 착륙 제어기 (Pegasus Backend)"""

    # ============================================================
    # ★★★ 튜닝 파라미터 (여기서 수정하세요!) ★★★
    # ============================================================

    # --- 디버깅 모드 ---
    DEBUG_MODE = True           # True로 설정하면 매 스텝 상세 출력

    # --- ArUco 사용 여부 ---
    USE_ARUCO = False           # False: ground truth 사용, True: ArUco 검출 사용

    # --- 추력 관련 ---
    THRUST_SCALE = 1.0           # 전체 추력 스케일 (1.0 = 원본)
    THRUST_OFFSET = 0.0          # 추력 오프셋 (0 = 원본, IsaacLab과 동일)

    # --- 토크/회전 관련 (action 출력 감쇠) ---
    ROLL_SCALE = 1.0             # Roll (좌우 기울기) 감쇠 (1.0 = 원본)
    PITCH_SCALE = 1.0            # Pitch (앞뒤 기울기) 감쇠 (1.0 = 원본)
    YAW_SCALE = 1.0              # Yaw (회전) 감쇠 (1.0 = 원본)

    # --- XY 이동 감쇠 (observation 입력 스케일) ---
    XY_GOAL_SCALE = 1.0          # XY 목표 거리 감쇠 (1.0 = 원본)
    Z_GOAL_SCALE = 0.3           # Z 목표 거리 감쇠 (작을수록 하강 느림)

    # --- 착륙 조건 ---
    XY_THRESHOLD = 0.2           # XY 오차가 이 이하일 때만 하강 (m)

    # --- 속도 감쇠 (observation 입력 스케일) ---
    VEL_SCALE = 1.0              # 속도 observation 스케일 (1.0 = 원본)
    ANG_VEL_SCALE = 1.0          # 각속도 observation 스케일 (1.0 = 원본, 사용 안함)

    # --- 물리 파라미터 ---
    IRIS_MASS = 1.5              # Iris 드론 질량 (kg)
    TRAIN_MASS = 0.033           # 학습 때 사용한 Crazyflie 질량 (kg)
    TRAIN_THRUST_TO_WEIGHT = 1.9 # 학습 때 thrust-to-weight ratio
    TRAIN_MOMENT_SCALE = 0.002   # 학습 때 moment scale (Nm)

    # --- 토크 스케일 오버라이드 ---
    TORQUE_MULTIPLIER = 1.0      # 토크 전체 배율 (자동계산 후 추가 조정)

    # ============================================================

    def __init__(self, rover_initial_pos, rover_velocity, model_path, device="cuda", detection_callback=None):
        # Backend 부모 클래스 초기화
        super().__init__(config=None)

        # 디바이스 설정
        self.rl_device = device

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

        # 튜닝 파라미터 출력
        print("\n" + "="*60)
        print("★ RL Controller 튜닝 파라미터 ★")
        print("="*60)
        print(f"  THRUST_SCALE:    {self.THRUST_SCALE}")
        print(f"  THRUST_OFFSET:   {self.THRUST_OFFSET}")
        print(f"  ROLL_SCALE:      {self.ROLL_SCALE}")
        print(f"  PITCH_SCALE:     {self.PITCH_SCALE}")
        print(f"  YAW_SCALE:       {self.YAW_SCALE}")
        print(f"  XY_GOAL_SCALE:   {self.XY_GOAL_SCALE}")
        print(f"  Z_GOAL_SCALE:    {self.Z_GOAL_SCALE}")
        print(f"  XY_THRESHOLD:    {self.XY_THRESHOLD}m (XY 정렬 후 하강)")
        print(f"  VEL_SCALE:       {self.VEL_SCALE}")
        print(f"  ANG_VEL_SCALE:   {self.ANG_VEL_SCALE}")
        print(f"  TORQUE_MULTIPLIER: {self.TORQUE_MULTIPLIER}")
        print(f"  DEBUG_MODE:      {self.DEBUG_MODE}")
        print(f"  USE_ARUCO:       {self.USE_ARUCO}")
        if not self.USE_ARUCO:
            print(f"  ⚠️  Ground truth 모드! 실제 로버 위치 사용")
        print("="*60 + "\n")
        
        # 상태
        self.dt = 0.01
        self.time = 0.0
        self.estimated_rover_pos = None
        self.detection_callback = detection_callback
        self._state = None
        
        # 착륙 상태
        self.phase = "APPROACH"
        self.approach_height = 5.5
        self.landing_height = 0.75
        self.current_target_height = self.approach_height  # 현재 목표 높이
        
        # 목표 위치 (world frame)
        self.desired_pos_w = np.array(rover_initial_pos, dtype=np.float32)
        self.desired_pos_w[2] = self.landing_height
        
        # 디버그 카운터
        self._obs_debug_count = 0
        self._action_debug_count = 0
        
    def update(self, dt: float):
        """Backend 필수 메서드"""
        self.dt = dt
        self.time += dt

        # 목표 XY 위치 업데이트
        if self.USE_ARUCO and self.estimated_rover_pos is not None:
            # ArUco 검출 결과 사용
            self.desired_pos_w[:2] = self.estimated_rover_pos[:2]
        else:
            # Ground truth 사용 (디버깅용)
            self.desired_pos_w[:2] = self.rover_pos[:2]

        # ★★★ 조건부 하강: XY 정렬 후에만 하강 ★★★
        if self._state is not None:
            drone_pos = np.array(self._state.position)
            xy_error = np.linalg.norm(drone_pos[:2] - self.desired_pos_w[:2])

            if xy_error < self.XY_THRESHOLD:
                # XY 정렬됨 → 하강
                descent_rate = 0.5  # m/s
                self.current_target_height = max(
                    self.landing_height,
                    self.current_target_height - descent_rate * dt
                )
                if self.phase != "LANDING":
                    self.phase = "LANDING"
                    print(f"[Phase] LANDING - XY error: {xy_error:.2f}m < {self.XY_THRESHOLD}m")
            else:
                # XY 미정렬 → 높이 유지
                if self.phase != "APPROACH":
                    self.phase = "APPROACH"
                    print(f"[Phase] APPROACH - XY error: {xy_error:.2f}m > {self.XY_THRESHOLD}m")

        self.desired_pos_w[2] = self.current_target_height

    def set_rover_pos(self, pos):
        """App에서 로버 위치를 직접 설정 (sync용)"""
        self.rover_pos[:] = pos
    
    def input_reference(self):
        """RL 모델로 액션 결정 후 4개 로터 속도 반환"""
        # 현재 상태 가져오기
        state = self._get_vehicle_state()
        
        # Observation 구성 (Isaac Lab과 동일한 16차원)
        obs = self._construct_observation(state)
        
        # RL 모델로 액션 예측
        action, _states = self.model.predict(obs, deterministic=True)
        
        # NumPy로 변환
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        action = action.flatten()
        
        # 액션을 로터 속도로 변환
        rotor_velocities = self._action_to_rotor_velocities(action, state)
        
        return rotor_velocities.tolist()
    
    def _construct_observation(self, state):
        """Isaac Lab 환경과 ★★★ 완전히 동일한 ★★★ 16차원 observation 구성

        drone_landing_env.py와 1:1 대응:
        - R.inv().apply() 사용 (scipy Rotation 메서드)
        - 각속도는 world frame 그대로 사용!
        - 중력은 [0, 0, -gravity] 사용 (정규화 안함)
        """

        # 드론 상태
        pos = np.array(state.position, dtype=np.float32)
        lin_vel = np.array(state.linear_velocity, dtype=np.float32)
        ang_vel = np.array(state.angular_velocity, dtype=np.float32)

        # ★★★ 핵심: Pegasus attitude는 [x,y,z,w] 순서 ★★★
        # IsaacLab의 drone_landing_env.py와 동일하게 처리
        quat_xyzw = np.array(state.attitude, dtype=np.float32)

        # Rotation 객체 생성 (scipy는 [x,y,z,w] 순서)
        R = Rotation.from_quat(quat_xyzw)

        # ★★★ 1. 드론 속도 (body frame) - R.inv().apply() 사용! ★★★
        lin_vel_b = R.inv().apply(lin_vel)
        lin_vel_b = lin_vel_b * self.VEL_SCALE

        # ★★★ 2. 각속도 - IsaacLab에서는 world frame 그대로 사용! ★★★
        # drone_landing_env.py: obs에 ang_vel 그대로 넣음 (변환 안함)
        ang_vel_obs = ang_vel  # body frame 변환 안함!

        # ★★★ 3. 중력 방향 (body frame) - 스케일 유지! ★★★
        gravity_world = np.array([0, 0, -self.gravity], dtype=np.float32)
        gravity_b = R.inv().apply(gravity_world)

        # ★★★ 4. 목표 위치 (body frame) ★★★
        goal_rel_world = self.desired_pos_w - pos
        desired_pos_b = R.inv().apply(goal_rel_world)

        # XY/Z 스케일 적용 (필요시)
        desired_pos_b[0] = desired_pos_b[0] * self.XY_GOAL_SCALE
        desired_pos_b[1] = desired_pos_b[1] * self.XY_GOAL_SCALE
        desired_pos_b[2] = desired_pos_b[2] * self.Z_GOAL_SCALE

        # ★★★ 5. 상대 속도 (body frame) ★★★
        rel_vel_world = lin_vel - self.rover_vel
        rel_vel_b = R.inv().apply(rel_vel_world)
        rel_vel_b = rel_vel_b * self.VEL_SCALE

        # ★★★ 6. Yaw 각도 - IsaacLab과 동일한 계산 ★★★
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        current_yaw = np.arctan2(
            2.0 * (quat_wxyz[0]*quat_wxyz[3] + quat_wxyz[1]*quat_wxyz[2]),
            1.0 - 2.0 * (quat_wxyz[2]**2 + quat_wxyz[3]**2)
        )

        # 디버깅 출력
        if (self.DEBUG_MODE or self._obs_debug_count < 5) and self._obs_debug_count % 50 == 1:
            print(f"\n{'='*70}")
            print(f"📊 Observation Debug (step {self._obs_debug_count})")
            print(f"{'='*70}")
            print(f"  Drone pos (world):    [{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}]")
            print(f"  Rover pos (world):    [{self.rover_pos[0]:6.2f}, {self.rover_pos[1]:6.2f}, {self.rover_pos[2]:6.2f}]")
            print(f"  Desired pos (world):  [{self.desired_pos_w[0]:6.2f}, {self.desired_pos_w[1]:6.2f}, {self.desired_pos_w[2]:6.2f}]")
            print(f"  Goal rel (world):     [{goal_rel_world[0]:6.2f}, {goal_rel_world[1]:6.2f}, {goal_rel_world[2]:6.2f}] (norm: {np.linalg.norm(goal_rel_world):.2f}m)")
            print(f"  Goal rel (body):      [{desired_pos_b[0]:6.2f}, {desired_pos_b[1]:6.2f}, {desired_pos_b[2]:6.2f}]")
            print(f"  Lin vel (body):       [{lin_vel_b[0]:6.2f}, {lin_vel_b[1]:6.2f}, {lin_vel_b[2]:6.2f}]")
            print(f"  Ang vel (world!):     [{ang_vel_obs[0]:6.2f}, {ang_vel_obs[1]:6.2f}, {ang_vel_obs[2]:6.2f}]")
            print(f"  Gravity (body):       [{gravity_b[0]:6.2f}, {gravity_b[1]:6.2f}, {gravity_b[2]:6.2f}]")
            print(f"  Yaw: {np.degrees(current_yaw):6.1f}°")
        self._obs_debug_count += 1

        # ★★★ 16차원 연결 - IsaacLab과 완전히 동일한 순서! ★★★
        obs = np.concatenate([
            lin_vel_b,        # 3: 선속도 (body) - R.inv().apply(vel)
            ang_vel_obs,      # 3: 각속도 (world!) - 변환 안함!
            gravity_b,        # 3: 중력 방향 (body)
            desired_pos_b,    # 3: 목표 위치 (body)
            rel_vel_b,        # 3: 상대 속도 (body)
            [current_yaw]     # 1: yaw 각도
        ])

        return obs.astype(np.float32)
    
    def _action_to_rotor_velocities(self, action, state):
        """RL 액션을 로터 속도로 변환"""
        # 액션 클리핑
        action = np.clip(action, -1.0, 1.0)
        
        # 원본 액션 저장 (디버깅용)
        original_action = action.copy()

        # 액션 추출
        thrust_action = action[0]
        roll_action = action[1]
        pitch_action = action[2]
        yaw_action = action[3]
        
        # 모멘트 스케일 적용
        roll_scaled = roll_action * self.ROLL_SCALE
        pitch_scaled = pitch_action * self.PITCH_SCALE
        yaw_scaled = yaw_action * self.YAW_SCALE
        moments_scaled = np.array([roll_scaled, pitch_scaled, yaw_scaled])

        # 질량비 계산
        mass_ratio = self.IRIS_MASS / self.TRAIN_MASS

        # 추력 변환
        thrust_ratio = self.TRAIN_THRUST_TO_WEIGHT * (thrust_action + 1.0) / 2.0
        thrust_ratio = thrust_ratio * self.THRUST_SCALE + self.THRUST_OFFSET
        thrust_force = thrust_ratio * self.IRIS_MASS * self.gravity

        # 토크 변환
        torques = moments_scaled * self.TRAIN_MOMENT_SCALE * mass_ratio * self.TORQUE_MULTIPLIER

        if (self.DEBUG_MODE or self._action_debug_count < 5) and self._action_debug_count % 50 == 1:
            print(f"\n{'='*70}")
            print(f"🎮 Action Debug (step {self._action_debug_count})")
            print(f"{'='*70}")
            print(f"  Raw action (RL):  [{original_action[0]:6.3f}, {original_action[1]:6.3f}, {original_action[2]:6.3f}, {original_action[3]:6.3f}]")
            print(f"  After invert:     [{thrust_action:6.3f}, {roll_action:6.3f}, {pitch_action:6.3f}, {yaw_action:6.3f}]")
            print(f"  Scaled moments:   [{roll_scaled:6.3f}, {pitch_scaled:6.3f}, {yaw_scaled:6.3f}]")
            print(f"  thrust_force:     {thrust_force:6.2f} N (hover: {self.IRIS_MASS * self.gravity:.2f} N)")
            print(f"  torques (Nm):     [{torques[0]:7.4f}, {torques[1]:7.4f}, {torques[2]:7.4f}]")
        self._action_debug_count += 1

        # Pegasus 내장 함수로 로터 속도 변환
        if self.vehicle:
            rotor_velocities = self.vehicle.force_and_torques_to_velocities(thrust_force, torques)
            return np.array(rotor_velocities)
        else:
            return np.array([500.0, 500.0, 500.0, 500.0])
    
    def update_estimator(self, marker_pos_world):
        """태그 감지 결과 업데이트"""
        self.estimated_rover_pos = marker_pos_world
        
        # ★★★ 마커 인식 시 위치 출력 강화 ★★★
        if hasattr(self, '_state') and self._state is not None:
            drone_pos = np.array(self._state.position)
            error_xy = np.linalg.norm(drone_pos[:2] - marker_pos_world[:2])
            error_z = abs(drone_pos[2] - marker_pos_world[2])
            
            # print(f"\n{'='*70}")
            # print(f"🎯 마커 인식 성공!")
            # print(f"{'='*70}")
            # print(f"  마커 위치 (world): [{marker_pos_world[0]:6.2f}, {marker_pos_world[1]:6.2f}, {marker_pos_world[2]:6.2f}]")
            # print(f"  드론 위치 (world): [{drone_pos[0]:6.2f}, {drone_pos[1]:6.2f}, {drone_pos[2]:6.2f}]")
            # print(f"  XY 오차: {error_xy:5.2f}m  |  Z 오차: {error_z:5.2f}m")
            # print(f"{'='*70}\n")
    
    def update_sensor(self, sensor_type: str, sensor_data: dict):
        """센서 데이터 수신"""
        pass
    
    def update_state(self, state: dict):
        """드론 상태 업데이트"""
        self._state = state
    
    def start(self):
        """Backend 시작"""
        print("[RL Controller] Started")
        print(f"[RL Controller] Model device: {self.rl_device}")
    
    def stop(self):
        """Backend 중지"""
        print(f"[RL Controller] Stopped - Final phase: {self.phase}")
    
    def reset(self):
        """Backend 리셋"""
        self.time = 0.0
        self.phase = "APPROACH"
        self.estimated_rover_pos = None
        self._obs_debug_count = 0
        self._action_debug_count = 0
        print("[RL Controller] Reset")
    
    def update_graphical_sensor(self, sensor_data: dict):
        """그래픽 센서 업데이트"""
        pass
        
    def _get_vehicle_state(self):
        """현재 드론 상태 반환"""
        if hasattr(self, '_state') and self._state is not None:
            return self._state
        
        # 기본값
        class DummyState:
            def __init__(self):
                self.position = np.zeros(3, dtype=np.float32)
                self.linear_velocity = np.zeros(3, dtype=np.float32)
                self.attitude = np.array([0, 0, 0, 1], dtype=np.float32)
                self.angular_velocity = np.zeros(3, dtype=np.float32)
        
        return DummyState()


class PegasusRLLandingApp:
    """Pegasus RL 착륙 시뮬레이션 앱"""
    
    def __init__(self, model_path):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        
        # 환경 로드
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])
        
        # 로버 설정
        self.rover_pos = np.array([0.0, 0.0, 0.375], dtype=np.float32)  # 큐브 0.75 / 2
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
        config.backends = [self.controller]
        
        initial_pos = [
            -1.5,
            -1.5,
            5.5
        ]
        
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
        
        # 조명 추가
        self._add_lighting()
        
        # 로버 생성
        self._create_rover()
        
        # 카메라 설정
        self._setup_camera()
        
        # ArUco 감지기 초기화
        if ARUCO_AVAILABLE:
            self._init_aruco()
        
        self.world.reset()
        
        # 상태
        self.step_count = 0
        self.detection_count = 0
        self.last_saved_frame = -1
        self.last_detection_time = 0.0
        
        print("\n[Verification] Checking drone initial position...")
        drone_state = self.drone.state
        actual_pos = np.array([drone_state.position[0], drone_state.position[1], drone_state.position[2]])
        print(f"  Expected: {initial_pos}")
        print(f"  Actual:   {actual_pos}")
        
        if not np.allclose(actual_pos, initial_pos, atol=0.1):
            print(f"  ⚠️  Position mismatch detected!")
        else:
            print(f"  ✓ Position correct!")
        
    def _add_lighting(self):
        """강화된 조명 시스템"""
        stage = omni.usd.get_context().get_stage()
        
        # DistantLight
        distant_light_path = "/World/DistantLight"
        distant_light = UsdLux.DistantLight.Define(stage, distant_light_path)
        distant_light.CreateIntensityAttr(5000.0)
        distant_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 0.95))
        distant_light.CreateAngleAttr(0.53)
        
        xform = UsdGeom.Xformable(distant_light)
        xform.ClearXformOpOrder()
        rotate_op = xform.AddRotateXYZOp()
        rotate_op.Set(Gf.Vec3f(-45, 45, 0))
        
        # DomeLight
        dome_light_path = "/World/DomeLight"
        dome_light = UsdLux.DomeLight.Define(stage, dome_light_path)
        dome_light.CreateIntensityAttr(1000.0)
        dome_light.CreateColorAttr(Gf.Vec3f(0.9, 0.95, 1.0))
        
        print("[Lighting] Added: DistantLight (5000 lux) + DomeLight (1000 lux)")
        
    def _create_rover(self):
        """AprilTag 로버 생성"""
        stage = omni.usd.get_context().get_stage()
        from pxr import UsdGeom, UsdPhysics

        rover_path = "/World/Rover"
        xform = UsdGeom.Xform.Define(stage, rover_path)

        # Cube - 1.5배 크기 (0.5 → 0.75)
        cube_path = rover_path + "/Cube"
        cube = UsdGeom.Cube.Define(stage, cube_path)
        cube.GetSizeAttr().Set(0.75)  # 1.5배 크기

        # 회색 재질
        cube_mtl_path = Sdf.Path(cube_path + "_Material")
        cube_mtl = UsdShade.Material.Define(stage, cube_mtl_path)
        cube_shader = UsdShade.Shader.Define(stage, cube_mtl_path.AppendPath("Shader"))
        cube_shader.CreateIdAttr("UsdPreviewSurface")
        cube_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.5, 0.5, 0.5))
        cube_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        cube_mtl.CreateSurfaceOutput().ConnectToSource(cube_shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(cube_mtl)

        # ★★★ 물리: Kinematic Body로 변경 (떠오르지 않게) ★★★
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
        rigid_api.CreateKinematicEnabledAttr(True)  # Kinematic = 물리 영향 안 받음
        collision_api = UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

        # 초기 위치
        xform_ops = xform.AddTranslateOp()
        xform_ops.Set(Gf.Vec3d(float(self.rover_pos[0]), float(self.rover_pos[1]), float(self.rover_pos[2])))
        
        # AprilTag 텍스처
        self._add_apriltag_texture()
        
        # 로버 위 조명
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
        """AprilTag 텍스처 생성"""
        stage = omni.usd.get_context().get_stage()
        
        mesh_path = "/World/Rover/TagMesh"
        mesh = UsdGeom.Mesh.Define(stage, mesh_path)

        # 태그 크기도 1.5배 (0.3 → 0.45)
        half = 0.45
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
        translate_op.Set(Gf.Vec3d(0, 0, 0.376))  # 큐브 높이 0.75/2 = 0.375 + 약간
        
        # AprilTag 이미지 생성
        tag_image_path = self._generate_apriltag_image()
        
        # 발광 재질
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
        
        print(f"[Rover] AprilTag texture added: {tag_image_path}")
        
    def _generate_apriltag_image(self):
        """AprilTag 이미지 생성"""
        if not ARUCO_AVAILABLE:
            return "/tmp/dummy_tag.png"
        
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
        tag_size = 512
        border_bits = 1
        
        tag_image = np.zeros((tag_size, tag_size), dtype=np.uint8)
        tag_image = aruco.generateImageMarker(aruco_dict, 0, tag_size, tag_image, border_bits)
        
        full_size = 600
        full_image = np.ones((full_size, full_size), dtype=np.uint8) * 255
        offset = (full_size - tag_size) // 2
        full_image[offset:offset+tag_size, offset:offset+tag_size] = tag_image
        
        output_path = "/tmp/apriltag_36h11_id0.png"
        cv2.imwrite(output_path, full_image)
        print(f"[AprilTag] Generated: {output_path}")
        
        return output_path
        
    def _setup_camera(self):
        """드론에 카메라 부착"""
        stage = omni.usd.get_context().get_stage()
        
        camera_path = "/World/Drone/body/Camera"
        camera_prim = UsdGeom.Camera.Define(stage, camera_path)
        
        camera_prim.GetFocalLengthAttr().Set(8.0)
        camera_prim.GetHorizontalApertureAttr().Set(60.0)
        camera_prim.GetVerticalApertureAttr().Set(33.75)
        camera_prim.GetFocusDistanceAttr().Set(1000.0)
        camera_prim.GetFStopAttr().Set(0.0)
        camera_prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10000.0))
        
        xform = UsdGeom.Xformable(camera_prim)
        xform.ClearXformOpOrder()
        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(0, 0, -0.15))
        
        if ARUCO_AVAILABLE:
            try:
                import omni.replicator.core as rep
                self.render_product = rep.create.render_product(camera_path, (1280, 720))
                self.annotator = rep.AnnotatorRegistry.get_annotator("rgb")
                self.annotator.attach([self.render_product])
                print("[Camera] 1280x720 @ 150° FOV")
            except Exception as e:
                print(f"[WARN] Camera setup failed: {e}")
                self.annotator = None
        
    def _init_aruco(self):
        """ArUco 감지기 초기화"""
        img_w, img_h = 1280, 720
        fov_deg = 150.0
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
        
        self.aruco_dicts = {
            "DICT_APRILTAG_36h11": aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11),
            "DICT_APRILTAG_25h9": aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_25h9),
            "DICT_APRILTAG_16h5": aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_16h5),
        }
        self.aruco_params = aruco.DetectorParameters()
        
        print(f"[ArUco] Initialized with camera matrix:")
        print(f"  fx={self.fx:.1f}, fy={self.fy:.1f}")
        print(f"  cx={self.cx:.1f}, cy={self.cy:.1f}")
        
    def _detect_aruco(self):
        """ArUco 태그 감지"""
        # if not ARUCO_AVAILABLE or not hasattr(self, 'annotator') or self.annotator is None:
        #     return
        
        # if self.step_count % 2 != 0:
        #     return
        
        try:
            image_data = self.annotator.get_data()
            
            if image_data is None:
                return
            
            if not isinstance(image_data, np.ndarray) or image_data.size == 0:
                return
            
            # 그레이스케일 변환
            if len(image_data.shape) == 3:
                gray = cv2.cvtColor(image_data[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
                color_image = image_data[:, :, :3].astype(np.uint8).copy()
            else:
                gray = image_data.astype(np.uint8)
                color_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            # 감지
            corners, ids = None, None
            detected_dict_name = None
            for dict_name, aruco_dict in self.aruco_dicts.items():
                detector = aruco.ArucoDetector(aruco_dict, self.aruco_params)
                corners, ids, _ = detector.detectMarkers(gray)
                if ids is not None and len(ids) > 0:
                    detected_dict_name = dict_name
                    break
            
            vis_img = color_image.copy()
            
            if ids is not None and len(ids) > 0:
                aruco.drawDetectedMarkers(vis_img, corners, ids)
                
                # 3D 자세 추정
                rvecs, tvecs = self._estimate_pose(corners, 0.6)
                
                if tvecs is not None and len(tvecs) > 0:
                    tvec = tvecs[0][0]

                    drone_state = self.drone.state
                    drone_pos = np.array(drone_state.position)
                    drone_quat = np.array(drone_state.attitude)

                    r = Rotation.from_quat(drone_quat)

                    # ★★★ 카메라→world 좌표 변환 수정 ★★★
                    # 카메라 좌표계 (OpenCV): X=오른쪽, Y=아래, Z=전방(거리)
                    # 카메라가 드론 아래에서 아래를 향함:
                    #   - 카메라 Z축 → 드론 -Z축 (아래)
                    #   - 카메라 X축 → 드론 Y축 (오른쪽)
                    #   - 카메라 Y축 → 드론 X축 (앞쪽)
                    marker_in_body = np.array([
                        tvec[1],    # body X = camera Y
                        tvec[0],    # body Y = camera X
                        -tvec[2]    # body Z = -camera Z (마커는 아래에 있으므로)
                    ])

                    # Body → World 변환
                    marker_in_world = drone_pos + r.apply(marker_in_body)

                    # 디버깅 출력
                    if self.step_count % 100 == 0:
                        print(f"[ArUco] tvec: [{tvec[0]:.2f}, {tvec[1]:.2f}, {tvec[2]:.2f}]")
                        print(f"[ArUco] body: [{marker_in_body[0]:.2f}, {marker_in_body[1]:.2f}, {marker_in_body[2]:.2f}]")
                        print(f"[ArUco] world: [{marker_in_world[0]:.2f}, {marker_in_world[1]:.2f}, {marker_in_world[2]:.2f}]")
                        print(f"[ArUco] actual rover: [{self.rover_pos[0]:.2f}, {self.rover_pos[1]:.2f}, {self.rover_pos[2]:.2f}]")

                    self._on_detection(marker_in_world[:2])
                    
                    self.detection_count += 1
                    self.last_detection_time = self.step_count * 0.01
                    
                    cv2.drawFrameAxes(vis_img, self.camera_matrix, self.dist_coeffs, 
                                     rvecs[0].reshape(3,1), tvecs[0].reshape(3,1), 0.3)
            
            # 십자선
            cv2.line(vis_img, (int(self.cx)-20, int(self.cy)), (int(self.cx)+20, int(self.cy)), (255,0,0), 2)
            cv2.line(vis_img, (int(self.cx), int(self.cy)-20), (int(self.cx), int(self.cy)+20), (255,0,0), 2)
            
            # 상태 텍스트
            num_markers = 0 if ids is None else len(ids)
            if num_markers > 0:
                status = f"Markers: {num_markers} ({detected_dict_name})"
                color = (0, 255, 0)
            else:
                status = "No markers detected"
                color = (0, 0, 255)
            
            cv2.putText(vis_img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # 시간 정보
            time_since_detection = self.step_count * 0.01 - self.last_detection_time
            time_text = f"Time: {self.step_count*0.01:.1f}s | Last detect: {time_since_detection:.1f}s ago"
            cv2.putText(vis_img, time_text, (10, vis_img.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # 이미지 저장
            if self.step_count % 50 == 0 and self.step_count != self.last_saved_frame:
                output_path = f"/tmp/aruco_rl_{self.step_count:06d}.png"
                cv2.imwrite(output_path, vis_img)
                self.last_saved_frame = self.step_count
                if self.step_count % 200 == 0:
                    print(f"[Debug] Saved: {output_path}")
            
        except Exception as e:
            if self.step_count % 100 == 0:
                print(f"[WARN] Detection error: {e}")
    
    def _estimate_pose(self, corners, marker_size):
        """마커 3D 자세 추정"""
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
        """태그 감지 콜백"""
        full_pos = np.array([marker_pos_xy[0], marker_pos_xy[1], self.rover_pos[2]])
        self.controller.update_estimator(full_pos)
    
    def _update_rover(self, dt):
        """로버 이동"""
        stage = omni.usd.get_context().get_stage()
        rover_prim = stage.GetPrimAtPath("/World/Rover")

        if not rover_prim.IsValid():
            return

        self.rover_pos += self.rover_vel * dt

        # Controller에도 로버 위치 동기화
        self.controller.set_rover_pos(self.rover_pos)

        xformable = UsdGeom.Xformable(rover_prim)
        translate_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break

        if translate_op:
            translate_op.Set(Gf.Vec3d(float(self.rover_pos[0]), float(self.rover_pos[1]), float(self.rover_pos[2])))
    
    def run(self):
        """메인 루프"""
        self.timeline.play()
        
        # 카메라 초기화 대기
        print("[Camera] Waiting for initialization (3 seconds)...")
        for _ in range(300):
            self.world.step(render=True)
            self.step_count += 1
        print("[Camera] ✓ Ready!")
        
        while simulation_app.is_running():
            # ArUco 감지
            self._detect_aruco()
            
            # 로버 업데이트
            self._update_rover(self.world.get_physics_dt())
            
            # 물리 스텝
            self.world.step(render=True)
            self.step_count += 1
            
            # 상태 출력
            if self.step_count % 100 == 0:
                drone_state = self.drone.state
                drone_pos = np.array([drone_state.position[0], drone_state.position[1], drone_state.position[2]])
                rover_xy_error = np.linalg.norm(drone_pos[:2] - self.rover_pos[:2])
                
                # XY 방향 오차 분석
                xy_error_vector = drone_pos[:2] - self.rover_pos[:2]
                x_err = xy_error_vector[0]
                y_err = xy_error_vector[1]
                
                if self.controller.estimated_rover_pos is not None:
                    detection_status = "✓ Tracking"
                    tracking_error = np.linalg.norm(drone_pos[:2] - self.controller.estimated_rover_pos[:2])
                    tracking_info = f"Track err: {tracking_error:.2f}m"
                else:
                    detection_status = "✗ No tag"
                    tracking_info = ""
                
                # ★★★ 치우침 경고 ★★★
                bias_warning = ""
                # if abs(y_err) > 0.5:
                #     if y_err > 0:
                #         bias_warning = "⚠️  드론이 +Y(오른쪽)으로 치우침!"
                #     else:
                #         bias_warning = "⚠️  드론이 -Y(왼쪽)으로 치우침!"
                
                # if abs(x_err) > 0.5:
                #     if x_err > 0:
                #         bias_warning += " +X(앞)으로 치우침!"
                #     else:
                #         bias_warning += " -X(뒤)으로 치우침!"
                
                # print(f"[{self.step_count*0.01:.1f}s] {detection_status} | "
                #       f"XY err: {rover_xy_error:.2f}m (X:{x_err:+.2f}, Y:{y_err:+.2f}) | "
                #       f"Height: {drone_pos[2]:.2f}m | "
                #       f"{tracking_info} | Detections: {self.detection_count} {bias_warning}")
        
        print(f"\n{'='*70}")
        print(f"[Summary] 시뮬레이션 종료")
        print(f"{'='*70}")
        print(f"  총 감지 횟수: {self.detection_count}")
        print(f"  총 프레임: {self.step_count}")
        print(f"  감지율: {self.detection_count / max(1, self.step_count/2) * 100:.1f}%")
        print(f"  디버그 이미지: /tmp/aruco_rl_*.png")
        print(f"{'='*70}\n")
        
        carb.log_warn("Simulation closing")
        self.timeline.stop()
        simulation_app.close()


def main():
    import sys
    
    # 모델 경로 설정
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "/home/rtx5080/s/ISAAC_LAB_DRONE/logs/sb3/Template-DroneLanding-v0/2026-01-20_15-52-16/model.zip"
    
    print(f"\n{'='*70}")
    print(f"RL 드론 착륙 시뮬레이션")
    print(f"{'='*70}")
    print(f"[Main] Model: {model_path}")
    print(f"\n현재 설정:")
    print(f"   DEBUG_MODE: {RLDroneLandingController.DEBUG_MODE}")
    print(f"   USE_ARUCO:  {RLDroneLandingController.USE_ARUCO}")
    if not RLDroneLandingController.USE_ARUCO:
        print(f"\n   Ground Truth 모드:")
        print(f"   - ArUco 검출 비활성화")
        print(f"   - 실제 로버 위치를 목표로 사용")
        print(f"   - Observation이 올바른지 테스트용")
    print(f"{'='*70}\n")
    
    # RL 모델 사용 가능 확인
    if not RL_AVAILABLE:
        print("[ERROR] stable-baselines3 not installed!")
        print("Install: pip install stable-baselines3")
        return
    
    app = PegasusRLLandingApp(model_path)
    app.run()


if __name__ == "__main__":
    main()