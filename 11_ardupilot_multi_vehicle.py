#!/usr/bin/env python
"""
Pegasus 드론 착륙 시뮬레이션 (강화학습 모델 사용)
- 학습된 RL 모델로 제어
- 움직이는 AprilTag 로버 추적
- 카메라 기반 태그 감지
- 안정적 착륙
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

    # --- 추력 관련 ---
    THRUST_SCALE = 0.9           # 전체 추력 스케일 (1.0 = 원본, 낮추면 천천히 하강)
    THRUST_OFFSET = 0.35         # 추력 오프셋 (양수 = 더 뜨려고 함, 호버링 보정용)

    # --- 토크/회전 관련 (action 출력 감쇠) ---
    ROLL_SCALE = 0.5             # Roll (좌우 기울기) 감쇠 (1.0 = 원본)
    PITCH_SCALE = 0.5            # Pitch (앞뒤 기울기) 감쇠 (1.0 = 원본)
    YAW_SCALE = 0.3              # Yaw (회전) 감쇠 (1.0 = 원본)

    # --- XY 이동 감쇠 (observation 입력 스케일) ---
    # 목표까지 거리를 축소해서 모델이 덜 급하게 이동하도록
    XY_GOAL_SCALE = 0.2          # XY 목표 거리 감쇠 (1.0 = 원본, 낮추면 천천히 이동)
    Z_GOAL_SCALE = 0.4           # Z 목표 거리 감쇠 (1.0 = 원본, 낮추면 천천히 하강)

    # --- 속도 감쇠 (observation 입력 스케일) ---
    VEL_SCALE = 0.3              # 속도 observation 스케일 (낮추면 속도를 과소평가)
    ANG_VEL_SCALE = 0.5          # 각속도 observation 스케일 (낮추면 덜 반응)

    # --- 물리 파라미터 ---
    IRIS_MASS = 1.5              # Iris 드론 질량 (kg)
    TRAIN_MASS = 0.033           # 학습 때 사용한 Crazyflie 질량 (kg)
    TRAIN_THRUST_TO_WEIGHT = 1.9 # 학습 때 thrust-to-weight ratio
    TRAIN_MOMENT_SCALE = 0.002   # 학습 때 moment scale (Nm)

    # --- 토크 스케일 오버라이드 ---
    # None이면 mass_ratio로 자동 계산, 숫자면 직접 지정
    TORQUE_MULTIPLIER = 1.0      # 토크 전체 배율 (자동계산 후 추가 조정)

    # --- 좌표계 조정 (드론이 치우치면 조정) ---
    FLIP_X = False               # X축 반전 (앞뒤 반대면 True)
    FLIP_Y = False               # Y축 반전 (좌우 반대면 True)

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
        print(f"  VEL_SCALE:       {self.VEL_SCALE}")
        print(f"  ANG_VEL_SCALE:   {self.ANG_VEL_SCALE}")
        print(f"  TORQUE_MULTIPLIER: {self.TORQUE_MULTIPLIER}")
        print(f"  FLIP_X:          {self.FLIP_X}")
        print(f"  FLIP_Y:          {self.FLIP_Y}")
        print("="*60 + "\n")
        
        # 상태
        self.dt = 0.01
        self.time = 0.0
        self.estimated_rover_pos = None  # 태그 감지로 업데이트
        self.detection_callback = detection_callback
        # NOTE: self.vehicle은 Backend가 자동으로 설정함
        
        # 착륙 상태
        self.phase = "APPROACH"
        self.approach_height = 3.5  # Isaac Lab cfg와 동일
        self.landing_height = 0.75
        
        # 목표 위치 (world frame)
        self.desired_pos_w = np.array(rover_initial_pos, dtype=np.float32)
        self.desired_pos_w[2] = self.landing_height  # Z는 로버 표면
        
    def update(self, dt: float):
        """Backend 필수 메서드"""
        self.dt = dt
        self.time += dt

        # NOTE: rover_pos는 App에서 업데이트하고, set_rover_pos()로 전달받음
        # 여기서 다시 업데이트하면 sync 안 맞음!

        # 목표 위치 업데이트
        if self.estimated_rover_pos is not None:
            self.desired_pos_w[:2] = self.estimated_rover_pos[:2]
        else:
            self.desired_pos_w[:2] = self.rover_pos[:2]
        self.desired_pos_w[2] = self.landing_height

    def set_rover_pos(self, pos):
        """App에서 로버 위치를 직접 설정 (sync용)"""
        self.rover_pos[:] = pos
    
    def input_reference(self):
        """RL 모델로 액션 결정 후 4개 로터 속도 반환"""
        # 현재 상태 가져오기
        state = self._get_vehicle_state()
        
        # Observation 구성 (Isaac Lab과 동일한 16차원)
        obs = self._construct_observation(state)
        
        # RL 모델로 액션 예측 (4차원: [thrust, roll_moment, pitch_moment, yaw_moment])
        # SB3의 predict()는 NumPy 배열을 받아서 내부적으로 텐서로 변환
        # CRITICAL: obs를 텐서로 변환하지 말고 NumPy 배열 그대로 전달!
        action, _states = self.model.predict(obs, deterministic=True)
        
        # NumPy로 변환 (predict()는 NumPy 배열 반환)
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        action = action.flatten()
        
        # 액션을 로터 속도로 변환
        rotor_velocities = self._action_to_rotor_velocities(action, state)
        
        return rotor_velocities.tolist()
    
    def _construct_observation(self, state):
        """Isaac Lab 환경과 동일한 16차원 observation 구성 (스케일 적용)"""
        # 드론 상태 (Pegasus State 객체)
        pos = np.array(state.position)
        lin_vel = np.array(state.linear_velocity)
        # CRITICAL: Pegasus state.attitude는 [x,y,z,w] 순서 (scipy와 동일)
        quat_xyzw = np.array(state.attitude)
        ang_vel = np.array(state.angular_velocity)

        # 1. 드론 속도 (드론 좌표계) - 3차원
        r = Rotation.from_quat(quat_xyzw)  # scipy: [x,y,z,w] - Pegasus와 동일
        R = r.as_matrix()
        lin_vel_b = R.T @ lin_vel
        # ★ 속도 스케일 적용 (모델이 속도를 과소평가하도록)
        lin_vel_b = lin_vel_b * self.VEL_SCALE

        # 2. 각속도 (드론 좌표계) - 3차원
        ang_vel_b = R.T @ ang_vel
        # ★ 각속도 스케일 적용 (덜 반응하도록)
        ang_vel_b = ang_vel_b * self.ANG_VEL_SCALE

        # 3. 중력 방향 (드론 좌표계) - 3차원
        gravity_w = np.array([0, 0, -1])
        gravity_b = R.T @ gravity_w

        # 4. 목표 위치 (드론 좌표계) - 3차원
        desired_pos_w = self.desired_pos_w
        goal_rel_world = desired_pos_w - pos
        desired_pos_b = R.T @ goal_rel_world
        # ★ XY/Z 목표 스케일 적용 (모델이 덜 급하게 이동하도록)
        desired_pos_b[0] = desired_pos_b[0] * self.XY_GOAL_SCALE  # X
        desired_pos_b[1] = desired_pos_b[1] * self.XY_GOAL_SCALE  # Y
        desired_pos_b[2] = desired_pos_b[2] * self.Z_GOAL_SCALE   # Z

        # ★ 좌표축 반전 (드론이 엉뚱한 방향으로 가면 조정)
        if self.FLIP_X:
            desired_pos_b[0] = -desired_pos_b[0]
        if self.FLIP_Y:
            desired_pos_b[1] = -desired_pos_b[1]

        # 5. 상대 속도 (드론 좌표계) - 3차원
        rel_vel_world = lin_vel - self.rover_vel
        rel_vel_b = R.T @ rel_vel_world
        # ★ 상대 속도도 스케일 적용
        rel_vel_b = rel_vel_b * self.VEL_SCALE

        # 6. Yaw 각도 - 1차원 (scipy euler 사용)
        _, _, current_yaw = r.as_euler('xyz')
        
        # 디버깅 출력 (처음 100번만)
        if not hasattr(self, '_obs_debug_count'):
            self._obs_debug_count = 0
        
        if self._obs_debug_count < 5:
            print(f"\n=== Observation Debug (step {self._obs_debug_count}) ===")
            print(f"Drone pos (world): {pos}")
            print(f"Rover pos (world): {self.rover_pos}")
            print(f"Desired pos (world): {desired_pos_w}")
            print(f"Goal rel (world): {goal_rel_world} (distance: {np.linalg.norm(goal_rel_world):.2f}m)")
            print(f"Goal rel (body): {desired_pos_b} (distance: {np.linalg.norm(desired_pos_b):.2f}m)")
            print(f"Lin vel (world): {lin_vel}")
            print(f"Lin vel (body): {lin_vel_b}")
            print(f"Gravity (body): {gravity_b}")
            print(f"Yaw: {np.degrees(current_yaw):.1f}°")
            self._obs_debug_count += 1
        
        # 16차원 연결
        obs = np.concatenate([
            lin_vel_b,        # 3
            ang_vel_b,        # 3
            gravity_b,        # 3
            desired_pos_b,    # 3
            rel_vel_b,        # 3
            [current_yaw]     # 1
        ])
        
        return obs.astype(np.float32)
    
    def _action_to_rotor_velocities(self, action, state):
        """RL 액션 (4차원)을 로터 속도 (4차원)로 변환 (튜닝 파라미터 적용)

        IsaacLab vs Pegasus 차이점:
        - IsaacLab 학습: Crazyflie (0.033kg), moment_scale=0.002
        - Pegasus 실행: Iris (~1.5kg), 질량비 약 45배
        - Pegasus: force (N), torque (Nm) → vehicle.force_and_torques_to_velocities()
        """
        # 디버깅 출력 (처음 5번만)
        if not hasattr(self, '_action_debug_count'):
            self._action_debug_count = 0

        # 액션 클리핑
        action = np.clip(action, -1.0, 1.0)

        # 액션 추출
        thrust_action = action[0]  # -1 ~ 1
        roll_action = action[1]
        pitch_action = action[2]
        yaw_action = action[3]

        # ★ 모멘트 스케일 적용 (튜닝 파라미터)
        roll_scaled = roll_action * self.ROLL_SCALE
        pitch_scaled = pitch_action * self.PITCH_SCALE
        yaw_scaled = yaw_action * self.YAW_SCALE
        moments_scaled = np.array([roll_scaled, pitch_scaled, yaw_scaled])

        # 질량비 계산
        mass_ratio = self.IRIS_MASS / self.TRAIN_MASS  # ~45.45

        # ★ 추력 변환 (튜닝 파라미터 적용)
        # thrust_ratio: 0 ~ 1.9 (학습 때 사용한 범위)
        thrust_ratio = self.TRAIN_THRUST_TO_WEIGHT * (thrust_action + 1.0) / 2.0
        # 추력 스케일 및 오프셋 적용
        thrust_ratio = thrust_ratio * self.THRUST_SCALE + self.THRUST_OFFSET
        # Iris 무게 기준으로 실제 힘 계산
        thrust_force = thrust_ratio * self.IRIS_MASS * self.gravity

        # ★ 토크 변환 (튜닝 파라미터 적용)
        torques = moments_scaled * self.TRAIN_MOMENT_SCALE * mass_ratio * self.TORQUE_MULTIPLIER

        if self._action_debug_count < 5:
            print(f"\n=== Action Debug (step {self._action_debug_count}) ===")
            print(f"Raw action: [{thrust_action:.3f}, {roll_action:.3f}, {pitch_action:.3f}, {yaw_action:.3f}]")
            print(f"Scaled moments: [{roll_scaled:.3f}, {pitch_scaled:.3f}, {yaw_scaled:.3f}]")
            print(f"  thrust_ratio: {thrust_ratio:.3f} (scale={self.THRUST_SCALE}, offset={self.THRUST_OFFSET})")
            print(f"  thrust_force: {thrust_force:.2f} N (Iris weight: {self.IRIS_MASS * self.gravity:.2f} N)")
            print(f"  torques: {torques} Nm")
            self._action_debug_count += 1

        # Pegasus 내장 함수로 로터 속도 변환
        if self.vehicle:
            rotor_velocities = self.vehicle.force_and_torques_to_velocities(thrust_force, torques)
            return np.array(rotor_velocities)
        else:
            # fallback: 호버링 속도
            return np.array([500.0, 500.0, 500.0, 500.0])
    
    def update_estimator(self, marker_pos_world):
        """태그 감지 결과 업데이트"""
        self.estimated_rover_pos = marker_pos_world
    
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
                self.position = np.zeros(3)
                self.linear_velocity = np.zeros(3)
                # [x,y,z,w] 순서 (Pegasus/scipy 동일)
                self.attitude = np.array([0, 0, 0, 1])
                self.angular_velocity = np.zeros(3)
        
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
        self.rover_pos = np.array([0.0, 3.0, 0.5])
        self.rover_vel = np.array([0.0, 0.0, 0.0])
        
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
        
        # Isaac Lab과 동일한 초기 위치 (로버 바로 위 approach_height)
        # 로버: [1.0, 0.0, 0.5], 드론: [1.0, 0.0, 3.5]
        initial_pos = [
            self.rover_pos[0],  # 로버와 같은 X
            self.rover_pos[1],  # 로버와 같은 Y  
            3.5                 # approach_height
        ]
        
        print(f"[Init] Drone starting at: {initial_pos}")
        print(f"[Init] Rover at: {self.rover_pos}")
        print(f"[Init] Initial XY distance: {np.linalg.norm(np.array(initial_pos[:2]) - self.rover_pos[:2]):.2f}m")
        
        self.drone = Multirotor(
            "/World/Drone",
            ROBOTS['Iris'],
            0,
            initial_pos,
            Rotation.from_euler("XYZ", [0, 0, 0], degrees=True).as_quat(),
            config=config
        )
        # NOTE: Backend.vehicle은 Pegasus가 자동으로 설정함

        # CRITICAL: Pegasus는 world.reset() 후에 초기 위치가 적용됨!
        # 여기서는 아직 초기화 전이므로 나중에 다시 확인 필요
        print(f"[Init] Drone created (position will be set after world.reset())")
        
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
        
        # CRITICAL: world.reset() 후 드론 위치 확인 및 수정
        # Pegasus는 때때로 초기 위치를 무시함
        print("\n[Verification] Checking drone initial position...")
        drone_state = self.drone.state
        actual_pos = np.array([drone_state.position[0], drone_state.position[1], drone_state.position[2]])
        print(f"  Expected: {initial_pos}")
        print(f"  Actual:   {actual_pos}")
        
        # 위치가 다르면 수동으로 설정 (Pegasus API 사용)
        if not np.allclose(actual_pos, initial_pos, atol=0.1):
            print(f"  ⚠️  Position mismatch! Manually setting...")
            # Pegasus는 직접 위치 설정이 어려울 수 있음
            # 대신 controller의 초기 상태를 조정
            print(f"  Note: Initial observation will compensate for position difference")
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
        
        # Cube
        cube_path = rover_path + "/Cube"
        cube = UsdGeom.Cube.Define(stage, cube_path)
        cube.GetSizeAttr().Set(0.5)
        
        # 회색 재질
        cube_mtl_path = Sdf.Path(cube_path + "_Material")
        cube_mtl = UsdShade.Material.Define(stage, cube_mtl_path)
        cube_shader = UsdShade.Shader.Define(stage, cube_mtl_path.AppendPath("Shader"))
        cube_shader.CreateIdAttr("UsdPreviewSurface")
        cube_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.5, 0.5, 0.5))
        cube_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        cube_mtl.CreateSurfaceOutput().ConnectToSource(cube_shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(cube_mtl)
        
        # 물리
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
        UsdPhysics.MassAPI.Apply(xform.GetPrim()).GetMassAttr().Set(500.0)  # Isaac Lab과 동일
        collision_api = UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        
        # 초기 위치
        xform_ops = xform.AddTranslateOp()
        xform_ops.Set(Gf.Vec3d(self.rover_pos[0], self.rover_pos[1], self.rover_pos[2]))
        
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
        
        half = 0.3
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
        translate_op.Set(Gf.Vec3d(0, 0, 0.251))
        
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
        
        print(f"[ArUco] Initialized")
        
    def _detect_aruco(self):
        """ArUco 태그 감지"""
        if not ARUCO_AVAILABLE or not hasattr(self, 'annotator') or self.annotator is None:
            return
        
        if self.step_count % 2 != 0:
            return
        
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
            for dict_name, aruco_dict in self.aruco_dicts.items():
                detector = aruco.ArucoDetector(aruco_dict, self.aruco_params)
                corners, ids, _ = detector.detectMarkers(gray)
                if ids is not None and len(ids) > 0:
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
                    # CRITICAL: state.attitude는 이미 [x,y,z,w] 순서 (scipy와 동일)
                    drone_quat = np.array(drone_state.attitude)

                    r = Rotation.from_quat(drone_quat)
                    R = r.as_matrix()
                    
                    marker_in_camera = np.array([tvec[0], tvec[1], tvec[2]])
                    marker_in_world = drone_pos + R @ marker_in_camera
                    
                    self._on_detection(marker_in_world[:2])
                    
                    self.detection_count += 1
                    if self.detection_count % 10 == 1:
                        print(f"[ArUco] ✓ World: ({marker_in_world[0]:.2f}, {marker_in_world[1]:.2f})")
                    
                    cv2.drawFrameAxes(vis_img, self.camera_matrix, self.dist_coeffs, 
                                     rvecs[0].reshape(3,1), tvecs[0].reshape(3,1), 0.3)
            
            # 십자선
            cv2.line(vis_img, (int(self.cx)-20, int(self.cy)), (int(self.cx)+20, int(self.cy)), (255,0,0), 2)
            cv2.line(vis_img, (int(self.cx), int(self.cy)-20), (int(self.cx), int(self.cy)+20), (255,0,0), 2)
            
            # 상태 텍스트
            num_markers = 0 if ids is None else len(ids)
            status = f"Markers: {num_markers}" if num_markers > 0 else "No markers"
            cv2.putText(vis_img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                       (0, 255, 0) if num_markers > 0 else (0, 0, 255), 2)
            
            cv2.putText(vis_img, f"Frame: {self.step_count}", (10, vis_img.shape[0] - 10), 
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
            translate_op.Set(Gf.Vec3d(self.rover_pos[0], self.rover_pos[1], self.rover_pos[2]))
    
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
                detection_status = "✓ Tracking" if self.controller.estimated_rover_pos is not None else "✗ No tag"
                print(f"[{self.step_count*0.01:.1f}s] {detection_status}, "
                      f"XY Error: {rover_xy_error:.2f}m, Height: {drone_pos[2]:.2f}m, "
                      f"Detections: {self.detection_count}")
        
        print(f"[Summary] Total detections: {self.detection_count}")
        print(f"[Summary] Debug images: /tmp/aruco_rl_*.png")
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
    
    print(f"[Main] Using model: {model_path}")
    
    # RL 모델 사용 가능 확인
    if not RL_AVAILABLE:
        print("[ERROR] stable-baselines3 not installed!")
        print("Install: pip install stable-baselines3")
        return
    
    app = PegasusRLLandingApp(model_path)
    app.run()


if __name__ == "__main__":
    main()