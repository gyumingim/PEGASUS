#!/usr/bin/env python
"""
Pegasus 드론 착륙 시뮬레이션 (Motor Mixing 완전 통합 + 안정화 개선)
- RL 모델로 고급 의사결정
- Motor Mixing으로 4개 모터 개별 제어 (PX4 스타일)
- PD 제어로 안정성 대폭 개선
- ArUco 마커 감지
- 이동 로버 추적
"""

import carb
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.timeline
import omni
from omni.isaac.core.world import World
import torch
import numpy as np
from scipy.spatial.transform import Rotation

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface import PegasusInterface
from pegasus.simulator.logic.backends.backend import Backend

from pxr import Sdf, UsdShade, UsdGeom, Gf, UsdLux

# Stable-Baselines3
try:
    from stable_baselines3 import PPO
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    print("[WARN] stable-baselines3 not available")

# OpenCV (ArUco)
try:
    import cv2
    import cv2.aruco as aruco
    ARUCO_AVAILABLE = True
except ImportError:
    ARUCO_AVAILABLE = False
    print("[WARN] OpenCV not available")


# ============================================================
# ★★★ Motor Mixing 클래스 ★★★
# ============================================================

class QuadcopterMotorMixer:
    """
    쿼드콥터 모터 믹싱 (X 구성)
    
    모터 배치:         회전 방향:
         1             1: CW (시계)
        / \            2: CCW (반시계)
       4   2           3: CW
        \ /            4: CCW
         3
    
    공식:
    f1 = T/4 + P/(4L) + Y/(4k)
    f2 = T/4 + R/(4L) - Y/(4k)
    f3 = T/4 - P/(4L) + Y/(4k)
    f4 = T/4 - R/(4L) - Y/(4k)
    
    T: 총 추력, R: Roll 토크, P: Pitch 토크, Y: Yaw 토크
    L: 암 길이, k: 토크 계수
    """
    
    def __init__(self, arm_length=0.13, thrust_coeff=1e-6, torque_coeff=1e-8):
        self.L = arm_length
        self.k_thrust = thrust_coeff
        self.k_torque = torque_coeff
        
        # Mixing matrix: [T, R, P, Y]^T = M * [f1, f2, f3, f4]^T
        self.mixing_matrix = np.array([
            [1, 1, 1, 1],
            [0, self.L, 0, -self.L],
            [self.L, 0, -self.L, 0],
            [self.k_torque, -self.k_torque, self.k_torque, -self.k_torque]
        ])
        
        self.inv_mixing_matrix = np.linalg.pinv(self.mixing_matrix)
    
    def attitude_to_motor_velocities(self, total_thrust, roll_torque, pitch_torque, yaw_torque):
        """Thrust + Torque → 4개 모터 각속도 (rad/s)"""
        desired = np.array([total_thrust, roll_torque, pitch_torque, yaw_torque])
        motor_forces = self.inv_mixing_matrix @ desired
        motor_forces = np.clip(motor_forces, 0, None)
        motor_velocities = np.sqrt(motor_forces / self.k_thrust)
        return motor_velocities


# ============================================================
# ★★★ RL Controller (안정화 버전) ★★★
# ============================================================

class RLDroneLandingController(Backend):
    """강화학습 + Motor Mixing 드론 제어기 (안정화 개선)"""

    # ============================================================
    # ★★★ 안정화 설정 파라미터 ★★★
    # ============================================================

    USE_MOTOR_MIXING = True      # ★ Motor Mixing 사용 (PX4 스타일)
    USE_ARUCO = True
    DEBUG_MODE = True
    
    # 물리 파라미터
    IRIS_MASS = 1.5              # ★ Iris 질량 (kg)
    TRAIN_MASS = 0.033           # Crazyflie 학습 질량
    TRAIN_THRUST_TO_WEIGHT = 1.9
    
    # Motor Mixing 파라미터
    MOTOR_ARM_LENGTH = 0.13
    MOTOR_THRUST_COEFF = 1e-6
    MOTOR_TORQUE_COEFF = 1e-8
    
    # ★★★ 핵심: 제어 게인 대폭 감소! ★★★
    K_P_ANGLE = 3.0              # 10.0 → 3.0 (70% 감소)
    K_D_ANGLE = 0.5              # 새로 추가! (댐핑)
    
    # ★★★ 관성 모멘트 증가 (안정성) ★★★
    I_XX = 0.05                  # 0.029 → 0.05
    I_YY = 0.05                  # 0.029 → 0.05
    I_ZZ = 0.08                  # 0.055 → 0.08
    
    # ★★★ 각도 제한 강화 ★★★
    MAX_ANGLE_DEG = 5.0          # 10도 → 5도
    MAX_YAW_RATE = 0.5           # 1.0 → 0.5
    
    # ★★★ Thrust 범위 축소 ★★★
    THRUST_MIN = 0.35            # 0.3 → 0.35
    THRUST_MAX = 0.55            # 0.6 → 0.55
    
    # 스케일
    VEL_SCALE = 1.0
    ROLL_SCALE = 1.0
    PITCH_SCALE = 1.0
    YAW_SCALE = 1.0

    # ============================================================

    def __init__(self, rover_initial_pos, rover_velocity, model_path, device="cuda", detection_callback=None):
        super().__init__(config=None)

        self.rl_device = device
        self.rover_pos = np.array(rover_initial_pos, dtype=np.float32)
        self.rover_vel = np.array(rover_velocity, dtype=np.float32)

        # RL 모델 로드
        if RL_AVAILABLE:
            print(f"[RL] Loading model from: {model_path}")
            self.model = PPO.load(model_path, device=device)
            print(f"[RL] Model loaded successfully")
        else:
            raise ImportError("stable-baselines3 not installed!")

        self.gravity = 9.81

        # ★★★ Motor Mixer 초기화 ★★★
        if self.USE_MOTOR_MIXING:
            self.motor_mixer = QuadcopterMotorMixer(
                arm_length=self.MOTOR_ARM_LENGTH,
                thrust_coeff=self.MOTOR_THRUST_COEFF,
                torque_coeff=self.MOTOR_TORQUE_COEFF
            )
            
            # Motor Mixing 명령 저장용
            self.current_thrust = 0.0
            self.current_roll_torque = 0.0
            self.current_pitch_torque = 0.0
            self.current_yaw_torque = 0.0

        # ★★★ 추가: 각속도 저장 (미분 제어용) ★★★
        self.last_ang_vel = np.zeros(3, dtype=np.float32)

        # 설정 출력
        print("\n" + "="*60)
        print("★ RL Controller 설정 (안정화 버전) ★")
        print("="*60)
        print(f"  USE_MOTOR_MIXING: {self.USE_MOTOR_MIXING}")
        print(f"  USE_ARUCO:        {self.USE_ARUCO}")
        print(f"  Drone Mass:       {self.IRIS_MASS} kg")
        print(f"  Hover Thrust:     {self.IRIS_MASS * self.gravity:.2f} N")
        if self.USE_MOTOR_MIXING:
            print(f"  Arm Length:       {self.MOTOR_ARM_LENGTH} m")
            print(f"  K_P_ANGLE:        {self.K_P_ANGLE}")
            print(f"  K_D_ANGLE:        {self.K_D_ANGLE}")
            print(f"  MAX_ANGLE:        {self.MAX_ANGLE_DEG}°")
            print(f"  THRUST_RANGE:     {self.THRUST_MIN:.2f} - {self.THRUST_MAX:.2f}")
        print(f"  DEBUG_MODE:       {self.DEBUG_MODE}")
        print("="*60 + "\n")
        
        self.dt = 0.01
        self.time = 0.0
        self.estimated_rover_pos = None
        self.detection_callback = detection_callback
        self._state = None
        self.landing_height = 0.75
        
        if self.USE_ARUCO:
            self.desired_pos_w = None
        else:
            self.desired_pos_w = np.array(rover_initial_pos, dtype=np.float32)
            self.desired_pos_w[2] = self.landing_height
        
        self._obs_debug_count = 0
        self._action_debug_count = 0

    def start(self):
        print("[Stabilized Motor Mixing Controller] Started")
    
    def stop(self):
        print("[Stabilized Motor Mixing Controller] Stopped")

    def reset(self):
        self.time = 0.0
        self.estimated_rover_pos = None
        self._obs_debug_count = 0
        self._action_debug_count = 0
        self.last_ang_vel = np.zeros(3, dtype=np.float32)  # ★ 추가

    def update(self, dt: float):
        self.dt = dt
        self.time += dt
        
        if self.USE_ARUCO:
            if self.estimated_rover_pos is not None:
                if self.desired_pos_w is None:
                    self.desired_pos_w = self.estimated_rover_pos.copy()
                else:
                    self.desired_pos_w[:2] = self.estimated_rover_pos[:2]
                    self.desired_pos_w[2] = self.rover_pos[2]
        else:
            if self.desired_pos_w is None:
                self.desired_pos_w = np.array(self.rover_pos, dtype=np.float32)
            else:
                self.desired_pos_w[:2] = self.rover_pos[:2]
                self.desired_pos_w[2] = self.rover_pos[2]

    def set_rover_pos(self, pos):
        self.rover_pos[:] = pos
    
    def input_reference(self):
        """★★★ Motor Mixing으로 로터 속도 반환 ★★★"""
        state = self._get_vehicle_state()

        # ArUco 대기
        if self.USE_ARUCO and self.desired_pos_w is None:
            hover_thrust = self.IRIS_MASS * self.gravity
            if self.vehicle:
                rotor_velocities = self.vehicle.force_and_torques_to_velocities(
                    hover_thrust, np.array([0.0, 0.0, 0.0])
                )
                return rotor_velocities.tolist()
            return [500.0, 500.0, 500.0, 500.0]
        
        # Observation 구성
        obs = self._construct_observation(state)
        
        # RL 모델 실행
        action, _states = self.model.predict(obs, deterministic=True)
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        action = action.flatten()
        
        if self.USE_MOTOR_MIXING:
            # ★★★ 안정화된 Motor Mixing 방식 ★★★
            self._compute_motor_mixing_command(action, state)
            
            motor_velocities = self.motor_mixer.attitude_to_motor_velocities(
                self.current_thrust,
                self.current_roll_torque,
                self.current_pitch_torque,
                self.current_yaw_torque
            )
            
            return motor_velocities.tolist()
        else:
            # 직접 제어 (기존 방식)
            rotor_velocities = self._action_to_rotor_velocities(action, state)
            return rotor_velocities.tolist()
    
    def _compute_motor_mixing_command(self, action, state):
        """안정화된 Thrust + Torque (PD 제어)"""
        
        thrust_norm = action[0]
        roll_moment = action[1]
        pitch_moment = action[2]
        yaw_moment = action[3]
        
        # 1. Thrust (좁은 범위)
        thrust_normalized = (thrust_norm + 1.0) / 2.0
        thrust_normalized = thrust_normalized * (self.THRUST_MAX - self.THRUST_MIN) + self.THRUST_MIN
        thrust_normalized = np.clip(thrust_normalized, 0.0, 1.0)
        
        max_thrust = self.IRIS_MASS * self.gravity * 2.0
        total_thrust = thrust_normalized * max_thrust
        
        # 2. Attitude (작은 각도)
        max_angle = np.radians(self.MAX_ANGLE_DEG)
        roll_angle = np.clip(roll_moment * 0.2, -max_angle, max_angle)
        pitch_angle = np.clip(pitch_moment * 0.2, -max_angle, max_angle)
        yaw_rate = np.clip(yaw_moment * 0.3, -self.MAX_YAW_RATE, self.MAX_YAW_RATE)
        
        # 3. PD 제어 (안정성!)
        ang_vel = np.array(state.angular_velocity, dtype=np.float32)
        
        # P (비례)
        roll_accel_p = roll_angle * self.K_P_ANGLE
        pitch_accel_p = pitch_angle * self.K_P_ANGLE
        yaw_accel_p = yaw_rate * self.K_P_ANGLE
        
        # D (미분 - 댐핑!)
        ang_vel_diff = ang_vel - self.last_ang_vel
        roll_accel_d = -ang_vel_diff[0] * self.K_D_ANGLE / max(self.dt, 0.001)
        pitch_accel_d = -ang_vel_diff[1] * self.K_D_ANGLE / max(self.dt, 0.001)
        yaw_accel_d = -ang_vel_diff[2] * self.K_D_ANGLE / max(self.dt, 0.001)
        
        roll_accel = roll_accel_p + roll_accel_d
        pitch_accel = pitch_accel_p + pitch_accel_d
        yaw_accel = yaw_accel_p + yaw_accel_d
        
        self.last_ang_vel = ang_vel.copy()
        
        # 토크
        roll_torque = self.I_XX * roll_accel
        pitch_torque = self.I_YY * pitch_accel
        yaw_torque = self.I_ZZ * yaw_accel
        
        # 4. 저주파 필터 (부드럽게!)
        alpha = 0.7
        self.current_thrust = alpha * total_thrust + (1 - alpha) * self.current_thrust
        self.current_roll_torque = alpha * roll_torque + (1 - alpha) * self.current_roll_torque
        self.current_pitch_torque = alpha * pitch_torque + (1 - alpha) * self.current_pitch_torque
        self.current_yaw_torque = alpha * yaw_torque + (1 - alpha) * self.current_yaw_torque
        
        # 디버그
        if self._action_debug_count % 10 == 0:
            height = self._state.position[2] if self._state else 0
            rpm = self.motor_mixer.attitude_to_motor_velocities(
                self.current_thrust, self.current_roll_torque,
                self.current_pitch_torque, self.current_yaw_torque
            ) * 60 / (2 * np.pi)
            
            print(f"[Stable] T={self.current_thrust:5.1f}N, "
                  f"ang=[{np.degrees(roll_angle):4.1f}°,{np.degrees(pitch_angle):4.1f}°], "
                  f"RPM=[{rpm[0]:.0f},{rpm[1]:.0f},{rpm[2]:.0f},{rpm[3]:.0f}], h={height:.2f}m")
        
        self._action_debug_count += 1
    
    def _construct_observation(self, state):
        """IsaacLab 동일 16차원 observation"""
        pos = np.array(state.position, dtype=np.float32)
        lin_vel = np.array(state.linear_velocity, dtype=np.float32)
        ang_vel = np.array(state.angular_velocity, dtype=np.float32)
        quat_xyzw = np.array(state.attitude, dtype=np.float32)

        R = Rotation.from_quat(quat_xyzw)
        lin_vel_b = R.inv().apply(lin_vel) * self.VEL_SCALE
        ang_vel_obs = ang_vel
        
        gravity_world = np.array([0, 0, -self.gravity], dtype=np.float32)
        gravity_b = R.inv().apply(gravity_world)

        if self.desired_pos_w is not None:
            goal_rel_world = self.desired_pos_w - pos
            desired_pos_b = R.inv().apply(goal_rel_world)
        else:
            desired_pos_b = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        rel_vel_world = lin_vel - self.rover_vel
        rel_vel_b = R.inv().apply(rel_vel_world) * self.VEL_SCALE

        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        current_yaw = np.arctan2(
            2.0 * (quat_wxyz[0]*quat_wxyz[3] + quat_wxyz[1]*quat_wxyz[2]),
            1.0 - 2.0 * (quat_wxyz[2]**2 + quat_wxyz[3]**2)
        )

        if (self.DEBUG_MODE or self._obs_debug_count < 5) and self._obs_debug_count % 50 == 1:
            dist = np.linalg.norm(goal_rel_world) if self.desired_pos_w is not None else 0
            print(f"\n📊 Obs: pos=[{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}], dist={dist:.2f}m")
        self._obs_debug_count += 1

        obs = np.concatenate([
            lin_vel_b, ang_vel_obs, gravity_b,
            desired_pos_b, rel_vel_b, [current_yaw]
        ])

        return obs.astype(np.float32)
    
    def _action_to_rotor_velocities(self, action, state):
        """직접 제어 방식 (USE_MOTOR_MIXING=False일 때)"""
        action = np.clip(action, -1.0, 1.0)

        thrust_action = action[0]
        roll_action = action[1] * self.ROLL_SCALE
        pitch_action = action[2] * self.PITCH_SCALE
        yaw_action = action[3] * self.YAW_SCALE

        mass_ratio = self.IRIS_MASS / self.TRAIN_MASS
        thrust_ratio = self.TRAIN_THRUST_TO_WEIGHT * (thrust_action + 1.0) / 2.0
        thrust_force = thrust_ratio * self.IRIS_MASS * self.gravity

        moments = np.array([roll_action, pitch_action, yaw_action])
        torques = moments * 0.002 * mass_ratio

        if self.vehicle:
            rotor_velocities = self.vehicle.force_and_torques_to_velocities(thrust_force, torques)
            return np.array(rotor_velocities)
        return np.array([500.0, 500.0, 500.0, 500.0])
    
    def update_estimator(self, marker_pos_world):
        self.estimated_rover_pos = marker_pos_world
    
    def update_sensor(self, sensor_type: str, sensor_data: dict):
        pass
    
    def update_state(self, state: dict):
        self._state = state
    
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


# ============================================================
# ★★★ 메인 앱 (완전한 코드) ★★★
# ============================================================

class PegasusRLLandingApp:
    """Pegasus RL 착륙 앱"""
    
    def __init__(self, model_path):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])
        
        self.rover_pos = np.array([0.0, 0.0, 0.375], dtype=np.float32)
        self.rover_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        self.controller = RLDroneLandingController(
            self.rover_pos.copy(),
            self.rover_vel.copy(),
            model_path=model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            detection_callback=self._on_detection
        )
        
        config = MultirotorConfig()
        config.backends = [self.controller]
        
        initial_pos = [-2.5, -2.5, 3.5]
        
        self.drone = Multirotor(
            "/World/Drone",
            ROBOTS['Iris'],
            0,
            initial_pos,
            Rotation.from_euler("XYZ", [0, 0, 0], degrees=True).as_quat(),
            config=config
        )
        
        self._add_lighting()
        self._create_rover()
        self._setup_camera()
        
        if ARUCO_AVAILABLE:
            self._init_aruco()
        
        self.world.reset()
        
        self.step_count = 0
        self.detection_count = 0
        self.last_saved_frame = -1
        self.last_detection_time = 0.0
        
    def _add_lighting(self):
        stage = omni.usd.get_context().get_stage()
        
        distant_light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
        distant_light.CreateIntensityAttr(5000.0)
        distant_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 0.95))
        distant_light.CreateAngleAttr(0.53)
        
        xform = UsdGeom.Xformable(distant_light)
        xform.ClearXformOpOrder()
        rotate_op = xform.AddRotateXYZOp()
        rotate_op.Set(Gf.Vec3f(-45, 45, 0))
        
        dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome_light.CreateIntensityAttr(1000.0)
        dome_light.CreateColorAttr(Gf.Vec3f(0.9, 0.95, 1.0))
        
    def _create_rover(self):
        stage = omni.usd.get_context().get_stage()
        from pxr import UsdPhysics

        rover_path = "/World/Rover"
        xform = UsdGeom.Xform.Define(stage, rover_path)

        cube_path = rover_path + "/Cube"
        cube = UsdGeom.Cube.Define(stage, cube_path)
        cube.GetSizeAttr().Set(0.75)

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
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

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
        
    def _add_apriltag_texture(self):
        stage = omni.usd.get_context().get_stage()
        
        mesh_path = "/World/Rover/TagMesh"
        mesh = UsdGeom.Mesh.Define(stage, mesh_path)

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
        translate_op.Set(Gf.Vec3d(0, 0, 0.376))
        
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
            
    def _generate_apriltag_image(self):
        if not ARUCO_AVAILABLE:
            return "/tmp/dummy_tag.png"
        
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
        tag_size = 512
        tag_image = np.zeros((tag_size, tag_size), dtype=np.uint8)
        tag_image = aruco.generateImageMarker(aruco_dict, 0, tag_size, tag_image, 1)
        
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
        
        camera_prim.GetFocalLengthAttr().Set(8.0)
        camera_prim.GetHorizontalApertureAttr().Set(60.0)
        camera_prim.GetVerticalApertureAttr().Set(33.75)
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
            except Exception as e:
                print(f"[WARN] Camera setup failed: {e}")
                self.annotator = None
        
    def _init_aruco(self):
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
        }
        self.aruco_params = aruco.DetectorParameters()
        
    def _detect_aruco(self):
        try:
            image_data = self.annotator.get_data()
            if image_data is None or not isinstance(image_data, np.ndarray):
                return
            
            if len(image_data.shape) == 3:
                gray = cv2.cvtColor(image_data[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = image_data.astype(np.uint8)
            
            corners, ids = None, None
            for dict_name, aruco_dict in self.aruco_dicts.items():
                detector = aruco.ArucoDetector(aruco_dict, self.aruco_params)
                corners, ids, _ = detector.detectMarkers(gray)
                if ids is not None and len(ids) > 0:
                    break
            
            if ids is not None and len(ids) > 0:
                rvecs, tvecs = self._estimate_pose(corners, 0.768)
                
                if tvecs is not None and len(tvecs) > 0:
                    tvec = tvecs[0][0]
                    drone_state = self.drone.state
                    drone_pos = np.array(drone_state.position)
                    drone_quat = np.array(drone_state.attitude)
                    r = Rotation.from_quat(drone_quat)

                    marker_in_body = np.array([-tvec[1]*1.3, tvec[0]*1.3, -tvec[2]])
                    marker_in_world = drone_pos + r.apply(marker_in_body)
                    
                    self._on_detection(marker_in_world[:2])
                    self.detection_count += 1
                    self.last_detection_time = self.step_count * 0.01
            
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
        full_pos = np.array([marker_pos_xy[0], marker_pos_xy[1], self.rover_pos[2]])
        self.controller.update_estimator(full_pos)

    def _update_rover(self, dt):
        stage = omni.usd.get_context().get_stage()
        rover_prim = stage.GetPrimAtPath("/World/Rover")
        if not rover_prim.IsValid():
            return

        self.rover_pos += self.rover_vel * dt
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
        self.timeline.play()
        
        print("[Camera] Initializing...")
        for _ in range(300):
            self.world.step(render=True)
            self.step_count += 1
        print("[Camera] Ready!")
        
        while simulation_app.is_running():
            self._detect_aruco()
            self._update_rover(self.world.get_physics_dt())
            self.world.step(render=True)
            self.step_count += 1
            
            if self.step_count % 100 == 0:
                drone_state = self.drone.state
                drone_pos = np.array([drone_state.position[0], drone_state.position[1], drone_state.position[2]])
                rover_xy_error = np.linalg.norm(drone_pos[:2] - self.rover_pos[:2])
                
                mode = "Stable-MotorMix" if self.controller.USE_MOTOR_MIXING else "Direct"
                detection_status = "✓" if self.controller.estimated_rover_pos is not None else "✗"
                
                print(f"[{self.step_count*0.01:.1f}s | {mode}] {detection_status} | "
                    f"XY err: {rover_xy_error:.2f}m | Height: {drone_pos[2]:.2f}m | "
                    f"Detections: {self.detection_count}")
        
        carb.log_warn("Closing...")
        self.timeline.stop()
        simulation_app.close()


def main():
    import sys
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "/home/rtx5080/s/ISAAC_LAB_DRONE/logs/sb3/Template-DroneLanding-v0/2026-01-20_15-52-16/model.zip"

    print(f"\n{'='*70}")
    print(f"안정화된 Motor Mixing 드론 착륙 시뮬레이션")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Motor Mixing: {RLDroneLandingController.USE_MOTOR_MIXING}")
    print(f"ArUco: {RLDroneLandingController.USE_ARUCO}")
    print(f"K_P_ANGLE: {RLDroneLandingController.K_P_ANGLE}")
    print(f"K_D_ANGLE: {RLDroneLandingController.K_D_ANGLE}")
    print(f"MAX_ANGLE: {RLDroneLandingController.MAX_ANGLE_DEG}°")
    print(f"{'='*70}\n")

    if not RL_AVAILABLE:
        print("[ERROR] stable-baselines3 not installed!")
        return

    app = PegasusRLLandingApp(model_path)
    app.run()
if __name__ == "__main__":
    main()