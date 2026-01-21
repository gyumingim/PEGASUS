#!/usr/bin/env python
"""
Pegasus 드론 착륙 시뮬레이션
- 움직이는 AprilTag 로버 추적
- 카메라 기반 태그 감지
- 안정적 착륙 제어
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

from pxr import Sdf, UsdShade, UsdGeom, Gf

# OpenCV (ArUco 감지)
try:
    import cv2
    import cv2.aruco as aruco
    ARUCO_AVAILABLE = True
except ImportError:
    ARUCO_AVAILABLE = False
    print("[WARN] OpenCV not available")


class DroneLandingController(Backend):
    """드론 착륙 제어기 (Pegasus Backend)"""
    
    def __init__(self, rover_initial_pos, rover_velocity, detection_callback=None):
        super().__init__()
        
        # 로버 설정
        self.rover_pos = np.array(rover_initial_pos, dtype=np.float32)
        self.rover_vel = np.array(rover_velocity, dtype=np.float32)
        
        # 제어 파라미터
        self.Kp_pos = np.array([1.5, 1.5, 2.0])  # XYZ 위치 게인
        self.Kd_pos = np.array([0.8, 0.8, 1.2])  # XYZ 속도 게인
        self.Kp_att = np.array([3.0, 3.0, 1.5])  # 롤/피치/요 게인
        self.Kd_att = np.array([0.5, 0.5, 0.3])  # 각속도 게인
        
        # 물리 파라미터
        self.mass = 0.033  # kg
        self.gravity = 9.81
        self.max_thrust = 0.6  # N
        self.max_torque = 0.002  # N⋅m
        
        # 상태
        self.dt = 0.01
        self.time = 0.0
        self.estimated_rover_pos = None  # 태그 감지로 업데이트
        self.detection_callback = detection_callback
        self.vehicle = None  # Pegasus vehicle 참조 (나중에 설정)
        
        # 착륙 상태
        self.phase = "APPROACH"  # APPROACH -> DESCEND -> LANDED
        self.approach_height = 2.0
        self.align_threshold = 0.3  # XY 정렬 임계값
        self.landing_height = 0.75  # 로버 표면 높이
        
        # 바람 효과
        self.wind_velocity = np.array([0.0, 0.0, 0.0])
        self._randomize_wind()
        
    def _randomize_wind(self):
        """바람 속도 랜덤화 (XY 평면, 0~2 m/s)"""
        wind_speed = np.random.rand() * 2.0  # 0~2 m/s
        wind_angle = np.random.rand() * 2 * np.pi  # 0~2π
        self.wind_velocity[0] = wind_speed * np.cos(wind_angle)
        self.wind_velocity[1] = wind_speed * np.sin(wind_angle)
        self.wind_velocity[2] = 0.0  # Z 방향 바람 없음
        
    def update(self, dt: float):
        """Backend 필수 메서드"""
        self.dt = dt
        self.time += dt
        
        # 로버 이동
        self.rover_pos += self.rover_vel * dt
        
    def input_reference(self):
        """목표 위치 반환"""
        if self.estimated_rover_pos is not None:
            # 태그 감지됨: 추정 위치 사용
            target_xy = self.estimated_rover_pos[:2]
        else:
            # 태그 미감지: 현재 위치 유지 (호버링)
            state = self._get_vehicle_state()
            target_xy = state["position"][:2]
        
        # Z 목표: Phase별 다른 높이
        if self.phase == "APPROACH":
            target_z = self.approach_height
        else:  # DESCEND or LANDED
            target_z = self.landing_height
        
        return np.array([target_xy[0], target_xy[1], target_z])
    
    def update_estimator(self, marker_pos_world):
        """태그 감지 결과 업데이트"""
        self.estimated_rover_pos = marker_pos_world
    
    def compute_control(self, state):
        """PID 제어 계산"""
        pos = state["position"]
        vel = state["velocity"]
        quat = state["attitude"]  # [w, x, y, z]
        ang_vel = state["angular_velocity"]
        
        # 목표 위치
        target_pos = self.input_reference()
        target_vel = self.rover_vel  # 로버 속도 매칭
        
        # 위치 오차
        pos_error = target_pos - pos
        vel_error = target_vel - vel
        
        # XY 정렬 확인
        xy_error = np.linalg.norm(pos_error[:2])
        if xy_error < self.align_threshold and self.phase == "APPROACH":
            self.phase = "DESCEND"
            print(f"[{self.time:.1f}s] Phase: APPROACH -> DESCEND")
        
        # 착륙 확인
        if abs(pos[2] - self.landing_height) < 0.1 and xy_error < 0.1:
            if self.phase == "DESCEND":
                self.phase = "LANDED"
                print(f"[{self.time:.1f}s] 착륙 성공!")
        
        # 추력 계산 (PD 제어)
        thrust_vec = (self.Kp_pos * pos_error + self.Kd_pos * vel_error) * self.mass
        thrust_vec[2] += self.mass * self.gravity  # 중력 보상
        
        # 바람 효과 추가 (드래그 모델)
        wind_drag = 0.01 * self.wind_velocity
        thrust_vec += wind_drag * self.mass
        
        # 추력 크기
        thrust = np.linalg.norm(thrust_vec)
        thrust = np.clip(thrust, 0, self.max_thrust)
        
        # 목표 자세 계산 (추력 방향)
        if thrust > 1e-6:
            thrust_dir = thrust_vec / np.linalg.norm(thrust_vec)
        else:
            thrust_dir = np.array([0, 0, 1])
        
        # Roll/Pitch 계산 (간단한 버전)
        target_roll = np.arcsin(-thrust_dir[1])
        target_pitch = np.arcsin(thrust_dir[0] / np.cos(target_roll))
        target_yaw = 0.0  # 북쪽 고정
        
        # 현재 자세 (쿼터니언 -> 오일러)
        r = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # [x,y,z,w]
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)
        
        # 자세 오차
        att_error = np.array([
            target_roll - roll,
            target_pitch - pitch,
            target_yaw - yaw
        ])
        
        # 토크 계산
        torques = self.Kp_att * att_error - self.Kd_att * ang_vel
        torques = np.clip(torques, -self.max_torque, self.max_torque)
        
        return thrust, torques
    
    def update_sensor(self, sensor_data: dict):
        """센서 데이터 수신 (Pegasus Backend 필수 메서드)"""
        pass
    
    def update_state(self, state: dict):
        """드론 상태 업데이트 (Pegasus Backend 필수 메서드)"""
        self._state = state
        
    def _get_vehicle_state(self):
        """현재 드론 상태 반환"""
        return getattr(self, '_state', {
            "position": np.zeros(3),
            "velocity": np.zeros(3),
            "attitude": np.array([1, 0, 0, 0]),
            "angular_velocity": np.zeros(3)
        })


class PegasusLandingApp:
    """Pegasus 착륙 시뮬레이션 앱"""
    
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        
        # 환경 로드
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])
        
        # 로버 설정
        self.rover_pos = np.array([2.0, -1.5, 0.5])  # 초기 위치
        self.rover_vel = np.array([0.2, 0.1, 0.0])  # 이동 속도
        
        # 제어기 생성
        self.controller = DroneLandingController(
            self.rover_pos.copy(),
            self.rover_vel.copy(),
            detection_callback=self._on_detection
        )
        
        # 드론 생성
        config = MultirotorConfig()
        config.backends = [self.controller]
        
        self.drone = Multirotor(
            "/World/Drone",
            ROBOTS['Iris'],
            0,
            [2.0, -1.5, 2.0],  # approach_height에서 시작
            Rotation.from_euler("XYZ", [0, 0, 0], degrees=True).as_quat(),
            config=config
        )
        
        # Controller에 vehicle 참조 설정
        self.controller.vehicle = self.drone
        
        # 로버(AprilTag) 생성
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
        
    def _create_rover(self):
        """AprilTag 로버 생성 (RigidPrim 대신 간단한 Cube)"""
        stage = omni.usd.get_context().get_stage()
        from pxr import UsdGeom, UsdPhysics
        
        # Xform 생성 (Transform 노드)
        rover_path = "/World/Rover"
        xform = UsdGeom.Xform.Define(stage, rover_path)
        
        # Cube 메시 추가
        cube_path = rover_path + "/Cube"
        cube = UsdGeom.Cube.Define(stage, cube_path)
        cube.GetSizeAttr().Set(0.5)  # 0.5m 크기
        
        # 물리 속성 (RigidBody)
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
        UsdPhysics.MassAPI.Apply(xform.GetPrim()).GetMassAttr().Set(1.0)
        
        # Collider (충돌 감지)
        collision_api = UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        
        # 초기 위치 설정
        xform_ops = xform.AddTranslateOp()
        xform_ops.Set(Gf.Vec3d(self.rover_pos[0], self.rover_pos[1], self.rover_pos[2]))
        
        # AprilTag 텍스처 추가
        self._add_apriltag_texture()
        
        print("[Rover] Created at", self.rover_pos)
        
    def _add_apriltag_texture(self):
        """AprilTag 텍스처 메시 생성"""
        stage = omni.usd.get_context().get_stage()
        
        # 평면 메시 (로버 윗면)
        mesh_path = "/World/Rover/TagMesh"
        mesh = UsdGeom.Mesh.Define(stage, mesh_path)
        
        # 정사각형 (0.5m x 0.5m)
        half = 0.25
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
        
        # UV 좌표
        texcoords = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        )
        texcoords.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])
        
        # 위치 (큐브 윗면)
        xform = UsdGeom.Xformable(mesh)
        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(0, 0, 0.251))
        
        # 재질
        mtl_path = Sdf.Path(mesh_path + "_Material")
        mtl = UsdShade.Material.Define(stage, mtl_path)
        
        shader = UsdShade.Shader.Define(stage, mtl_path.AppendPath("Shader"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        
        # 텍스처 (model.zip 압축 해제 후 경로 수정 필요!)
        texture_path = "/path/to/tag586_ariel.png"  # TODO: 실제 경로로 수정
        
        st_reader = UsdShade.Shader.Define(stage, mtl_path.AppendPath("stReader"))
        st_reader.CreateIdAttr("UsdPrimvarReader_float2")
        st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        
        diffuse_tex = UsdShade.Shader.Define(stage, mtl_path.AppendPath("DiffuseTexture"))
        diffuse_tex.CreateIdAttr("UsdUVTexture")
        diffuse_tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_path)
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
        
        print("[Rover] AprilTag texture added")
        
    def _setup_camera(self):
        """드론에 하향 카메라 부착"""
        stage = omni.usd.get_context().get_stage()
        
        camera_path = "/World/Drone/body/Camera"
        camera_prim = UsdGeom.Camera.Define(stage, camera_path)
        
        # 150도 광각 FOV
        camera_prim.GetFocalLengthAttr().Set(8.0)
        camera_prim.GetHorizontalApertureAttr().Set(60.0)
        camera_prim.GetVerticalApertureAttr().Set(33.75)  # 16:9
        camera_prim.GetFocusDistanceAttr().Set(1000.0)
        camera_prim.GetFStopAttr().Set(0.0)
        camera_prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10000.0))
        
        # 위치 (드론 아래 15cm)
        xform = UsdGeom.Xformable(camera_prim)
        xform.ClearXformOpOrder()
        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(0, 0, -0.15))
        
        # 렌더 프로덕트
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
        # 카메라 행렬
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
        
        # ArUco 사전들
        self.aruco_dicts = {
            "DICT_APRILTAG_36h11": aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11),
            "DICT_APRILTAG_25h9": aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_25h9),
            "DICT_APRILTAG_16h5": aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_16h5),
        }
        self.aruco_params = aruco.DetectorParameters()
        
        print(f"[ArUco] Initialized with {len(self.aruco_dicts)} dictionaries")
        
    def _detect_aruco(self):
        """ArUco 태그 감지"""
        if not ARUCO_AVAILABLE or not hasattr(self, 'annotator') or self.annotator is None:
            return
        
        # 2스텝마다만 감지
        if self.step_count % 2 != 0:
            return
        
        try:
            image_data = self.annotator.get_data()
            if image_data is None:
                return
            
            # 그레이스케일 변환
            if len(image_data.shape) == 3:
                gray = cv2.cvtColor(image_data[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = image_data.astype(np.uint8)
            
            # 감지 시도
            corners, ids = None, None
            for dict_name, aruco_dict in self.aruco_dicts.items():
                detector = aruco.ArucoDetector(aruco_dict, self.aruco_params)
                corners, ids, _ = detector.detectMarkers(gray)
                if ids is not None and len(ids) > 0:
                    break
            
            # 시각화
            vis_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            if ids is not None and len(ids) > 0:
                aruco.drawDetectedMarkers(vis_img, corners, ids)
                
                # 3D 자세 추정
                rvecs, tvecs = self._estimate_pose(corners, 0.5)
                
                if tvecs is not None and len(tvecs) > 0:
                    # 첫 마커 사용
                    tvec = tvecs[0][0]
                    
                    # 드론 상태 (Pegasus API 사용)
                    drone_state = self.drone.state
                    drone_pos = np.array([drone_state.position[0], drone_state.position[1], drone_state.position[2]])
                    drone_quat = np.array([drone_state.attitude[1], drone_state.attitude[2], 
                                          drone_state.attitude[3], drone_state.attitude[0]])  # [x,y,z,w]
                    
                    # 회전 행렬
                    r = Rotation.from_quat(drone_quat)
                    R = r.as_matrix()
                    
                    # 카메라 좌표 -> 세계 좌표
                    marker_in_camera = np.array([tvec[0], tvec[1], tvec[2]])
                    marker_in_world = drone_pos + R @ marker_in_camera
                    
                    # 제어기에 전달
                    self._on_detection(marker_in_world[:2])  # XY만
                    
                    self.detection_count += 1
                    if self.detection_count % 10 == 1:
                        print(f"[ArUco] Detected at world: ({marker_in_world[0]:.2f}, {marker_in_world[1]:.2f})")
                    
                    # 축 그리기
                    cv2.drawFrameAxes(vis_img, self.camera_matrix, self.dist_coeffs, 
                                     rvecs[0].reshape(3,1), tvecs[0].reshape(3,1), 0.25)
            
            # 십자선
            cv2.line(vis_img, (int(self.cx)-20, int(self.cy)), (int(self.cx)+20, int(self.cy)), (255,0,0), 2)
            cv2.line(vis_img, (int(self.cx), int(self.cy)-20), (int(self.cx), int(self.cy)+20), (255,0,0), 2)
            
            # 상태 텍스트
            num_markers = 0 if ids is None else len(ids)
            status = f"Markers: {num_markers}" if num_markers > 0 else "No markers"
            cv2.putText(vis_img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                       (0, 255, 0) if num_markers > 0 else (0, 0, 255), 2)
            
            cv2.imshow("ArUco Detection", vis_img)
            cv2.waitKey(1)
            
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
        # XY 좌표만 업데이트 (Z는 고정)
        full_pos = np.array([marker_pos_xy[0], marker_pos_xy[1], self.rover_pos[2]])
        self.controller.update_estimator(full_pos)
    
    def _update_rover(self, dt):
        """로버 이동 (USD Prim 직접 조작)"""
        stage = omni.usd.get_context().get_stage()
        rover_prim = stage.GetPrimAtPath("/World/Rover")
        
        if not rover_prim.IsValid():
            return
        
        # 위치 업데이트
        self.rover_pos += self.rover_vel * dt
        
        # USD Transform 업데이트
        xformable = UsdGeom.Xformable(rover_prim)
        translate_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
        
        if translate_op:
            translate_op.Set(Gf.Vec3d(self.rover_pos[0], self.rover_pos[1], self.rover_pos[2]))
        
        # 속도 설정 (PhysX)
        try:
            from pxr import PhysxSchema
            rigid_body = PhysxSchema.PhysxRigidBodyAPI(rover_prim)
            if rigid_body:
                rigid_body.GetVelocityAttr().Set(Gf.Vec3f(self.rover_vel[0], self.rover_vel[1], self.rover_vel[2]))
        except:
            pass  # PhysX 속도 설정 실패해도 Transform은 작동
    
    def run(self):
        """메인 루프"""
        self.timeline.play()
        
        while simulation_app.is_running():
            # ArUco 감지
            self._detect_aruco()
            
            # 로버 업데이트
            self._update_rover(self.world.get_physics_dt())
            
            # 물리 스텝
            self.world.step(render=True)
            self.step_count += 1
            
            # 상태 출력 (1초마다)
            if self.step_count % 100 == 0:
                drone_state = self.drone.state
                drone_pos = np.array([drone_state.position[0], drone_state.position[1], drone_state.position[2]])
                rover_xy_error = np.linalg.norm(drone_pos[:2] - self.rover_pos[:2])
                print(f"[{self.step_count*0.01:.1f}s] Phase: {self.controller.phase}, "
                      f"XY Error: {rover_xy_error:.2f}m, Height: {drone_pos[2]:.2f}m, "
                      f"Wind: ({self.controller.wind_velocity[0]:.2f}, {self.controller.wind_velocity[1]:.2f}) m/s")
        
        carb.log_warn("Simulation closing")
        self.timeline.stop()
        simulation_app.close()


def main():
    app = PegasusLandingApp()
    app.run()


if __name__ == "__main__":
    main()