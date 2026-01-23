# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

# PyTorch: GPU 가속 텐서 연산 라이브러리
import torch

# NumPy: 수치 계산 라이브러리 (카메라 행렬 등)
import numpy as np

# Sequence: 타입 힌팅용 (리스트, 튜플 등)
from collections.abc import Sequence

# Isaac Lab 시뮬레이션 유틸리티
import isaaclab.sim as sim_utils

# 로봇(관절체)과 강체 물체 클래스
from isaaclab.assets import Articulation, RigidObject

# 강화학습 환경 베이스 클래스
from isaaclab.envs import DirectRLEnv

# 바닥 생성 함수
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

# 쿼터니언(회전) 수학 함수들
from isaaclab.utils.math import quat_apply, quat_conjugate

# 환경 설정 클래스 import
from .drone_landing_env_cfg import DroneLandingEnvCfg

# ArUco/AprilTag 마커 감지를 위한 OpenCV
try:
    import cv2  # OpenCV 메인 라이브러리
    import cv2.aruco as aruco  # ArUco 마커 감지 모듈

    ARUCO_AVAILABLE = True  # OpenCV 사용 가능
except ImportError:
    ARUCO_AVAILABLE = False  # OpenCV 없음
    print("[WARN] opencv-python not available. Install: pip install opencv-python")


class DroneLandingEnv(DirectRLEnv):
    """
    드론이 움직이는 로버를 추적하고 착륙하는 강화학습 환경.

    목표:
    - XY 평면에서 로버 추적 (로버 바로 위에 위치)
    - 로버 표면까지 안전하게 착륙
    - 로버 속도와 매칭
    - 안정적인 자세 유지
    """

    cfg: DroneLandingEnvCfg  # 환경 설정 객체 (타입 힌트)

    def __init__(
        self, cfg: DroneLandingEnvCfg, render_mode: str | None = None, **kwargs
    ):
        """
        환경 초기화 함수 (생성자).

        Args:
            cfg: 환경 설정 객체
            render_mode: 렌더링 모드 (None, "human", "rgb_array" 등)
            **kwargs: 추가 인자들
        """
        # ========== 카메라 관련 변수 초기화 (super().__init__() 전에 필요!) ==========
        # 왜냐하면 _setup_scene()이 super().__init__() 내부에서 호출되기 때문
        self.apriltag_detector = None  # AprilTag 감지기 (사용 안 함, 추후 확장용)
        self.camera_annotators = []  # 카메라 이미지 캡처 객체들
        self.detection_step_counter = 0  # 감지 시도 횟수 카운터
        self.last_detection_count = 0  # 마지막 감지된 마커 개수

        # ========== ArUco/AprilTag 감지 시스템 초기화 ==========
        if ARUCO_AVAILABLE:
            # 카메라 파라미터 설정 (150° 광각 FOV)
            img_w, img_h = 1280, 720  # 이미지 해상도
            fov_deg = 150.0  # 시야각 150도 (매우 넓음!)

            # 카메라 내부 행렬 계산 (핀홀 카메라 모델)
            # fx, fy: 초점 거리 (픽셀 단위)
            self.fx = img_w / (2 * np.tan(np.radians(fov_deg / 2)))
            self.fy = self.fx  # 정사각형 픽셀 가정
            # cx, cy: 이미지 중심점 (주점)
            self.cx = img_w / 2
            self.cy = img_h / 2

            # 여러 종류의 ArUco/AprilTag 사전 준비 (tag586은 AprilTag일 가능성 높음)
            self.aruco_dicts = {
                "DICT_APRILTAG_36h11": aruco.getPredefinedDictionary(
                    aruco.DICT_APRILTAG_36h11
                ),  # AprilTag 36h11
                "DICT_APRILTAG_25h9": aruco.getPredefinedDictionary(
                    aruco.DICT_APRILTAG_25h9
                ),  # AprilTag 25h9
                "DICT_APRILTAG_16h5": aruco.getPredefinedDictionary(
                    aruco.DICT_APRILTAG_16h5
                ),  # AprilTag 16h5
                "DICT_ARUCO_ORIGINAL": aruco.getPredefinedDictionary(
                    aruco.DICT_ARUCO_ORIGINAL
                ),  # ArUco Original
                "DICT_4X4_50": aruco.getPredefinedDictionary(
                    aruco.DICT_4X4_50
                ),  # ArUco 4x4
                "DICT_5X5_100": aruco.getPredefinedDictionary(
                    aruco.DICT_5X5_100
                ),  # ArUco 5x5
                "DICT_6X6_250": aruco.getPredefinedDictionary(
                    aruco.DICT_6X6_250
                ),  # ArUco 6x6
            }
            self.aruco_params = aruco.DetectorParameters()  # 감지 파라미터 (기본값)
            self.detected_dict_type = None  # 성공한 사전 타입 (캐싱용)

            # 카메라 내부 행렬 (3x3 행렬)
            # [fx  0  cx]
            # [ 0 fy  cy]
            # [ 0  0   1]
            self.camera_matrix = np.array(
                [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]],
                dtype=np.float32,
            )
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)  # 왜곡 계수 (없음)

            print(
                f"[ArUco] Detectors initialized with {len(self.aruco_dicts)} dictionaries (FOV={fov_deg}°, fx={self.fx:.1f})"
            )
        else:
            print("[ArUco] Detector NOT available - install opencv-python")

        # ========== 부모 클래스 초기화 (중요!) ==========
        # 여기서 _setup_scene() 호출됨 → 로봇, 바닥, 카메라 생성
        super().__init__(cfg, render_mode, **kwargs)

        # ========== 드론 물리 속성 계산 ==========
        self.robot_mass = 0.033  # Crazyflie 드론 질량 (kg)
        # 드론 무게 = 질량 × 중력가속도 (abs는 방향 무시)
        self.robot_weight = self.robot_mass * abs(self.sim.cfg.gravity[2])

        # ========== AprilTag(로버) 위치 관리 ==========
        # 초기 위치를 텐서로 변환 (모든 환경에 복제)
        self.apriltag_initial_pos = torch.tensor(
            self.cfg.apriltag_position, device=self.device, dtype=torch.float32
        ).repeat(
            self.num_envs, 1
        )  # shape: (num_envs, 3)

        # 현재 위치 (매 스텝 업데이트됨 - 로버가 움직임!)
        self.apriltag_pos = self.apriltag_initial_pos.clone()

        # 로버 속도 (상수, XY 평면 이동)
        self.apriltag_velocity = torch.tensor(
            self.cfg.apriltag_velocity, device=self.device, dtype=torch.float32
        ).repeat(
            self.num_envs, 1
        )  # shape: (num_envs, 3)

        # ========== 목표 위치 (드론이 도달해야 할 곳) ==========
        # XY: 로버 중심, Z: 로버 표면 (착륙!)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # ========== 성공 여부 추적 ==========
        # True = 착륙 성공, False = 실패/진행 중
        self.tracking_success = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

        # ========== 에피소드별 보상 누적 버퍼 (로깅용) ==========
        # 각 보상 항목의 합계를 추적
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "xy_distance",  # XY 거리 보상
                "z_distance",  # Z 거리 보상
                "velocity_tracking",  # 속도 추적 보상
                "angular_vel_penalty",  # 각속도 패널티
                "orientation_penalty",  # 자세 패널티
                "descent_reward",  # 하강 보상
                "excessive_descent_penalty",  # 과도한 하강 패널티
                "premature_descent",  # 조기 하강 패널티 (Staged Landing)
                "initial_stability",  # 초기 안정성 보너스
                "gentle_approach",  # 표면 근처 천천히 접근 보상
                "alignment",  # 방향 정렬 보상 (0°/90°/180°/270°)
                "premature_movement",  # 정렬 전 이동 패널티
                "yaw_reward"  # yaw(Z축 회전) 정렬 보상 (0°/90°/180°/270° 중 가까운 각도)
            ]
        }

    def _setup_scene(self):
        """3D 씬(장면) 구성: 드론, 로버, 바닥, 카메라, 조명 등."""

        # ========== 1. 드론(로봇) 생성 ==========
        self.robot = Articulation(self.cfg.robot_cfg)

        # ========== 2. AprilTag(로버) 생성 ==========
        self.apriltag = RigidObject(self.cfg.apriltag_cfg)

        # ========== 3. 바닥 생성 ==========
        # 단색 회색 바닥 (격자 무늬 없음 - AprilTag 오감지 방지)
        ground_cfg = GroundPlaneCfg(
            color=(0.5, 0.5, 0.5),  # 중간 회색
            size=(1000.0, 1000.0),  # 1km × 1km 거대한 바닥
        )
        spawn_ground_plane(prim_path="/World/ground", cfg=ground_cfg)

        # ========== 4. 환경 복제 (병렬 학습) ==========
        # num_envs개의 환경을 복제하여 병렬 시뮬레이션
        self.scene.clone_environments(copy_from_source=False)

        # ========== 5. 충돌 필터링 (CPU 시뮬레이션용) ==========
        if self.device == "cpu":
            # CPU에서는 충돌 필터링 필요
            self.scene.filter_collisions(global_prim_paths=[])

        # ========== 6. 씬에 객체 등록 ==========
        self.scene.articulations["robot"] = self.robot  # 드론
        self.scene.rigid_objects["apriltag"] = self.apriltag  # 로버

        # ========== 7. AprilTag 텍스처 추가 ==========
        # 각 로버 위에 AprilTag 이미지 배치
        self._setup_aruco_texture()

        # ========== 8. 드론에 카메라 부착 ==========
        # 아래를 향하는 광각 카메라 (마커 감지용)
        self._setup_cameras()

        # ========== 9. 바닥 격자 제거 ==========
        # ArUco 오감지 방지를 위해 깔끔한 회색 바닥
        self._remove_ground_grid()

        # ========== 10. 조명 추가 ==========
        # 은은한 조명 (AprilTag 가시성 향상)
        light_cfg = sim_utils.DomeLightCfg(intensity=800.0, color=(0.9, 0.9, 0.9))
        light_cfg.func("/World/Light", light_cfg)

        # ========== 11. 뷰포트 격자선 비활성화 (주석 처리됨) ==========
        # import carb
        # carb.settings.get_settings().set("persistent/app/viewport/displayOptions", 0)
        # print("[DroneLandingEnv] Disabled viewport grid lines")

    def _setup_aruco_texture(self):
        """각 로버 위에 AprilTag 텍스처(이미지) 메시 생성."""
        import omni  # Omniverse USD 라이브러리
        from pxr import Sdf, UsdShade, UsdGeom, Gf  # USD 씬 그래프 조작

        # USD 스테이지 가져오기 (3D 씬 데이터베이스)
        stage = omni.usd.get_context().get_stage()

        # AprilTag 이미지 파일 경로
        texture_path = "/home/rtx5080/s/ISAAC_LAB_DRONE/tag586_ariel.png"

        # ========== 각 환경마다 텍스처 메시 생성 ==========
        for env_id in range(self.num_envs):
            # 메시 경로: /World/envs/env_0/AprilTag/TagMesh
            apriltag_mesh_path = f"/World/envs/env_{env_id}/AprilTag/TagMesh"

            # ========== 1. 평면 메시 생성 ==========
            mesh = UsdGeom.Mesh.Define(stage, apriltag_mesh_path)

            # 정사각형 평면 정의 (0.5m × 0.5m)
            half = 0.25  # 반지름
            # 4개의 꼭지점 (왼쪽 아래부터 시계 반대 방향)
            mesh.GetPointsAttr().Set(
                [
                    Gf.Vec3f(-half, -half, 0),  # 왼쪽 아래
                    Gf.Vec3f(half, -half, 0),  # 오른쪽 아래
                    Gf.Vec3f(half, half, 0),  # 오른쪽 위
                    Gf.Vec3f(-half, half, 0),  # 왼쪽 위
                ]
            )
            # 면 정의 (4개 꼭지점으로 이루어진 면)
            mesh.GetFaceVertexCountsAttr().Set([4])
            # 면의 꼭지점 인덱스 순서
            mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
            # 법선 벡터 (위쪽 방향)
            mesh.GetNormalsAttr().Set([Gf.Vec3f(0, 0, 1)] * 4)
            mesh.SetNormalsInterpolation("vertex")

            # ========== 2. UV 좌표 설정 (텍스처 매핑용) ==========
            # UV: 2D 이미지를 3D 메시에 매핑하는 좌표
            texcoords = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
            )
            # 각 꼭지점의 UV 좌표 (0~1 범위)
            texcoords.Set(
                [Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]
            )

            # ========== 3. 메시 위치 설정 ==========
            # 큐브 윗면에 살짝 띄워서 배치 (Z-fighting 방지)
            xform = UsdGeom.Xformable(mesh)
            # 기존 변환 연산자 확인
            translate_op = None
            for op in xform.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                    break

            # 없으면 새로 생성
            if translate_op is None:
                translate_op = xform.AddTranslateOp()

            # 큐브 윗면(0.25m) + 약간 위(0.001m) = 0.251m
            translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.251))

            # ========== 4. 재질(Material) 생성 ==========
            mtl_path = Sdf.Path(apriltag_mesh_path + "_Material")
            mtl = UsdShade.Material.Define(stage, mtl_path)

            # ========== 5. 셰이더 생성 (어두운 AprilTag 가시성 최적화) ==========
            shader = UsdShade.Shader.Define(stage, mtl_path.AppendPath("Shader"))
            shader.CreateIdAttr("UsdPreviewSurface")  # PBR 기본 셰이더
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
                0.8
            )  # 거친 표면 (반사 적음)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)  # 비금속
            shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)  # 불투명

            # ========== 6. 텍스처 좌표 리더 생성 ==========
            st_reader = UsdShade.Shader.Define(stage, mtl_path.AppendPath("stReader"))
            st_reader.CreateIdAttr("UsdPrimvarReader_float2")
            st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

            # ========== 7. 텍스처 샘플러 생성 ==========
            diffuse_tex = UsdShade.Shader.Define(
                stage, mtl_path.AppendPath("DiffuseTexture")
            )
            diffuse_tex.CreateIdAttr("UsdUVTexture")
            # 텍스처 파일 경로 설정
            diffuse_tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_path)
            # UV 좌표 연결
            diffuse_tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                st_reader.ConnectableAPI(), "result"
            )
            # 텍스처 래핑 모드 (clamp = 경계 반복 안 함)
            diffuse_tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
            diffuse_tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
            diffuse_tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

            # ========== 8. 텍스처를 셰이더에 연결 ==========
            # Diffuse 색상
            shader.CreateInput(
                "diffuseColor", Sdf.ValueTypeNames.Color3f
            ).ConnectToSource(diffuse_tex.ConnectableAPI(), "rgb")

            # Emissive 색상 (어두운 부분이 밝은 조명에도 어둡게 유지)
            shader.CreateInput(
                "emissiveColor", Sdf.ValueTypeNames.Color3f
            ).ConnectToSource(diffuse_tex.ConnectableAPI(), "rgb")

            # ========== 9. 셰이더를 재질에 연결 ==========
            mtl.CreateSurfaceOutput().ConnectToSource(
                shader.ConnectableAPI(), "surface"
            )

            # ========== 10. 재질을 메시에 바인딩 ==========
            UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(mtl)

        print(
            f"[DroneLandingEnv] Created AprilTag textured meshes for {self.num_envs} markers"
        )

    def _setup_cameras(self):
        """각 드론에 아래를 향하는 광각 카메라 부착."""
        print("[DEBUG] _setup_cameras() called")

        import omni
        from pxr import Gf, UsdGeom

        # USD 스테이지 가져오기
        stage = omni.usd.get_context().get_stage()

        # ========== 카메라 FOV 설정 (150도 광각) ==========
        # FOV 공식: FOV = 2 × arctan(aperture / (2 × focal_length))
        FOCAL_LENGTH_MM = 8.0  # 짧은 초점 거리 (광각)
        HORIZONTAL_APERTURE_MM = 60.0  # 넓은 조리개

        # FOV 계산 검증
        fov_rad = 2 * np.arctan(HORIZONTAL_APERTURE_MM / (2 * FOCAL_LENGTH_MM))
        fov_deg = np.degrees(fov_rad)
        print(f"[DEBUG] Calculated FOV: {fov_deg:.1f}°")

        # ========== 첫 번째 환경에만 카메라 생성 (테스트용) ==========
        # 나중에 모든 환경으로 확장 가능
        env_id = 0
        camera_prim_path = f"/World/envs/env_{env_id}/Robot/body/Camera"
        print(f"[DEBUG] Creating camera at: {camera_prim_path}")

        # ========== 카메라 생성 ==========
        camera_prim = UsdGeom.Camera.Define(stage, camera_prim_path)
        print(f"[DEBUG] Camera prim created: {camera_prim.GetPrim().IsValid()}")

        # ========== 카메라 광학 속성 설정 (USD는 mm 단위 사용) ==========
        camera_prim.GetFocalLengthAttr().Set(FOCAL_LENGTH_MM)
        camera_prim.GetHorizontalApertureAttr().Set(HORIZONTAL_APERTURE_MM)

        # 16:9 종횡비에 맞춰 수직 조리개 계산
        aspect_ratio = 9.0 / 16.0
        vertical_aperture_mm = HORIZONTAL_APERTURE_MM * aspect_ratio
        camera_prim.GetVerticalApertureAttr().Set(vertical_aperture_mm)

        # ========== 피사계 심도(Depth of Field) 비활성화 ==========
        camera_prim.GetFocusDistanceAttr().Set(1000.0)  # 초점 거리 무한대
        camera_prim.GetFStopAttr().Set(0.0)  # F-stop 0 = DOF 없음

        # ========== 클리핑 범위 설정 ==========
        # 0.01m ~ 10,000m 사이의 물체만 렌더링
        camera_prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10000.0))

        # ========== 카메라 위치 설정 (드론 본체 기준) ==========
        xform = UsdGeom.Xformable(camera_prim)
        xform.ClearXformOpOrder()  # 기존 변환 제거

        # 드론 중심에서 15cm 아래로 이동
        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(0.0, 0.0, -0.0))

        # ========== 렌더 프로덕트 설정 (이미지 캡처용) ==========
        if ARUCO_AVAILABLE:
            try:
                import omni.replicator.core as rep

                print("[DEBUG] Importing omni.replicator.core...")
                # 1280×720 해상도로 렌더 프로덕트 생성
                render_product = rep.create.render_product(
                    camera_prim_path, (1280, 720)
                )
                print("[DEBUG] Render product created")

                # RGB 이미지 어노테이터 가져오기
                annotator = rep.AnnotatorRegistry.get_annotator("rgb")
                print("[DEBUG] Annotator obtained")

                # 어노테이터를 렌더 프로덕트에 연결
                annotator.attach([render_product])
                print("[DEBUG] Annotator attached")

                # 나중에 이미지 가져오기 위해 저장
                self.camera_annotators.append((annotator, render_product))
                print(
                    f"[Camera] Render product created for detection (FOV: {fov_deg:.1f}°)"
                )
            except Exception as e:
                print(f"[WARN] Could not setup camera annotator: {e}")
                import traceback

                traceback.print_exc()

        print(f"[DroneLandingEnv] Added camera with wide FOV (FOV: {fov_deg:.1f}°)")
        print(f"[DEBUG] camera_annotators count: {len(self.camera_annotators)}")
        print(f"[DEBUG] ARUCO_AVAILABLE: {ARUCO_AVAILABLE}")
        print(
            f"[DEBUG] aruco_dicts: {len(self.aruco_dicts) if hasattr(self, 'aruco_dicts') else 0} dictionaries"
        )

    def _remove_ground_grid(self):
        """바닥의 격자 무늬 제거 (ArUco 오감지 방지)."""
        import omni
        from pxr import Sdf, UsdShade, Gf

        # USD 스테이지 가져오기
        stage = omni.usd.get_context().get_stage()
        ground_prim = stage.GetPrimAtPath("/World/ground")

        # 바닥이 없으면 경고
        if not ground_prim.IsValid():
            print("[WARN] Ground prim not found at /World/ground")
            return

        # ========== 재질 생성 또는 가져오기 ==========
        mtl_path = Sdf.Path("/World/ground/GroundMaterial")
        mtl = UsdShade.Material.Define(stage, mtl_path)

        # ========== 단순한 회색 셰이더 생성 (격자 없음) ==========
        shader = UsdShade.Shader.Define(stage, mtl_path.AppendPath("Shader"))
        shader.CreateIdAttr("UsdPreviewSurface")

        # 중간 회색 색상 (0.5, 0.5, 0.5)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.5, 0.5, 0.5)
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)  # 거친 표면
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)  # 비금속

        # ========== 셰이더를 재질에 연결 ==========
        mtl.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        # ========== 재질을 바닥에 바인딩 ==========
        UsdShade.MaterialBindingAPI(ground_prim).Bind(mtl)

        print("[DroneLandingEnv] Removed ground grid pattern - solid gray applied")

    def _estimate_pose_single_markers(
        self, corners, marker_size, camera_matrix, dist_coeffs
    ):
        """
        ArUco 마커의 3D 자세 추정 (cv2.solvePnP 사용).

        cv2.aruco.estimatePoseSingleMarkers()가 deprecated되어 직접 구현.

        Args:
            corners: detectMarkers()에서 반환된 마커 코너들
            marker_size: 마커 크기 (미터 단위)
            camera_matrix: 카메라 내부 행렬 (3x3)
            dist_coeffs: 왜곡 계수 (5x1)

        Returns:
            rvecs: 회전 벡터 배열 (N, 1, 3)
            tvecs: 변환 벡터 배열 (N, 1, 3)
        """
        # ========== 마커 코너 3D 좌표 정의 (원점 중심) ==========
        # 정사각형 마커의 4개 코너 (왼쪽 위부터 시계 방향)
        marker_points = np.array(
            [
                [-marker_size / 2, marker_size / 2, 0],  # 왼쪽 위
                [marker_size / 2, marker_size / 2, 0],  # 오른쪽 위
                [marker_size / 2, -marker_size / 2, 0],  # 오른쪽 아래
                [-marker_size / 2, -marker_size / 2, 0],  # 왼쪽 아래
            ],
            dtype=np.float32,
        )
        rvecs = []  # 회전 벡터 리스트
        tvecs = []  # 변환 벡터 리스트

        # ========== 각 마커마다 자세 추정 ==========
        for corner in corners:
            try:
                # PnP (Perspective-n-Point) 문제 해결
                # SOLVEPNP_IPPE_SQUARE: 정사각형 평면 마커에 최적화된 알고리즘
                retval, rvec, tvec = cv2.solvePnP(
                    marker_points,  # 3D 점들
                    corner,  # 2D 이미지 점들
                    camera_matrix,  # 카메라 행렬
                    dist_coeffs,  # 왜곡 계수
                    None,  # 초기 추정 없음
                    None,  # 초기 추정 없음
                    False,  # extrinsic 추정 안 함
                    cv2.SOLVEPNP_IPPE_SQUARE,  # 정사각형 마커용 알고리즘
                )
                if retval:
                    # 성공: (3, 1) → (1, 3)으로 reshape
                    rvecs.append(rvec.reshape(1, 3))
                    tvecs.append(tvec.reshape(1, 3))
                else:
                    # 실패: None 추가
                    rvecs.append(None)
                    tvecs.append(None)
            except Exception as e:
                print(f"[WARN] solvePnP failed: {e}")
                rvecs.append(None)
                tvecs.append(None)

        # ========== None 값 필터링 ==========
        valid_rvecs = [r for r in rvecs if r is not None]
        valid_tvecs = [t for t in tvecs if t is not None]

        # 유효한 결과가 없으면 None 반환
        if len(valid_rvecs) == 0:
            return None, None

        # NumPy 배열로 변환 (N, 1, 3) 형태
        return np.array(valid_rvecs), np.array(valid_tvecs)

    def _detect_aruco(self):
        """카메라에서 ArUco/AprilTag 마커 감지 및 시각화."""
        # ========== 감지 가능 여부 확인 ==========
        if (
            not ARUCO_AVAILABLE  # OpenCV 없음
            or not self.aruco_dicts  # 사전 없음
            or not self.camera_annotators  # 카메라 없음
        ):
            if self.detection_step_counter == 0:
                print("[ArUco] Detection disabled: detector or camera not available")
            return

        # ========== 2스텝마다만 감지 (프레임레이트 향상) ==========
        self.detection_step_counter += 1
        if self.detection_step_counter % 2 != 0:
            return

        try:
            # ========== 첫 번째 환경의 카메라 이미지 가져오기 ==========
            annotator, render_product = self.camera_annotators[0]
            image_data = annotator.get_data()

            # 이미지가 없으면 리턴
            if image_data is None:
                if self.detection_step_counter % 100 == 2:
                    print(
                        f"[ArUco] No image data at step {self.detection_step_counter}"
                    )
                return

            # ========== 첫 이미지 캡처 로그 ==========
            if self.detection_step_counter == 2:
                print(
                    f"[ArUco] First image captured: shape={image_data.shape}, dtype={image_data.dtype}"
                )

            # ========== 그레이스케일 변환 (ArUco 감지는 흑백 이미지 사용) ==========
            if len(image_data.shape) == 3 and image_data.shape[2] >= 3:
                # RGB → GRAY 변환
                gray = cv2.cvtColor(
                    image_data[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY
                )
            else:
                # 이미 그레이스케일
                gray = image_data.astype(np.uint8)

            # ========== 디버그 이미지 저장 (처음 몇 장) ==========
            if (
                self.detection_step_counter <= 20
                and self.detection_step_counter % 10 == 0
            ):
                debug_path = f"/tmp/aruco_debug_{self.detection_step_counter:04d}.png"
                cv2.imwrite(debug_path, gray)
                print(f"[ArUco] Saved debug image: {debug_path}")

            # ========== 여러 사전으로 마커 감지 시도 ==========
            corners, ids = None, None
            used_dict_name = None

            # 이전에 성공한 사전이 있으면 먼저 시도
            if self.detected_dict_type:
                dict_order = [self.detected_dict_type] + [
                    k for k in self.aruco_dicts.keys() if k != self.detected_dict_type
                ]
            else:
                dict_order = list(self.aruco_dicts.keys())

            # 각 사전으로 감지 시도
            for dict_name in dict_order:
                aruco_dict = self.aruco_dicts[dict_name]
                detector = aruco.ArucoDetector(aruco_dict, self.aruco_params)
                corners, ids, rejected = detector.detectMarkers(gray)

                # 마커 발견!
                if ids is not None and len(ids) > 0:
                    used_dict_name = dict_name
                    # 성공한 사전 기억 (다음에 먼저 시도)
                    if self.detected_dict_type != dict_name:
                        print(f"[ArUco] SUCCESS! Marker type: {dict_name}")
                        self.detected_dict_type = dict_name
                    break

            # ========== 시각화 이미지 준비 ==========
            vis_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # ========== 자세 추정 (마커가 감지된 경우) ==========
            rvecs, tvecs = None, None
            if ids is not None and len(ids) > 0:
                # 0.5m 마커 크기로 3D 자세 추정
                rvecs, tvecs = self._estimate_pose_single_markers(
                    corners, 0.5, self.camera_matrix, self.dist_coeffs
                )

                # ========== 감지 결과 로그 (변화가 있을 때만) ==========
                if len(ids) != self.last_detection_count:
                    print(f"\n{'='*60}")
                    print(
                        f"[ArUco] DETECTED {len(ids)} marker(s)! Type: {used_dict_name}"
                    )
                    for i, marker_id in enumerate(ids.flatten()):
                        corner = corners[i][0]
                        center = corner.mean(axis=0)  # 코너 중심점
                        print(f"  - Marker ID: {marker_id}")
                        print(f"    Center: ({center[0]:.0f}, {center[1]:.0f})px")
                        if tvecs is not None and len(tvecs) > i:
                            t = tvecs[i][0]  # 3D 위치
                            print(
                                f"    Position: ({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})m"
                            )
                    print(f"{'='*60}\n")
                    self.last_detection_count = len(ids)

                # ========== 감지된 마커 그리기 ==========
                aruco.drawDetectedMarkers(vis_img, corners, ids)

                # ========== 각 마커의 좌표축 그리기 ==========
                if rvecs is not None and tvecs is not None and len(rvecs) > 0:
                    for i in range(min(len(ids), len(rvecs))):
                        try:
                            # drawFrameAxes는 (3, 1) 형태 필요
                            rvec = rvecs[i].reshape(3, 1)
                            tvec = tvecs[i].reshape(3, 1)
                            # 0.25m 길이의 축 그리기 (빨강=X, 초록=Y, 파랑=Z)
                            cv2.drawFrameAxes(
                                vis_img,
                                self.camera_matrix,
                                self.dist_coeffs,
                                rvec,
                                tvec,
                                0.25,
                            )
                        except Exception as e:
                            if self.detection_step_counter % 100 == 0:
                                print(f"[WARN] drawFrameAxes error: {e}")

            else:
                # ========== 감지 실패 로그 ==========
                if self.last_detection_count > 0:
                    print(
                        f"[ArUco] No markers detected (step {self.detection_step_counter})"
                    )
                    self.last_detection_count = 0
                elif (
                    self.detection_step_counter <= 50
                    and self.detection_step_counter % 20 == 0
                ):
                    print(
                        f"[ArUco] Detection attempt {self.detection_step_counter}: 0 markers found (tried {len(self.aruco_dicts)} dicts)"
                    )

            # ========== 이미지 중심에 십자선 그리기 (항상) ==========
            cv2.line(
                vis_img,
                (int(self.cx) - 20, int(self.cy)),
                (int(self.cx) + 20, int(self.cy)),
                (255, 0, 0),
                2,
            )  # 수평선
            cv2.line(
                vis_img,
                (int(self.cx), int(self.cy) - 20),
                (int(self.cx), int(self.cy) + 20),
                (255, 0, 0),
                2,
            )  # 수직선

            # ========== 상태 텍스트 추가 ==========
            num_markers = 0 if ids is None else len(ids)
            status_text = (
                f"Markers: {num_markers}" if num_markers > 0 else "No markers detected"
            )
            cv2.putText(
                vis_img,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0) if num_markers > 0 else (0, 0, 255),
                2,
            )

            # ========== 이미지 표시 (항상) ==========
            cv2.imshow("ArUco Detection", vis_img)
            cv2.waitKey(1)  # 1ms 대기 (GUI 이벤트 처리)

        except Exception as e:
            if self.detection_step_counter % 100 == 0:
                print(f"[WARN] Detection error: {e}")

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        물리 시뮬레이션 전에 실행되는 전처리 단계.

        Args:
            actions: 정책 네트워크의 출력 (num_envs, 4)
                    [추진력, 롤모멘트, 피치모멘트, 요모멘트]
        """
        # ========== 1. 행동 저장 (나중에 _apply_action()에서 사용) ==========
        self.actions = actions.clone()

        # ========== 2. ArUco 마커 감지 (카메라 이미지 처리) ==========
        self._detect_aruco()

        # ========== 3. 로버 위치 업데이트 (움직이는 로버!) ==========
        # 현재위치 += 속도 × 시간간격
        self.apriltag_pos += self.apriltag_velocity * self.step_dt

        # ========== 4. 로버의 물리 상태 업데이트 ==========
        # 상대 좌표 → 절대 좌표 변환 (환경 원점 추가)
        apriltag_world_pos = self.apriltag_pos + self.scene.env_origins

        # 로버의 전체 상태 복사 (13차원)
        apriltag_state = self.apriltag.data.default_root_state.clone()
        apriltag_state[:, :3] = apriltag_world_pos  # 위치 (x, y, z)
        apriltag_state[:, 3:7] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device
        )  # 회전 (쿼터니언) - Isaac Lab: [w, x, y, z] 형태
        apriltag_state[:, 7:10] = self.apriltag_velocity  # 선속도
        apriltag_state[:, 10:] = 0.0  # 각속도 (회전 없음)

        # 물리 엔진에 상태 쓰기
        self.apriltag.write_root_state_to_sim(apriltag_state)

        # ========== 5. 목표 위치 설정 (착륙 모드!) ==========
        # XY: 로버의 중심 위치
        self._desired_pos_w[:, :2] = apriltag_world_pos[:, :2]
        # Z: 로버 표면 = 중심(0.5m) + 큐브 높이의 절반(0.25m) = 0.75m
        # apriltag_pos는 큐브 중심이므로 height/2를 더해야 윗면!
        rover_surface_z = self.apriltag_pos[:, 2] + 0.25  # 0.5 + 0.25 = 0.75m
        self._desired_pos_w[:, 2] = rover_surface_z

    def _apply_action(self) -> None:
        """저장된 행동을 드론에 실제로 적용 (힘/토크)."""

        # ========== 1. 행동 추출 ==========
        thrust_action = self.actions[:, 0]  # 추진력 (-1 ~ 1)
        moments_normalized = self.actions[:, 1:4]  # 모멘트 (롤, 피치, 요)

        # ========== 2. 추진력 변환 ==========
        # [-1, 1] → [0, thrust_to_weight] 범위로 매핑
        # thrust_ratio = 0 ~ 1.9 (드론 무게의 0배 ~ 1.9배)
        thrust_ratio = self.cfg.thrust_to_weight * (thrust_action + 1.0) / 2.0

        # 실제 힘 계산 (뉴턴 단위)
        thrust_force_mag = thrust_ratio * self.robot_weight

        # ========== 3. 추진력 방향 계산 (드론 좌표계 기준) ==========
        # 드론 좌표계 Z축 = [0, 0, 1] (위쪽)
        local_up = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(
            self.num_envs, 1
        )

        # 드론 좌표계 → 세계 좌표계 변환 (쿼터니언 회전 적용)
        # 드론이 기울어져 있으면 추진력도 기울어짐!
        thrust_vector_w = quat_apply(
            self.robot.data.root_quat_w, local_up
        ) * thrust_force_mag.unsqueeze(-1)

        # ========== 4. 모멘트(토크) 스케일링 ==========
        # [-1, 1] → [-0.002, 0.002] N·m 범위
        torques = moments_normalized * self.cfg.moment_scale

        # ========== 5. 힘과 토크를 물리 엔진에 적용 ==========
        # unsqueeze(1): (num_envs, 3) → (num_envs, 1, 3) 형태 변환
        thrust_vector_expanded = thrust_vector_w.unsqueeze(1)
        torques_expanded = torques.unsqueeze(1)

        # body_ids=[0]: 드론의 본체(첫 번째 링크)에 적용
        # env_ids=None: 모든 환경에 적용
        self.robot.set_external_force_and_torque(
            thrust_vector_expanded, torques_expanded, body_ids=[0], env_ids=None
        )

    def _get_observations(self) -> dict:
        """
        정책 네트워크에 입력될 관측값 계산.

        Returns:
            {"policy": obs} - obs는 (num_envs, 16) 텐서
        """
        # ========== 1. 목표 위치를 드론 좌표계로 변환 ==========
        # 쿼터니언의 켤레 (역회전)
        root_quat_conjugate = quat_conjugate(self.robot.data.root_quat_w)

        # 세계 좌표계에서의 목표까지 벡터
        goal_rel_world = self._desired_pos_w - self.robot.data.root_pos_w

        # 드론 좌표계로 변환 (드론이 봤을 때 목표가 어디에?)
        desired_pos_b = quat_apply(root_quat_conjugate, goal_rel_world)

        # ========== 2. 상대 속도를 드론 좌표계로 변환 ==========
        # 드론 속도 - 로버 속도 (세계 좌표계)
        rel_vel_world = self.robot.data.root_lin_vel_w - self.apriltag_velocity

        # 드론 좌표계로 변환
        rel_vel_b = quat_apply(root_quat_conjugate, rel_vel_world)

        # ========== 3. Yaw 각도 계산 (드론이 자신의 회전 상태를 알 수 있도록) ==========
        # 쿼터니언에서 yaw 추출: Isaac Lab은 [w, x, y, z] 형태
        quat = self.robot.data.root_quat_w
        qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        current_yaw = torch.atan2(siny_cosp, cosy_cosp).unsqueeze(-1)  # (num_envs, 1)

        # ========== 4. 모든 관측값 연결 (16차원) ==========
        obs = torch.cat(
            [
                self.robot.data.root_lin_vel_b,  # 3차원: 드론 속도 (드론 좌표계)
                self.robot.data.root_ang_vel_b,  # 3차원: 회전 속도 (드론 좌표계)
                self.robot.data.projected_gravity_b,  # 3차원: 중력 방향 (어느 쪽이 아래?)
                desired_pos_b,  # 3차원: 목표까지 거리/방향
                rel_vel_b,  # 3차원: 상대 속도 (드론 - 로버)
                current_yaw,  # 1차원: 현재 yaw 각도 (Z축 회전)
            ],
            dim=-1,  # 마지막 차원으로 연결
        )

        # ========== 5. 딕셔너리로 반환 ==========
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """
        현재 상태에 대한 보상 계산 (Staged Landing Approach - 완화 버전).

        핵심 원칙: "XY가 맞으면 일단 내려가라, 자세는 내려가면서 잡아라"
        - 기울기(Tilt)는 착륙 허가 조건에서 제외 (Gating X)
        - 기울기는 연속적인 패널티로만 관리
        - XY 정렬만 되면 하강 시작

        Returns:
            reward: (num_envs,) 텐서 - 각 환경의 보상값
        """
        time_in_episode = self.episode_length_buf.float() * self.step_dt
        early_episode = time_in_episode < self.cfg.stability_bonus_duration

        ang_vel_magnitude = torch.norm(self.robot.data.root_ang_vel_b, dim=1)
        stability_bonus = torch.where(
            early_episode,
            torch.exp(-5.0 * ang_vel_magnitude**2),  # 회전 안 하면 보상 (제곱)
            torch.zeros_like(ang_vel_magnitude)
        )
                
        # ========== 상태 계산 ==========
        apriltag_world_pos = self.apriltag_pos + self.scene.env_origins
        drone_pos = self.robot.data.root_pos_w
        drone_height = drone_pos[:, 2] - self.scene.env_origins[:, 2]
        rover_surface_z = self.apriltag_pos[:, 2] + 0.25

        # ========== 방향 정렬 보상 (0° 고정) ==========
        # 정렬 유도 시간 내인지 확인
        is_alignment_phase = time_in_episode < self.cfg.alignment_duration

        # 드론 → 로버 상대 위치 벡터 (XY 평면)
        rel_pos_xy = apriltag_world_pos[:, :2] - drone_pos[:, :2]

        # 로버 방향 각도 계산 (atan2: -π ~ π)
        angle_to_rover = torch.atan2(rel_pos_xy[:, 1], rel_pos_xy[:, 0])

        # 0도와의 오차 (절대값, -π~π 범위이므로 그대로 사용)
        angle_error = torch.abs(angle_to_rover)

        # 정렬 보상: 0도에 가까울수록 높은 보상
        is_well_aligned_angle = angle_error < self.cfg.alignment_angle_threshold
        alignment_reward = torch.where(
            is_alignment_phase,
            torch.exp(-10.0 * angle_error**2),  # 0도에 가까울수록 보상 (제곱)
            torch.zeros_like(angle_error)
        )

        # 정렬 전 이동 패널티: 정렬 안 됐는데 XY/Z 이동하면 패널티
        drone_vel = self.robot.data.root_lin_vel_w
        movement_speed = torch.norm(drone_vel, dim=1)  # 전체 속도
        premature_movement_penalty = torch.where(
            is_alignment_phase & (~is_well_aligned_angle) & (movement_speed > 0.3),
            movement_speed,  # 속도에 비례한 패널티
            torch.zeros_like(movement_speed)
        )

        # ========== XY 오차 및 Phase 판단 ==========
        # 기울기 조건 제거! XY만 맞으면 내려가기 시작
        xy_error = torch.norm(drone_pos[:, :2] - apriltag_world_pos[:, :2], dim=1)
        is_aligned = xy_error < self.cfg.align_threshold

        # ========== 동적 목표 Z (XY만 보고 결정) ==========
        approach_height_tensor = torch.full_like(rover_surface_z, self.cfg.approach_height)
        target_z = torch.where(is_aligned, rover_surface_z, approach_height_tensor)

        # ========== 핵심 보상 1: XY 거리 ==========
        xy_distance_reward = torch.exp(-1.5 * xy_error**2)

        # ========== 핵심 보상 2: Z 거리 (정렬 시 보상 증폭) ==========
        z_error = torch.abs(drone_height - target_z)
        # 정렬된 상태에서는 Z 보상 2배 → 착륙 유도
        z_scale = torch.where(
            is_aligned,
            torch.tensor(2.0, device=self.device),
            torch.tensor(1.0, device=self.device),
        )
        z_distance_reward = torch.exp(-2.0 * z_error**2) * z_scale

        # ========== 핵심 보상 3: XY 속도 매칭 ==========
        drone_vel_xy = self.robot.data.root_lin_vel_w[:, :2]
        target_vel_xy = self.apriltag_velocity[:, :2]
        velocity_error_xy = torch.norm(drone_vel_xy - target_vel_xy, dim=1)
        velocity_tracking_reward = torch.exp(-2.0 * velocity_error_xy**2)

        # ========== 조기 하강 패널티 (강화) ==========
        # XY 정렬 안 됐는데 approach_height 아래로 내려가면 패널티
        premature_descent_threshold = self.cfg.approach_height - 0.3  # 더 엄격하게
        height_below_threshold = premature_descent_threshold - drone_height
        premature_descent_penalty = torch.where(
            (~is_aligned) & (drone_height < premature_descent_threshold),
            torch.clamp(height_below_threshold, min=0.0),  # 내려간 만큼 비례 패널티
            torch.zeros_like(drone_height),
        )

        # ========== 안정성 패널티 (연속적, Gating 아님) ==========
        # 각속도 패널티
        ang_vel_penalty = torch.sum(torch.square(self.robot.data.root_ang_vel_b), dim=1)

        # 기울기 패널티: 허용 각도 초과 시에만 적용
        gravity_b = self.robot.data.projected_gravity_b
        tilt_raw = 1.0 - gravity_b[:, 2]  # 0=수직, 1=수평
        # allowed_tilt_angle(rad)를 tilt_raw 스케일로 변환: 1 - cos(angle)
        allowed_tilt = 1.0 - torch.cos(torch.tensor(self.cfg.allowed_tilt_angle, device=self.device))
        orientation_penalty = torch.where(
            tilt_raw > allowed_tilt,
            tilt_raw - allowed_tilt,  # 초과분만 패널티
            torch.zeros_like(tilt_raw)
        )

        # ========== Yaw 정렬 보상 (로버 기준 0°/90°/180°/270° 중 가까운 각도) ==========
        # 드론 다리와 로버가 안정적으로 맞물리도록 로버 기준 90도 단위 정렬

        # 드론 yaw 계산
        drone_quat = self.robot.data.root_quat_w  # Isaac Lab: [w, x, y, z] 형태
        dqw, dqx, dqy, dqz = drone_quat[:, 0], drone_quat[:, 1], drone_quat[:, 2], drone_quat[:, 3]
        drone_siny = 2.0 * (dqw * dqz + dqx * dqy)
        drone_cosy = 1.0 - 2.0 * (dqy * dqy + dqz * dqz)
        drone_yaw = torch.atan2(drone_siny, drone_cosy)  # -π ~ π

        # 로버 yaw 계산
        rover_quat = self.apriltag.data.root_quat_w  # Isaac Lab: [w, x, y, z] 형태
        rqw, rqx, rqy, rqz = rover_quat[:, 0], rover_quat[:, 1], rover_quat[:, 2], rover_quat[:, 3]
        rover_siny = 2.0 * (rqw * rqz + rqx * rqy)
        rover_cosy = 1.0 - 2.0 * (rqy * rqy + rqz * rqz)
        rover_yaw = torch.atan2(rover_siny, rover_cosy)  # -π ~ π

        # 상대 yaw 계산 (드론 yaw - 로버 yaw)
        relative_yaw = drone_yaw - rover_yaw
        # -π ~ π 범위로 정규화
        relative_yaw = torch.atan2(torch.sin(relative_yaw), torch.cos(relative_yaw))

        # 로버 기준 0°/90°/180°/270° 중 가장 가까운 각도와의 오차 계산
        yaw_normalized = torch.where(relative_yaw < 0, relative_yaw + 2 * 3.14159, relative_yaw)  # 0 ~ 2π
        yaw_remainder = torch.remainder(yaw_normalized, 3.14159 / 2)  # 90도 단위 나머지
        yaw_error = torch.minimum(yaw_remainder, 3.14159 / 2 - yaw_remainder)  # 가까운 쪽 오차

        # yaw 정렬 보상: 로버 기준 0/90/180/270도에 가까울수록 보상
        yaw_reward = torch.exp(-2.0 * yaw_error**2)

        # ========== 하강 보상 (Phase 2에서만) ==========
        vel_z = self.robot.data.root_lin_vel_w[:, 2]
        target_vel_z = self.cfg.target_descent_velocity
        is_above_rover = drone_height > (rover_surface_z + 0.1)
        vel_z_error = torch.abs(vel_z - target_vel_z)

        # 기울기 조건 없이, XY 정렬되고 위에 있으면 하강 보상
        descent_reward = torch.where(
            is_aligned & is_above_rover & (vel_z < 0),
            torch.exp(-5.0 * vel_z_error**2),
            torch.zeros_like(vel_z),
        )

        # ========== 과도한 하강 속도 패널티 ==========
        max_vel_z = self.cfg.max_descent_velocity
        excessive_descent_penalty = torch.where(
            vel_z < max_vel_z,
            torch.abs(vel_z - max_vel_z),  # 초과 속도 크기
            torch.zeros_like(vel_z),
        )

        # ========== 착륙 전 속도 제어 보상 (표면 근처에서 천천히!) ==========
        # 표면 근처에 있을 때, 전체 속도가 낮으면 보상
        height_from_surface = drone_height - rover_surface_z
        is_near_surface = (height_from_surface > 0.0) & (height_from_surface < self.cfg.gentle_approach_distance_threshold)

        # XY 정렬 조건 (천천히 착륙 보너스용, 더 엄격)
        xy_distance_for_gentle = torch.norm(drone_pos[:, :2] - apriltag_world_pos[:, :2], dim=1)
        is_well_aligned = xy_distance_for_gentle < self.cfg.gentle_approach_xy_threshold

        # 전체 속도 계산 (XY + Z)
        total_speed = torch.norm(self.robot.data.root_lin_vel_w, dim=1)
        # 속도가 낮을수록 보상: exp(-속도*10) → 0.2m/s에서 0.135, 0.1m/s에서 0.368
        gentle_approach_reward = torch.where(
            is_well_aligned & is_near_surface,  # 엄격한 XY 정렬 + 표면 근처
            torch.exp(-10.0 * total_speed**2),  # 제곱으로 천천히 접근 유도 강화
            torch.zeros_like(total_speed),
        )

        # ========== 보상 결합 ==========
        rewards = {
            "xy_distance": xy_distance_reward * self.cfg.xy_distance_reward_scale * self.step_dt,
            "z_distance": z_distance_reward * self.cfg.z_distance_penalty_scale * self.step_dt,
            "velocity_tracking": velocity_tracking_reward * self.cfg.velocity_tracking_reward_scale * self.step_dt,
            "angular_vel_penalty": ang_vel_penalty * self.cfg.angular_velocity_penalty_scale * self.step_dt,
            "orientation_penalty": orientation_penalty * self.cfg.orientation_penalty_scale * self.step_dt,
            "descent_reward": descent_reward * self.cfg.descent_reward_scale * self.step_dt,
            "excessive_descent_penalty": excessive_descent_penalty * self.cfg.excessive_descent_penalty_scale * self.step_dt,
            "premature_descent": premature_descent_penalty * self.cfg.premature_descent_penalty_scale * self.step_dt,
            "initial_stability": stability_bonus * self.cfg.initial_stability_bonus_scale * self.step_dt,
            "gentle_approach": gentle_approach_reward * self.cfg.gentle_approach_reward_scale * self.step_dt,
            "alignment": alignment_reward * self.cfg.alignment_reward_scale * self.step_dt,
            "premature_movement": premature_movement_penalty * self.cfg.premature_movement_penalty_scale * self.step_dt,
            "yaw_reward": yaw_reward * self.cfg.yaw_reward_scale * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # ========== 종료 보상 ==========
        landing_reward = self.cfg.landing_success_reward * self.tracking_success.float()

        # 빠른 착륙 시간 보너스 (남은 시간 비례)
        remaining_time = (self.max_episode_length - self.episode_length_buf).float() / self.max_episode_length
        time_bonus = self.cfg.landing_time_bonus_scale * remaining_time * self.tracking_success.float()

        # 부드러운 착륙 보너스 (속도가 낮을수록 보너스)
        # 착륙 성공 시 드론의 전체 속도(선속도 크기) 계산
        landing_speed = torch.norm(self.robot.data.root_lin_vel_w, dim=1)
        # exp(-5*속도²): 속도 0 → 보너스 1.0, 제곱으로 유도 강화
        soft_landing_bonus = torch.exp(-5.0 * landing_speed**2) * self.cfg.soft_landing_bonus_scale * self.tracking_success.float()

        reward += landing_reward + time_bonus + soft_landing_bonus

        # 추락 패널티
        died = self.robot.data.root_pos_w[:, 2] < (self.cfg.min_height + self.scene.env_origins[:, 2])
        reward += self.cfg.crash_penalty * died.float()

        # ========== 로깅 ==========
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        에피소드 종료 조건 확인.

        Returns:
            terminated: (num_envs,) bool - 에피소드 종료 여부
            time_out: (num_envs,) bool - 시간 초과 여부
        """
        # ========== 1. 시간 초과 확인 ==========
        # 15초 = 900 스텝 (60Hz 기준)
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # ========== 2. 추락 또는 fly away 확인 ==========
        # 0.15m 아래로 떨어지거나 3m 위로 올라감
        died = torch.logical_or(
            self.robot.data.root_pos_w[:, 2]
            < (self.cfg.min_height + self.scene.env_origins[:, 2]),
            self.robot.data.root_pos_w[:, 2] > self.cfg.max_height,
        )

        # ========== 3. 뒤집힘 확인 ==========
        # 세계 Z축
        up_vec_world = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(
            self.num_envs, 1
        )
        # 드론 좌표계에서 본 Z축
        up_vec_body = quat_apply(self.robot.data.root_quat_w, up_vec_world)

        # 기울기 각도 계산
        # acos(1.0) = 0° (수평)
        # acos(0.0) = 90° (옆으로 누움)
        # acos(-1.0) = 180° (완전히 뒤집힘)
        tilt_angle = torch.acos(torch.clamp(up_vec_body[:, 2], -1.0, 1.0))

        # 60도 이상 기울면 뒤집힌 것으로 판정
        is_flipped = tilt_angle > self.cfg.max_tilt_limit

        # 추락에 뒤집힘 포함
        died = torch.logical_or(died, is_flipped)

        # ========== 4. 착륙 성공 확인 ==========
        apriltag_world_pos = self.apriltag_pos + self.scene.env_origins

        # 4-1. XY 정렬 확인 (10cm 이내)
        xy_distance = torch.norm(
            self.robot.data.root_pos_w[:, :2] - apriltag_world_pos[:, :2], dim=1
        )
        xy_aligned = xy_distance < self.cfg.tracking_distance_threshold
        # 4-2. 속도 매칭 확인 (0.15m/s 이내)
        rel_vel_xy = (
            self.robot.data.root_lin_vel_w[:, :2] - self.apriltag_velocity[:, :2]
        )
        rel_vel_magnitude = torch.norm(rel_vel_xy, dim=1)
        velocity_matched = rel_vel_magnitude < self.cfg.tracking_max_velocity

        # 4-3. 자세 확인 (17도 이내)
        upright = tilt_angle < self.cfg.max_tilt

        # 4-4. 표면 접촉 확인 (10cm 이내)
        # 로버 표면 = 중심(0.5m) + 큐브 높이의 절반(0.25m) = 0.75m
        rover_surface_height = self.apriltag_pos[:, 2] + 0.25
        drone_height = self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        height_diff = torch.abs(drone_height - rover_surface_height)
        on_surface = height_diff < self.cfg.landing_height_threshold

        # ========== 모든 조건 만족 시 성공! ==========
        self.tracking_success = xy_aligned & velocity_matched & upright & on_surface

        if self.tracking_success.any():
            # 성공한 환경이 몇 개인지 계산
            success_count = torch.sum(self.tracking_success).item()
            print(f"현재 스텝 착륙 성공 개수: {success_count} / {self.num_envs}")
            
            # (선택 사항) 첫 번째로 성공한 환경의 상태만 디버깅용으로 출력
            first_idx = torch.where(self.tracking_success)[0][0]
            print(f"상세(0번성공): XY:{xy_aligned[first_idx]}, Vel:{velocity_matched[first_idx]}, Up:{upright[first_idx]}, Surface:{on_surface[first_idx]}")
        else:
            print("성공한게없음")
        # ========== 에피소드 종료 ==========
        # 사망 또는 성공 시 종료
        terminated = died | self.tracking_success

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """
        특정 환경들을 리셋.

        Args:
            env_ids: 리셋할 환경 ID 리스트 (None이면 전체)
        """
        # ========== 전체 환경 리셋 ==========
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # ========== 로깅: 에피소드 통계 계산 ==========
        # 최종 XY 거리 계산
        apriltag_world_pos = (
            self.apriltag_pos[env_ids] + self.scene.env_origins[env_ids]
        )
        final_xy_distance = torch.norm(
            self.robot.data.root_pos_w[env_ids, :2] - apriltag_world_pos[:, :2], dim=1
        ).mean()

        # 각 보상 항목의 평균 계산
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            # 에피소드 길이로 나누어 정규화
            extras["Episode_Reward/" + key] = (
                episodic_sum_avg / self.max_episode_length_s
            )
            # 버퍼 초기화
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        # 종료 원인 통계
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()  # 추락/뒤집힘 횟수
        extras["Episode_Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids]
        ).item()  # 시간 초과 횟수
        extras["Episode_Termination/tracking_success"] = torch.count_nonzero(
            self.tracking_success[env_ids]
        ).item()  # 착륙 성공 횟수
        extras["Metrics/final_xy_distance"] = final_xy_distance.item()  # 최종 거리
        self.extras["log"].update(extras)

        # ========== 부모 클래스 리셋 호출 ==========
        super()._reset_idx(env_ids)

        # ========== 로버 위치 리셋 ==========
        # 초기 위치로 되돌리기
        self.apriltag_pos[env_ids] = self.apriltag_initial_pos[env_ids].clone()

        # 물리 엔진에 로버 상태 쓰기
        apriltag_world_pos = (
            self.apriltag_pos[env_ids] + self.scene.env_origins[env_ids]
        )
        apriltag_state = self.apriltag.data.default_root_state[env_ids].clone()
        apriltag_state[:, :3] = apriltag_world_pos  # 위치
        apriltag_state[:, 3:7] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device
        )  # 회전 - Isaac Lab: [w, x, y, z] 형태
        apriltag_state[:, 7:10] = self.apriltag_velocity[env_ids]  # 속도
        apriltag_state[:, 10:] = 0.0  # 각속도
        self.apriltag.write_root_state_to_sim(apriltag_state, env_ids)

        # ========== 목표 위치 설정 ==========
        self._desired_pos_w[env_ids, :2] = apriltag_world_pos[:, :2]  # XY = 로버
        # Z = 로버 표면 = 중심(0.5m) + 큐브 높이의 절반(0.25m) = 0.75m
        self._desired_pos_w[env_ids, 2] = (
            self.apriltag_pos[env_ids, 2] + 0.25  # Z = 표면
        )

        # ========== 드론 위치 리셋 (Staged Landing: approach_height에서 시작) ==========
        root_state = self.robot.data.default_root_state[env_ids].clone()

        # XY 랜덤 오프셋 생성 (로버 위치 기준!)
        xy_range = self.cfg.initial_xy_range
        num_resets = len(env_ids)
        random_x = (torch.rand(num_resets, device=self.device) * 2 - 1) * xy_range[0]
        random_y = (torch.rand(num_resets, device=self.device) * 2 - 1) * xy_range[1]

        # 위치 설정: XY는 로버 위치 + 랜덤, Z는 approach_height로 고정
        # apriltag_pos는 이미 env_ids로 슬라이싱된 상태 (리셋 직후 초기 위치)
        root_state[:, 0] = self.apriltag_pos[env_ids, 0] + random_x  # 로버 X + 랜덤
        root_state[:, 1] = self.apriltag_pos[env_ids, 1] + random_y  # 로버 Y + 랜덤
        root_state[:, 2] = self.cfg.approach_height  # Z = approach_height (고정)

        # 환경 원점 추가 (각 환경마다 다른 위치)
        root_state[:, 0] += self.scene.env_origins[env_ids, 0]
        root_state[:, 1] += self.scene.env_origins[env_ids, 1]

        # ========== 드론 자세 리셋 (수평) ==========
        # Isaac Lab: [w, x, y, z] 형태 - identity quaternion = [1, 0, 0, 0]
        root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        # ========== 드론 속도 리셋 (정지) ==========
        root_state[:, 7:] = 0.0

        # 물리 엔진에 드론 상태 쓰기ㅁㄴㅇ
        self.robot.write_root_state_to_sim(root_state, env_ids)

        # ========== 조인트 상태 리셋 ==========
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, :] = 0.0  # 모든 조인트 각도 0
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        joint_vel[:, :] = 0.0  # 모든 조인트 속도 0
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # ========== 성공 플래그 리셋 ==========
        self.tracking_success[env_ids] = False