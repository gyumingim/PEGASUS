#!/usr/bin/env python
"""
ArUco 카메라 디버깅 스크립트
- 드론 카메라 이미지 저장
- ArUco 인식 위치 출력
- 좌표계 이해를 위한 시각화
"""

import cv2
import cv2.aruco as aruco
import numpy as np
from pathlib import Path

# ============================================================
# ArUco가 카메라 중앙에 있을 때 예상되는 값
# ============================================================
"""
카메라 좌표계 (OpenCV 기준):
    - X축: 오른쪽이 양수
    - Y축: 아래쪽이 양수
    - Z축: 카메라가 바라보는 방향이 양수

ArUco가 카메라 정중앙에 있을 때:
    tvec = [0, 0, distance]

    - tvec[0] ≈ 0: 좌우 중앙
    - tvec[1] ≈ 0: 상하 중앙
    - tvec[2] = 거리 (양수, 미터)

예시 (마커가 3m 아래에 있고 정중앙일 때):
    tvec = [0.0, 0.0, 3.0]

마커가 오른쪽에 있으면: tvec[0] > 0
마커가 왼쪽에 있으면:  tvec[0] < 0
마커가 아래에 있으면:  tvec[1] > 0
마커가 위에 있으면:    tvec[1] < 0
"""


class ArUcoDebugger:
    def __init__(self, image_width=1280, image_height=720, fov_deg=150.0):
        self.img_w = image_width
        self.img_h = image_height
        self.fov_deg = fov_deg

        # 카메라 내부 파라미터 계산
        self.fx = image_width / (2 * np.tan(np.radians(fov_deg / 2)))
        self.fy = self.fx
        self.cx = image_width / 2
        self.cy = image_height / 2

        self.camera_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        # ArUco 딕셔너리
        self.aruco_dicts = {
            "DICT_APRILTAG_36h11": aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11),
            "DICT_APRILTAG_25h9": aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_25h9),
            "DICT_APRILTAG_16h5": aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_16h5),
            "DICT_ARUCO_ORIGINAL": aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL),
            "DICT_4X4_50": aruco.getPredefinedDictionary(aruco.DICT_4X4_50),
        }
        self.aruco_params = aruco.DetectorParameters()

        print("=" * 60)
        print("ArUco Debugger 초기화")
        print("=" * 60)
        print(f"이미지 크기: {image_width} x {image_height}")
        print(f"FOV: {fov_deg}°")
        print(f"초점 거리: fx={self.fx:.2f}, fy={self.fy:.2f}")
        print(f"주점: cx={self.cx:.2f}, cy={self.cy:.2f}")
        print("=" * 60)
        print()
        self._print_coordinate_guide()

    def _print_coordinate_guide(self):
        """좌표계 가이드 출력"""
        print("=" * 60)
        print("★ ArUco 좌표계 가이드 ★")
        print("=" * 60)
        print()
        print("카메라 좌표계 (tvec 의미):")
        print("  ┌─────────────────────────────────────┐")
        print("  │          카메라 이미지              │")
        print("  │                                     │")
        print("  │    tvec[0] < 0     tvec[0] > 0     │")
        print("  │       (왼쪽)   ●   (오른쪽)        │")
        print("  │              중앙                   │")
        print("  │                                     │")
        print("  │    tvec[1] < 0 (위쪽)              │")
        print("  │    tvec[1] > 0 (아래쪽)            │")
        print("  │                                     │")
        print("  │    tvec[2] = 거리 (항상 양수)      │")
        print("  └─────────────────────────────────────┘")
        print()
        print("정중앙에 있을 때 예상값:")
        print("  tvec = [0.0, 0.0, distance]")
        print()
        print("예시:")
        print("  - 마커가 3m 아래, 정중앙: tvec ≈ [0, 0, 3.0]")
        print("  - 마커가 3m 아래, 오른쪽 0.5m: tvec ≈ [0.5, 0, 3.0]")
        print("  - 마커가 3m 아래, 왼쪽 0.5m: tvec ≈ [-0.5, 0, 3.0]")
        print()
        print("=" * 60)

    def analyze_image(self, image_path, marker_size=0.6, save_output=True):
        """이미지 분석 및 ArUco 감지"""
        print(f"\n분석 중: {image_path}")
        print("-" * 40)

        # 이미지 로드
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  [ERROR] 이미지를 불러올 수 없습니다: {image_path}")
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        vis_img = image.copy()

        # 이미지 정보
        h, w = gray.shape
        print(f"  이미지 크기: {w} x {h}")

        # ArUco 감지 시도
        detected = False
        for dict_name, aruco_dict in self.aruco_dicts.items():
            detector = aruco.ArucoDetector(aruco_dict, self.aruco_params)
            corners, ids, rejected = detector.detectMarkers(gray)

            if ids is not None and len(ids) > 0:
                detected = True
                print(f"  [OK] {dict_name}로 감지됨!")
                print(f"  감지된 마커 수: {len(ids)}")

                # 마커 그리기
                aruco.drawDetectedMarkers(vis_img, corners, ids)

                for i, (corner, marker_id) in enumerate(zip(corners, ids.flatten())):
                    # 코너 좌표
                    c = corner[0]
                    center_x = np.mean(c[:, 0])
                    center_y = np.mean(c[:, 1])

                    # 중앙으로부터의 픽셀 오프셋
                    offset_x = center_x - self.cx
                    offset_y = center_y - self.cy

                    print(f"\n  마커 ID {marker_id}:")
                    print(f"    픽셀 중심: ({center_x:.1f}, {center_y:.1f})")
                    print(f"    이미지 중앙으로부터 오프셋: ({offset_x:.1f}, {offset_y:.1f}) px")

                    # 3D 자세 추정
                    marker_points = np.array([
                        [-marker_size/2, marker_size/2, 0],
                        [marker_size/2, marker_size/2, 0],
                        [marker_size/2, -marker_size/2, 0],
                        [-marker_size/2, -marker_size/2, 0]
                    ], dtype=np.float32)

                    retval, rvec, tvec = cv2.solvePnP(
                        marker_points, corner, self.camera_matrix, self.dist_coeffs,
                        None, None, False, cv2.SOLVEPNP_IPPE_SQUARE
                    )

                    if retval:
                        tvec = tvec.flatten()
                        rvec = rvec.flatten()

                        print(f"    ★ tvec (카메라 좌표): [{tvec[0]:.3f}, {tvec[1]:.3f}, {tvec[2]:.3f}]")
                        print(f"    ★ 거리: {tvec[2]:.3f} m")

                        # 해석
                        print(f"\n    해석:")
                        if abs(tvec[0]) < 0.1:
                            print(f"      X: 거의 중앙 (오차 {tvec[0]:.3f}m)")
                        elif tvec[0] > 0:
                            print(f"      X: 오른쪽으로 {tvec[0]:.3f}m 치우침")
                        else:
                            print(f"      X: 왼쪽으로 {-tvec[0]:.3f}m 치우침")

                        if abs(tvec[1]) < 0.1:
                            print(f"      Y: 거의 중앙 (오차 {tvec[1]:.3f}m)")
                        elif tvec[1] > 0:
                            print(f"      Y: 아래쪽으로 {tvec[1]:.3f}m 치우침")
                        else:
                            print(f"      Y: 위쪽으로 {-tvec[1]:.3f}m 치우침")

                        # 축 그리기
                        cv2.drawFrameAxes(vis_img, self.camera_matrix, self.dist_coeffs,
                                         rvec.reshape(3,1), tvec.reshape(3,1), marker_size * 0.5)

                        # tvec 텍스트 추가
                        text = f"tvec: [{tvec[0]:.2f}, {tvec[1]:.2f}, {tvec[2]:.2f}]"
                        cv2.putText(vis_img, text, (int(center_x) - 100, int(center_y) + 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                break  # 감지 성공시 종료

        if not detected:
            print("  [FAIL] ArUco 마커를 감지하지 못했습니다")
            print(f"  거부된 후보: {len(rejected) if rejected else 0}개")

        # 십자선 그리기 (이미지 중앙)
        cv2.line(vis_img, (int(self.cx) - 30, int(self.cy)),
                (int(self.cx) + 30, int(self.cy)), (0, 0, 255), 2)
        cv2.line(vis_img, (int(self.cx), int(self.cy) - 30),
                (int(self.cx), int(self.cy) + 30), (0, 0, 255), 2)
        cv2.putText(vis_img, "CENTER", (int(self.cx) + 35, int(self.cy) + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 결과 저장
        if save_output:
            output_path = Path(image_path).parent / f"debug_{Path(image_path).name}"
            cv2.imwrite(str(output_path), vis_img)
            print(f"\n  결과 저장: {output_path}")

        return {
            'detected': detected,
            'image': vis_img,
            'corners': corners if detected else None,
            'ids': ids if detected else None,
        }

    def analyze_directory(self, directory, pattern="aruco_rl_*.png", marker_size=0.6):
        """디렉토리 내 모든 이미지 분석"""
        dir_path = Path(directory)
        images = sorted(dir_path.glob(pattern))

        print(f"\n{directory} 디렉토리 분석")
        print(f"패턴: {pattern}")
        print(f"발견된 이미지: {len(images)}개")
        print("=" * 60)

        results = []
        for img_path in images:
            result = self.analyze_image(img_path, marker_size)
            if result:
                results.append(result)

        # 요약
        detected_count = sum(1 for r in results if r['detected'])
        print(f"\n요약: {detected_count}/{len(results)}개 이미지에서 마커 감지")

        return results


def create_test_image_with_marker():
    """테스트용 ArUco 마커 이미지 생성"""
    # ArUco 마커 생성
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    marker_size = 400
    marker_img = np.zeros((marker_size, marker_size), dtype=np.uint8)
    marker_img = aruco.generateImageMarker(aruco_dict, 0, marker_size, marker_img, 1)

    # 배경 이미지 (1280x720, 흰색)
    bg_img = np.ones((720, 1280, 3), dtype=np.uint8) * 200

    # 마커를 중앙에 배치
    marker_color = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
    x_offset = (1280 - marker_size) // 2
    y_offset = (720 - marker_size) // 2
    bg_img[y_offset:y_offset+marker_size, x_offset:x_offset+marker_size] = marker_color

    # 저장
    output_path = "/tmp/test_aruco_center.png"
    cv2.imwrite(output_path, bg_img)
    print(f"테스트 이미지 생성: {output_path}")
    print("  - 마커가 정확히 중앙에 위치")
    print("  - 예상 tvec[0], tvec[1] ≈ 0")

    return output_path


def main():
    import sys

    print("=" * 60)
    print("ArUco 카메라 디버거")
    print("=" * 60)

    debugger = ArUcoDebugger(
        image_width=1280,
        image_height=720,
        fov_deg=150.0
    )

    if len(sys.argv) > 1:
        # 인자로 이미지 경로가 주어진 경우
        image_path = sys.argv[1]
        marker_size = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
        debugger.analyze_image(image_path, marker_size)
    else:
        # 기본: /tmp 디렉토리의 aruco 이미지 분석
        print("\n사용법:")
        print("  python debug_aruco_camera.py <이미지경로> [마커크기(m)]")
        print("  python debug_aruco_camera.py /tmp/aruco_rl_000050.png 0.6")
        print()

        # 테스트 이미지 생성 및 분석
        print("테스트 이미지로 검증:")
        test_path = create_test_image_with_marker()
        debugger.analyze_image(test_path, marker_size=0.6)

        # /tmp의 실제 이미지도 분석
        print("\n" + "=" * 60)
        print("/tmp 디렉토리의 실제 시뮬레이션 이미지 분석")
        print("=" * 60)

        from pathlib import Path
        aruco_images = sorted(Path("/tmp").glob("aruco_rl_*.png"))

        if aruco_images:
            print(f"발견된 이미지: {len(aruco_images)}개")
            # 최근 5개만 분석
            for img_path in aruco_images[-5:]:
                debugger.analyze_image(str(img_path), marker_size=0.6)
        else:
            print("시뮬레이션 이미지가 없습니다.")
            print("11_ardupilot_multi_vehicle.py를 먼저 실행하세요.")


if __name__ == "__main__":
    main()
