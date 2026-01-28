#!/usr/bin/env python3
"""
드론 착륙용 AprilTag 인식 및 상대 위치/각도 계산
Intel RealSense D435i 사용
"""

import pyrealsense2 as rs
import numpy as np
import cv2
from pupil_apriltags import Detector

class AprilTagLanding:
    def __init__(self, tag_size=0.165):  # AprilTag 실제 크기 (미터, 예: 16.5cm)
        """
        tag_size: AprilTag의 실제 크기 (미터 단위)
        """
        self.tag_size = tag_size
        
        # RealSense 파이프라인 설정
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # 파이프라인 시작
        profile = self.pipeline.start(config)
        
        # 카메라 내부 파라미터 가져오기
        color_stream = profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        
        self.camera_params = [
            intrinsics.fx,  # focal length x
            intrinsics.fy,  # focal length y
            intrinsics.ppx,  # principal point x
            intrinsics.ppy   # principal point y
        ]
        
        # AprilTag 검출기 초기화
        self.detector = Detector(
            families='tag36h11',
            nthreads=4,
            quad_decimate=2.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )
        
        print(f"초기화 완료 - Tag 크기: {tag_size}m")
        print(f"카메라 파라미터: fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}")
        
    def rotation_matrix_to_euler(self, R):
        """회전 행렬을 오일러 각도로 변환 (Roll, Pitch, Yaw)"""
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        
        if sy > 1e-6:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0
            
        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)
    
    def get_relative_pose(self):
        """
        AprilTag 상대 위치 및 각도 반환
        Returns:
            dict: {
                'detected': bool,
                'tag_id': int,
                'x': float (미터, 오른쪽+),
                'y': float (미터, 아래+),
                'z': float (미터, 앞+),
                'roll': float (도),
                'pitch': float (도),
                'yaw': float (도),
                'distance': float (미터),
                'image': numpy array (항상 포함)
            }
        """
        try:
            # 프레임 획득
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                print("⚠ 프레임을 받지 못함")
                return {'detected': False, 'image': np.zeros((480, 640, 3), dtype=np.uint8)}
            
            # 이미지 변환
            color_image = np.asanyarray(color_frame.get_data())
            
            # 그레이스케일 변환
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            
            # AprilTag 검출
            detections = self.detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=self.camera_params,
                tag_size=self.tag_size
            )
            
            if len(detections) == 0:
                return {'detected': False, 'image': color_image}
            
            # 첫 번째 태그 사용 (여러 개면 가장 가까운 것 선택 가능)
            detection = detections[0]
            
            # 상대 위치 (카메라 좌표계)
            x, y, z = detection.pose_t.flatten()
            
            # 회전 행렬에서 오일러 각도 추출
            R = detection.pose_R
            roll, pitch, yaw = self.rotation_matrix_to_euler(R)
            
            # 거리 계산
            distance = np.linalg.norm(detection.pose_t)
            
            result = {
                'detected': True,
                'tag_id': detection.tag_id,
                'x': float(x),      # 오른쪽이 양수
                'y': float(y),      # 아래가 양수
                'z': float(z),      # 앞이 양수
                'roll': float(roll),
                'pitch': float(pitch),
                'yaw': float(yaw),
                'distance': float(distance),
                'image': color_image,
                'corners': detection.corners,
                'rotation_matrix': R  # 3D 축 그리기용
            }
            
            return result
            
        except Exception as e:
            print(f"⚠ get_relative_pose 에러: {e}")
            return {'detected': False, 'image': np.zeros((480, 640, 3), dtype=np.uint8)}
    
    def draw_3d_axes(self, image, rvec, tvec, camera_params, axis_length=0.05):
        """3D 좌표축 그리기 (Roll, Pitch, Yaw 시각화)"""
        # 카메라 행렬 구성
        fx, fy, cx, cy = camera_params
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        dist_coeffs = np.zeros(4)
        
        # 3D 좌표축 점들 (원점 + X, Y, Z축)
        axis_points_3d = np.array([
            [0, 0, 0],              # 원점
            [axis_length, 0, 0],    # X축 (빨강)
            [0, axis_length, 0],    # Y축 (초록)
            [0, 0, axis_length]     # Z축 (파랑)
        ], dtype=np.float32)
        
        # 3D -> 2D 투영
        image_points, _ = cv2.projectPoints(
            axis_points_3d, rvec, tvec, camera_matrix, dist_coeffs
        )
        image_points = image_points.reshape(-1, 2).astype(int)
        
        origin = tuple(image_points[0])
        
        # X축 (빨강) - Roll
        cv2.arrowedLine(image, origin, tuple(image_points[1]), (0, 0, 255), 3, tipLength=0.3)
        cv2.putText(image, 'X(Roll)', tuple(image_points[1] + [10, 0]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Y축 (초록) - Pitch  
        cv2.arrowedLine(image, origin, tuple(image_points[2]), (0, 255, 0), 3, tipLength=0.3)
        cv2.putText(image, 'Y(Pitch)', tuple(image_points[2] + [10, 0]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Z축 (파랑) - Yaw
        cv2.arrowedLine(image, origin, tuple(image_points[3]), (255, 0, 0), 3, tipLength=0.3)
        cv2.putText(image, 'Z(Yaw)', tuple(image_points[3] + [10, 0]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return image
    
    def visualize(self, result):
        """검출 결과 시각화 (3D 축 포함)"""
        if not result['detected']:
            return None
            
        image = result['image'].copy()
        corners = result['corners'].astype(int)
        
        # AprilTag 테두리 그리기
        cv2.polylines(image, [corners], True, (0, 255, 0), 2)
        
        # 중심점 표시
        center = corners.mean(axis=0).astype(int)
        cv2.circle(image, tuple(center), 5, (0, 0, 255), -1)
        
        # 3D 좌표축 그리기
        try:
            R = result.get('rotation_matrix')
            if R is not None:
                # 회전 행렬 -> 회전 벡터
                rvec, _ = cv2.Rodrigues(R)
                tvec = np.array([[result['x']], [result['y']], [result['z']]], dtype=np.float32)
                
                self.draw_3d_axes(image, rvec, tvec, self.camera_params, axis_length=self.tag_size * 0.5)
        except Exception as e:
            print(f"⚠ 3D 축 그리기 실패: {e}")
        
        # 정보 표시
        text_y = 30
        cv2.putText(image, f"ID: {result['tag_id']}", (10, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        text_y += 25
        cv2.putText(image, f"Distance: {result['distance']:.2f}m", (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        text_y += 25
        cv2.putText(image, f"Pos: X={result['x']:+.2f}m  Y={result['y']:+.2f}m  Z={result['z']:+.2f}m", 
                    (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        text_y += 25
        cv2.putText(image, f"Roll: {result['roll']:+.1f}deg  Pitch: {result['pitch']:+.1f}deg  Yaw: {result['yaw']:+.1f}deg", 
                    (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 각도별 상태 표시 (시각적 가이드)
        text_y += 30
        
        # Pitch 상태
        if abs(result['pitch']) < 5:
            pitch_status = "LEVEL ✓"
            pitch_color = (0, 255, 0)
        elif result['pitch'] < -5:
            pitch_status = "TILT UP ↑"
            pitch_color = (0, 165, 255)
        else:
            pitch_status = "TILT DOWN ↓"
            pitch_color = (0, 165, 255)
        
        # Roll 상태
        if abs(result['roll']) < 5:
            roll_status = "LEVEL ✓"
            roll_color = (0, 255, 0)
        elif result['roll'] < -5:
            roll_status = "LEFT ←"
            roll_color = (0, 165, 255)
        else:
            roll_status = "RIGHT →"
            roll_color = (0, 165, 255)
        
        # Yaw 상태
        if abs(result['yaw']) < 5:
            yaw_status = "ALIGNED ✓"
            yaw_color = (0, 255, 0)
        elif result['yaw'] < -5:
            yaw_status = "CCW ↺"
            yaw_color = (0, 165, 255)
        else:
            yaw_status = "CW ↻"
            yaw_color = (0, 165, 255)
        
        cv2.putText(image, f"Pitch: {pitch_status}", (10, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, pitch_color, 2)
        text_y += 25
        cv2.putText(image, f"Roll: {roll_status}", (10, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, roll_color, 2)
        text_y += 25
        cv2.putText(image, f"Yaw: {yaw_status}", (10, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, yaw_color, 2)
        
        return image
    
    def stop(self):
        """카메라 종료"""
        self.pipeline.stop()


def main():
    # AprilTag 크기 설정 (미터 단위, 실제 측정값 입력)
    TAG_SIZE = 0.165  # 16.5cm
    
    print("\n=== 드론 착륙 AprilTag 추적 시작 ===")
    print("ESC 키를 눌러 종료\n")
    
    try:
        landing = AprilTagLanding(tag_size=TAG_SIZE)
    except Exception as e:
        print(f"\n❌ 초기화 실패: {e}")
        print("\n해결 방법:")
        print("1. RealSense 카메라 USB 연결 확인")
        print("2. 먼저 test_realsense.py로 카메라 테스트")
        print("3. pip install pyrealsense2 재설치")
        import traceback
        traceback.print_exc()
        return
    
    frame_count = 0
    detection_count = 0
    
    try:
        while True:
            frame_count += 1
            
            try:
                # 상대 위치 및 각도 획득
                result = landing.get_relative_pose()
                
                if result['detected']:
                    detection_count += 1
                    print(f"\n[Frame {frame_count} - Tag ID: {result['tag_id']}]")
                    print(f"상대 위치: X={result['x']:+.3f}m, Y={result['y']:+.3f}m, Z={result['z']:+.3f}m")
                    print(f"거리: {result['distance']:.3f}m")
                    print(f"각도: Yaw={result['yaw']:+.1f}°, Pitch={result['pitch']:+.1f}°, Roll={result['roll']:+.1f}°")
                    
                    # 착륙 가이던스
                    if abs(result['x']) < 0.05 and abs(result['y']) < 0.05:
                        print("✓ 중앙 정렬 완료!")
                    else:
                        if result['x'] > 0.05:
                            print("→ 오른쪽으로 이동 필요")
                        elif result['x'] < -0.05:
                            print("← 왼쪽으로 이동 필요")
                        
                        if result['y'] > 0.05:
                            print("↓ 아래로 이동 필요")
                        elif result['y'] < -0.05:
                            print("↑ 위로 이동 필요")
                    
                    # 시각화
                    vis_image = landing.visualize(result)
                    if vis_image is not None:
                        cv2.imshow('AprilTag Drone Landing', vis_image)
                else:
                    if frame_count % 30 == 0:  # 30프레임마다 출력
                        print(f"[Frame {frame_count}] 태그 없음 (검출: {detection_count}회)")
                    
                    # 카메라 영상은 계속 표시
                    if 'image' in result:
                        img = result['image'].copy()
                        cv2.putText(img, "No AprilTag detected", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(img, f"Frame: {frame_count}", (10, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.imshow('AprilTag Drone Landing', img)
                    else:
                        # 빈 화면이라도 표시
                        blank = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(blank, "No camera frame", (10, 240),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.imshow('AprilTag Drone Landing', blank)
                
            except Exception as e:
                print(f"⚠ 프레임 처리 에러: {e}")
                if frame_count % 30 == 0:
                    import traceback
                    traceback.print_exc()
            
            # ESC로 종료
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
    except KeyboardInterrupt:
        print("\n사용자 중단")
    except Exception as e:
        print(f"\n❌ 실행 중 에러: {e}")
        import traceback
        traceback.print_exc()
    finally:
        landing.stop()
        cv2.destroyAllWindows()
        print(f"\n종료됨 (총 {frame_count}프레임, {detection_count}회 검출)")


if __name__ == "__main__":
    main()