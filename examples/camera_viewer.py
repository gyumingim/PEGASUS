#!/usr/bin/env python3
"""
드론 카메라 뷰어 - 실시간으로 저장된 이미지를 표시
시스템 Python으로 실행: python3 camera_viewer.py
"""

import cv2
import time
import os

IMAGE_PATH = "/tmp/drone_camera_view.png"

def main():
    print("=" * 50)
    print("Drone Camera Viewer")
    print("=" * 50)
    print(f"Watching: {IMAGE_PATH}")
    print("Press 'q' to quit")
    print("=" * 50)

    cv2.namedWindow("Drone Camera View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drone Camera View", 1280, 720)

    last_mtime = 0

    while True:
        try:
            if os.path.exists(IMAGE_PATH):
                mtime = os.path.getmtime(IMAGE_PATH)

                # 파일이 업데이트되었을 때만 읽기
                if mtime != last_mtime:
                    img = cv2.imread(IMAGE_PATH)
                    if img is not None:
                        cv2.imshow("Drone Camera View", img)
                        last_mtime = mtime
            else:
                # 파일이 없으면 대기 메시지
                blank = cv2.imread(IMAGE_PATH) if os.path.exists(IMAGE_PATH) else None
                if blank is None:
                    blank = 255 * cv2.UMat(480, 640, cv2.CV_8UC3)

        except Exception as e:
            pass

        key = cv2.waitKey(10) & 0xFF  # 10ms = ~100fps max
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Viewer closed.")

if __name__ == "__main__":
    main()
