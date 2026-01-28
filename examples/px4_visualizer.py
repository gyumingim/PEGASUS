#!/usr/bin/env python3
"""
PX4 SITL 텔레메트리 시각화

사용법:
1. 터미널 1: cd ~/PX4-Autopilot && make px4_sitl
2. 터미널 2: python px4_visualizer.py

기능:
- 드론 위치/속도/자세 실시간 표시
- 2D 탑뷰 + 고도바
- Attitude 표시 (roll/pitch/yaw)
"""

import asyncio
import numpy as np
import cv2
import time
from collections import deque
import argparse

from mavsdk import System
from mavsdk.offboard import OffboardError, Attitude


class PX4Visualizer:
    """PX4 텔레메트리 OpenCV 시각화"""

    def __init__(self, width=1000, height=700):
        self.width = width
        self.height = height

        # 상태
        self.position = np.zeros(3)  # NED
        self.velocity = np.zeros(3)  # NED
        self.attitude_euler = np.zeros(3)  # roll, pitch, yaw (deg)
        self.attitude_quat = np.array([0, 0, 0, 1])
        self.angular_velocity = np.zeros(3)
        self.armed = False
        self.flight_mode = "UNKNOWN"
        self.connected = False

        # 궤적
        self.trajectory = deque(maxlen=1000)

        # 스케일
        self.scale = 30  # 1m = 30px

    def draw(self) -> np.ndarray:
        """시각화 이미지 생성"""
        img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 30

        # === 좌측: 탑뷰 (XY 평면) ===
        top_view_w = 500
        top_view_h = 500
        top_cx = top_view_w // 2
        top_cy = top_view_h // 2

        # 그리드
        for i in range(-10, 11):
            x = top_cx + i * self.scale
            y = top_cy + i * self.scale
            color = (60, 60, 60) if i != 0 else (100, 100, 100)
            thickness = 1 if i != 0 else 2
            cv2.line(img, (x, 0), (x, top_view_h), color, thickness)
            cv2.line(img, (0, y), (top_view_w, y), color, thickness)

        # 원점 라벨
        cv2.putText(img, "N", (top_cx - 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, "E", (top_view_w - 20, top_cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 드론 위치 (NED → 화면)
        dx = int(top_cx + self.position[1] * self.scale)  # East → 오른쪽
        dy = int(top_cy - self.position[0] * self.scale)  # North → 위

        # 궤적 저장
        if 0 <= dx < top_view_w and 0 <= dy < top_view_h:
            self.trajectory.append((dx, dy))

        # 궤적 그리기
        for i in range(1, len(self.trajectory)):
            alpha = i / len(self.trajectory)
            color = (int(100 * alpha), int(255 * alpha), int(100 * alpha))
            cv2.line(img, self.trajectory[i-1], self.trajectory[i], color, 1)

        # 드론 그리기
        drone_color = (0, 255, 0) if self.armed else (0, 100, 255)
        cv2.circle(img, (dx, dy), 12, drone_color, 2)

        # 드론 방향 (yaw)
        yaw_rad = np.radians(self.attitude_euler[2])
        arrow_len = 25
        ax = int(dx + arrow_len * np.sin(yaw_rad))
        ay = int(dy - arrow_len * np.cos(yaw_rad))
        cv2.arrowedLine(img, (dx, dy), (ax, ay), drone_color, 2, tipLength=0.3)

        # 속도 벡터
        vel_scale = 5
        vx = int(dx + self.velocity[1] * vel_scale)
        vy = int(dy - self.velocity[0] * vel_scale)
        cv2.arrowedLine(img, (dx, dy), (vx, vy), (255, 255, 0), 1, tipLength=0.2)

        # === 우측 상단: 자세 표시 ===
        att_x = 550
        att_y = 30

        # Attitude 인디케이터 배경
        cv2.rectangle(img, (520, 10), (700, 200), (50, 50, 50), -1)
        cv2.rectangle(img, (520, 10), (700, 200), (100, 100, 100), 1)

        cv2.putText(img, "ATTITUDE", (att_x, att_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        att_y += 30

        # Roll
        roll_color = (0, 255, 0) if abs(self.attitude_euler[0]) < 5 else (0, 165, 255)
        cv2.putText(img, f"Roll:  {self.attitude_euler[0]:+7.2f} deg", (att_x, att_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, roll_color, 1)
        att_y += 25

        # Pitch
        pitch_color = (0, 255, 0) if abs(self.attitude_euler[1]) < 5 else (0, 165, 255)
        cv2.putText(img, f"Pitch: {self.attitude_euler[1]:+7.2f} deg", (att_x, att_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, pitch_color, 1)
        att_y += 25

        # Yaw
        cv2.putText(img, f"Yaw:   {self.attitude_euler[2]:+7.2f} deg", (att_x, att_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        att_y += 30

        # Angular velocity
        cv2.putText(img, "Angular Vel (rad/s):", (att_x, att_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        att_y += 20
        cv2.putText(img, f"  p: {self.angular_velocity[0]:+.3f}", (att_x, att_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        att_y += 18
        cv2.putText(img, f"  q: {self.angular_velocity[1]:+.3f}", (att_x, att_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        att_y += 18
        cv2.putText(img, f"  r: {self.angular_velocity[2]:+.3f}", (att_x, att_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # === 우측 중단: 위치/속도 ===
        pos_x = 550
        pos_y = 230

        cv2.rectangle(img, (520, 210), (700, 380), (50, 50, 50), -1)
        cv2.rectangle(img, (520, 210), (700, 380), (100, 100, 100), 1)

        cv2.putText(img, "POSITION (NED)", (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        pos_y += 25
        cv2.putText(img, f"N: {self.position[0]:+8.2f} m", (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        pos_y += 22
        cv2.putText(img, f"E: {self.position[1]:+8.2f} m", (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        pos_y += 22
        cv2.putText(img, f"D: {self.position[2]:+8.2f} m", (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        pos_y += 30

        cv2.putText(img, "VELOCITY (NED)", (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        pos_y += 25
        cv2.putText(img, f"Vn: {self.velocity[0]:+7.2f} m/s", (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        pos_y += 22
        cv2.putText(img, f"Ve: {self.velocity[1]:+7.2f} m/s", (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        pos_y += 22
        cv2.putText(img, f"Vd: {self.velocity[2]:+7.2f} m/s", (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # === 우측: 고도바 ===
        bar_x = 720
        bar_top = 50
        bar_bottom = 450
        bar_height = bar_bottom - bar_top

        # 배경
        cv2.rectangle(img, (bar_x - 15, bar_top - 20), (bar_x + 50, bar_bottom + 30), (50, 50, 50), -1)
        cv2.putText(img, "ALT", (bar_x - 10, bar_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 바
        cv2.rectangle(img, (bar_x, bar_top), (bar_x + 20, bar_bottom), (80, 80, 80), -1)
        cv2.rectangle(img, (bar_x, bar_top), (bar_x + 20, bar_bottom), (150, 150, 150), 1)

        # 현재 고도 (0~20m 범위)
        alt = -self.position[2]  # NED에서 down이 양수
        alt_ratio = np.clip(alt / 20.0, 0, 1)
        alt_y = int(bar_bottom - alt_ratio * bar_height)

        # 고도 막대
        cv2.rectangle(img, (bar_x + 2, alt_y), (bar_x + 18, bar_bottom), (0, 255, 0), -1)

        # 고도 숫자
        cv2.putText(img, f"{alt:.1f}m", (bar_x - 5, alt_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 스케일 표시
        for h in [0, 5, 10, 15, 20]:
            y = int(bar_bottom - (h / 20.0) * bar_height)
            cv2.line(img, (bar_x + 20, y), (bar_x + 25, y), (150, 150, 150), 1)
            cv2.putText(img, f"{h}", (bar_x + 28, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

        # === 하단: 상태 ===
        status_y = 520

        # 연결 상태
        conn_color = (0, 255, 0) if self.connected else (0, 0, 255)
        conn_text = "CONNECTED" if self.connected else "DISCONNECTED"
        cv2.circle(img, (30, status_y), 8, conn_color, -1)
        cv2.putText(img, conn_text, (45, status_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, conn_color, 1)

        # Armed 상태
        arm_color = (0, 255, 0) if self.armed else (0, 100, 255)
        arm_text = "ARMED" if self.armed else "DISARMED"
        cv2.circle(img, (200, status_y), 8, arm_color, -1)
        cv2.putText(img, arm_text, (215, status_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, arm_color, 1)

        # Flight mode
        cv2.putText(img, f"Mode: {self.flight_mode}", (350, status_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # === 사이드뷰 (XZ 평면) ===
        side_y_offset = 550
        side_h = 140
        side_cx = 250

        cv2.rectangle(img, (0, side_y_offset - 10), (500, self.height), (40, 40, 40), -1)
        cv2.putText(img, "SIDE VIEW (N-D)", (10, side_y_offset + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 지면
        ground_y = side_y_offset + side_h - 20
        cv2.line(img, (0, ground_y), (500, ground_y), (100, 100, 100), 2)

        # 드론
        side_dx = int(side_cx + self.position[0] * self.scale)
        side_dy = int(ground_y + self.position[2] * self.scale)  # down이 양수

        cv2.circle(img, (side_dx, side_dy), 8, drone_color, 2)

        # Pitch 표시
        pitch_rad = np.radians(self.attitude_euler[1])
        px = int(side_dx + 20 * np.cos(pitch_rad))
        py = int(side_dy - 20 * np.sin(pitch_rad))
        cv2.arrowedLine(img, (side_dx, side_dy), (px, py), drone_color, 2, tipLength=0.3)

        # === Attitude 인디케이터 (인공수평선) ===
        ahi_cx = 850
        ahi_cy = 150
        ahi_r = 80

        cv2.rectangle(img, (ahi_cx - ahi_r - 20, ahi_cy - ahi_r - 30),
                     (ahi_cx + ahi_r + 20, ahi_cy + ahi_r + 30), (50, 50, 50), -1)
        cv2.putText(img, "HORIZON", (ahi_cx - 35, ahi_cy - ahi_r - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 원형 테두리
        cv2.circle(img, (ahi_cx, ahi_cy), ahi_r, (150, 150, 150), 2)

        # Roll/Pitch에 따른 수평선
        roll_rad = np.radians(self.attitude_euler[0])
        pitch_offset = self.attitude_euler[1] * 2  # 스케일

        # 수평선 계산
        cos_r, sin_r = np.cos(roll_rad), np.sin(roll_rad)

        # 하늘 (파란색) / 땅 (갈색) 영역
        sky_color = (200, 150, 50)
        ground_color = (50, 100, 150)

        # 간단한 수평선
        h_len = ahi_r - 10
        lx1 = int(ahi_cx - h_len * cos_r)
        ly1 = int(ahi_cy + pitch_offset + h_len * sin_r)
        lx2 = int(ahi_cx + h_len * cos_r)
        ly2 = int(ahi_cy + pitch_offset - h_len * sin_r)

        cv2.line(img, (lx1, ly1), (lx2, ly2), (255, 255, 255), 2)

        # 중심 마커
        cv2.line(img, (ahi_cx - 20, ahi_cy), (ahi_cx - 5, ahi_cy), (255, 200, 0), 2)
        cv2.line(img, (ahi_cx + 5, ahi_cy), (ahi_cx + 20, ahi_cy), (255, 200, 0), 2)
        cv2.circle(img, (ahi_cx, ahi_cy), 3, (255, 200, 0), -1)

        # Roll 눈금
        for angle in [-60, -30, 0, 30, 60]:
            a_rad = np.radians(angle - 90)
            tx = int(ahi_cx + (ahi_r + 10) * np.cos(a_rad))
            ty = int(ahi_cy + (ahi_r + 10) * np.sin(a_rad))
            cv2.putText(img, f"{angle}", (tx - 10, ty + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

        # === 키 안내 ===
        cv2.putText(img, "Keys: ESC=Quit, A=Arm, D=Disarm, T=Takeoff, L=Land, O=Offboard",
                   (10, self.height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        return img


class PX4Controller:
    """PX4 SITL 연결 및 제어"""

    def __init__(self, connection: str = "udp://:14540"):
        self.connection = connection
        self.drone = System()
        self.visualizer = PX4Visualizer()
        self.running = False

    async def connect(self):
        print(f"[MAVSDK] 연결 중: {self.connection}")
        await self.drone.connect(system_address=self.connection)

        print("[MAVSDK] 연결 대기...")
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("[MAVSDK] 연결 완료!")
                self.visualizer.connected = True
                break

    async def _telemetry_loops(self):
        """텔레메트리 수신"""
        async def pos_vel():
            async for pv in self.drone.telemetry.position_velocity_ned():
                self.visualizer.position = np.array([
                    pv.position.north_m, pv.position.east_m, pv.position.down_m
                ])
                self.visualizer.velocity = np.array([
                    pv.velocity.north_m_s, pv.velocity.east_m_s, pv.velocity.down_m_s
                ])
                if not self.running:
                    break

        async def attitude():
            async for att in self.drone.telemetry.attitude_euler():
                self.visualizer.attitude_euler = np.array([
                    att.roll_deg, att.pitch_deg, att.yaw_deg
                ])
                if not self.running:
                    break

        async def quat():
            async for q in self.drone.telemetry.attitude_quaternion():
                self.visualizer.attitude_quat = np.array([q.x, q.y, q.z, q.w])
                if not self.running:
                    break

        async def angular():
            async for r in self.drone.telemetry.attitude_angular_velocity_body():
                self.visualizer.angular_velocity = np.array([
                    r.roll_rad_s, r.pitch_rad_s, r.yaw_rad_s
                ])
                if not self.running:
                    break

        async def armed():
            async for is_armed in self.drone.telemetry.armed():
                self.visualizer.armed = is_armed
                if not self.running:
                    break

        async def flight_mode():
            async for mode in self.drone.telemetry.flight_mode():
                self.visualizer.flight_mode = str(mode).split('.')[-1]
                if not self.running:
                    break

        await asyncio.gather(pos_vel(), attitude(), quat(), angular(), armed(), flight_mode())

    async def run(self):
        """메인 루프"""
        await self.connect()

        self.running = True
        telemetry_task = asyncio.create_task(self._telemetry_loops())

        print("\n" + "="*50)
        print("PX4 Visualizer 시작")
        print("="*50)
        print("  ESC: 종료")
        print("  A: Arm")
        print("  D: Disarm")
        print("  T: Takeoff")
        print("  L: Land")
        print("  O: Offboard (hover)")
        print("="*50 + "\n")

        try:
            while self.running:
                # 시각화
                img = self.visualizer.draw()
                cv2.imshow("PX4 Visualizer", img)

                # 키 처리
                key = cv2.waitKey(20) & 0xFF

                if key == 27:  # ESC
                    break
                elif key == ord('a'):
                    print("[CMD] Arming...")
                    await self.drone.action.arm()
                elif key == ord('d'):
                    print("[CMD] Disarming...")
                    await self.drone.action.disarm()
                elif key == ord('t'):
                    print("[CMD] Takeoff...")
                    await self.drone.action.takeoff()
                elif key == ord('l'):
                    print("[CMD] Landing...")
                    await self.drone.action.land()
                elif key == ord('o'):
                    print("[CMD] Offboard hover...")
                    try:
                        await self.drone.offboard.set_attitude(Attitude(0, 0, 0, 0.5))
                        await self.drone.offboard.start()
                    except OffboardError as e:
                        print(f"  Offboard 실패: {e}")

                await asyncio.sleep(0.02)

        finally:
            self.running = False
            telemetry_task.cancel()
            cv2.destroyAllWindows()
            print("[종료]")


async def main():
    parser = argparse.ArgumentParser(description='PX4 SITL Visualizer')
    parser.add_argument('--connection', type=str, default='udp://:14540', help='연결 주소')
    args = parser.parse_args()

    print("\n" + "="*50)
    print("PX4 SITL 텔레메트리 시각화")
    print("="*50)
    print(f"연결: {args.connection}")
    print("\n먼저 PX4 SITL을 실행하세요:")
    print("  cd ~/PX4-Autopilot && make px4_sitl")
    print("\n준비되면 Enter...")
    input()

    controller = PX4Controller(connection=args.connection)
    await controller.run()


if __name__ == "__main__":
    asyncio.run(main())
