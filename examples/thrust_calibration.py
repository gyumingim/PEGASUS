#!/usr/bin/env python
"""
Iris 드론 추력 캘리브레이션 스크립트
호버링에 필요한 정확한 thrust 값을 측정합니다.
"""

import carb
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni.timeline
from omni.isaac.core.world import World
import numpy as np
from scipy.spatial.transform import Rotation
import asyncio
import threading

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig

from mavsdk import System
from mavsdk.offboard import Attitude, OffboardError


class ThrustCalibrationApp:
    """추력 캘리브레이션 앱"""

    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])

        # 드론 생성
        config = MultirotorConfig()
        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": 0,
            "px4_autolaunch": True,
            "px4_dir": self.pg.px4_path,
            "px4_vehicle_model": self.pg.px4_default_airframe
        })
        config.backends = [PX4MavlinkBackend(mavlink_config)]

        initial_pos = [0, 0, 1.0]  # 1m 높이에서 시작

        self.drone = Multirotor(
            "/World/Drone",
            ROBOTS['Iris'],
            0,
            initial_pos,
            Rotation.from_euler("XYZ", [0, 0, 0], degrees=True).as_quat(),
            config=config
        )

        self.world.reset()
        self.stop_sim = False

        # 캘리브레이션 상태
        self.current_thrust = 0.5  # 시작 추력
        self.thrust_step = 0.01    # 조정 단위
        self.hover_thrust = None   # 측정된 호버 추력

        # 측정 데이터
        self.height_history = []
        self.thrust_history = []
        self.stable_count = 0
        self.stable_threshold = 50  # 50 스텝 동안 안정되면 호버링으로 판정

    async def calibrate(self):
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

        await drone.offboard.set_attitude(Attitude(0.0, 0.0, 0.0, 0.0))

        print("[MAVSDK] -- Offboard 모드 시작")
        try:
            await drone.offboard.start()
        except OffboardError as error:
            print(f"[MAVSDK] Offboard 모드 시작 실패: {error._result.result}")
            await drone.action.disarm()
            return

        print("\n" + "=" * 70)
        print("추력 캘리브레이션 시작")
        print("=" * 70)
        print("목표: 드론이 1m 높이에서 안정적으로 호버링하는 thrust 값 찾기")
        print("=" * 70)

        # 초기 상승 (안전하게 시작)
        print("\n[Phase 1] 초기 상승...")
        for _ in range(100):
            await drone.offboard.set_attitude(Attitude(0.0, 0.0, 0.0, 0.6))
            await asyncio.sleep(0.02)

        # 캘리브레이션 루프
        print("\n[Phase 2] 자동 캘리브레이션 시작...")
        print("키보드 입력: 'u'=추력 증가, 'd'=추력 감소, 'q'=종료\n")

        target_height = 1.5  # 목표 고도
        step_count = 0

        while not self.stop_sim and simulation_app.is_running():
            # 현재 상태 읽기
            drone_state = self.drone.state
            current_height = drone_state.position[2]
            vertical_vel = drone_state.linear_velocity[2]

            # 높이 기록
            self.height_history.append(current_height)
            self.thrust_history.append(self.current_thrust)

            # 자동 조정: P 컨트롤러
            height_error = target_height - current_height

            # 높이 오차와 수직 속도 기반 추력 조정
            # 너무 낮으면 추력 증가, 너무 높으면 감소
            adjustment = 0.0
            if abs(height_error) > 0.1:  # 10cm 이상 오차
                adjustment = height_error * 0.01  # P gain

            # 수직 속도 댐핑
            adjustment -= vertical_vel * 0.02  # D gain

            self.current_thrust += adjustment
            self.current_thrust = np.clip(self.current_thrust, 0.3, 0.8)

            # 안정성 체크
            if abs(height_error) < 0.05 and abs(vertical_vel) < 0.1:
                self.stable_count += 1
            else:
                self.stable_count = 0

            # 호버링 감지
            if self.stable_count >= self.stable_threshold:
                self.hover_thrust = self.current_thrust
                print(f"\n{'=' * 70}")
                print(f"호버링 감지! Thrust = {self.hover_thrust:.4f}")
                print(f"높이: {current_height:.3f}m, 수직속도: {vertical_vel:.3f}m/s")
                print(f"{'=' * 70}")
                self.stable_count = 0  # 리셋하고 계속 측정

            # 명령 전송
            await drone.offboard.set_attitude(
                Attitude(0.0, 0.0, 0.0, float(self.current_thrust))
            )

            # 상태 출력 (50 스텝마다)
            if step_count % 50 == 0:
                status = "STABLE" if self.stable_count > 10 else "ADJUSTING"
                print(f"[{step_count:5d}] Thrust: {self.current_thrust:.4f} | "
                      f"Height: {current_height:.3f}m | "
                      f"Vel_Z: {vertical_vel:+.3f}m/s | "
                      f"Status: {status}")

                if self.hover_thrust:
                    print(f"        >>> 측정된 호버 추력: {self.hover_thrust:.4f}")

            step_count += 1
            await asyncio.sleep(0.02)

            # 안전 체크
            if current_height < 0.3:
                print("[WARN] 고도 너무 낮음! 추력 증가")
                self.current_thrust = min(self.current_thrust + 0.05, 0.8)
            elif current_height > 3.0:
                print("[WARN] 고도 너무 높음! 추력 감소")
                self.current_thrust = max(self.current_thrust - 0.05, 0.3)

        # 결과 출력
        print("\n" + "=" * 70)
        print("캘리브레이션 결과")
        print("=" * 70)

        if self.hover_thrust:
            print(f"측정된 호버 추력: {self.hover_thrust:.4f}")
            print(f"\nIsaacLabToPX4Converter에 적용할 값:")
            print(f"  IRIS_HOVER_THRUST = {self.hover_thrust:.4f}")
        else:
            # 히스토리에서 추정
            if len(self.height_history) > 100:
                # 높이가 가장 안정적이었던 구간의 추력 평균
                heights = np.array(self.height_history[-200:])
                thrusts = np.array(self.thrust_history[-200:])

                # 높이 변화가 작은 구간 찾기
                height_diff = np.abs(np.diff(heights))
                stable_idx = np.where(height_diff < 0.01)[0]

                if len(stable_idx) > 10:
                    estimated_thrust = np.mean(thrusts[stable_idx])
                    print(f"추정된 호버 추력: {estimated_thrust:.4f}")
                    print(f"\nIsaacLabToPX4Converter에 적용할 값:")
                    print(f"  IRIS_HOVER_THRUST = {estimated_thrust:.4f}")

        print("=" * 70)

        # 착륙
        print("\n[MAVSDK] -- 착륙")
        try:
            await drone.offboard.stop()
        except:
            pass
        await drone.action.land()
        await asyncio.sleep(3)

    def run_calibration_thread(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.calibrate())
        finally:
            loop.close()

    def run(self):
        calibration_thread = threading.Thread(target=self.run_calibration_thread, daemon=True)
        calibration_thread.start()

        self.timeline.play()

        print("[Init] 시뮬레이션 초기화 중...")
        for _ in range(200):
            self.world.step(render=True)

        print("[Init] 캘리브레이션 준비 완료")

        while simulation_app.is_running() and not self.stop_sim:
            self.world.step(render=True)

        self.timeline.stop()
        simulation_app.close()


def main():
    import signal

    app = None

    def cleanup_handler(signum, frame):
        print("\n[Signal] 종료 신호 수신...")
        if app is not None:
            app.stop_sim = True

    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)

    print("\n" + "=" * 70)
    print("Iris 드론 추력 캘리브레이션")
    print("=" * 70)
    print("목적: PX4에서 Iris 드론이 호버링하는 정확한 thrust 값 측정")
    print("=" * 70 + "\n")

    try:
        app = ThrustCalibrationApp()
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
