#!/usr/bin/env python
"""
통합 PX4 시뮬레이션 및 제어 스크립트
Isaac Sim으로 시뮬레이션을 실행하고 MAVSDK로 드론을 제어합니다.
"""

import asyncio
import threading
import carb
from isaacsim import SimulationApp

# Isaac Sim 시작
simulation_app = SimulationApp({"headless": False})

import omni.timeline
from omni.isaac.core.world import World
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from scipy.spatial.transform import Rotation

# MAVSDK 임포트
from mavsdk import System
from mavsdk.offboard import Attitude, OffboardError


class PegasusApp:
    """
    Isaac Sim 시뮬레이션과 MAVSDK 제어를 통합한 클래스
    """

    def __init__(self):
        """시뮬레이션 환경 초기화"""
        
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # 환경 로드
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])

        # 드론 생성
        config_multirotor = MultirotorConfig()
        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": 0,
            "px4_autolaunch": True,
            "px4_dir": self.pg.px4_path,
            "px4_vehicle_model": self.pg.px4_default_airframe
        })
        config_multirotor.backends = [PX4MavlinkBackend(mavlink_config)]

        Multirotor(
            "/World/quadrotor",
            ROBOTS['Iris'],
            0,
            [0.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )

        self.world.reset()
        self.stop_sim = False
        self.control_started = False

    async def control_drone(self):
        """MAVSDK를 사용한 드론 제어 (별도 스레드의 이벤트 루프에서 실행)"""
        
        drone = System()
        await drone.connect(system_address="udp://:14540")

        print("드론 연결 대기 중...")
        async for state in drone.core.connection_state():
            if state.is_connected:
                print("-- 드론 연결 완료!")
                break

        print("GPS 위치 추정 대기 중...")
        async for health in drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                print("-- GPS 위치 추정 완료")
                break

        print("-- Arming")
        await drone.action.arm()

        print("-- 초기 setpoint 설정")
        await drone.offboard.set_attitude(Attitude(0.0, 0.0, 0.0, 0.0))

        print("-- Offboard 모드 시작")
        try:
            await drone.offboard.start()
        except OffboardError as error:
            print(f"Offboard 모드 시작 실패: {error._result.result}")
            print("-- Disarming")
            await drone.action.disarm()
            return

        print("-- 70% 추력으로 상승")
        await drone.offboard.set_attitude(Attitude(0.0, 0.0, 0.0, 0.7))
        await asyncio.sleep(2)

        print("-- Roll 30도, 60% 추력")
        await drone.offboard.set_attitude(Attitude(30.0, 0.0, 0.0, 0.6))
        await asyncio.sleep(2)

        print("-- Roll -30도, 60% 추력")
        await drone.offboard.set_attitude(Attitude(-30.0, 0.0, 0.0, 0.6))
        await asyncio.sleep(2)

        print("-- Hover, 60% 추력")
        await drone.offboard.set_attitude(Attitude(0.0, 0.0, 0.0, 0.6))
        await asyncio.sleep(2)

        print("-- Offboard 모드 중지")
        try:
            await drone.offboard.stop()
        except OffboardError as error:
            print(f"Offboard 모드 중지 실패: {error._result.result}")

        print("-- 착륙")
        await drone.action.land()
        
        # 착륙 완료 대기
        await asyncio.sleep(5)
        
        # 시뮬레이션 종료 신호
        self.stop_sim = True

    def run_control_thread(self):
        """별도 스레드에서 asyncio 이벤트 루프를 실행하여 드론 제어"""
        
        # 시뮬레이션 초기화 대기
        import time
        print("시뮬레이션 초기화 중... (5초 대기)")
        time.sleep(5)
        
        # 새로운 이벤트 루프 생성 및 실행
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self.control_drone())
        finally:
            loop.close()

    def run(self):
        """시뮬레이션 메인 루프"""
        
        # 드론 제어를 별도 스레드에서 시작
        control_thread = threading.Thread(target=self.run_control_thread, daemon=True)
        control_thread.start()
        
        # 시뮬레이션 시작
        self.timeline.play()
        
        # 메인 시뮬레이션 루프
        while simulation_app.is_running() and not self.stop_sim:
            self.world.step(render=True)
        
        # 종료
        carb.log_warn("시뮬레이션 종료 중...")
        self.timeline.stop()
        simulation_app.close()


def main():
    pg_app = PegasusApp()
    pg_app.run()


if __name__ == "__main__":
    main()