#!/usr/bin/env python
"""
긴급 수정 버전 - 가벼운 드론 사용
Pegasus에서 사용 가능한 가장 가벼운 드론으로 시도
"""

import carb
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.timeline
import omni
from omni.isaac.core.world import World
import numpy as np
from scipy.spatial.transform import Rotation

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.backends.backend import Backend

from pxr import Sdf, UsdShade, UsdGeom, Gf, UsdLux

print("\n" + "="*60)
print("사용 가능한 드론 목록:")
for robot_name in ROBOTS.keys():
    print(f"  - {robot_name}")
print("="*60 + "\n")

# 가벼운 드론 우선순위
DRONE_PRIORITY = [
    'Crazyflie',      # 가장 가벼움 (33g)
    'Quadcopter',     # 일반 쿼드콥터
    'Hummingbird',    # 작은 드론
    'Firefly',        # 작은 드론
    'Iris',           # 마지막 선택
]

SELECTED_DRONE = None
for drone_name in DRONE_PRIORITY:
    if drone_name in ROBOTS:
        SELECTED_DRONE = drone_name
        print(f"[선택] 드론: {SELECTED_DRONE}")
        break

if SELECTED_DRONE is None:
    SELECTED_DRONE = list(ROBOTS.keys())[0]
    print(f"[선택] 기본 드론: {SELECTED_DRONE}")


class SimplePIDController(Backend):
    """간단한 PID 제어기 - 호버링용 (NonlinearController 참고)"""

    def __init__(self):
        super().__init__(config=None)

        # PID 게인 (위치 제어)
        self.Kp = np.diag([6.0, 6.0, 10.0])
        self.Kd = np.diag([4.5, 4.5, 6.0])
        self.Ki = np.diag([0.1, 0.1, 0.2])

        # 자세 제어 게인
        self.Kr = np.diag([3.5, 3.5, 3.5])
        self.Kw = np.diag([0.5, 0.5, 0.5])

        # 드론 파라미터
        self.m = 1.5  # 질량 (kg) - Iris 기준
        self.g = 9.81

        # 목표 위치
        self.target_pos = np.array([0.0, 0.0, 1.5])

        # 적분항
        self.int_error = np.zeros(3)

        # 상태
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.R = Rotation.identity()
        self.w = np.zeros(3)

        self.dt = 0.01
        self.time = 0.0
        self.received_first_state = False
        self.input_ref = [0.0, 0.0, 0.0, 0.0]

    def update_state(self, state):
        """상태 업데이트 콜백 (Pegasus가 호출)"""
        self.p = np.array(state.position)
        self.v = np.array(state.linear_velocity)
        # Pegasus의 attitude는 scipy와 같은 [x,y,z,w] 순서
        self.R = Rotation.from_quat(state.attitude)
        self.w = np.array(state.angular_velocity)
        self.received_first_state = True

    def update(self, dt: float):
        """제어 업데이트 (매 물리 스텝마다 호출)"""
        self.dt = dt
        self.time += dt

        if not self.received_first_state:
            return

        # 목표 (호버링: 고정 위치)
        p_ref = self.target_pos
        v_ref = np.zeros(3)
        a_ref = np.zeros(3)
        yaw_ref = 0.0

        # 위치/속도 오차
        ep = self.p - p_ref
        ev = self.v - v_ref
        self.int_error = self.int_error + ep * dt
        self.int_error = np.clip(self.int_error, -1.0, 1.0)  # 적분 와인드업 방지

        # 원하는 힘 계산 (F = m*a 형태)
        F_des = -(self.Kp @ ep) - (self.Kd @ ev) - (self.Ki @ self.int_error) \
                + np.array([0.0, 0.0, self.m * self.g]) + (self.m * a_ref)

        # 현재 Z축 (바디 프레임)
        Z_B = self.R.as_matrix()[:, 2]

        # 총 추력 (Z축 방향 투영)
        u_1 = F_des @ Z_B

        # 원하는 Z축 방향
        F_norm = np.linalg.norm(F_des)
        if F_norm < 0.1:
            Z_b_des = np.array([0.0, 0.0, 1.0])
        else:
            Z_b_des = F_des / F_norm

        # 원하는 자세 계산
        X_c_des = np.array([np.cos(yaw_ref), np.sin(yaw_ref), 0.0])
        Z_b_cross_X_c = np.cross(Z_b_des, X_c_des)
        Z_b_cross_norm = np.linalg.norm(Z_b_cross_X_c)
        if Z_b_cross_norm < 0.01:
            Y_b_des = np.array([0.0, 1.0, 0.0])
        else:
            Y_b_des = Z_b_cross_X_c / Z_b_cross_norm
        X_b_des = np.cross(Y_b_des, Z_b_des)

        R_des = np.c_[X_b_des, Y_b_des, Z_b_des]
        R_mat = self.R.as_matrix()

        # 자세 오차 (vee map)
        e_R_mat = (R_des.T @ R_mat) - (R_mat.T @ R_des)
        e_R = 0.5 * np.array([-e_R_mat[1, 2], e_R_mat[0, 2], -e_R_mat[0, 1]])

        # 각속도 오차
        e_w = self.w  # 목표 각속도 = 0

        # 토크 계산
        tau = -(self.Kr @ e_R) - (self.Kw @ e_w)

        # 드론의 내장 함수로 로터 속도 변환
        if self.vehicle:
            self.input_ref = self.vehicle.force_and_torques_to_velocities(u_1, tau)

        # 디버깅
        if int(self.time * 10) % 50 == 0:
            print(f"[{self.time:.1f}s] Pos: [{self.p[0]:.2f}, {self.p[1]:.2f}, {self.p[2]:.2f}], "
                  f"Target: {self.target_pos[2]:.1f}m, Force: {u_1:.1f}N, "
                  f"Rotors: [{self.input_ref[0]:.0f}, {self.input_ref[1]:.0f}, {self.input_ref[2]:.0f}, {self.input_ref[3]:.0f}]")

    def input_reference(self):
        """로터 속도(rad/s) 반환"""
        return self.input_ref

    def update_sensor(self, sensor_type: str, data):
        pass

    def start(self):
        print(f"[Controller] 시작 - 목표 위치: {self.target_pos}")

    def stop(self):
        print(f"[Controller] 종료 - 최종 위치: [{self.p[0]:.2f}, {self.p[1]:.2f}, {self.p[2]:.2f}]")
        if abs(self.p[2] - self.target_pos[2]) < 0.3:
            print("  성공!")
        else:
            print("  실패")

    def reset(self):
        self.time = 0.0
        self.int_error = np.zeros(3)
        self.received_first_state = False

    def update_graphical_sensor(self, sensor_type: str, data):
        pass


class QuickTestApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])
        
        # 간단한 PID 제어기
        self.controller = SimplePIDController()
        
        config = MultirotorConfig()
        config.backends = [self.controller]
        
        # 선택된 드론으로 생성
        print(f"\n[드론 생성] {SELECTED_DRONE} 사용")
        # 쿼터니언: Pegasus는 [qx, qy, qz, qw] 순서 사용
        init_quat = [0.0, 0.0, 0.0, 1.0]  # 항등 쿼터니언 (정방향)
        self.drone = Multirotor(
            "/World/Drone",
            ROBOTS[SELECTED_DRONE],
            0,
            [0.0, 0.0, 1.5],  # 1.5m 높이에서 시작
            init_quat,
            config=config
        )

        # 조명
        stage = omni.usd.get_context().get_stage()
        distant_light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
        distant_light.CreateIntensityAttr(3000.0)
        
        self.world.reset()
        self.step_count = 0
        
        print("\n" + "="*60)
        print("긴급 호버링 테스트 시작!")
        print("드론이 1.5m 높이에서 10초간 호버링을 시도합니다.")
        print("="*60 + "\n")
        
    def run(self):
        self.timeline.play()
        
        # 1초 대기
        for _ in range(100):
            self.world.step(render=True)
            self.step_count += 1
        
        # 10초 테스트
        max_steps = 1000
        while simulation_app.is_running() and self.step_count < max_steps:
            self.world.step(render=True)
            self.step_count += 1
        
        # 결과
        drone_state = self.drone.state
        final_pos = np.array([drone_state.position[0], drone_state.position[1], drone_state.position[2]])
        
        print("\n" + "="*60)
        print("테스트 완료!")
        print(f"드론: {SELECTED_DRONE}")
        print(f"목표 고도: 1.5m")
        print(f"최종 고도: {final_pos[2]:.2f}m")
        print(f"오차: {abs(final_pos[2] - 1.5):.2f}m")
        
        if abs(final_pos[2] - 1.5) < 0.5:
            print("\n✓✓✓ 성공! 드론이 안정적입니다! ✓✓✓")
            print("이제 RL 모델을 이 드론에 적용할 수 있습니다.")
        else:
            print(f"\n✗✗✗ 실패: 드론이 {final_pos[2]:.2f}m에 있습니다 ✗✗✗")
            if final_pos[2] < 0.3:
                print("추락했습니다. 다른 드론을 시도하거나 Pegasus 설정을 확인하세요.")
            else:
                print("너무 높이 날아갔습니다. 제어 게인을 조정하세요.")
        print("="*60 + "\n")
        
        self.timeline.stop()
        simulation_app.close()


def main():
    app = QuickTestApp()
    app.run()


if __name__ == "__main__":
    main()