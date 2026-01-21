#!/usr/bin/env python
"""학습된 RL 모델을 Pegasus로 테스트"""

import carb
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.timeline
from omni.isaac.core.world import World
from omni.isaac.core.objects import DynamicCuboid
import numpy as np
import torch

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from scipy.spatial.transform import Rotation

# ========== RL 모델 로드 ==========
from stable_baselines3 import PPO
model = PPO.load("path/to/model.zip")  # 여기에 실제 경로 입력!

class RLController:
    """RL 정책 기반 컨트롤러"""
    
    def __init__(self, vehicle, target_pos=[0, 0, 0.5]):
        self.vehicle = vehicle
        self.target = np.array(target_pos, dtype=np.float32)
        self.gravity = 9.81
        self.mass = 0.033  # kg
        
    def update(self, dt: float):
        """매 스텝 호출"""
        # ========== 관측값 수집 (16차원) ==========
        state = self.vehicle.state
        pos = np.array(state.position)
        vel = np.array(state.linear_velocity)
        ang_vel = np.array(state.angular_velocity)
        quat = np.array([state.attitude[3], state.attitude[0], 
                        state.attitude[1], state.attitude[2]])  # w,x,y,z
        
        # 중력 방향 (body frame)
        R = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # x,y,z,w
        gravity_world = np.array([0, 0, -self.gravity])
        gravity_body = R.inv().apply(gravity_world)
        
        # 목표까지 상대 위치 (body frame)
        rel_pos_world = self.target - pos
        rel_pos_body = R.inv().apply(rel_pos_world)
        
        # 상대 속도 (목표는 정지)
        rel_vel_body = R.inv().apply(vel)
        
        # Yaw 각도
        yaw = np.arctan2(2.0 * (quat[0]*quat[3] + quat[1]*quat[2]),
                        1.0 - 2.0 * (quat[2]**2 + quat[3]**2))
        
        # 관측값 조합 (16차원)
        obs = np.concatenate([
            R.inv().apply(vel),      # 3: 선속도 (body)
            ang_vel,                  # 3: 각속도 (body)
            gravity_body,             # 3: 중력 방향
            rel_pos_body,             # 3: 목표 위치
            rel_vel_body,             # 3: 상대 속도
            [yaw]                     # 1: yaw 각도
        ])
        
        # ========== RL 정책 실행 ==========
        action, _ = model.predict(obs, deterministic=True)
        
        # ========== 행동 해석 (4차원) ==========
        # [thrust, roll_moment, pitch_moment, yaw_moment]
        thrust_norm = action[0]  # -1~1
        moments = action[1:4]    # -1~1
        
        # 추력 변환: [-1,1] -> [0, 1.9*weight]
        thrust_ratio = 1.9 * (thrust_norm + 1.0) / 2.0
        thrust_N = thrust_ratio * self.mass * self.gravity
        
        # 토크 변환
        torques = moments * 0.002  # N·m
        
        # ========== 힘을 드론에 적용 ==========
        # Pegasus는 world frame 기준 힘 필요
        thrust_world = R.apply([0, 0, thrust_N])
        
        # 외력 설정 (Pegasus API)
        self.vehicle.apply_force(thrust_world, torques)
        
        # 디버그 출력
        if int(1.0/dt) % 10 == 0:  # 0.1초마다
            print(f"Pos: {pos}, Action: {action}")

class PegasusApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        
        # 환경 로드
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])
        
        # 목표 큐브 생성
        self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/target",
                name="target",
                position=np.array([0.0, 0.0, 0.5]),
                scale=np.array([0.5, 0.5, 0.5]),
                size=0.5,
                color=np.array([0, 255, 0])
            )
        )
        
        # 드론 생성
        config = MultirotorConfig()
        self.drone = Multirotor(
            "/World/quadrotor",
            ROBOTS['Iris'],
            0,
            [2.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config
        )
        
        self.world.reset()
        
        # RL 컨트롤러 생성
        self.controller = RLController(self.drone, target_pos=[0, 0, 0.5])
        
    def run(self):
        self.timeline.play()
        dt = 1.0/60.0  # 60Hz
        
        while simulation_app.is_running():
            # RL 정책 실행
            self.controller.update(dt)
            
            # 물리 시뮬레이션
            self.world.step(render=True)
        
        carb.log_warn("Closing...")
        self.timeline.stop()
        simulation_app.close()

def main():
    pg_app = PegasusApp()
    pg_app.run()

if __name__ == "__main__":
    main()