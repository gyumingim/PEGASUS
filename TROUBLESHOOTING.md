# Pegasus RL 드론 착륙 프로젝트 - 트러블슈팅 가이드

## 프로젝트 개요

IsaacLab에서 학습된 RL 모델을 Pegasus 시뮬레이터로 이전하여 드론 착륙을 수행하는 프로젝트입니다.

### 주요 파일

| 파일 | 설명 |
|------|------|
| `11_ardupilot_multi_vehicle.py` | Pegasus RL 착륙 메인 코드 |
| `drone_landing_env.py` | IsaacLab 학습 환경 (참조용) |
| `10_graphs.py` | PID 호버링 테스트 코드 |
| `debug_aruco_camera.py` | ArUco 카메라 디버깅 도구 |

---

## 핵심 주의사항

### 1. 쿼터니언 순서

**Pegasus와 scipy 모두 `[x, y, z, w]` 순서를 사용합니다!**

```python
# ✅ 올바른 방법
quat_xyzw = np.array(state.attitude)  # Pegasus: [x, y, z, w]
R = Rotation.from_quat(quat_xyzw)     # scipy도 [x, y, z, w]

# ❌ 잘못된 방법 (순서 변환하면 안됨!)
# quat_wxyz = [state.attitude[3], state.attitude[0], ...]  # 불필요
```

**Multirotor 생성 시:**
```python
# 쿼터니언: [qx, qy, qz, qw] 순서
init_quat = [0.0, 0.0, 0.0, 1.0]  # 항등 쿼터니언 (정방향)
drone = Multirotor(..., init_quat, ...)
```

---

### 2. Observation 구성 - IsaacLab과 동일하게!

**IsaacLab (drone_landing_env.py) 방식:**

```python
# ★★★ 핵심: 각속도는 world frame 그대로 사용! ★★★
obs = np.concatenate([
    R.inv().apply(vel),      # 3: 선속도 (body frame)
    ang_vel,                  # 3: 각속도 (world frame!) ← 변환 안함!
    gravity_body,             # 3: 중력 방향 (body frame)
    rel_pos_body,             # 3: 목표 위치 (body frame)
    rel_vel_body,             # 3: 상대 속도 (body frame)
    [yaw]                     # 1: yaw 각도
])
```

**흔한 실수:**
```python
# ❌ 각속도를 body frame으로 변환하면 안됨!
ang_vel_b = R.T @ ang_vel  # 이렇게 하면 드론이 이상하게 움직임

# ✅ 각속도는 world frame 그대로
ang_vel_obs = ang_vel  # 변환 없이 사용
```

---

### 3. 좌표 변환 방식

**IsaacLab 방식 (R.inv().apply 사용):**
```python
R = Rotation.from_quat(quat_xyzw)

# World → Body 변환
lin_vel_b = R.inv().apply(lin_vel)
gravity_b = R.inv().apply(gravity_world)
goal_pos_b = R.inv().apply(goal_rel_world)
```

**행렬 방식 (동등하지만 미묘한 차이 가능):**
```python
R_mat = R.as_matrix()
lin_vel_b = R_mat.T @ lin_vel  # 이론적으로 동일
```

> **권장:** IsaacLab과 동일하게 `R.inv().apply()` 사용

---

### 4. 중력 벡터

**IsaacLab:**
```python
gravity_world = np.array([0, 0, -9.81])  # 실제 중력값
gravity_body = R.inv().apply(gravity_world)
```

**주의:**
```python
# ❌ 정규화된 중력 사용하면 안됨!
gravity_w = np.array([0, 0, -1])  # 스케일이 다름

# ✅ 실제 중력값 사용
gravity_w = np.array([0, 0, -self.gravity])  # -9.81
```

---

### 5. Yaw 계산

**IsaacLab 방식:**
```python
# quat_wxyz: [w, x, y, z] 순서로 변환 필요
quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
yaw = np.arctan2(
    2.0 * (quat_wxyz[0]*quat_wxyz[3] + quat_wxyz[1]*quat_wxyz[2]),
    1.0 - 2.0 * (quat_wxyz[2]**2 + quat_wxyz[3]**2)
)
```

---

### 6. 로터 속도 변환

**Pegasus 내장 함수 사용:**
```python
# force: 추력 (N), torques: [roll, pitch, yaw] 토크 (Nm)
rotor_velocities = self.vehicle.force_and_torques_to_velocities(force, torques)
```

**질량 스케일링:**
```python
# IsaacLab: Crazyflie 0.033kg
# Pegasus: Iris 1.5kg
mass_ratio = IRIS_MASS / TRAIN_MASS  # ~45

# 추력
thrust_ratio = 1.9 * (thrust_action + 1.0) / 2.0  # 0 ~ 1.9
thrust_force = thrust_ratio * IRIS_MASS * gravity  # N

# 토크
torques = moments * 0.002 * mass_ratio  # Nm
```

---

## 흔한 문제와 해결책

### 문제 1: 드론이 한쪽으로 치우침

**원인:** Observation 구성이 IsaacLab과 다름

**확인 방법:**
1. `DEBUG_MODE = True` 설정
2. `Goal rel (world)`와 `Goal rel (body)` 비교
3. 부호가 반대면 좌표 변환 문제

**해결:**
- `R.inv().apply()` 사용 확인
- 각속도가 world frame인지 확인
- 중력 벡터 스케일 확인

---

### 문제 2: 드론이 추락함

**원인 1:** 추력 부족
```python
# 호버링에 필요한 추력
hover_thrust = IRIS_MASS * 9.81  # ~14.7N
```

**원인 2:** gravity 값 오류
```python
# ❌ 잘못된 값
self.gravity = 6  # 오타!

# ✅ 올바른 값
self.gravity = 9.81
```

**원인 3:** THRUST_OFFSET 잘못 설정
```python
# ❌ 오프셋이 너무 크면 과도한 추력
THRUST_OFFSET = 1  # 위험!

# ✅ 기본값
THRUST_OFFSET = 0.0
```

---

### 문제 3: 드론이 뒤집힘

**원인:** 쿼터니언 순서 오류

**해결:**
```python
# Multirotor 생성 시
init_quat = [0.0, 0.0, 0.0, 1.0]  # [qx, qy, qz, qw]

# ❌ 잘못된 순서
init_quat = [1.0, 0.0, 0.0, 0.0]  # qw가 1이면 180° 회전!
```

---

### 문제 4: 로버 위치 동기화 오류

**원인:** 여러 곳에서 rover_pos 업데이트

**해결:**
```python
# App에서만 업데이트하고 Controller에 전달
def _update_rover(self, dt):
    self.rover_pos += self.rover_vel * dt
    self.controller.set_rover_pos(self.rover_pos)  # 동기화!
```

---

### 문제 5: ArUco 마커 인식 안됨

**확인:**
```bash
python debug_aruco_camera.py /tmp/aruco_rl_000050.png 0.6
```

**예상 출력 (마커가 중앙에 있을 때):**
```
tvec = [0.0, 0.0, distance]
```

**조명 부족 시:**
- DomeLight 강도 증가
- 마커에 emissive 재질 적용

---

## 튜닝 파라미터 가이드

```python
class RLDroneLandingController(Backend):
    # --- 디버깅 ---
    DEBUG_MODE = False           # True: 매 스텝 출력

    # --- 추력 ---
    THRUST_SCALE = 1.0           # 추력 배율 (1.0 = 원본)
    THRUST_OFFSET = 0.0          # 추력 오프셋 (0 = 원본)

    # --- 토크 ---
    ROLL_SCALE = 1.0             # Roll 감쇠
    PITCH_SCALE = 1.0            # Pitch 감쇠
    YAW_SCALE = 1.0              # Yaw 감쇠

    # --- Observation 스케일 ---
    XY_GOAL_SCALE = 1.0          # XY 목표 거리 감쇠
    Z_GOAL_SCALE = 1.0           # Z 목표 거리 감쇠
    VEL_SCALE = 1.0              # 속도 스케일

    # --- 물리 ---
    IRIS_MASS = 1.5              # Iris 질량 (kg)
    TRAIN_MASS = 0.033           # 학습 시 질량 (kg)
    TRAIN_THRUST_TO_WEIGHT = 1.9 # 학습 시 thrust-to-weight
    TRAIN_MOMENT_SCALE = 0.002   # 학습 시 moment scale
```

---

## 디버깅 체크리스트

1. [ ] `gravity = 9.81` 확인
2. [ ] `THRUST_OFFSET = 0` 확인
3. [ ] 쿼터니언 순서 `[x, y, z, w]` 확인
4. [ ] 각속도가 world frame인지 확인
5. [ ] `R.inv().apply()` 사용 확인
6. [ ] 중력 벡터 `[0, 0, -9.81]` 확인
7. [ ] 로버 위치 동기화 확인

---

## 참고 자료

- Pegasus 쿼터니언: `state.attitude` = `[x, y, z, w]`
- scipy Rotation: `from_quat([x, y, z, w])`
- IsaacLab 학습 환경: `drone_landing_env.py`
- PID 호버링 테스트: `10_graphs.py`
