# PX4 통합 실행 가이드

## 🎯 변경 사항 요약

### 기존 코드에서 추가된 기능
- ✅ PX4 통합 (ON/OFF 가능)
- ✅ ArUco 마커 감지 (유지)
- ✅ 이동 로버 추적 (유지)
- ✅ 카메라 시스템 (유지)

### 핵심 수정 사항
```python
class RLDroneLandingController(Backend):
    # ★ PX4 사용 여부 설정
    USE_PX4 = False  # True로 변경하면 PX4 사용!
    
    # 기존 파라미터 모두 유지
    USE_ARUCO = True
    DEBUG_MODE = True
    # ...
```

---

## 🚀 실행 방법

### 방법 1: PX4 비활성화 (기존 방식)

**코드 수정 없이 실행**:
```bash
python your_script.py /path/to/model.zip
```

`USE_PX4 = False`가 기본값이므로 기존처럼 작동합니다.

---

### 방법 2: PX4 활성화 (새로운 방식)

#### 1단계: 코드 수정
```python
class RLDroneLandingController(Backend):
    # ★ 이 줄만 수정!
    USE_PX4 = True  # False → True로 변경
```

#### 2단계: PX4 SITL 실행
**터미널 1**:
```bash
cd ~/PX4-Autopilot
make px4_sitl none_iris
```

출력 확인:
```
INFO  [mavlink] mode: Normal, data rate: 4000000 B/s on udp port 14540
```

#### 3단계: 시뮬레이션 실행
**터미널 2**:
```bash
python your_script.py /path/to/model.zip
```

---

## 📊 실행 중 출력 비교

### PX4 비활성화 시
```
================================================================================
★ RL Controller 설정 ★
================================================================================
  USE_PX4:         False
  USE_ARUCO:       True
  THRUST_SCALE:    1.0
  ...
================================================================================

직접 제어 모드 (PX4 비활성화)
```

### PX4 활성화 시
```
================================================================================
★ RL Controller 설정 ★
================================================================================
  USE_PX4:         True
  USE_ARUCO:       True
  PX4_CONNECTION:  udp:127.0.0.1:14540
  PX4_MAX_ANGLE:   0.3 rad (17.2°)
  ...
================================================================================

[PX4] Connecting to: udp:127.0.0.1:14540
[PX4] Waiting for heartbeat...
[PX4] ✓ Connected to System 1

[Main] Initializing PX4...
[PX4] Initializing...
[PX4] Arming...
[PX4] Setting OFFBOARD mode...
[PX4] ✓ Ready
```

---

## ⚙️ 튜닝 파라미터

### PX4 관련 설정
```python
class RLDroneLandingController(Backend):
    # PX4 연결 주소
    PX4_CONNECTION = 'udp:127.0.0.1:14540'
    
    # Attitude 명령 최대 각도 (rad)
    PX4_MAX_ANGLE = 0.3  # 약 17도
    # 값이 크면: 기동성↑, 안정성↓
    # 값이 작으면: 기동성↓, 안정성↑
```

### 기존 파라미터 (PX4 여부와 무관)
```python
    USE_ARUCO = True         # ArUco 마커 사용 여부
    DEBUG_MODE = True        # 디버그 출력
    THRUST_SCALE = 1.0       # 추력 스케일
    ROLL_SCALE = 1.0         # Roll 감쇠
    PITCH_SCALE = 1.0        # Pitch 감쇠
    YAW_SCALE = 1.0          # Yaw 감쇠
```

---

## 🔍 동작 원리

### 직접 제어 모드 (USE_PX4 = False)
```
RL 모델 → Action (thrust, moments)
    ↓
직접 로터 속도 계산
    ↓
Pegasus 물리 엔진
```

### PX4 모드 (USE_PX4 = True)
```
RL 모델 → Action (thrust, moments)
    ↓
PX4 Attitude Command 변환
    ↓
PX4 제어 알고리즘
    ↓
Pegasus 물리 엔진 (HIL 모드)
```

---

## ❓ 문제 해결

### 1. PX4 연결 실패
```
[PX4] Connection failed: ...
[PX4] Falling back to direct control
```

**원인**: PX4 SITL이 실행되지 않음

**해결**:
1. PX4 SITL 실행 확인
2. 포트 확인: `netstat -an | grep 14540`
3. 방화벽 설정: `sudo ufw allow 14540/udp`

---

### 2. 드론이 떨어짐 (PX4 모드)
```
Drone altitude decreasing continuously
```

**원인**: Thrust 매핑 오류

**해결**: `PX4_MAX_ANGLE` 조정
```python
PX4_MAX_ANGLE = 0.2  # 0.3에서 줄임 (더 보수적)
```

또는 thrust 오프셋 추가:
```python
# _send_px4_attitude_command() 메서드에서
thrust_cmd = (thrust_norm + 1.0) / 2.0 + 0.1  # +10% 추가
thrust_cmd = np.clip(thrust_cmd, 0.0, 1.0)
```

---

### 3. ArUco 감지 안 됨
```
✗ No tag | Detections: 0
```

**원인**: 조명 부족 또는 카메라 설정 문제

**해결**:
1. 조명 강도 확인 (코드에서 이미 설정됨)
2. 카메라 초기화 대기 시간 증가:
```python
# run() 메서드에서
for _ in range(500):  # 300 → 500으로 증가
    self.world.step(render=True)
```

---

### 4. PX4와 직접 제어 성능 비교

| 항목 | 직접 제어 | PX4 |
|------|----------|-----|
| 안정성 | ★★★☆☆ | ★★★★★ |
| 정확도 | ★★★★☆ | ★★★★★ |
| 설정 복잡도 | 쉬움 | 보통 |
| 실제 배포 | 어려움 | 쉬움 |
| 디버깅 | 쉬움 | 보통 |

---

## 🎛️ 추천 설정

### 테스트 단계
```python
USE_PX4 = False         # 직접 제어로 빠른 iteration
USE_ARUCO = False       # Ground truth로 정확성 확인
DEBUG_MODE = True       # 상세 출력
```

### 검증 단계
```python
USE_PX4 = True          # PX4로 실제 제어 테스트
USE_ARUCO = True        # ArUco 감지 활성화
DEBUG_MODE = True       # 디버깅 유지
PX4_MAX_ANGLE = 0.2     # 보수적 설정
```

### 배포 단계
```python
USE_PX4 = True
USE_ARUCO = True
DEBUG_MODE = False      # 출력 최소화
PX4_MAX_ANGLE = 0.3     # 최적화된 값
```

---

## 📈 성능 최적화 팁

### 1. PX4 제어 파라미터 튜닝
```python
# 안정성 우선 (착륙 정확도↑)
PX4_MAX_ANGLE = 0.2

# 기동성 우선 (착륙 속도↑)
PX4_MAX_ANGLE = 0.4
```

### 2. ArUco 감지 성능 향상
```python
# _detect_aruco() 메서드에서
if self.step_count % 2 != 0:  # 매 프레임 감지
    return
# → 매 프레임으로 변경
```

### 3. RL Action 스케일 조정
```python
ROLL_SCALE = 0.8   # Roll 감소
PITCH_SCALE = 0.8  # Pitch 감소
# → 더 부드러운 움직임
```

---

## 🔄 빠른 전환 방법

### PX4 ON/OFF 빠르게 전환
```python
# 코드 맨 위에 전역 변수 추가
import sys

USE_PX4_GLOBAL = '--px4' in sys.argv

class RLDroneLandingController(Backend):
    USE_PX4 = USE_PX4_GLOBAL
```

**실행**:
```bash
# PX4 없이
python your_script.py /path/to/model.zip

# PX4 사용
python your_script.py /path/to/model.zip --px4
```

---

## 📝 체크리스트

### PX4 모드 실행 전 확인사항
- [ ] PX4-Autopilot 설치됨
- [ ] pymavlink 설치됨 (`pip install pymavlink`)
- [ ] PX4 SITL 실행 중 (포트 14540 대기)
- [ ] `USE_PX4 = True`로 설정
- [ ] 방화벽 설정 확인

### 시뮬레이션 실행 전 확인사항
- [ ] stable-baselines3 설치됨
- [ ] OpenCV 설치됨 (ArUco 사용 시)
- [ ] 모델 경로 올바름
- [ ] Isaac Sim 정상 작동

---

## 🎯 예상 결과

### 성공 시
```
[10.0s | PX4] ✓ Tracking | XY err: 0.15m | Height: 0.82m | Detections: 156
[PX4] roll=-1.2°, pitch=2.3°, thrust=0.52

[Summary] 시뮬레이션 종료
  총 감지 횟수: 523
  총 프레임: 1000
```

### PX4 장점 확인 방법
1. **안정성**: 드론이 흔들림 없이 부드럽게 이동
2. **정확도**: 목표 위치에 정확히 착륙
3. **실제 배포**: 같은 코드를 실제 드론에 적용 가능

---

## 💡 핵심 요약

### 한 줄로 PX4 활성화
```python
USE_PX4 = True  # 이것만 바꾸면 됨!
```

### 실행 순서
1. PX4 SITL 실행 (터미널 1)
2. `USE_PX4 = True` 설정
3. 시뮬레이션 실행 (터미널 2)

### 기존 기능 유지
- ArUco 마커 감지 ✓
- 이동 로버 추적 ✓
- 카메라 시스템 ✓
- 디버그 출력 ✓

모든 기능이 그대로 작동하면서, PX4의 안정성만 추가됩니다! 🚀