SceneUpdate - Mobile Manipulator with REACT Scene Graph

Isaac Sim 기반 로봇 시뮬레이션 시스템.
REACT 실시간 씬그래프 + LLM 태스크 플래닝 + VFH/ROS2 네비게이션.

## 지원 기능

**1. 플래너 (Planner)** (`--planner` 옵션)
- **SayPlan** (`sayplan`, 기본값): LLM 기반 기본 태스크 플래닝
- **PRED** (`pred`): Pre-emptive action Revision by Environmental Feedback. DTA/APM/ASR 메커니즘 + 미발견 물체 탐색 + 자동 replanning
- **Custom** (`custom`): 사용자 정의 확장 플래너

**2. LLM 백엔드** (`--llm` 옵션)
- **Gemini** (`gemini`, 기본값): Vertex AI 기반 Gemini 2.5 Flash
- **Qwen** (`qwen`): Ollama를 이용한 Qwen2.5 7B 로컬 구동
- **Llama** (`llama`): Ollama를 이용한 Llama3.1 8B 로컬 구동

**3. 네비게이션 (Navigation)** (`--nav` 옵션)
- **DWA / VFH** (`dwa`, 기본값): 자체 구현 VFH 기반 장애물 회피 (LocalNavigator)
- **ROS2** (`ros2`): ROS2 Nav2 스택 연동 네비게이션

**4. 모바일 매니퓰레이터 실행 (실행 파일)**
- **main.py** - Jackal + Franka (모바일 매니퓰레이터, 휠 구동)

## 실행

```bash
cd /home/sungbin/isaac/sg/sceneupdate

# [권장] 최신 빈공간(Freespace V2) 스코어링 및 Custom 플래너 + ROS2 네비게이션 실행
python3 main.py --planner custom --nav ros2 --llm gemini

# 기타 옵션 조합 예시: PRED 플래너 + Llama 로컬 모델 + 자체 DWA 네비게이션
python3 main.py --planner pred --llm llama --nav dwa
```

`react_venv` 가상환경이 `../react_venv/`에 있어야 REACT Worker가 동작함.
로컬 LLM(`qwen`, `llama`) 사용 시 Ollama 서버(`ollama serve`)가 실행 중이어야 함.

## 폴더 구조

```text
sceneupdate/
├── main.py                   # 엔트리포인트: Jackal + Franka
├── config.py                 # 모든 설정값 + 경로 + 모드별 프리셋
├── robot_utils.py            # 로봇 제어 유틸 (USD transform, LiDAR, 휠)
├── llm_wrapper.py            # LLM 통합 인터페이스 (Gemini, Qwen, Llama)
├── arm_controller.py         # FrankaArmController: pick/place 제어
├── navigator.py              # 자체 구현 VFH + 휠 차동구동 네비게이션
├── navigator_ros2.py         # ROS2 Nav2 스택 연동 네비게이션
├── nav2_bridge.py            # ROS2 통신용 브릿지 (액션 클라이언트)
├── react_bridge.py           # REACT 워커 프로세스 IPC 통신 브릿지
├── react_worker.py           # REACT 워커 (YOLO-World + EfficientNet 실시간 처리)
├── sayplan_brain.py          # 기본 SayPlan 베이스라인 플래너
├── pred_brain.py             # PRED 기반 탐색/추론 특화 플래너
├── custom_brain.py           # 사용자 정의 확장이 가능한 커스텀 플래너
├── sayplan_ui.py             # Omni UI + 상태머신 (전체 파이프라인 관리)
├── precompute_freespace.py   # [Freespace] ConceptGraph 데이터로 빈공간 오프라인 사전 계산
├── depth_freespace.py        # [Freespace] Live Depth 기반 비동기 실시간 하이브리드 빈공간 추적
├── surface_analyzer_v2.py    # [Freespace] V2 그리드 기반 3-Factor(Edge, Object, Efficiency) 안전 배치 스코어링 알고리즘
├── generate_surface_json_v2.py # 표면 및 빈공간 정보를 surface_json으로 생성하는 스크립트
├── docs/                     # 프레임워크 핵심 알고리즘 설명서 (Freespace 등)
├── models/                   # 학습된 모델 가중치 (YOLO-World, EfficientNet)
├── scene/                    # USD 씬 파일 보관 (mm, frnkaarm 등)
└── data/                     # 초기 씬그래프 및 프리컴퓨트 JSON 데이터
```

## 주요 기능 및 아키텍처

### 모듈 및 IPC 구조
| 역할 | 방식 | 상세 |
|-----|-----|-----|
| **UI ↔ 플래너** | 상태머신 | `IDLE → THINKING → EXECUTING → NAVIGATING → VERIFYING` |
| **플래너 ↔ LLM** | `llm_wrapper` | Gemini API 호출 또는 로컬 Ollama REST API 호출 |
| **메인 ↔ REACT** | IPC 파일 교환 | `/tmp/react_*` 파일을 통한 이미지/포즈 전송 및 씬그래프 갱신 수신 |
| **REACT 파이프라인**| YOLO+임베딩 | 프레임 수신 → YOLO-World 객체 탐지 → EfficientNet-B2 임베딩 매칭 → NEW/MOVED/ABSENT 판정 |

### 주요 기능 및 특징 (현재 사용 중인 방식)
권장 실행 명령어(`python3 main.py --planner custom --nav ros2 --llm gemini`)를 통해 다음의 핵심 기능들이 동작합니다.

- **V2 빈공간 스코어링 (`surface_analyzer_v2.py`)**: 
  - V2 그리드 기반 3-Factor(Edge Safety, Object Safety, Efficiency) 스코어링 알고리즘.
  - 단순 빈공간의 크기를 구하는 것이 아니라, 가장자리 낙하 위험과 타 객체와의 충돌을 피하면서 가장 배치하기 좋은 최적의 Sweet Spot을 선정합니다.
- **실시간 빈공간 비동기 관리 (`depth_freespace.py`)**: 
  - 비동기 Multi-frame Sliding Window를 통해 프레임 저하 없이 실시간으로 Occupancy Grid 갱신. 가림 현상은 Scene Graph와 Persistent Cache 기반 Fallback으로 하이브리드 처리.
- **Precomputed Free Space 연동 (`precompute_freespace.py`)**: [자세한 알고리즘 설명 보기](docs/freespace_algorithm.md)
  - ConceptGraph 오프라인 데이터를 사용해 초기 빈공간을 정밀하게 사전 계산(2-Pass 표면 Z값 자동 감지).
  - 실시간 배치 반영: 로봇이 물체를 놓는 즉시 Persistent Cache에 해당 영역을 Occupied로 강제 마킹하여 다음 프레임 연산을 기다리지 않고 즉각적으로 씬그래프와 LLM에 반영.
- **ROS2 Nav2 연동 네비게이션 (`navigator_ros2.py`)**: `--nav ros2` 옵션으로 안정적인 글로벌/로컬 경로 계획 수행.
- **Custom 플래너 (`custom_brain.py`)**: `--planner custom`을 통한 최적화된 태스크 플래닝 수행.

### 💡 시행착오 및 폐기된 방법 (과거의 빈공간 탐색 알고리즘)
과거에 사용했으나 현재는 폐기된 방식들입니다. 현재 V2 알고리즘이 탄생하게 된 배경을 이해하는 데 도움이 됩니다.

- **MER (Maximal Empty Rectangle) 알고리즘 (`surface_analyzer.py`)**
  - **방법**: 표면 위에서 물체가 없는 가장 큰 직사각형(빈공간)을 수학적으로 찾아내어 중앙에 물체를 배치하려고 했던 방식.
  - **폐기 이유**: 직사각형의 크기가 크더라도 직사각형의 중심이 책상 모서리 끝단에 걸치거나, 얇고 긴 형태로 인해 실제 물체를 놓기에는 매우 불안정한 위치가 도출되는 치명적 문제가 발생했습니다. "가장 큰 공간"이 항상 "가장 안전한 공간"은 아니라는 점을 깨닫고, 3가지 안전 가중치를 둔 현재의 **V2 그리드 스코어링**으로 전면 교체되었습니다.

## 의존성

**Isaac Sim 환경 (main.py):**
- Isaac Sim 2024.2+
- `google-generativeai` (Gemini API)
- `urllib` (Ollama 통신용 기본 내장 모듈)

**REACT 환경 (react_venv):**
- `PyTorch` (nightly)
- `ultralytics` (YOLOv8-World)
- `torchvision` (EfficientNet-B2)

## 환경 변수
- `GOOGLE_API_KEY`: Gemini API 키 (없을 경우 config.py 설정 참조)
