# SceneUpdate - Free Space-Aware Scene Graph for Mobile Manipulators

Isaac Sim 기반 모바일 매니퓰레이터 시뮬레이션 시스템.  
씬그래프(Scene Graph)에 빈공간(Free Space) 정보를 통합하여 LLM 기반 배치 태스크 플래닝의 정확도를 높이는 것을 목표로 합니다.

---

## 연구 배경 및 동기

### 기존 연구의 흐름

ConceptGraph, SayPlan 등의 연구를 통해 씬그래프(Scene Graph)가 공간 정보를 LLM/VLM에게 효과적으로 전달하고, 이를 통해 로봇의 태스크 플래닝을 가능하게 한다는 것이 입증되어 왔다. 초기 연구들은 정적인 환경을 가정했지만, REACT 와 같은 연구를 통해 씬그래프의 실시간 갱신 및 동적 환경에서의 적용 가능성도 보여주었다.

### 기존 연구의 한계

그러나 이러한 발전에도 불구하고 씬그래프는 **"어디에 물건을 놓을 수 있는가"** 라는 핵심적인 공간 정보, 즉 **빈공간(Free Space) 정보가 부족하다**는 한계를 보인다.

모바일 매니퓰레이터(Mobile Manipulator)의 주된 태스크는 물건을 집어 특정 위치에 배치하는 pick-and-place 작업이다. 이 태스크를 원활하게 수행하려면 단순히 어떤 물체가 어디에 있는지를 아는 것을 넘어, **어느 표면의 어떤 위치에 물체를 안전하게 놓을 수 있는지**에 대한 정보가 필요하다. 기존 씬그래프는 객체 중심(object-centric)으로 구성되어 있어 이러한 빈공간에 대한 표현이 없으며, LLM이 배치 위치를 추론하더라도 실제 안전한 위치인지 검증할 방법이 없다.

본 연구는 이 공백에 주목하여, **씬그래프에 빈공간 정보를 통합**하고 이를 LLM 태스크 플래닝에 활용하는 방법을 탐구한다.

### 제안 접근법

기존 씬그래프의 한계를 해결하기 위해 두 가지 방식으로 빈공간 정보를 씬그래프에 통합한다.

1. **오프라인 사전 계산 (Precomputed Free Space)**  
   ConceptGraph 데이터를 기반으로, 로봇이 작업하기 전에 환경의 각 표면에 대한 빈공간을 미리 계산하여 씬그래프에 포함시킨다. 2-Pass 방식으로 표면 Z값을 자동 감지하고, 3-Factor(가장자리 안전성, 물체 충돌 안전성, 배치 효율성) 스코어링을 통해 최적의 배치 위치를 선정한다.

2. **실시간 빈공간 관리 (Real-time Free Space Update)**  
   로봇이 작업하는 동안 depth 카메라를 통해 비동기적으로 Occupancy Grid를 실시간 갱신한다. 로봇이 물체를 놓는 즉시 해당 영역을 Occupied로 마킹하여 LLM이 이미 사용된 공간을 다음 배치 위치로 제안하는 오류를 방지한다.

### 한계점

본 연구의 주된 한계는 **실제 환경(Real World)에서의 노이즈 대응**이다. 현재 구현은 시뮬레이션 환경(Isaac Sim)에서의 클린한 depth 데이터를 전제로 하기 때문에, 실제 depth 카메라에서 발생하는 노이즈(반사, 투명도, 거리 오차 등)가 있는 환경에서도 빈공간 탐지가 정확하게 동작할지는 추가적인 검증이 필요하다.

---

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

---

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

---

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

---

## 주요 기능 및 아키텍처

### 모듈 및 IPC 구조

| 역할 | 방식 | 상세 |
|-----|-----|-----|
| **UI ↔ 플래너** | 상태머신 | `IDLE → THINKING → EXECUTING → NAVIGATING → VERIFYING` |
| **플래너 ↔ LLM** | `llm_wrapper` | Gemini API 호출 또는 로컬 Ollama REST API 호출 |
| **메인 ↔ REACT** | IPC 파일 교환 | `/tmp/react_*` 파일을 통한 이미지/포즈 전송 및 씬그래프 갱신 수신 |
| **REACT 파이프라인**| YOLO+임베딩 | 프레임 수신 → YOLO-World 객체 탐지 → EfficientNet-B2 임베딩 매칭 → NEW/MOVED/ABSENT 판정 |

### 실시간 씬그래프 갱신 (Real-time Scene Graph Update) 로직 및 코드

동적 환경에서 로봇이 태스크를 수행하기 위해서는 사물의 이동, 생성, 소멸을 실시간으로 감지하고 씬그래프에 반영하는 것이 필수적입니다. 본 시스템은 독립된 워커 프로세스를 통해 실시간으로 씬그래프를 갱신합니다.

**1. 주요 코드 파일**
- **`react_bridge.py`**: Isaac Sim 메인 프로세스에서 동작. 로봇의 RGB-D 카메라 프레임과 현재 포즈(Pose) 데이터를 수집하여 `/tmp/react_*.npy` 형태의 IPC 파일로 전달합니다.
- **`react_worker.py`**: 별도의 `react_venv` 가상환경에서 동작. 무거운 PyTorch, YOLO, EfficientNet 연산을 메인 시뮬레이션 루프와 분리하여 병목을 방지합니다.

**2. 갱신 로직 (REACT 파이프라인)**
1. **프레임 수집 및 객체 탐지**: `react_worker.py`의 `process_frame()`이 호출되면, 수신된 RGB 이미지에 대해 **YOLO-World** (Open-vocabulary Object Detection) 모델을 실행하여 이미지 내의 모든 객체 바운딩 박스를 추출합니다.
2. **3D 위치 추정 및 특징 추출**:
   - Depth 이미지를 이용해 바운딩 박스 픽셀들의 실제 3D 월드 좌표(Position)와 크기(Extent)를 계산합니다 (`estimate_3d_position`, `estimate_3d_extent`).
   - 객체 영역(Crop)에 대해 **EfficientNet-B2** 모델을 적용하여 1408차원의 임베딩 벡터(특징)를 추출합니다.
3. **매칭 및 변화 판정 (`_match_and_update`, `_check_absent_objects`)**:
   - 기존 씬그래프의 객체들과 임베딩 코사인 유사도(Cosine Similarity) 및 3D 공간적 거리(L2 Distance)를 비교합니다.
   - **MOVED (이동)**: 기존 객체와 임베딩이 일치하지만 위치가 일정 임계값 이상 변경된 경우.
   - **NEW (신규)**: 기존 씬그래프에 매칭되는 임베딩이 없는 새로운 객체가 발견된 경우.
   - **ABSENT (사라짐)**: 로봇의 시야(FOV) 안에 있어야 할 객체가 탐지되지 않는 경우, 해당 객체를 부재 상태로 마킹합니다.
4. **씬그래프 동기화 (`_apply_sg_updates`, `_sync_sg_to_disk`)**:
   - 변경된 좌표, 신규 노드 추가, 상태 플래그 등을 반영하여 `hierarchical_scene_graph.json` 파일을 실시간으로 덮어씁니다.
   - 플래너(SayPlan, PRED 등)는 갱신된 씬그래프를 읽어들여 항상 최신 환경 정보를 바탕으로 다음 행동을 계획(Replanning)하게 됩니다.

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

### 시행착오 및 폐기된 방법 (과거의 빈공간 탐색 알고리즘)

과거에 사용했으나 현재는 폐기된 방식들입니다. 현재 V2 알고리즘이 탄생하게 된 배경을 이해하는 데 도움이 됩니다.

- **MER (Maximal Empty Rectangle) 알고리즘 (`surface_analyzer.py`)**
  - **방법**: 표면 위에서 물체가 없는 가장 큰 직사각형(빈공간)을 수학적으로 찾아내어 중앙에 물체를 배치하려고 했던 방식.
  - **폐기 이유**: 직사각형의 크기가 크더라도 직사각형의 중심이 책상 모서리 끝단에 걸치거나, 얇고 긴 형태로 인해 실제 물체를 놓기에는 매우 불안정한 위치가 도출되는 치명적 문제가 발생했습니다. "가장 큰 공간"이 항상 "가장 안전한 공간"은 아니라는 점을 깨닫고, 3가지 안전 가중치를 둔 현재의 **V2 그리드 스코어링**으로 전면 교체되었습니다.

---

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
