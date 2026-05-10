# Hierarchical Scene Graph + LLM Robot Interaction

이 프로젝트는 다음 두 폴더를 중심으로 구성됩니다.

- `hierarchy_robot_interaction_scene_graph_submission/hierarchy-scene-graphs`: RGB-D/pose 데이터와 ConceptGraph 결과를 결합해 계층형(건물-층-방-객체) scene graph 생성
- `hierarchy_robot_interaction_scene_graph_submission/my_long_term_mem_project`: Habitat 기반 Fetch 로봇 실행, 메모리(장/단기), LLM planner, rearrangement 중 동적 scene graph 업데이트

## 1. 프로젝트 개요

핵심 목표는 다음과 같습니다.

1. 오프라인/초기 단계에서 계층형 scene graph를 생성한다.
2. 런타임에서 로봇이 task를 수행하는 동안 객체 변화를 scene graph에 반영한다.
3. LLM으로 task decomposition 함수를 생성하고 실제 로봇 실행 루프와 연결한다.

## 2. 배경 및 연구 동기 (Background & Motivation)

이 프로젝트는 로봇의 Navigation 및 Rearrangement task 성공률을 높이기 위한 메모리 구조 연구에서 출발하여, 실제 환경 적용을 위한 다양한 기술적 한계를 극복하는 과정으로 발전했습니다.

* **KARMA 모델과 오라클 데이터의 한계 극복:** 초기 아이디어는 단기/장기 메모리를 활용하는 KARMA 논문에서 영감을 받았습니다. 장기 메모리를 3D Scene Graph로, 단기 메모리를 하위 작업(Subtask) 수행을 위한 작업 기억(Working Memory)으로 활용했습니다. 하지만 시뮬레이터의 오라클 데이터(메타데이터)에 의존하는 기존 방식은 실제 환경으로 확장하기 어렵다는 명확한 한계가 있었습니다. 이를 해결하기 위해 로봇의 순수 관찰(Observation) 데이터만으로 Scene Graph를 구축하는 방향으로 전환했습니다.
* **관찰 기반 Scene Graph의 구조적 한계 (ConceptGraph):** 시각 정보 기반의 3D Scene Graph 구축을 위해 초기에는 `ConceptGraph` 파이프라인을 도입했습니다. 그러나 해당 방법론은 객체들을 단순 나열하는 평면적(Flat) json 형태를 출력한다는 치명적인 단점이 있었습니다. 예를 들어, "1층 주방의 사과를 집어서 1층 거실 커피 테이블에 올려놔"와 같은 명령을 수행하려면 객체가 어느 방에 속해 있는지 알아야 하지만, 동일한 물체가 여러 방에 있을 경우 로봇이 이를 구분하지 못했습니다.
* **계층형 구조(Hierarchical Structure)로의 발전:** 위 문제를 해결하기 위해 `Hierarchical Open-Vocabulary 3D Scene Graphs (HOV-SG)` 논문의 Floor/Room 분할 로직을 참고했습니다. 이를 바탕으로 ConceptGraph로 추출한 객체들을 특정 Room에 할당하는 계층형 파이프라인(건물-층-방-객체)을 구축했습니다. 기존 연구의 코드가 특정 데이터셋에 강하게 종속되어 있는 문제를 해결하고자, 본 프로젝트의 데이터셋 및 환경에 맞게 구조를 전면 수정하여 호환성을 확보했습니다.
* **정적 환경의 한계와 동적(Dynamic) 업데이트 고안:** 기존 3D Scene Graph 연구들은 환경의 변화를 실시간으로 반영하지 못하는 정적인 맵이라는 공통된 한계가 있었습니다. 이를 해결하고자 `Dynamic Open-Vocabulary 3D Scene Graphs` 연구를 참고하여 동적 업데이트 로직을 추가했습니다. 본 프로젝트에서는 "현재 환경에서 변화를 일으키는 유일한 주체는 Rearrangement task를 수행하는 Fetch 로봇뿐이다"라는 핵심 가정을 설정했습니다. 이에 따라 로봇이 타겟 지점에 도착했을 때, 객체를 집어 들었을 때(pickup), 내려놓았을 때(put) 발생하는 이벤트를 기준으로 Scene Graph가 즉각 갱신되도록 파이프라인을 완성했습니다.

## 3. 사전 준비물 (중요)

`create_hybrid_scene_graph.py`를 실행하려면 **ConceptGraph 결과물**이 먼저 필요합니다.

필수 입력:

1. posed RGB-D + pose(`traj.txt`) 데이터셋
2. ConceptGraph에서 미리 생성한 mapping 결과 파일 (`pcd_*.pkl.gz`)
3. `create_hybrid_scene_graph.yaml`의 경로 설정

`hierarchy-scene-graphs/config/create_hybrid_scene_graph.yaml`에서 최소 아래 항목이 유효해야 합니다.

```yaml
main:
  dataset_path: /.../hssd/<scene_id>
  scene_id: "108736872_177263607"

conceptgraph:
  mapping_path: /.../hssd/<scene_id>/exps/r_mapping_stride5/pcd_r_mapping_stride5.pkl.gz
```

즉, 이 단계는 "RGB-D만으로 단독 생성"이 아니라, **ConceptGraph 객체 정보를 함께 사용해 hybrid hierarchy scene graph를 구성**합니다.

## 4. 폴더 구조

```text
.
└── hierarchy_robot_interaction_scene_graph_submission/
    ├── hierarchy-scene-graphs/
    │   ├── config/
    │   ├── scripts/
    │   └── utils/
    └── my_long_term_mem_project/
        ├── scripts/
        ├── prompts/
        ├── resources/
        ├── memory/
        ├── logs/
        ├── experience/
        └── history_tasks/
```

## 5. 파이프라인 A: 계층형 Scene Graph 생성

진입점:

- `hierarchy_robot_interaction_scene_graph_submission/hierarchy-scene-graphs/scripts/create_hybrid_scene_graph.py`

주요 처리:

1. RGB-D 프레임 누적으로 full point cloud 생성
2. floor/room segmentation
3. ConceptGraph 객체를 room에 할당
4. room naming + room adjacency 구성
5. building-floor-room-object graph 저장

실행:

```bash
python hierarchy_robot_interaction_scene_graph_submission/hierarchy-scene-graphs/scripts/create_hybrid_scene_graph.py
```

옵션:

- `--hov`: HOV 데이터셋 경로 사용
- `--step-through-full-pcd`: 프레임 단위 시각화 스텝 실행
- `--reuse-existing-layout`: 기존 floor/room 레이아웃 재사용

## 6. 파이프라인 B: Rearrangement + 동적 Scene Graph 업데이트

진입점:

- GUI: `hierarchy_robot_interaction_scene_graph_submission/my_long_term_mem_project/scripts/GUI_karma.py`
- 실행 코어: `hierarchy_robot_interaction_scene_graph_submission/my_long_term_mem_project/scripts/execute_LLM_plan.py`
- 동적 업데이트 코어: `hierarchy_robot_interaction_scene_graph_submission/my_long_term_mem_project/scripts/dynamic_scene_graph_updater.py`

핵심 아이디어:

- 로봇이 rearrangement task를 수행할 때(`pickup`, `put`, `goto_align`),
- 이벤트 단위로 `apply_dynamic_scene_graph_update(...)`를 호출해
- scene graph overlay를 현재 상태로 갱신합니다.

생성 산출물 예시:

- 이벤트별 업데이트 로그:
  - `.../memory/dynamic_scene_updates/frame_000xx_<event>/scene_graph_update.json`
- 현재 누적 상태:
  - `.../memory/dynamic_scene_graph/<scene_id>/current_state.json`
  - `.../memory/dynamic_scene_graph/<scene_id>/current_scene_manifest.json`
  - `.../memory/dynamic_scene_graph/<scene_id>/current_scene_full.ply`

이 섹션이 바로 "task 수행 중 scene 업데이트" 로직을 설명하는 부분입니다.

## 7. 실행 방법 (Quick Start)

### 7.1 필수 환경

- Python 3.10 권장
- Habitat/Habitat-Sim 환경
- Open3D, NumPy, SciPy, OpenCV, networkx, OmegaConf 등
- OpenAI API 키

```bash
export OPENAI_API_KEY=<YOUR_KEY>
```

### 7.2 (선택) 수동 RGB-D 수집

```bash
python hierarchy_robot_interaction_scene_graph_submission/my_long_term_mem_project/scripts/manual_control_posed_rgbd.py --hov
```

### 7.3 Scene Graph 생성

```bash
python hierarchy_robot_interaction_scene_graph_submission/hierarchy-scene-graphs/scripts/create_hybrid_scene_graph.py
```

### 7.4 GUI 기반 Task 실행

```bash
python hierarchy_robot_interaction_scene_graph_submission/my_long_term_mem_project/scripts/GUI_karma.py
```

## 8. 데모 미디어 (이미지/영상)

### 8.1 방 분리 결과 이미지
| 1 | 2 | 3 | 4 |
|:---:|:---:|:---:|:---:|
| <img src="https://github.com/user-attachments/assets/7df81276-5b42-450e-80fd-db56d06c4672" width="100%"> | <img src="https://github.com/user-attachments/assets/fd9f7897-26d1-40f4-9ea7-f96aff4e29f4" width="100%"> | <img src="https://github.com/user-attachments/assets/4c4a1582-b464-4bd6-9245-c3db899da223" width="100%"> | <img src="https://github.com/user-attachments/assets/aee2a039-f5be-4c46-8c86-43087035b166" width="100%"> |
| <img src="https://github.com/user-attachments/assets/a05f5b52-b137-4c49-9282-a24b14fc4c00" width="100%"> | <img src="https://github.com/user-attachments/assets/7d870580-b98e-4b76-a55c-707be50fa546" width="100%"> | <img src="https://github.com/user-attachments/assets/0a615ba2-0ebe-43a9-80dc-ffe3f911b23b" width="100%"> | <img src="https://github.com/user-attachments/assets/72bc62a4-aeca-4b1d-badf-0bf826dbaf57" width="100%"> |

### 8.2 Task 수행 동영상
https://youtu.be/n9BuyLzmbgs


## 9. 메모리/로그 산출물

- short-term memory:
  - `.../my_long_term_mem_project/memory/memory3.json`
  - `.../my_long_term_mem_project/prompts/short_term_memory.txt`
- long-term memory:
  - `.../my_long_term_mem_project/memory/longterm_memory.json`
- planner 관련:
  - `.../my_long_term_mem_project/logs/messages.json`
  - `.../my_long_term_mem_project/logs/generated_function_name.json`
- dynamic scene graph:
  - `.../my_long_term_mem_project/memory/dynamic_scene_updates/`
  - `.../my_long_term_mem_project/memory/dynamic_scene_graph/`

## 10. 주의사항

1. 코드에 절대경로(`/home/yuchaehee/...`)가 다수 포함되어 있습니다.
2. 다른 환경에서는 경로 상수와 config를 먼저 수정해야 합니다.
3. OpenAI API 키가 없으면 planner/이미지 분석 단계가 실패합니다.
4. Habitat 데이터셋, navmesh, sensor 설정이 없으면 시뮬레이터가 초기화되지 않습니다.

## 11. 참고 스크립트

ConceptGraph 객체 기준 텔레포트 디버그:

```bash
python hierarchy_robot_interaction_scene_graph_submission/my_long_term_mem_project/scripts/teleport_to_conceptgraph_object.py --list
```
