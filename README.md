# Hierarchical Scene Graph + LLM Robot Interaction

이 프로젝트는 다음 두 폴더를 중심으로 구성됩니다.

- `hierarchy_robot_interaction_scene_graph_submission/hierarchy-scene-graphs`: RGB-D/pose 데이터와 ConceptGraph 결과를 결합해 계층형(건물-층-방-객체) scene graph 생성
- `hierarchy_robot_interaction_scene_graph_submission/my_long_term_mem_project`: Habitat 기반 Fetch 로봇 실행, 메모리(장/단기), LLM planner, rearrangement 중 동적 scene graph 업데이트

## 1. 프로젝트 개요

핵심 목표는 다음과 같습니다.

1. 오프라인/초기 단계에서 계층형 scene graph를 생성한다.
2. 런타임에서 로봇이 task를 수행하는 동안 객체 변화를 scene graph에 반영한다.
3. LLM으로 task decomposition 함수를 생성하고 실제 로봇 실행 루프와 연결한다.

## 2. 사전 준비물 (중요)

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

## 3. 폴더 구조

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

## 4. 파이프라인 A: 계층형 Scene Graph 생성

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

## 5. 파이프라인 B: Rearrangement + 동적 Scene Graph 업데이트

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

## 6. 실행 방법 (Quick Start)

### 6.1 필수 환경

- Python 3.10 권장
- Habitat/Habitat-Sim 환경
- Open3D, NumPy, SciPy, OpenCV, networkx, OmegaConf 등
- OpenAI API 키

```bash
export OPENAI_API_KEY=<YOUR_KEY>
```

### 6.2 (선택) 수동 RGB-D 수집

```bash
python hierarchy_robot_interaction_scene_graph_submission/my_long_term_mem_project/scripts/manual_control_posed_rgbd.py --hov
```

### 6.3 Scene Graph 생성

```bash
python hierarchy_robot_interaction_scene_graph_submission/hierarchy-scene-graphs/scripts/create_hybrid_scene_graph.py
```

### 6.4 GUI 기반 Task 실행

```bash
python hierarchy_robot_interaction_scene_graph_submission/my_long_term_mem_project/scripts/GUI_karma.py
```

## 7. 데모 미디어 (이미지/영상)

### 7.1 방 분리 결과 이미지
```md
<img width="708" height="654" alt="9" src="https://github.com/user-attachments/assets/7df81276-5b42-450e-80fd-db56d06c4672" />
<img width="708" height="654" alt="8" src="https://github.com/user-attachments/assets/fd9f7897-26d1-40f4-9ea7-f96aff4e29f4" />
<img width="708" height="654" alt="6" src="https://github.com/user-attachments/assets/4c4a1582-b464-4bd6-9245-c3db899da223" />
<img width="708" height="654" alt="5" src="https://github.com/user-attachments/assets/aee2a039-f5be-4c46-8c86-43087035b166" />
<img width="708" height="654" alt="4" src="https://github.com/user-attachments/assets/a05f5b52-b137-4c49-9282-a24b14fc4c00" />
<img width="708" height="654" alt="3" src="https://github.com/user-attachments/assets/7d870580-b98e-4b76-a55c-707be50fa546" />
<img width="708" height="654" alt="2" src="https://github.com/user-attachments/assets/0a615ba2-0ebe-43a9-80dc-ffe3f911b23b" />
<img width="708" height="654" alt="1" src="https://github.com/user-attachments/assets/72bc62a4-aeca-4b1d-badf-0bf826dbaf57" />
```

### 7.2 Task 수행 동영상
```md
[![Watch Demo](docs/media/rooms/room_01.png)](https://youtu.be/<VIDEO_ID>)
```

## 8. 메모리/로그 산출물

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

## 9. 주의사항

1. 코드에 절대경로(`/home/yuchaehee/...`)가 다수 포함되어 있습니다.
2. 다른 환경에서는 경로 상수와 config를 먼저 수정해야 합니다.
3. OpenAI API 키가 없으면 planner/이미지 분석 단계가 실패합니다.
4. Habitat 데이터셋, navmesh, sensor 설정이 없으면 시뮬레이터가 초기화되지 않습니다.

## 10. 참고 스크립트

ConceptGraph 객체 기준 텔레포트 디버그:

```bash
python hierarchy_robot_interaction_scene_graph_submission/my_long_term_mem_project/scripts/teleport_to_conceptgraph_object.py --list
```
