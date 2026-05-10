# 2026 Capstone Team 43
[![Website](https://img.shields.io/badge/Website-Project_Page-181717?style=flat-square&logo=github)](https://kookmin-sw.github.io/2026-capstone-43/)
🏠 [프로젝트 홈페이지 바로가기](https://kookmin-sw.github.io/2026-capstone-43/)

## 1. 프로젝트 소개

본 프로젝트는 로봇의 실세계 인지와 상호작용을 위한 여러 연구형 모듈을 통합한 캡스톤 디자인 프로젝트입니다.

본 저장소는 하나의 단일 애플리케이션이 아니라, 로봇이 실제 환경에서 사람과 상호작용하고, 주변 공간을 이해하며, 음향·시각·장면 정보를 활용해 행동할 수 있도록 하기 위한 여러 연구 주제를 포함합니다.
로봇이 세상을 이해하고 인지하기 위한 멀티모달 인식 기능에 대한 연구 모음입니다.

주요 연구 방향은 다음과 같습니다.

- 로봇 동작 중 발생하는 noise를 고려한 speech enhancement
- 공간 음향 기반 로봇 인지
- RGB image와 pose 기반 3D scene reconstruction
- 계층형 Scene Graph 기반 환경 이해
- LLM 기반 로봇 task planning 및 interaction
- Isaac Sim 기반 모바일 매니퓰레이터 시뮬레이션

각 하위 폴더는 독립적인 연구 주제를 가지며, 전체 프로젝트는 로봇의 perception, mapping, scene understanding, planning, manipulation을 다루는 통합 연구 저장소입니다.

### 프로젝트 구성

| 폴더 | 연구 주제 | 설명 |
| --- | --- | --- |
| `LRDSE` | Robot Noise Speech Enhancement | 로봇 동작 중 발생하는 비정상적 noise를 제거하기 위한 speech enhancement 연구입니다. SGMSE/RDDM 기반 모델과 foot force condition을 활용합니다. |
| `SpatialAudio` | Spatial Audio Perception | 로봇의 공간 음향 인지를 위한 audio feature, FOA/AmbiX, SpatialAST 기반 실험을 포함합니다. |
| `catkin_ws` | RGB-Pose Collection & Gaussian Splatting | 로봇에서 RGB image와 pose를 수집하고, 이를 Gaussian Splatting 및 continual mapping 실험에 활용합니다. |
| `hierarchy_robot_interaction_scene_graph_submission` | Hierarchical Scene Graph | RGB-D/pose 기반 계층형 Scene Graph를 생성하고, LLM 기반 로봇 interaction에 활용합니다. |
| `sceneupdate` | Isaac Sim Robot Interaction | Isaac Sim 환경에서 모바일 매니퓰레이터, Scene Graph, LLM planner, navigation, manipulation을 결합한 시뮬레이션 프로젝트입니다. |

---

## 2. 소개 영상

프로젝트 소개 영상은 아래 링크를 통해 확인할 수 있습니다.

- 소개 영상 링크: 추후 추가 예정

---

## 3. 팀 소개

| 이름 | 역할 | 담당 내용 | GitHub 담당 하위 폴더 |
| --- | --- | --- | --- |
| 류재우 | 팀장 | Legged Robot noise speech enhancement 연구 및 LRDSE 구현 | `LRDSE` |
| 이성빈 | 팀원 | Scene Graph 기반 환경 인식 및 시뮬레이션 기능 구현 | `sceneupdate` |
| 유동현 | 팀원 | Spatial Audio 기반 로봇 공간 음향 인지 연구 및 구현 | `SpatialAudio` |
| 유채희 | 팀원 | Hierarchical Scene Graph 기반 로봇 상호작용 연구 및 구현 | `hierarchy_robot_interaction_scene_graph_submission` |
| 장근서 | 팀원 | RGB-Pose 수집 및 Gaussian Splatting 기반 3D scene reconstruction 연구 | `catkin_ws` |

---

## 4. 사용법

본 저장소의 각 하위 프로젝트는 서로 다른 실행 환경과 dependency를 가지고 있습니다. 따라서 루트 디렉토리에서 한 번에 실행하는 방식이 아니라, 필요한 하위 프로젝트로 이동한 뒤 해당 README를 참고하여 실행해야 합니다.

### 4.1 저장소 clone

```bash
git clone https://github.com/kookmin-sw/2026-capstone-43.git
cd 2026-capstone-43
```

### 4.2 하위 프로젝트 실행

각 프로젝트의 자세한 설치 및 실행 방법은 해당 폴더의 README를 참고하세요.


### 4.3 주의사항

- 각 하위 프로젝트는 독립적인 환경을 가집니다.
- Python version, CUDA version, ROS/ROS2, Isaac Sim, simulator asset 등이 프로젝트마다 다를 수 있습니다.
- 일부 dataset, checkpoint, simulator asset, API key는 저장소에 포함되어 있지 않을 수 있습니다.
- 실행 전 각 하위 폴더의 README와 config 파일을 먼저 확인해야 합니다.

---
