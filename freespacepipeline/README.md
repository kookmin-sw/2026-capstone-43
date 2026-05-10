# FreeSpace Pipeline: Free Space-Aware Scene Graph via VLM Fine-tuning

로봇의 pick-and-place 태스크에서 핵심이 되는 **"어디에 물건을 놓을 수 있는가"** 라는 빈 공간(Free Space) 정보를, 씬그래프(Scene Graph)에 통합하기 위한 VLM 파인튜닝 데이터 생성 파이프라인입니다.

---

## 연구 배경 및 동기

### 씬그래프의 구조적 한계

ConceptGraph, SayPlan 등 기존 연구에서 씬그래프는 LLM/VLM이 공간 정보를 이해하고 로봇 태스크 플래닝을 수행하는 핵심 표현으로 자리잡았습니다. 그러나 기존 씬그래프는 **"어디에 물건이 있는가"** 는 잘 표현하지만, **"어디에 물건을 놓을 수 있는가"** 라는 빈 공간 정보는 거의 다루지 않습니다.

이는 모바일 매니퓰레이터의 주된 태스크인 pick-and-place 작업에 있어 치명적인 빈틈입니다.

### 이전 연구(SceneUpdate)에서 확인한 한계

이 문제를 해결하고자 이전 프로젝트([SceneUpdate](https://github.com/LEESB17/sceneupdate/tree/sungbin))에서는 씬그래프에 빈 공간 정보를 통합하는 두 가지 방식을 구현했습니다:

- **오프라인 Precomputed Free Space**: ConceptGraph 데이터 기반 표면 빈 공간 사전 계산
- **실시간 Free Space Update**: Depth 카메라 + Occupancy Grid 비동기 갱신

시뮬레이션(Isaac Sim) 환경에서는 어느 정도 동작했지만, **실제 환경(Real World)으로 넘어갈 경우 Depth 센서 노이즈, 반사재질, 조명 변화 등으로 인한 오차**가 누적되어 신뢰할 수 있는 빈 공간 추정이 어렵다는 근본적인 한계가 있었습니다.

결론: **기하학 기반(Depth + 포인트 클라우드)의 파이프라인만으로는 한계가 있으며, VLM을 활용한 의미론적 이해가 반드시 필요하다.**

### 왜 VLM인가

최근 VLM(Vision Language Model)은 이미지를 보고 공간 관계를 추론하는 능력이 비약적으로 발전했습니다. 그러나 범용 VLM(GPT-4V, LLaVA 등)은:

1. **빈 공간 인식 능력이 부족함**: 물체가 없는 영역을 폴리곤으로 정량화하는 능력이 거의 없음
2. **씬그래프 생성에 특화된 모델이 없음**: 빈 공간을 씬그래프 노드로 생성하는 파인튜닝 사례 자체가 드묾

### RoboSpatial에서의 착안

[RoboSpatial](https://arxiv.org/abs/2411.11537)은 로봇 조작 환경에서의 공간 이해에 특화된 VLM 연구로, 빈 공간 인식 성능에서 뛰어난 결과를 보입니다. 그러나 RoboSpatial은 **씬그래프 생성 자체가 목적이 아니며**, 씬그래프의 노드/엣지 구조로 빈 공간 정보를 통합하는 방향을 직접 다루지 않습니다.

**따라서 이 프로젝트는 RoboSpatial의 공간 인식 방법론을 참조하되, 씬그래프 통합을 위한 나만의 데이터 생성 파이프라인과 QA 포맷을 설계하여 VLM을 파인튜닝하는 방향으로 발전시킵니다.**

---

## 접근 방법 요약

```
기존 씬그래프 (Object-only)
    ↓  한계: 빈 공간 정보 없음
SceneUpdate (Depth 기반 Freespace)
    ↓  한계: 실제 환경 노이즈에 취약
FreeSpace Pipeline (VLM 파인튜닝)
    → 빈 공간을 이미지에서 직접 폴리곤으로 추론하는 VLM 학습
    → 씬그래프 노드로 통합 가능한 구조화된 QA 포맷
```

---

## 전체 파이프라인 흐름

```
[GraspNet / HOPE / SUN RGB-D 원시 데이터]
  RGB 이미지 + Depth Map + 카메라 파라미터 + 3D OBB 어노테이션
           │
           ▼
  [Stage 1] run_generation.py  (RoboSpatial 원본 파이프라인)
  └─ 3D 씬 파싱 → 카메라별 물체 6DoF Pose → bbox_3d 계산
           │
           ▼
  [Stage 2] graspnet_surface_qa.py  (핵심: Depth 기반 빈 공간 QA 생성)
  ├─ Depth Map → 테이블 표면 픽셀 마스크 생성
  ├─ 물체 3D OBB → 2D 투영 후 표면에서 차감 (Safety Margin 15px)
  └─ 출력: table_layout / object_layout / freespace_layout QA 3종

  [Stage 2b] sunrgbd_surface_qa.py  (눈높이 카메라 환경 확장)
  ├─ SUN RGB-D OBB Top-face 직접 2D 투영 (Depth 불필요)
  ├─ LAB 색상 기반 Visual Snapping + Distance-Adaptive Morphology
  └─ 출력: 6,596 씬 × freespace QA (34,347건)
           │
           ▼
  [Stage 3] combine_datasets.py
  └─ GraspNet + HOPE + SUN RGB-D + RoboSpatial 기존 QA 병합
     → dataset/train.json (143,474건), validation.json (14,943건)
           │
           ▼
  [Stage 4] finetune_7b_freespace.py
  ├─ Qwen2-VL-7B-Instruct 로드
  ├─ freespace / table / object layout 카테고리 필터링 (~53,000건)
  ├─ LoRA (r=16, alpha=32, 1 Epoch) 파인튜닝
  └─ 출력: checkpoints/qwen2vl_7b_freespace/final/  (LoRA 어댑터)
           │
           ▼
  [검증] compare_freespace.py
  └─ Base Model vs LoRA 모델 폴리곤 출력 시각적 비교
```

---

## 핵심 알고리즘: Depth Map 기반 테이블 표면 추출

### Step 1 — Table Z 기준점 추정

모든 물체의 3D OBB 하단 Z좌표 Median을 테이블 높이로 설정합니다. Median을 사용하는 이유는 공중에 뜬 물체나 긴 병처럼 높이 이상치를 가진 물체의 영향을 제거하기 위함입니다.

### Step 2 — Depth → 테이블 픽셀 마스크

픽셀별 카메라 좌표를 월드 좌표로 변환 후, 테이블 표면 높이 근방(`±8/20cm` 비대칭 범위)의 픽셀만 보존합니다. 이 과정에서 어노테이션에 없는 물체(손, 전선 등)도 자동으로 제외됩니다.

### Step 3 — Morphological Closing으로 구멍 복원

반사재질(유리, 금속)이나 검은 표면에서 Depth 센서가 반환하는 NaN 픽셀을 31px 커널 Closing으로 채운 뒤, Contour → Shapely Polygon으로 테이블 폴리곤을 완성합니다.

### Step 4 — 물체 OBB 2D 투영 후 차감

```python
empty_poly = table_poly.difference(unary_union(obj_polys))
# obj_polys: 각 물체 OBB를 2D 투영 후 15px Safety Margin 확장
```

---

## 생성되는 QA 3종 포맷

하나의 이미지 프레임에서 씬그래프 통합을 위한 구조화된 QA 3종이 자동 생성됩니다.

**1. `table_layout`** — 테이블 전체 경계
```json
{
  "qa_type": "table_layout",
  "question": "<image>What is the total area of the table surface and what is its polygon shape? Output coordinates in [0, 1000] scale.",
  "answer": "The total table surface occupies 833466 pixels. Its shape is bounded by the polygon: [(12,0), (998,0), (998,998), (12,998)]."
}
```

**2. `object_layout`** — 개별 물체 위치 (Safety Margin 포함)
```json
{
  "qa_type": "object_layout",
  "question": "<image>Where is the pudding box and how much space does it occupy on the table assuming a 15 pixel safety margin?",
  "answer": "Including the 15px safety margin, the pudding box occupies 47823 pixels. Its location is defined by the polygon: [(329,349), (757,349), (757,539), (329,539)]."
}
```

**3. `freespace_layout`** — 가장 큰 빈 공간 (학습 핵심 타깃)
```json
{
  "qa_type": "freespace_layout",
  "question": "<image>How much empty space is available on the table and where is the largest empty area, assuming a 15 pixel safety margin around all objects?",
  "answer": "With a 15px safety margin, the objects occupy 422152 pixels of the table. Subtracting this from the table area (833466 pixels) leaves 411314 pixels of free space. The largest continuous empty space is defined by the polygon: [(0,577), (221,929), (306,675), ...]."
}
```

모든 좌표는 `[0, 1000]` 정수 스케일로 정규화됩니다. Qwen2-VL 등 최신 VLM 토크나이저에서 좌표 토큰화 효율을 최대화하기 위한 설계입니다.

---

## 데이터 현황

| 소스 | 씬 수 | freespace류 QA | 특징 |
|---|---|---|---|
| GraspNet train_4 | 10씬 (0090~0099) | ~4,546건 | 오버헤드 카메라, 근접 탁상 |
| GraspNet train_1 | 30씬 (0000~0029) | ~16,818건 | 오버헤드 카메라, 근접 탁상 |
| HOPE image/video | - | ~2,865건 | 오버헤드 카메라, 근접 탁상 |
| **SUN RGB-D** | **6,596씬** | **~34,347건** | **눈높이 카메라, 다양한 환경** |
| **전체 (train)** | - | **~53,000건** | **기존 RoboSpatial 대비 7배** |

전체 학습 데이터: `train.json` 143,474건 / `validation.json` 14,943건

---

## 학습 결과 (Base vs LoRA)

GraspNet + HOPE 약 200샘플, 25 step PoC 파인튜닝 결과:

**Base Model (Zero-shot)**
> "The largest empty area on the table is approximately at the coordinates (0, 1000) with a size of approximately (1000, 1000)."
→ 폴리곤 포맷 출력 불가. 완전한 환각(Hallucination).

**Fine-Tuned Model (LoRA, 25 step)**
> "With a 15px safety margin, the objects occupy 422152 pixels of the table. Subtracting this from the table area (833466 pixels) leaves 411314 pixels of free space. The largest continuous empty space is defined by the polygon: `[(0,577), (221,929), (306,675), (261,620), ...]`."
→ 테이블 면적에서 물체 점유 면적을 정확히 차감하고, 빈 공간을 폴리곤으로 완벽히 출력. 5장 테스트 이미지 전부 정상 추출 확인.

---

## 파이프라인 실행 방법

```bash
# Stage 1 — annotations.json 생성
python run_generation.py --config configs/graspnet.yaml

# Stage 2 — GraspNet/HOPE freespace QA 생성
python scripts/graspnet_surface_qa.py \
  --annotations_dir /home/sungbin/Robospatial/annotations/graspnet

# Stage 2b — SUN RGB-D freespace QA 생성
python scripts/sunrgbd_surface_qa.py \
  --mat  /path/to/SUNRGBDMeta3DBB_v2.mat \
  --root /path/to/SUNRGBD \
  --out  annotations/sunrgbd \
  --safety 15

# Stage 3 — 데이터셋 병합
python combine_datasets.py

# Stage 4 — LoRA 파인튜닝
python finetune_7b_freespace.py

# 검증 — Base vs LoRA 시각적 비교
python compare_freespace.py --image /path/to/image.png
```

---

## 디렉토리 구조

```
Robospatial/
├── annotations/
│   ├── graspnet/           ← Stage 2 출력 (GraspNet)
│   ├── hope_image/
│   ├── hope_video/
│   └── sunrgbd/            ← Stage 2b 출력 (SUN RGB-D, 34,347 QA)
├── dataset/
│   ├── train.json          (143,474건)
│   └── validation.json     (14,943건)
├── checkpoints/
│   └── qwen2vl_7b_freespace/final/    ← LoRA 어댑터
├── scripts/
│   ├── graspnet_surface_qa.py   ← GraspNet/HOPE용 (Depth 기반)
│   └── sunrgbd_surface_qa.py    ← SUN RGB-D용 (OBB 투영)
├── compare_freespace.py
├── combine_datasets.py
└── finetune_7b_freespace.py
```

---

## 관련 연구

- [SceneUpdate](https://github.com/LEESB17/sceneupdate/tree/sungbin) — 이 프로젝트의 전신. 씬그래프 + Depth 기반 빈 공간 통합 (Isaac Sim)
- [RoboSpatial](https://arxiv.org/abs/2411.11537) — 로봇 조작 환경 VLM 공간 이해 (데이터 파이프라인 참조)
- [ConceptGraph](https://github.com/concept-graphs/concept-graphs) — Open-vocabulary 3D 씬그래프
- [SayPlan](https://sayplan.github.io/) — 씬그래프 기반 LLM 태스크 플래닝
