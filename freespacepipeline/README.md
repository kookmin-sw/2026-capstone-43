# RoboSpatial: Tabletop Freespace & Spatial Layout Pipeline

이 저장소는 로봇 비전 모델(VLM)이 단순히 객체를 탐지하는 것을 넘어, **3D 기하학적 형태와 비율, 여백(Safety Margin)을 고려한 '빈 공간(Freespace)'을 인간 수준으로 이해하고 추론**할 수 있도록 설계된 데이터 자동 생성 및 파인튜닝 파이프라인입니다.

기존 방식(ScanNet의 부정확한 3D Mesh에 의존)을 탈피하고, 탁상 조작(Tabletop Manipulation) 환경에 최적화된 **HOPE 및 GraspNet-1Billion 데이터셋**과 **Depth Map 기반 필터링 알고리즘**을 도입하여 오차 없는 공간 데이터를 완전 자동으로 생성합니다.

---

## 전체 파이프라인 흐름

```
[GraspNet / HOPE 원시 데이터]
  RGB 이미지 + Depth Map + 카메라 파라미터 (Intrinsic/Extrinsic)
           │
           ▼
  [Stage 1] run_generation.py  (RoboSpatial 원본 파이프라인)
  ├─ GraspNet Loader: 3D 씬 파싱, 카메라별 물체 6DoF Pose → bbox_3d 계산
  └─ 출력: annotations/graspnet/{scene}/{frame}.annotations.json
           (image_path, camera_annotations, object_grounding[bbox_3d, bbox_2d], spatial_relationships)
           │
           ▼
  [Stage 2] scripts/graspnet_surface_qa.py  (본 프로젝트 핵심)
  ├─ annotations.json 읽기
  ├─ Depth Map 기반 테이블 표면 폴리곤 추출  ← 핵심 알고리즘
  ├─ 물체 3D OBB → 2D 투영 후 표면에서 차감
  └─ 출력: annotations/graspnet/{scene}/{frame}.graspnet_qa.json
           (table_layout / object_layout / freespace_layout QA 3종)
           │
           ▼
  [Stage 2b] scripts/sunrgbd_surface_qa.py  (일반 환경 확장)
  ├─ SUNRGBDMeta3DBB_v2.mat 직접 파싱 (Stage 1 불필요)
  ├─ table/desk/counter OBB Top-face 투영 → 테이블 폴리곤 추출
  ├─ 물체 OBB 전체 투영 → 테이블과 교차하는 장애물만 차감
  └─ 출력: annotations/sunrgbd/{scene_id}.json (6,596 씬 × 평균 5.2 QA)
           (34,347건: table_layout 6,596 / object_layout 22,795 / freespace_layout 4,956)
           │
           ▼
  [Stage 3] combine_datasets.py
  ├─ GraspNet QA + HOPE QA + SUN RGB-D QA + 기존 RoboSpatial QA 병합
  └─ 출력: dataset/train.json (143,474건), dataset/validation.json (14,943건)
           │
           ▼
  [Stage 4] finetune_7b_freespace.py
  ├─ Qwen2-VL-7B-Instruct 로드
  ├─ freespace/layout 카테고리 레코드만 필터링 (~53,000건, 기존 대비 7배)
  ├─ LoRA (r=16, alpha=32, 1 Epoch) 파인튜닝
  └─ 출력: checkpoints/qwen2vl_7b_freespace/final/  (LoRA 어댑터)
           │
           ▼
  [검증] compare_freespace.py
  └─ Base Model vs LoRA 모델 폴리곤 출력 시각적 비교
```

---

## 핵심 알고리즘: Depth Map 기반 테이블 표면 추출

`scripts/graspnet_surface_qa.py`의 `process_hope_annotation()` 함수가 핵심이며, 총 4단계로 동작합니다.

### Step 1 — Table Z 기준점 추정

```python
# 모든 물체의 3D OBB 하단 Z좌표를 수집하여 중앙값(Median)을 테이블 높이로 설정
bottom_z_list = []
for obj in objects:
    corners = np.asarray(obj['obb'].get_box_points())  # Open3D OBB → 8개 꼭짓점
    bottom_z_list.append(corners[:, 2].min())          # 가장 낮은 Z = 물체 바닥
table_z = np.median(bottom_z_list)
```

- `bbox_3d` 포맷 `[cx, cy, cz, dx, dy, dz, rx, ry, rz]`를 `_bbox3d_to_obb()`로 Open3D `OrientedBoundingBox`로 변환
- 개별 물체 바닥 Z가 아닌 **Median**을 사용하는 이유: 공중에 떠 있는 물체나 높이 이상치(예: 긴 병, 칼)의 영향을 제거하기 위함

### Step 2 — Depth Map → 테이블 픽셀 마스크 생성

```python
depth = depth_raw.astype(np.float32) / 1000.0  # mm → m

# 픽셀별 카메라 좌표계 → 월드 좌표계 변환
ys, xs = np.where(depth > 0.05)
zs = depth[ys, xs]
Xc = (xs - cx) * zs / fx
Yc = (ys - cy) * zs / fy
pts_cam = np.stack([Xc, Yc, zs, 1], axis=1)
pts_world = (extrinsic @ pts_cam.T).T       # 카메라 → 월드 좌표

# 테이블 표면 높이 근방 픽셀만 보존 (위로 8cm, 아래로 20cm 허용)
keep = (world_Z >= table_z - 0.20) & (world_Z <= table_z + 0.08)
```

- **왜 -20cm / +8cm 비대칭인가:** 테이블 두께나 카메라 틸트에 의한 아랫방향 오차는 크고, 위쪽(물체가 올려진 방향)은 오차가 작기 때문
- **Implicit Obstacle Avoidance:** 어노테이션에 없는 물체(손, 전선 등)도 테이블 높이 범위를 벗어나므로 자동으로 마스크에서 제외됨

### Step 3 — Morphological Closing으로 구멍 복원

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
table_mask = cv2.morphologyEx(table_mask, cv2.MORPH_CLOSE, kernel)
```

- 반사 재질(유리, 금속)이나 검은 물체 표면에서 Depth 센서가 반환하는 `NaN` / `0` 픽셀을 31px 커널 Closing으로 메움
- 이후 `cv2.findContours` → `Shapely Polygon.simplify(2.0)` → `convex_hull`로 최종 테이블 폴리곤 완성

### Step 4 — 물체 OBB 2D 투영 후 차감

```python
for obj in objects:
    corners = np.asarray(obj['obb'].get_box_points())   # 3D OBB의 8개 꼭짓점
    poly_2d = _get_2d_convex_hull(corners, extrinsic, intrinsic, img_w, img_h)
    obj_polys.append(poly_2d.buffer(15.0))              # 15px Safety Margin 확장

# Shapely 집합 연산으로 빈 공간 계산
empty_poly = table_poly.difference(unary_union(obj_polys))
```

- `_project_pts()`: 3D 월드 좌표 → 카메라 좌표 (`w2c = inv(extrinsic)`) → 픽셀 좌표 (`fx * X/Z + cx`)
- `.buffer(15.0)`: Minkowski sum으로 물체 외곽에 15px 안전 여백 추가
- `unary_union(obj_polys)`: 겹치는 물체 폴리곤들을 하나로 병합 후 `difference`

---

## 생성되는 QA 3종 포맷

하나의 이미지 프레임에서 아래 3종류의 QA가 자동 생성됩니다.

### 1. `table_layout` — 테이블 전체 경계
```json
{
  "qa_type": "table_layout",
  "question": "<image>What is the total area of the table surface and what is its polygon shape? Output coordinates in [0, 1000] scale.",
  "answer": "The total table surface occupies 833466 pixels. Its shape is bounded by the polygon: [(12,0), (998,0), (998,998), (12,998)]."
}
```

### 2. `object_layout` — 개별 물체 위치 (Safety Margin 포함)
```json
{
  "qa_type": "object_layout",
  "question": "<image>Where is the pudding box and how much space does it occupy on the table assuming a 15 pixel safety margin? Output coordinates in [0, 1000] scale.",
  "answer": "Including the 15px safety margin, the pudding box occupies 47823 pixels. Its location is defined by the polygon: [(329,349), (757,349), (757,539), (329,539)]."
}
```

### 3. `freespace_layout` — 가장 큰 빈 공간 (학습 핵심 타깃)
```json
{
  "qa_type": "freespace_layout",
  "question": "<image>How much empty space is available on the table and where is the largest empty area, assuming a 15 pixel safety margin around all objects? Output coordinates in [0, 1000] scale.",
  "answer": "With a 15px safety margin, the objects occupy 422152 pixels of the table. Subtracting this from the table area (833466 pixels) leaves 411314 pixels of free space. The largest continuous empty space is defined by the polygon: [(0,577), (221,929), (306,675), ...]."
}
```

---

## 1000-Scale 좌표 정규화

모든 폴리곤 좌표는 픽셀 좌표가 아닌 `[0, 1000]` 정수 스케일로 출력됩니다.

```python
# 정규화 공식
x_norm = int(x_pixel / img_w * 1000)
y_norm = int(y_pixel / img_h * 1000)
```

- **이유:** Qwen2-VL, LLaVA 등 최신 VLM 토크나이저는 `0.366` 같은 소수점 좌표나 `1280` 같은 큰 절대 픽셀 값을 여러 토큰으로 쪼개서 학습 효율이 나빠짐. `[0, 1000]` 정수는 대부분 2~3자리로 안정적으로 토큰화됨
- **역변환:** `x_pixel = x_norm / 1000 * img_w` (추론 후 로봇 제어 시 사용)
- **Shapely 연동:** 역변환 좌표를 `Shapely.contains()`, `difference()` 등에 직접 입력하여 실시간 충돌 검사 가능

---

## 파이프라인 실행 방법

### Stage 1 — annotations.json 생성 (씬당 1회)
```bash
cd /home/sungbin/RoboSpatial/robospatial

# GraspNet train_4 (scene_0090~0099)
python run_generation.py --config configs/graspnet.yaml

# GraspNet train_1 (scene_0000~0029)  ← 현재 진행 중
python run_generation.py --config configs/graspnet_train1.yaml
```

### Stage 2 — freespace QA 생성
```bash
cd /home/sungbin/RoboSpatial

# GraspNet train_4
python scripts/graspnet_surface_qa.py \
  --annotations_dir /home/sungbin/Robospatial/annotations/graspnet

# GraspNet train_1
python scripts/graspnet_surface_qa.py \
  --annotations_dir /home/sungbin/Robospatial/annotations/graspnet_train1
```

### Stage 2b — SUN RGB-D freespace QA 생성 (일반 환경)
```bash
cd /home/sungbin/Robospatial

python scripts/sunrgbd_surface_qa.py \
  --mat  /home/sungbin/Downloads/SUNRGBDMeta3DBB_v2.mat \
  --root /home/sungbin/Downloads \
  --out  /home/sungbin/Robospatial/annotations/sunrgbd \
  --safety 15
# → annotations/sunrgbd/sunrgbd_qa.json  (34,347건, ~12초)
```

### Stage 3 — 데이터셋 병합
```bash
cd /home/sungbin/Robospatial
python combine_datasets.py
# → dataset/train.json (143,474건), dataset/validation.json (14,943건) 생성
```

### Stage 4 — 파인튜닝
```bash
python finetune_7b_freespace.py
# Qwen2-VL-7B + LoRA r=16, 1 Epoch
# 학습 대상: freespace_layout / table_layout / object_layout 카테고리만 필터링
```

### 검증 (Base vs LoRA 시각적 비교)
```bash
python compare_freespace.py --image /path/to/image.png
# → compare_result.jpg: 좌(Base) / 우(LoRA) 폴리곤 오버레이 비교 이미지
```

---

## 데이터 현황

| 소스 | 씬 수 | 처리 상태 | freespace류 QA 수 | 특징 |
|---|---|---|---|---|
| GraspNet train_4 | 10씬 (0090~0099) | 완료 | ~4,546건 | 오버헤드 카메라, 근접 탁상 |
| GraspNet train_1 | 30씬 (0000~0029) | 완료 | ~16,818건 | 오버헤드 카메라, 근접 탁상 |
| HOPE image/video | - | 완료 | ~2,865건 | 오버헤드 카메라, 근접 탁상 |
| **SUN RGB-D** | **6,596씬** | **완료** | **~34,347건** | **눈높이 카메라, 다양한 거리/각도** |
| **전체 학습 데이터** | - | **train 143,474건** | **freespace류 ~53,000건** | **기존 대비 7배 증가** |

### SUN RGB-D 데이터셋 특징
- **10,335개 RGB-D 프레임**: kv1/kv2/realsense/xtion 4종 센서
- **6,668개 테이블/책상 포함 씬** 중 6,596개 성공 처리 (99% 성공률)
- **다양한 환경**: 침실, 주방, 사무실, 교실, 회의실, 거실 등
- **눈높이 카메라**: 기존 GraspNet/HOPE와 달리 실제 로봇/사용자 시점
- **투영 방식**: Depth 필터링 대신 OBB Top-face 직접 2D 투영 (카메라 기울기 보정 포함)

---

## SUN RGB-D OBB 투영 알고리즘

`scripts/sunrgbd_surface_qa.py`는 GraspNet 파이프라인과 달리 Depth Map 필터링 없이 직접 3D OBB를 투영합니다.

### 좌표계 변환 (MATLAB project3dPtsTo2d 재현)

SUN RGB-D의 3D 어노테이션은 **Upright 좌표계** (중력 방향 정렬)에 저장됩니다. 이미지 픽셀로 투영하려면:

```python
# 1. Rtilt.T 적용: upright → annotation/camera 좌표계
q = Rtilt.T @ pt_upright          # shape: (3,)

# 2. 축 재배치: MATLAB swap cols 2↔3 + negate
x3 = q[0]          # lateral
y3 = -q[2]         # MATLAB: -col3_after_swap = original -q[2]
z3 = q[1]          # MATLAB: col2_after_swap = original q[1]

# 3. 핀홀 투영
u = fx * x3 / z3 + cx
v = fy * y3 / z3 + cy
```

### 테이블 폴리곤 생성

```python
# 테이블 Top-face 4개 꼭짓점 (basis[2] = [0,0,1] 방향 최상단)
for sx, sy in [(-1,-1),(-1,+1),(+1,-1),(+1,+1)]:
    corner = centroid + sx*coeffs[0]*basis[0] + sy*coeffs[1]*basis[1] + coeffs[2]*basis[2]
top_corners = np.array(corners)  # (4, 3) in upright space

# 투영 후 Shapely 볼록 껍질
px2d      = _project(top_corners, Rtilt, fx, fy, cx, cy)
table_poly = Polygon(px2d).convex_hull.intersection(image_boundary)
```

### GraspNet 방식과의 차이점 및 개선 사항 (Distance-Adaptive Visual Snapping)

기존 SUN RGB-D 및 ARKitScenes 파이프라인은 부정확한 3D OBB 투영(단순 사각형)에 크게 의존하여 책상이 멀리 떨어져 있을 경우(눈높이 카메라) 배경을 포함해버리거나 형태가 붕 뜨는 치명적인 오차가 있었습니다. 이를 해결하기 위해 다음과 같은 첨단 필터링 알고리즘이 파이프라인에 통합되었습니다.

1. **LAB 색상 기반 Visual Refinement (Color Snapping)**
   - OBB 투영 결과 또는 RANSAC Depth 마스크의 한가운데(Core)에서 순수 책상 픽셀의 LAB 중앙값(Median) 색상을 추출합니다.
   - `L` (명도) 채널의 가중치를 대폭 낮춰 조명이나 그림자 변화를 무시하고, `a, b` (색상) 채널을 기준으로 색상이 30 이상 차이나는 이질적 배경이나 옆 책상을 마스크에서 완전히 도려냅니다.
2. **동적 스케일링(Distance-Adaptive Morphology)**
   - 고정된 픽셀 크기가 아닌, 이미지상 책상의 너비(`np.sqrt(area)`)에 비례하여 Morphological 커널 크기를 동적으로 조절합니다. 책상이 멀리 있으면 작고 정교하게, 가까이 있으면 거대하게 모니터 암 같은 부가물을 필터링합니다.
3. **Convex Hull 복원**
   - 위 필터링으로 인해 둥글게 파인 책상 모서리와 내부 패턴(로고 등)의 구멍들을 `.convex_hull` 연산으로 팽팽하게 당겨, 실제 픽셀의 시각적 경계(Visual boundary)에 완벽하게 달라붙는 정교한 2D 폴리곤으로 변환합니다.

| 항목 | GraspNet / HOPE | SUN RGB-D / ARKitScenes (Eye-level) |
|---|---|---|
| 테이블 표면 검출 | Depth Map → LAB Color Filtering → Contour | OBB/RANSAC Depth → LAB Color Snapping → Contour |
| 카메라 각도 | 오버헤드 (수직), 노이즈가 적음 | 눈높이 (수평), 원거리 왜곡 및 노이즈 심함 |
| 주요 필터링 기술 | 거대 Morphological Opening + 색상 기반 필터 | 동적 스케일링(Adaptive Kernel) + 색상 기반 필터 |

---

## 디렉토리 구조

```
Robospatial/
├── annotations/
│   ├── graspnet/           ← Stage 2 출력 (train_4, 10씬)
│   ├── graspnet_train1/    ← Stage 2 출력 (train_1, 30씬)
│   ├── hope_image/
│   ├── hope_video/
│   └── sunrgbd/            ← Stage 2b 출력 (6,596씬, 34,347 QA)
│       ├── SUNRGBD_kv2_...json   (씬별 QA 캐시)
│       └── sunrgbd_qa.json       (전체 병합 파일)
├── dataset/
│   ├── train.json          (143,474건, Stage 3 출력)
│   └── validation.json     (14,943건)
├── checkpoints/
│   └── qwen2vl_7b_freespace/final/    ← LoRA 어댑터 (Stage 4 출력)
├── lora_final/final/       ← 서버 배포용 LoRA 복사본
├── scripts/
│   ├── graspnet_surface_qa.py  ← GraspNet/HOPE용 (Depth 기반)
│   └── sunrgbd_surface_qa.py   ← SUN RGB-D용 (OBB 투영)
├── compare_freespace.py    ← Base vs LoRA 시각화 비교
├── combine_datasets.py     ← Stage 3
├── finetune_7b_freespace.py ← Stage 4
└── export_dataset.py       ← DLPC 서버 배포용 패키징
```

---

## 학습 결과 요약 (Zero-shot vs 1-Epoch LoRA)

GraspNet + HOPE 데이터 약 200샘플, 25 step PoC 파인튜닝 후 추론 결과 비교:

### Base Model (Zero-shot)
**답변:** "The largest empty area on the table is approximately at the coordinates (0, 1000) with a size of approximately (1000, 1000)."
> 폴리곤 포맷 출력 불가. 물체 위치를 멋대로 지어내는 완전한 환각(Hallucination).

### Fine-Tuned Model (LoRA, 25 step)
**답변:** "With a 15px safety margin, the objects occupy 422152 pixels of the table. Subtracting this from the table area (833466 pixels) leaves 411314 pixels of free space. The largest continuous empty space is defined by the polygon: `[(0,577), (221,929), (306,675), (261,620), (236,626), (207,575), (324,41), (372,66), (450,27), (491,217), (687,0), (0,0), (0,577)]`."
> 테이블 면적에서 물체 점유 면적을 정확히 차감하고, 빈 공간 형태를 12개 꼭짓점 폴리곤으로 완벽히 출력. 5장 테스트 이미지 전부에서 폴리곤 1개씩 정상 추출 확인.