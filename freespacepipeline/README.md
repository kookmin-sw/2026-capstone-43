# Surface QA 데이터셋 생성 파이프라인

로봇 조작 환경에서 **물체를 올려놓을 수 있는 표면(support surface) 이해**를 위한 Visual QA 데이터셋 생성 파이프라인입니다.

두 가지 파이프라인이 상호 보완적으로 QA 쌍을 생성하여, VLM(비전-언어 모델)이 지지 표면과 그 위에 놓인 물체를 인식하도록 학습시킵니다.

| 파이프라인 | 데이터셋 | 씬 유형 | 표면 감지 방식 |
|---|---|---|---|
| **GraspNet Tabletop v2** | GraspNet-1Billion | 테이블탑 근접 촬영 | 깊이 + 카메라-테이블 변환 행렬 |
| **ScanNet++ Indoor** | ScanNet++ | 실내 공간 전체 | 3D OBB + 메쉬 + 레이 캐스팅 |

---

## 저장소 구조

```
scripts/
  graspnet_surface_qa_v2.py       # GraspNet 테이블탑 QA 생성기 (v2)
  scannetpp_indoor_surface_qa.py  # ScanNet++ 실내 표면 QA 생성기
annotations/
  graspnet/          # GraspNet 테스트 씬 QA (scene_0090~)
  graspnet_train1/   # GraspNet 학습 씬 QA (scene_0000~)
  scannetpp/         # ScanNet++ 실내 QA
```

---

## 파이프라인 1 — GraspNet Tabletop v2

### 개요

GraspNet-1Billion 데이터셋의 테이블탑 씬에 대해 `table_layout`, `object_layout`, `freespace` QA를 생성합니다. 카메라-테이블 외부 행렬과 픽셀 단위 물체 레이블을 사용하여 메쉬나 OBB 근사 없이 픽셀 정확도의 표면 추출이 가능합니다.

### v2로 개선한 이유

**v1의 문제점:**
- 깊이만으로 추정한 테이블 Z + 볼록 껍질(convex hull) → 테이블 바깥까지 freespace로 잘못 표시
- OBB 투영으로만 물체를 차감 → 경계가 부정확

**v2 개선 사항:**
- `cam0_wrt_table.npy` (카메라→테이블 프레임 변환) 사용 → 테이블 Z=0 평면이 정확하게 정의됨
- 깊이를 테이블 프레임으로 변환 후 |z| < 3cm 픽셀만 유지 → 픽셀 단위 정확도의 테이블 마스크
- `label/{frame}.png` (픽셀별 물체 ID) 사용 → 물체 경계 100% 정확, OBB 불필요
- 볼록 껍질 단계 제거 → 모서리가 둥근 테이블이나 부분적으로 보이는 테이블 경계를 올바르게 처리
- 이미지 가장자리에 닿은 픽셀을 freespace에서 제외

### 필요 데이터

[GraspNet-1Billion](https://graspnet.net/) 데이터셋, 씬별 구조:

```
{scene_id}/
  realsense/
    rgb/{frame:04d}.png
    depth/{frame:04d}.png
    label/{frame:04d}.png
    camera_poses.npy          # (N, 4, 4) 프레임별 카메라 포즈
    cam0_wrt_table.npy        # (4, 4) 카메라-0 → 테이블 변환 행렬
    camK.npy                  # (3, 3) 내부 파라미터 행렬
```

### 사용법

```bash
python scripts/graspnet_surface_qa_v2.py \
    --graspnet_root /path/to/graspnet \
    --scene scene_0093 \
    --out_dir annotations/graspnet/scene_0093 \
    --debug_dir debug/graspnet_qa_v2 \
    [--max_frames 50] \
    [--frame_step 5]
```

### 출력 QA 형식

```json
{
  "qa_type": "table_layout",
  "question": "<image>What is the total area of the table surface and what is its polygon shape? Output coordinates in [0, 1000] scale.",
  "answer": "The total table surface occupies 897833 pixels. Its shape is bounded by the polygon: [(75,0), ...]."
}
```

생성되는 QA 유형:
- `table_layout` — 보이는 테이블 표면 폴리곤 및 면적
- `object_layout` — 물체별 발자국 폴리곤 (15px 안전 여백 포함)
- `freespace` — 테이블 위 빈 배치 가능 영역

---

## 파이프라인 2 — ScanNet++ Indoor Surface

### 개요

ScanNet++ 실내 씬에 대해 `surface_layout`, `objects_on_surface` QA를 생성합니다. 밀집 재구성 어노테이션의 3D 방향성 바운딩 박스(OBB)로 지지 표면(테이블, 책상, 카운터 등)을 감지하고, 씬 메쉬로 경계를 정제하며, 레이 캐스팅으로 가시성을 검증합니다.

### 핵심 설계 결정

#### 깊이 추정 미사용
`segments_anno.json`의 OBB 어노테이션이 정확한 3D 크기를 제공합니다. 각 OBB의 상단 면이 정밀한 표면 평면을 바로 제공하므로 깊이 맵이나 평면 피팅이 필요 없습니다.

#### 메쉬 기반 경계 정제
직사각형 OBB 면을 투영하는 대신, 해당 물체에 속한 메쉬 꼭짓점을 추출하고 가장 높은 꼭짓점들의 볼록 껍질을 구합니다. 이를 통해 원형 테이블, L자형 카운터 등 비직사각형 표면을 올바르게 표현할 수 있습니다.

#### 레이 캐스팅 가시성 필터
표면에 대한 QA를 생성하기 전, 카메라에서 표면 상단 면의 샘플 포인트를 향해 25개의 광선을 쏩니다. 20% 미만의 광선만 표면에 도달하면(벽, 가구 등에 가려진 경우) 해당 표면은 건너뜁니다. 이를 통해 벽 뒤에 숨거나 완전히 가려진 표면에 대한 잘못된 QA 생성을 방지합니다.

#### 3단계 가시성 검사
1. **카메라 높이** — 카메라가 표면보다 위에 있어야 함 (아래에서 올려다보는 경우 QA 생성 안 함)
2. **중심점 투영** — 표면 중심점이 이미지 프레임 안에 투영되어야 함
3. **레이 캐스팅** — 표면 샘플 포인트의 20% 이상이 직접 가시여야 함

### 지원 표면 카테고리

```python
SUPPORT_SURFACE_LABELS = {
    'table',          # 테이블
    'desk',           # 책상
    'kitchen counter', # 주방 조리대
    'counter',        # 카운터
    'coffee table',   # 커피 테이블
    'dining table',   # 식탁
    'end table',      # 사이드 테이블
    'side table',     # 사이드 테이블
    'nightstand',     # 협탁
    'tv stand',       # TV 스탠드
    'bench',          # 벤치
}
```

### 필요 데이터

[ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/) 데이터셋, 씬별 구조:

```
{scene_id}/
  scans/
    mesh_aligned_0.05.ply       # 밀집 재구성 메쉬
    segments.json               # 꼭짓점 → 세그먼트 ID 매핑
    segments_anno.json          # OBB + 레이블이 포함된 세그먼트 그룹
  dslr/
    colmap/
      cameras.txt               # OPENCV_FISHEYE 카메라 모델
      images.txt                # 이미지별 포즈 (쿼터니언 + 평행이동)
    resized_images/
      {image_name}.JPG          # DSLR RGB 프레임
```

### 사용법

```bash
# 단일 씬 처리 (디버그 이미지 포함)
python scripts/scannetpp_indoor_surface_qa.py \
    --data_root /path/to/ScanNetPP/data \
    --scenes 09c1414f1b \
    --max_frames 20 \
    --workers 1 \
    --out_json annotations/scannetpp/scannetpp_indoor_qa.json \
    --debug_dir debug/scannetpp_indoor_qa

# 전체 씬 병렬 처리
python scripts/scannetpp_indoor_surface_qa.py \
    --data_root /path/to/ScanNetPP/data \
    --out_json annotations/scannetpp/scannetpp_indoor_qa.json \
    --debug_dir debug/scannetpp_indoor_qa \
    --max_frames 15 \
    --workers 4

# 빠른 모드 (메쉬 정제 생략, OBB 상단 면만 사용)
python scripts/scannetpp_indoor_surface_qa.py \
    --data_root /path/to/ScanNetPP/data \
    --out_json annotations/scannetpp/scannetpp_indoor_qa.json \
    --no_mesh_refine
```

### 출력 QA 형식

```json
{
  "scene_id": "09c1414f1b",
  "image_name": "DSC05469.JPG",
  "image_path": "/path/to/resized_images/DSC05469.JPG",
  "qa": [
    {
      "qa_type": "surface_layout",
      "question": "<image>What is the visible area and boundary of the coffee table surface? Output coordinates in [0, 1000] scale.",
      "answer": "The visible coffee table surface occupies 219900 pixels. Its boundary polygon is: [(810,1000), (847,781), ...]."
    },
    {
      "qa_type": "objects_on_surface",
      "question": "<image>What objects are on the coffee table?",
      "answer": "The following objects are on the coffee table: remote controller, coaster."
    }
  ]
}
```

생성되는 QA 유형:
- `surface_layout` — 보이는 표면 경계 폴리곤 및 픽셀 면적
- `objects_on_surface` — 표면 위에 실제로 놓인 물체 목록

---

## 의존성 패키지

```bash
pip install open3d opencv-python numpy shapely scipy tqdm
```

| 패키지 | 용도 |
|---|---|
| `open3d` | 메쉬 로딩, 레이 캐스팅 (`o3d.t.geometry.RaycastingScene`) |
| `opencv-python` | 이미지 입출력, 어안렌즈 투영 (`cv2.fisheye.projectPoints`) |
| `shapely` | 2D 폴리곤 연산 (교집합, 차집합, 볼록 껍질) |
| `scipy` | 메쉬 경계 추출을 위한 3D 볼록 껍질 |
| `tqdm` | 진행 표시줄 |

---

## QA 좌표 체계

모든 폴리곤 좌표는 이미지 해상도에 관계없이 **[0, 1000]** 스케일로 정규화됩니다:

```
pixel_x_norm = int(pixel_x / image_width  * 1000)
pixel_y_norm = int(pixel_y / image_height * 1000)
```

이를 통해 QA 답변이 해상도 독립적이 되며, 정규화된 좌표를 출력하는 VLM(Qwen2-VL, InternVL 등) 파인튜닝에 바로 사용할 수 있습니다.

---

## 데이터셋 통계 (ScanNet++ 50개 씬, 씬당 최대 15프레임)

| 항목 | 수치 |
|---|---|
| 처리된 씬 수 | 50개 |
| 지지 표면이 있는 씬 수 | 약 37개 |
| 표면이 보이는 프레임 수 | 271개 |
| 총 QA 쌍 수 | 1,004개 |
| 프레임당 평균 QA 수 | 약 3.7개 |
| 감지된 표면 유형 | table, desk, kitchen counter, coffee table, dining table, nightstand |
