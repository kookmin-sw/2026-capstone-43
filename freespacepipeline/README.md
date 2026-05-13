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

## 🛑 시행착오 및 파이프라인 발전 과정 (Trial & Errors)

Eye-level 실내 환경(ScanNet, ARKitScenes, ScanNet++)에서 테이블 위 소물체와 빈 공간을 검출하기 위해 여러 접근법을 시도하였으며, 최종적으로 현재의 파이프라인 구조에 이르게 되었습니다.

### 1차 시도: 3D BBox 및 LiDAR Depth 직접 활용 (실패)
- **접근:** ScanNet 및 ARKitScenes의 3D OBB와 LiDAR 센서 Depth를 이용해 RANSAC 평면 피팅을 시도.
- **문제점:** Eye-level 카메라 특성상 테이블 측면(Side face)이 상판으로 오검출되거나, LiDAR의 고질적인 노이즈로 인해 표면 Normal이 깨짐. 특히 테이블 위의 컵이나 펜 같은 '소물체'는 Depth 센서가 아예 잡지 못하거나 3D BBox가 너무 크게(Loose) 잡혀 빈 공간이 과도하게 깎여나가는 치명적 문제가 발생.

### 2차 시도: 2D Foundation Model (Grounded SAM) 도입 (한계 봉착)
- **접근:** 3D 노이즈를 피하기 위해 Grounding DINO로 물체를 찾고 SAM(Segment Anything)으로 픽셀 단위 마스크를 추출.
- **문제점:** SAM은 3D 공간(깊이)에 대한 이해가 없기 때문에, "table" 프롬프트를 주면 상판뿐만 아니라 **테이블 다리, 그림자, 책상 아래 바닥까지 통째로 분할**해버리는 문제 발생.

### 3차 시도: 하이브리드 파이프라인 (Depth RANSAC + SAM)
- **접근:** 테이블 상판은 기존의 Depth RANSAC으로 정밀하게 추출하고, 그 위에 올라간 소물체들만 Grounding DINO + SAM으로 분할.
- **문제점:** 추출 품질은 훌륭했으나, 전체 데이터셋에 무거운 2D Foundation Model 추론을 돌리기에는 연산 비용이 지나치게 높고 ARKitScenes의 원본 데이터 접근 한계가 존재.

### 4차 시도: ScanNet++ RoboSpatial 기반 파이프라인 (방향 전환)
- **접근:** ScanNet++의 dense mesh + 3D OBB를 활용해 RoboSpatial의 `analyse_surfaces` 함수 기반으로 표면 감지를 시도.
- **문제점:** `analyse_surfaces`는 tabletop 근접 씬을 가정하고 설계된 함수여서, 실내 공간 전체가 담긴 eye-level 씬에서는 표면 scoring이 맞지 않음. 화장실, 복도 등 테이블 없는 씬에서도 엉뚱한 표면을 감지하거나, 벽 너머에 있는 책상도 카메라에 보이는 것으로 잘못 판단하는 문제 발생.

### 💡 현재 구조: 두 파이프라인 병렬 운영

- **Tabletop (GraspNet v2)**: 근접 촬영 테이블탑 씬 → 깊이 기반 픽셀 단위 정확도. 외부 라이브러리 의존 없이 카메라-테이블 변환 행렬로 완벽한 표면 추출.
- **Indoor (ScanNet++ v1)**: 실내 공간 전체 씬 → OBB 직접 활용 + mesh refinement + ray casting 가시성 필터. RoboSpatial 라이브러리 완전 제거, 처음부터 새로 설계.

---

## 현재 파이프라인 구조

두 가지 파이프라인이 상호 보완적으로 QA 쌍을 생성합니다.

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

### v1 대비 v2 개선 사항

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

### 출력 QA 유형

- `table_layout` — 보이는 테이블 표면 폴리곤 및 면적
- `object_layout` — 물체별 발자국 폴리곤 (15px 안전 여백 포함)
- `freespace` — 테이블 위 빈 배치 가능 영역

---

## 파이프라인 2 — ScanNet++ Indoor Surface

### 개요

ScanNet++ 실내 씬에 대해 `surface_layout`, `objects_on_surface` QA를 생성합니다. 밀집 재구성 어노테이션의 3D OBB로 지지 표면을 감지하고, 씬 메쉬로 경계를 정제하며, 레이 캐스팅으로 가시성을 검증합니다. RoboSpatial 라이브러리에 대한 의존성이 전혀 없습니다.

### 핵심 설계

#### 깊이 추정 미사용
`segments_anno.json`의 OBB 어노테이션이 정확한 3D 크기를 제공합니다. 각 OBB의 상단 면이 정밀한 표면 평면을 바로 제공하므로 깊이 맵이나 평면 피팅이 필요 없습니다.

#### 메쉬 기반 경계 정제
직사각형 OBB 면을 투영하는 대신, 해당 물체에 속한 메쉬 꼭짓점을 추출하고 가장 높은 꼭짓점들의 볼록 껍질을 구합니다. 원형 테이블, L자형 카운터 등 비직사각형 표면을 올바르게 표현합니다.

#### 레이 캐스팅 가시성 필터 (핵심)
표면에 대한 QA를 생성하기 전, 카메라에서 표면 상단 면의 샘플 포인트를 향해 25개의 광선을 쏩니다. 20% 미만의 광선만 표면에 도달하면(벽, 가구 등에 가려진 경우) 해당 표면은 건너뜁니다. 이전 파이프라인에서 벽 너머 책상에도 QA가 생성되던 문제를 해결합니다.

#### 3단계 가시성 검사
1. **카메라 높이** — 카메라가 표면보다 위에 있어야 함
2. **중심점 투영** — 표면 중심점이 이미지 프레임 안에 투영되어야 함
3. **레이 캐스팅** — 표면 샘플 포인트의 20% 이상이 직접 가시여야 함

### 지원 표면 카테고리

```python
SUPPORT_SURFACE_LABELS = {
    'table', 'desk', 'kitchen counter', 'counter',
    'coffee table', 'dining table', 'end table', 'side table',
    'nightstand', 'tv stand', 'bench',
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

# 빠른 모드 (메쉬 정제 생략)
python scripts/scannetpp_indoor_surface_qa.py \
    --data_root /path/to/ScanNetPP/data \
    --out_json annotations/scannetpp/scannetpp_indoor_qa.json \
    --no_mesh_refine
```

### 출력 QA 유형

- `surface_layout` — 보이는 표면 경계 폴리곤 및 픽셀 면적
- `objects_on_surface` — 표면 위에 실제로 놓인 물체 목록

---

## QA 좌표 체계

모든 폴리곤 좌표는 이미지 해상도에 관계없이 **[0, 1000]** 스케일로 정규화됩니다:

```
pixel_x_norm = int(pixel_x / image_width  * 1000)
pixel_y_norm = int(pixel_y / image_height * 1000)
```

Qwen2-VL 등 최신 VLM 토크나이저에서 좌표 토큰화 효율을 최대화하기 위한 설계입니다.

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

## 데이터셋 통계

**GraspNet (Tabletop v2)**

| 소스 | 씬 수 | QA 수 | 특징 |
|---|---|---|---|
| GraspNet 테스트 씬 | 4씬 (0090~0093) | ~4,546건 | 오버헤드 카메라, 근접 탁상 |
| GraspNet 학습 씬 | 32씬 (0000~0031) | ~16,818건 | 오버헤드 카메라, 근접 탁상 |

**ScanNet++ Indoor (v1)**

| 항목 | 수치 |
|---|---|
| 처리된 씬 수 | 50개 |
| 지지 표면이 있는 씬 수 | 약 37개 |
| 표면이 보이는 프레임 수 | 271개 |
| 총 QA 쌍 수 | 1,004개 |
| 감지된 표면 유형 | table, desk, kitchen counter, coffee table, dining table, nightstand |

---

## 관련 연구

- [SceneUpdate](https://github.com/LEESB17/sceneupdate/tree/sungbin) — 이 프로젝트의 전신. 씬그래프 + Depth 기반 빈 공간 통합 (Isaac Sim)
- [RoboSpatial](https://arxiv.org/abs/2411.11537) — 로봇 조작 환경 VLM 공간 이해 (데이터 파이프라인 참조)
- [ConceptGraph](https://github.com/concept-graphs/concept-graphs) — Open-vocabulary 3D 씬그래프
- [SayPlan](https://sayplan.github.io/) — 씬그래프 기반 LLM 태스크 플래닝
