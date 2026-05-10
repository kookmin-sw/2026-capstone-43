# Freespace Detection & Management Algorithm (V2)

본 문서는 로봇이 물체를 안전하게 배치할 수 있는 빈공간(Freespace)을 계산하고 실시간으로 관리하는 시스템의 핵심 알고리즘과 구현 방식을 구체적으로 설명합니다.

현재 시스템은 **오프라인 초기 빈공간 계산(`precompute_freespace.py`)**, **실시간 하이브리드 빈공간 추적(`depth_freespace.py`)**, 그리고 **V2 그리드 기반 안전 배치 스코어링(`surface_analyzer_v2.py`)** 의 세 가지 핵심 파이프라인으로 구성되어 있습니다.

---

## 1. 초기 Freespace 오프라인 계산 파이프라인 (`precompute_freespace.py`)

로봇이 시뮬레이션 환경에 배치되기 전에 ConceptGraph 등에서 수집한 오프라인 데이터(RGB-D 이미지, 카메라 포즈, 초기 Scene Graph)를 활용하여 각 표면(책상, 테이블 등)의 초기 빈공간을 미리 정밀하게 계산합니다. 

### Pass 1: 실제 Surface Z값 감지 (True Surface Z Detection)
Bounding Box의 높이 값은 부정확할 수 있기 때문에, 실제 포인트 클라우드를 분석하여 표면의 정확한 Z(높이) 값을 감지합니다.

1. **Depth Backprojection**: 뎁스 이미지와 카메라 인트린직, 카메라 포즈를 사용하여 3D World 좌표계의 포인트 클라우드로 변환합니다.
2. **ROI 필터링**: 각 Surface의 X, Y Bounding Box 내부에 떨어지는 포인트 중, 대략적인 Z 범위(bbox_z $\pm 0.5m$) 내의 포인트들의 Z값 샘플을 수집합니다.
3. **히스토그램 피크 탐색**: 수집된 Z값들의 1cm 단위 세밀한 히스토그램을 생성합니다.
   - 책상이나 캐비닛의 윗면은 포인트가 가장 조밀하게 모인 **수평 밴드(Peak)** 로 나타납니다.
   - 이 히스토그램의 Peak Z값을 찾고, 주변 가중 평균을 통해 실제 표면의 Z 좌표(`refined_z`)를 정밀하게 교정합니다.

### Pass 2: Occupancy Grid 누적
교정된 정밀 Surface Z값을 바탕으로, 2D Top-down 형태의 Occupancy Grid(2cm 단위 셀)를 구성하고 뎁스 프레임에 걸쳐 누적합니다.

주어진 포인트 클라우드 포인트 $p = (x, y, z) \in P$에 대하여, 해당 표면의 Z좌표를 $z_{surf}$, 마진 파라미터를 각각 $h_{min} = 0.02m$, $h_{max} = 0.30m$라 할 때, 공간을 다음과 같이 분류합니다.

* $P_{surf} = \{ p \in P \mid z_{surf} - 0.10 \le z < z_{surf} + h_{min} \}$ (표면 자체)
* $P_{obj} = \{ p \in P \mid z_{surf} + h_{min} \le z \le z_{surf} + h_{max} \}$ (표면 위 물체)

최소 2프레임 이상 관측된 셀을 유효하다고 판단하며, 물체 영역($P_{obj}$)은 모폴로지 팽창(Dilation) 연산을 통해 안전 마진을 확보합니다. 최종적으로 "시야에 들어왔으며 물체가 없는 공간"을 초기 빈공간으로 산출하여 JSON으로 저장합니다.

---

## 2. 실시간 Freespace 관리 알고리즘 (`depth_freespace.py`)

로봇이 실시간으로 환경을 탐색하며 물체를 집거나 내려놓는 과정에서 동적으로 변하는 빈공간을 관리합니다. **비동기 백그라운드 스레딩**을 적용하여 Isaac Sim의 물리 엔진(UI) 프리징을 완벽히 방지합니다.

### 2.1. 비동기 Multi-frame Sliding Window (Live Depth)
매 프레임 들어오는 Depth 이미지를 분석해 실시간 Occupancy Grid를 갱신합니다. 이 무거운 연산은 메인 스레드를 차단하지 않도록 백그라운드에서 실행됩니다.

1. **Point Cloud 변환 및 ROI 필터링**: 현재 로봇의 위치와 틸트된 카메라 마운트 정보를 반영하여 뎁스 맵을 월드 좌표계로 변환하고, 표면의 X, Y Bounds 및 얇은 높이 대역(-10cm ~ +30cm) 안의 포인트만 자릅니다.
2. **Sliding Window Voting**: 단일 프레임의 노이즈를 제거하기 위해 최근 $N$개(기본 5개)의 프레임을 Sliding Window 큐에 저장하고 투표(Voting)를 진행합니다.
   - **Occupied**: 전체 관측 중 30% 이상이 물체로 판단한 셀
   - **Free**: 전체 관측 중 40% 이상이 빈공간으로 판단하고, Occupied가 아닌 셀

### 2.2. Hybrid 융합 기법 (Depth + Scene Graph)
카메라 시야각(FOV) 한계나 가림(Occlusion)으로 인해 표면 전체를 보지 못하는 문제를 해결합니다.

* **시야 내 영역 ($V_{depth} = 1$)**: Live Depth의 Sliding Window 결과를 100% 신뢰합니다.
* **시야 밖 영역 ($V_{depth} = 0$)**: Scene Graph(SG)의 Bounding Box 정보를 Fallback으로 사용합니다. SG 상에 객체가 있으면 Occupied로 처리하고, 없으면 Persistent Cache(과거 관측 데이터 또는 Precomputed 데이터)를 유지합니다.

---

## 3. V2 그리드 기반 안전 배치 스코어링 (`surface_analyzer_v2.py`) [현재 채택된 방식]

기존의 MER(Maximal Empty Rectangle) 방식 대신, 표면 전체를 2cm 단위의 그리드로 나누어 **모든 가능한 배치 후보군을 샘플링하고 안전도를 평가(Scoring)** 하는 V2 알고리즘을 도입했습니다.

### 3.1. Hard Rejection (배치 불가 조건)
후보 좌표 $(x, y)$에 물체를 놓았을 때, 물체의 Footprint(여유 공간 포함)가 다음 중 하나라도 해당되면 즉시 기각됩니다.
1. 기존 Scene Graph 물체의 Footprint와 겹침
2. 실시간 Depth 기반 Occupancy Grid에서 `Occupied`로 판정된 셀과 겹침 (70% 이상 Free 셀이어야 통과)

### 3.2. 3-Factor Scoring System
통과된 후보 좌표들은 다음 3가지 항목으로 0~1 사이의 점수를 부여받으며, 가중 합산을 통해 최적의 위치를 선정합니다.

$$ Total Score = W_{edge} \cdot S_{edge} + W_{obj} \cdot S_{obj} + W_{eff} \cdot S_{eff} $$

1. **Edge Safety ($S_{edge}$, 가중치 $W_{edge} = 4.0$) - 낙하 방지 (최우선)**
   - 물체의 끝단이 책상 모서리로부터 얼마나 떨어져 있는지를 평가합니다.
   - 모서리에 닿을락 말락 하면 0점, 표면 안쪽으로 깊숙이 들어갈수록 1점에 가까워집니다.
2. **Object Safety ($S_{obj}$, 가중치 $W_{obj} = 3.5$) - 충돌 방지**
   - 표면 위에 이미 존재하는 다른 물체들과의 최소 거리를 평가합니다.
   - 다른 물체와 닿을 위험이 있으면 0점, 안전 거리(약 15cm) 이상 떨어져 있으면 1점을 부여합니다.
3. **Efficiency ($S_{eff}$, 가중치 $W_{eff} = 1.5$) - 공간 효율성**
   - 특정 물체 "옆에(next to)" 놓아야 할 경우, 해당 물체와의 거리가 가까울수록 높은 점수를 줍니다.
   - 기준 물체가 없을 경우, 표면의 정중앙(공간 낭비)이나 극단적인 가장자리를 피하고, 모서리에서 약 30% 들어온 "Sweet Spot"에 배치되도록 유도합니다.

---

## 4. 요약: 로봇 이동에 따른 동적 Freespace Life Cycle

1. **시작 시점 (Initialization)**: 로봇이 아직 카메라로 주변을 확인하지 못했지만, `load_precomputed()` 덕분에 모든 가구의 `surface_z`와 초기 빈공간 그리드가 준비되어 있습니다.
2. **이동 및 탐색 (Exploration)**: 로봇이 이동하며 새로운 표면을 바라보면, 백그라운드 스레드에서 Z값을 정밀 교정하고 Live Depth를 통해 Occupancy Grid를 실시간으로 덮어씌웁니다.
3. **배치 직후 (Placement Update)**: 로봇 팔이 물체를 표면에 내려놓으면 즉시 Scene Graph 상에 위치가 기록되며, Persistent Cache 그리드에 해당 영역이 강제로 `Occupied`로 마킹(`mark_cells_occupied`)됩니다. 이를 통해 다음 프레임 연산을 기다리지 않고도 즉각적으로 빈공간이 차감되어 LLM 컨텍스트와 RViz에 반영됩니다.

---

## 5. 💡 시행착오 및 폐기된 방법 (과거의 알고리즘)

현재의 V2 알고리즘이 도입되기 이전, 초기 버전에서 시도했던 방식과 한계점을 기록합니다.

### 5.1. MER (Maximal Empty Rectangle) 알고리즘 (`surface_analyzer.py`)
초기 시스템에서는 2D 평면에서 가장 면적이 큰 직사각형 빈공간을 찾아내는 수학적 알고리즘(MER)을 사용했습니다.

* **동작 원리**: 장애물(기존 물체)들을 피해 겹치지 않으면서 생성될 수 있는 가장 큰 넓이의 직사각형 영역을 계산하고, 해당 사각형의 중심을 로봇의 내려놓기(Place) 목표 좌표로 설정했습니다.
* **한계 및 폐기 사유 (안전성 문제)**:
  1. **가장자리 위험**: "가장 큰 면적"에만 집중하다 보니, 직사각형의 중심(배치 좌표)이 책상 끄트머리나 모서리에 위치하는 경우가 빈번하게 발생했습니다. 로봇이 물체를 놓을 때 약간의 오차만 있어도 물체가 바닥으로 떨어지는(Drop) 치명적 문제가 있었습니다.
  2. **비현실적인 모양**: 넓이는 넓지만 길고 얇은 틈새 형태의 사각형이 선택될 때, 물체가 안전하게 지지되지 못하는 물리적 불안정성이 존재했습니다.

결론적으로 단순 공간의 '크기'보다 주변 환경(모서리 낙하 방지, 다른 객체와의 충돌 회피)을 고려한 **종합적 안전 점수(Safety Score)** 가 훨씬 더 중요하다는 것을 깨닫고, 현재의 V2 알고리즘(그리드 기반 3-Factor Scoring)으로 완전히 대체되었습니다.
