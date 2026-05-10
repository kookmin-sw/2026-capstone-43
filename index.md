# 로봇을 위한 실세계 강인한 멀티모달 시공간 인지 기반의 전역적 동적 환경 인식 원천기술 개발

> 본 프로젝트는 시각, 청각, 촉각 및 계층적 기억(Memory)을 융합하여 로봇이 복잡하고 동적인 실세계 환경을 전역적으로 이해하고, 이를 바탕으로 지능적인 행동을 수행하도록 돕는 인지 중심의 원천기술을 다룹니다.

---

## 📌 Project Overview Q&A

**1) 이 기술은 누가 사용하는 건가요?**
* 자율주행 서비스 로봇 및 모바일 매니퓰레이터를 개발하는 로보틱스 엔지니어와 멀티모달 AI 연구자들이 사용합니다.

**2) 이 기술은 언제 사용하는 건가요?**
* 로봇 자체 소음이 크거나, 물체가 실시간으로 이동하고, 가려진 공간이 존재하는 실제 가정 및 산업 현장에서 로봇의 안정적인 임무 수행이 필요할 때 사용합니다.

**3) 이 기술을 사용하면 무엇이 해결(개선)되나요?**
* **소음 극복**: 사족보행 로봇의 구동 소음 속에서도 깨끗한 음성 명령을 추출합니다.
* **입체적 인지**: 단순히 보는 것을 넘어 소리의 위치와 공간의 기하학적 구조를 결합해 이해합니다.
* **기억의 고도화**: 한 번 본 환경을 계층적으로 저장하고, 변화하는 물체 상태를 실시간으로 씬그래프에 반영합니다.
* **안전한 행동**: '어디에 물건을 놓아야 가장 안전한지'와 같은 고차원적인 판단을 실시간 빈공간 분석을 통해 수행합니다.

**4) 이 기술이 되면 사람들이 사용할까요?**
* 통제되지 않은 실환경(In-the-wild)에서 로봇의 자율성을 비약적으로 높여주기 때문에, 차세대 가전 및 물류 로봇 시장에서 핵심 솔루션으로 활용될 것입니다.

---

## 👥 Team Profiles

| 이름 | 역할 | 담당 컴포넌트 | 주요 기술 스택 |
| :---: | :---: | :---: | :--- |
| **동현** | 인지 (Audio) | **SpatialAudio** | FOA, SpatialAST, V_sphere/A_sphere |
| **재우** | 인지 (Denoising) | **Denoising** | SGMSE, Diffusion, Aux Condition (Foot Force) |
| **채희** | 인지 (Memory) | **Memory** | Hierarchical Scene Graph, Dynamic Update |
| **근서** | 인지 (Tactile/SLAM) | **Tactile** | Gaussian Splatting, Online SLAM, Material Embedding |
| **성빈** | 추론 & 액션 | **Planning** | LLM Planner (PRED), Freespace V2, Isaac Sim |

---

## 🧩 Components Detail

### 🎧 1. SpatialAudio (동현)
로봇 관점의 기하학적 맥락과 앰비소닉(Ambisonics) 오디오를 결합한 공간 음향 인지 시스템입니다.
* **주요 기능**: FOA(First-Order Ambisonics) 기반 음원 위치 추적 및 SpatialAST 모델을 통한 성능 고도화를 수행합니다.
* **차별점**: 시각 정보(`V_sphere`)와 오디오 정보(`A_sphere`)를 정합하여 가시 영역뿐 아니라 비가시 영역의 소리까지 통합적으로 인지합니다.

### 🔇 2. Denoising (재우)
로봇 구동 시 발생하는 강력한 하드웨어 노이즈를 제거하는 로봇 특화 음성 향상 기술입니다.
* **주요 기능**: SGMSE 및 RDDM 기반의 Diffusion 모델을 사용하여 깨끗한 음성을 복원합니다.
* **차별점**: 로봇의 다리 관절 힘(Foot force) 데이터를 보조 조건(Auxiliary Condition)으로 입력받아, 구동 타이밍에 맞춘 정밀한 소음 제거가 가능합니다.

### 🧠 3. Memory (채희)
환경을 '건물-층-방-객체'로 구조화하여 관리하는 계층형 씬그래프 메모리 시스템입니다.
* **주요 기능**: ConceptGraph를 활용한 초기 맵 생성 및 로봇의 활동 중 발생하는 객체 변화를 실시간으로 그래프에 반영합니다.
* **차별점**: 장단기 메모리 구조를 통해 로봇이 과거의 경험을 바탕으로 현재 작업(Rearrangement 등)을 효율적으로 계획하도록 돕습니다.

### 🤲 4. Tactile (근서)
시각 정보와 촉각/재질 정보를 융합하기 위한 3D Gaussian Splatting 기반 SLAM 시스템입니다.
* **주요 기능**: 로봇 주행 중 수집된 RGB-Pose 데이터를 바탕으로 3D 가우시안 맵을 생성하고 업데이트합니다.
* **차별점**: 향후 가우시안 맵에 재질(Material) 정보를 임베딩하여, 로봇이 물체의 시각적 형태뿐만 아니라 만졌을 때의 질감까지 이해하도록 확장 중입니다.

### 🤖 5. Planning (성빈)
LLM과 실시간 씬그래프를 결합하여 복잡한 명령을 수행하는 로봇 작업 지능 모듈입니다.
* **주요 기능**: Gemini/Llama 등의 LLM을 백엔드로 활용하여 태스크 플래닝을 수행하고, VFH/ROS2 기반 네비게이션을 제어합니다.
* **차별점**: **V2 빈공간 스코어링** 알고리즘을 통해 낙하 위험과 충돌 위험을 고려한 최적의 배치 지점(Sweet Spot)을 실시간으로 계산합니다.

---

## 🎬 DEMO (팀원별 결과 및 시각 자료)

### 🎧 1. SpatialAudio (동현)
> 앰비소닉(Ambisonics) 오디오와 기하학적 맥락을 결합한 시공간 음향 인지 결과입니다.
* **오디오 인지 파이프라인 데모**
  * (여기에 동현 님의 데모 영상 링크나 결과물 이미지를 추가해 주세요. 예: `[시연 영상 보기](링크)`)

<br>

### 🔇 2. Denoising (재우)
> 사족보행 로봇의 구동 소음 환경에서 음성 명령만을 깨끗하게 추출한 결과입니다.
* **음성 향상(Speech Enhancement) 전/후 비교**
  * (여기에 재우 님의 오디오 샘플 링크나 디노이징 전후 스펙트로그램 이미지를 추가해 주세요.)

<br>

### 🧠 3. Memory (채희)
> 계층형 씬그래프 구축 및 장단기 메모리를 활용한 로봇의 동적 환경 인지 결과입니다.

#### 방 분리 결과 이미지
<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; align-items: center;">
  <img src="https://github.com/user-attachments/assets/7df81276-5b42-450e-80fd-db56d06c4672" style="width: 100%;">
  <img src="https://github.com/user-attachments/assets/fd9f7897-26d1-40f4-9ea7-f96aff4e29f4" style="width: 100%;">
  <img src="https://github.com/user-attachments/assets/4c4a1582-b464-4bd6-9245-c3db899da223" style="width: 100%;">
  <img src="https://github.com/user-attachments/assets/aee2a039-f5be-4c46-8c86-43087035b166" style="width: 100%;">
  <img src="https://github.com/user-attachments/assets/a05f5b52-b137-4c49-9282-a24b14fc4c00" style="width: 100%;">
  <img src="https://github.com/user-attachments/assets/7d870580-b98e-4b76-a55c-707be50fa546" style="width: 100%;">
  <img src="https://github.com/user-attachments/assets/0a615ba2-0ebe-43a9-80dc-ffe3f911b23b" style="width: 100%;">
  <img src="https://github.com/user-attachments/assets/72bc62a4-aeca-4b1d-badf-0bf826dbaf57" style="width: 100%;">
</div>

#### Task 수행 동영상
https://youtu.be/n9BuyLzmbgs

<br>

### 🤲 4. Tactile (근서)
> RGB-Pose 데이터를 바탕으로 실시간 3D 환경을 렌더링한 Continual Gaussian Splatting 매핑 결과입니다.
* **3D Gaussian Splatting 실시간 매핑 데모**
  * (여기에 근서 님의 SLAM 시각화 이미지나 렌더링 결과 영상 링크를 추가해 주세요.)

<br>

### 🤖 5. Planning (성빈)
> LLM과 실시간 씬그래프, 그리고 V2 빈공간 스코어링을 활용하여 로봇이 작업을 계획하고 주행하는 결과입니다.
* **모바일 매니퓰레이터 태스크 플래닝 및 주행 데모**
  * (여기에 성빈 님의 Isaac Sim 시뮬레이션 주행 영상이나 플래닝 결과 화면을 추가해 주세요.)

---
