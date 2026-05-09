# 01_l3das

HM3D 기반 단일 소스 공간음향 데이터셋 생성 파이프라인입니다.  
gLOS/gNLOS, FOV, 방향 레이블(3-way/8-way/front 계열), FOA 오디오 렌더링, manifest 생성을 한 번에 처리합니다.

## 디렉토리 구성

- `hm3d_l3das23_single_mic_dataset_gen/configs`: 생성 시나리오별 YAML 설정
- `hm3d_l3das23_single_mic_dataset_gen/src/hm3d_l3das23_single_mic`: 코어 생성 로직
- `hm3d_l3das23_single_mic_dataset_gen/scripts`: 대량 생성/점검용 유틸 스크립트
- `hm3d_l3das23_single_mic_dataset_gen/tests`: manifest/FOA remap 회귀 테스트

## 주요 엔트리포인트

- CLI: `hm3d-l3das23-generate`
- 서브커맨드:
  - `build-splits`: scene split 생성
  - `generate`: 샘플 생성
  - `qc`: 기존 결과에서 QC 리포트 생성
  - `dump-config`: 해석된 설정 덤프

## 빠른 시작

```bash
cd 01_l3das/hm3d_l3das23_single_mic_dataset_gen
pip install -e ".[dev]"

hm3d-l3das23-generate build-splits \
  --config configs/hm3d_losnlos_100k_balanced.yaml \
  --mode full

hm3d-l3das23-generate generate \
  --config configs/hm3d_losnlos_100k_balanced.yaml \
  --mode full

hm3d-l3das23-generate qc \
  --dataset-root /path/to/generated_dataset
```

## 자주 쓰는 보조 스크립트

- `scripts/run_100k_dual_habitat.sh`: 대규모 생성 실행
- `scripts/run_audio_only_100k_ambix.sh`: 오디오 전용 AmbiX 생성
- `scripts/check_dataset_spatial_consistency.py`: 방향/기하 일관성 점검
- `scripts/update_ambix_direction_labels.py`: AmbiX 방향 라벨 후처리
