# 02_pipeline

데이터셋 샘플을 RGB/Depth/PointCloud/FOA 신호 기준으로 진단 시각화하는 파이프라인입니다.  
한 샘플에서 geometry와 audio direction 추정을 함께 확인할 수 있도록 다중 출력 이미지를 생성합니다.

## 구성

- `run_pipeline.py`: 전체 시각화 파이프라인 실행
- `run_audio_only_maps.py`: 오디오 방향맵만 빠르게 생성
- `output01_*` ~ `output15_*`: 단계별 시각화 출력 모듈
- `collect_overviews.py`, `collect_output15_gifs.py`: 배치 결과 취합

## 주요 출력

- `01_rgb_gt.png`, `02_depth.png`
- `03_raw_pointcloud.png`, `04_raw_overlay.png`
- `05_intensity_map.png`, `06_beam_map.png`
- `12_beam_filtered_overlay.png`
- `14_overview.png`
- (옵션) `15_windowed_beam` GIF 세트

## 실행 예시

```bash
cd 02_pipeline

python run_pipeline.py \
  --dataset-root /path/to/dataset_root \
  --manifest val \
  --limit 10 \
  --output-root /path/to/output_dir \
  --foa-channel-order WYZX \
  --enable-output15
```

오디오 맵만 확인할 때:

```bash
python run_audio_only_maps.py \
  --sample-audio /path/to/sample.wav \
  --output-root /path/to/output_dir \
  --channel-order WYZX
```
