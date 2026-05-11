# Online Gaussian Splatting 등록 파이프라인

이 문서는 Piper 카메라 수집 결과를 online Gaussian Splatting SLAM prototype에 등록하는 방법을 정리한다.

## 전체 구조

```text
ROS collector
  /camera/color/image_raw
  /camera/aligned_depth_to_color/image_raw
  /camera/color/camera_info
  /robot/odom + base_link -> camera TF
    -> rgb/*.png
    -> depth/*.png
    -> poses.csv
    -> camera_info.json

online_gs_slam
  LiveRgbPoseDataset
    -> Frame dataclass
    -> GaussianTracker
    -> OnlineMapper
    -> GaussianInserter
    -> GaussianMap checkpoint
```

## 1. 3Hz RGB-D + Pose 수집 시작

권장 방식은 로봇에서 image topic을 3Hz로 throttle하고, 노트북에서 collector를 실행하는 것이다.

로봇 `192.168.0.2`에서:

```bash
cd ~/catkin_ws
THROTTLE_DEPTH=True ./scripts/start_robot_image_throttle.sh
```

노트북 `192.168.0.100`에서:

```bash
cd ~/catkin_ws
OUTPUT_DIR=/home/harudev/rgb_pose_dataset_01 \
SAVE_DEPTH=True \
./scripts/start_laptop_rgb_pose_collection.sh
```

수동 실행은 아래와 같다.

로봇:

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://192.168.0.4:11311
export ROS_HOSTNAME=192.168.0.2

rosrun topic_tools throttle messages \
  /camera/color/image_raw 3.0 /camera/color/image_raw_3hz
```

노트북:

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://192.168.0.4:11311
export ROS_IP=192.168.0.100
unset ROS_HOSTNAME

roslaunch uni_navigation collect_rgb_pose.launch \
  image_topic:=/camera/color/image_raw_3hz \
  depth_topic:=/camera/aligned_depth_to_color/image_raw_3hz \
  save_depth:=true \
  output_dir:=/home/harudev/rgb_pose_dataset_01
```

기본 설정:

```text
min_interval: 0.333333
min_translation: 0.0
min_rotation: 0.0
```

즉 약 1초에 3장을 저장한다.

## 1.5. COLMAP 없이 Depth 기반 sparse point cloud 만들기

RealSense aligned depth가 있으면 COLMAP/SfM을 돌리지 않고도 Gaussian Splatting 초기 point cloud를 만들 수 있다.

```bash
cd ~/catkin_ws
DATA_DIR=/home/harudev/rgb_pose_dataset_01 \
KEEP_EVERY=3 \
INCLUDE_DEPTH=True \
GENERATE_POINT_CLOUD=True \
POINT_STRIDE=6 \
MAX_POINTS_PER_FRAME=12000 \
MAX_TOTAL_POINTS=1500000 \
VOXEL_SIZE=0.01 \
./scripts/export_nerfstudio_dataset.sh
```

생성 파일:

```text
/home/harudev/rgb_pose_dataset_01/transforms.json
/home/harudev/rgb_pose_dataset_01/sparse_pc.ply
```

동작:

```text
1. aligned depth pixel을 camera intrinsics로 3D back-projection
2. poses.csv의 camera pose로 world frame에 변환
3. RGB image에서 같은 pixel color를 가져와 point color로 저장
4. voxel downsample 후 sparse_pc.ply 생성
5. transforms.json에 ply_file_path를 기록해서 Nerfstudio splatfacto 초기점으로 사용
```

시각화:

```bash
python3 scripts/view_depth_point_cloud.py \
  /home/harudev/rgb_pose_dataset_01/sparse_pc.ply
```

## 2. Online GS 등록 실행

Ubuntu 20.04의 기본 Python은 3.8이므로 최신 PyTorch/typing-extensions가 안 맞을 수 있다. 아래처럼 Python 3.8 호환 버전을 설치한다.

```bash
python3 -m pip install --user --upgrade "pip<25" wheel
python3 -m pip install --user "typing_extensions<4.14"
python3 -m pip install --user torch==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121
python3 -m pip install --user gsplat==1.5.3
python3 scripts/check_gsplat_backend.py
```

주의: `gsplat`은 PyTorch CUDA runtime만으로는 부족하고 CUDA toolkit의 `nvcc`가 필요하다. `gsplat: No CUDA toolkit found. gsplat will be disabled.`가 뜨면 CUDA toolkit 12.1 설치와 `PATH` 설정이 필요하다.

`nvcc --version`이 CUDA 10.1처럼 오래된 버전이면 gsplat 빌드가 실패한다.

```text
nvcc fatal: Unknown option '-generate-dependencies-with-compile'
```

이 경우 CUDA toolkit 12.1을 설치하고 아래 환경변수를 잡는다.

```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$HOME/.local/bin:$PATH
nvcc --version
python3 scripts/check_gsplat_backend.py
```

그 다음 실행:

```bash
python run_online_gs_slam.py \
  --data_dir /home/harudev/rgb_pose_dataset_01 \
  --config configs/online_gs_slam.yaml \
  --output_dir outputs/scene01
```

짧게 테스트:

```bash
python run_online_gs_slam.py \
  --data_dir /home/harudev/rgb_pose_dataset_01 \
  --config configs/online_gs_slam.yaml \
  --output_dir outputs/scene01 \
  --max_frames 30 \
  --idle_timeout 5
```

## 3. 현재 등록 방식

현재 prototype은 gsplat renderer를 아직 붙이지 않은 최소 구조다.

프레임이 새로 들어오면:

```text
1. poses.csv의 새 row 감지
2. rgb/000xxx.png 로드
3. camera_info.json에서 intrinsics 로드
4. ROS optical pose를 OpenGL/Nerfstudio camera-to-world로 변환
5. Frame dataclass 생성
6. Tracker는 초기 pose를 그대로 사용
7. Mapper가 visible Gaussian observation_count/opacity 갱신
8. RGB-only pseudo-depth ray sampling으로 새 Gaussian 삽입
9. gsplat renderer로 현재 Gaussian map 렌더링
10. rendered RGB와 입력 RGB의 L1 loss로 local Gaussian parameter 최적화
11. uncertainty 업데이트
12. checkpoint/trajectory 저장
```

RGB-D depth가 연결되면 `GaussianInserter.propose_from_frame()`에서 depth backprojection으로 바꾸면 된다.

## 4. Output

```text
outputs/scene01/
  checkpoints/
    gaussians_000009.pt
    gaussians_latest.pt
  trajectory.json
  summary.json
  gaussians_latest.ply
  debug/
    compare_latest.png
    compare_000009.png
  high_uncertainty_centers.npy
  uncertainty_debug.png
```

## 5. 시각화

현재 prototype은 `gsplat` 렌더링과 간단한 키보드 viewer를 지원하지만, Nerfstudio만큼 보기 편한 web viewer는 아니다. 빠른 확인은 debug 이미지 또는 PLY로 하고, 편하게 둘러보려면 아래 Nerfstudio/Splatfacto 흐름을 사용한다.

Online GS 실행 중 아래 파일이 계속 갱신된다.

```text
outputs/scene01/gaussians_latest.ply
```

간단히 파일 갱신을 watch:

```bash
PLY_PATH=/home/harudev/catkin_ws/outputs/scene01/gaussians_latest.ply \
./scripts/watch_gaussian_map.sh
```

볼 수 있는 프로그램:

```text
MeshLab
CloudCompare
Open3D
```

Open3D가 있으면:

```bash
python3 -c "import open3d as o3d; p=o3d.io.read_point_cloud('outputs/scene01/gaussians_latest.ply'); o3d.visualization.draw_geometries([p])"
```

이 PLY는 Gaussian의 center와 RGB color만 보여준다. Nerfstudio viewer용 Gaussian splat 파일이 아니라 단순 확인용 point cloud다.

실제 Gaussian Splatting 렌더 결과는 debug comparison 이미지로 확인한다.

```text
outputs/scene01/debug/compare_latest.png
```

이미지는 좌에서 우로:

```text
input RGB | gsplat rendered RGB | absolute error
```

간단한 `gsplat` checkpoint viewer:

```bash
python3 scripts/view_gsplat_checkpoint.py \
  --output_dir outputs/scene01 \
  --data_dir /home/harudev/rgb_pose_dataset_01 \
  --checkpoint outputs/scene01/checkpoints/gaussians_001939.pt
```

## 6. Nerfstudio / Splatfacto로 보기

직접 만든 viewer가 보기 불편하면 수집한 `rgb + pose`를 Nerfstudio dataset으로 변환해서 `splatfacto`를 돌리는 것이 가장 편하다.

변환:

```bash
cd ~/catkin_ws
DATA_DIR=/home/harudev/rgb_pose_dataset_01 \
KEEP_EVERY=5 \
INCLUDE_DEPTH=True \
GENERATE_POINT_CLOUD=True \
./scripts/export_nerfstudio_dataset.sh
```

생성 파일:

```text
/home/harudev/rgb_pose_dataset_01/transforms.json
/home/harudev/rgb_pose_dataset_01/sparse_pc.ply
```

Nerfstudio 설치 후 학습:

```bash
DATA_DIR=/home/harudev/rgb_pose_dataset_01 \
KEEP_EVERY=5 \
MAX_NUM_ITERATIONS=7000 \
./scripts/train_nerfstudio_splatfacto.sh
```

`ns-train splatfacto`는 Nerfstudio web viewer 주소를 터미널에 출력한다. 브라우저에서 그 주소를 열면 마우스로 카메라를 조종하면서 Gaussian Splatting 결과를 볼 수 있다.

주의:

- 현재 repo의 `gaussians_latest.ply`는 Nerfstudio splat viewer용 파일이 아니다.
- Nerfstudio에는 `rgb/` 이미지와 `transforms.json`을 넘겨서 `splatfacto`가 다시 학습하게 하는 흐름이 더 안정적이다.
- 6739프레임 전체를 바로 넣으면 오래 걸릴 수 있으니 처음에는 `KEEP_EVERY=5` 또는 `KEEP_EVERY=10`으로 시작한다.

확인:

```bash
xdg-open outputs/scene01/debug/compare_latest.png
```

카메라를 직접 움직이면서 실제 gsplat 렌더를 보고 싶으면:

```bash
python3 scripts/view_gsplat_checkpoint.py \
  --output_dir outputs/scene01 \
  --data_dir /home/harudev/rgb_pose_dataset_01
```

키:

```text
W/S: forward/back
A/D: left/right
Q/E: down/up
Arrow keys: yaw/pitch
R: reset pose
P: save snapshot
ESC: quit
```

한 장만 렌더해서 저장:

```bash
python3 scripts/view_gsplat_checkpoint.py \
  --output_dir outputs/scene01 \
  --data_dir /home/harudev/rgb_pose_dataset_01 \
  --once
```

## 6. 다음 구현 단계

우선순위:

```text
1. gsplat backend renderer 구현
2. rendered RGB vs frame RGB residual 계산
3. SE(3) pose refinement loop 구현
4. residual/high-coverage 기반 Gaussian insertion
5. local keyframe window optimization
6. uncertainty-guided exploration target 출력
7. tactile embedding update 연결
```

Renderer만 `NullGaussianRenderer`에서 gsplat wrapper로 교체하면 tracking/mapping 코드는 최대한 유지되도록 설계했다.
