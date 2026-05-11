# Piper RGB-D + Camera Pose 수집

이 문서는 Piper에 달린 RealSense RGB-D 카메라에서 aligned RGB/depth image와 camera pose를 함께 저장하는 방법을 정리한다.

목표는 online / incremental Gaussian Splatting SLAM의 입력으로 사용할 수 있는 frame stream을 만드는 것이다.

## 현재 수집 방식

수집 노드:

```text
uni_navigation/scripts/collect_rgb_pose.py
```

Launch:

```text
uni_navigation/launch/collect_rgb_pose.launch
```

기본 입력:

```text
RGB image: /camera/color/image_raw
aligned depth image: /camera/aligned_depth_to_color/image_raw
camera info: /camera/color/camera_info
odom: /robot/odom
camera frame: camera_color_optical_frame
base frame: base_link
```

기본 저장 속도는 약 3Hz이다.

```text
min_interval = 0.333333 sec
```

현재 `/robot/odom` timestamp가 카메라 timestamp와 맞지 않는 문제가 있어서, 기본 pose source는 다음 방식이다.

```text
latest /robot/odom pose
+ base_link -> camera_color_optical_frame TF
= odom 기준 camera pose
```

즉 정확한 timestamp sync 방식은 아니지만, 로봇/팔을 천천히 움직이며 online GS용 frame을 쌓는 목적에는 사용할 수 있다.

## 실행

권장 구조는 로봇에서 image topic을 3Hz로 throttle하고, 노트북 `192.168.0.100`에서 collector를 실행하는 방식이다.

```text
robot 192.168.0.2:
  /camera/color/image_raw -> /camera/color/image_raw_3hz

laptop 192.168.0.100:
  /camera/color/image_raw_3hz 구독
  RGB + pose를 노트북 디스크에 저장
```

만약 로봇 디스크에 직접 저장하고 싶다면 로봇 ROS 환경에서:

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://192.168.0.4:11311
export ROS_HOSTNAME=192.168.0.2
```

로봇 저장 수집 시작:

```bash
roslaunch uni_navigation collect_rgb_pose.launch output_dir:=/home/hd/rgb_pose_dataset_01
```

`~` 경로가 roslaunch argument에서 헷갈릴 수 있으므로, 확실하게 절대 경로를 권장한다.

종료:

```text
Ctrl + C
```

## 노트북 GPU에서 수집하기

로봇에는 GPU가 없으므로, 실제 online GS 학습은 노트북에서 실행하는 것을 권장한다.

### 1. 로봇에서 3Hz RGB + aligned depth throttle

로봇에서:

```bash
cd ~/catkin_ws
THROTTLE_DEPTH=True ./scripts/start_robot_image_throttle.sh
```

직접 실행하려면:

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://192.168.0.4:11311
export ROS_HOSTNAME=192.168.0.2

rosrun topic_tools throttle messages \
  /camera/color/image_raw 3.0 /camera/color/image_raw_3hz

rosrun topic_tools throttle messages \
  /camera/aligned_depth_to_color/image_raw 3.0 \
  /camera/aligned_depth_to_color/image_raw_3hz
```

확인:

```bash
rostopic hz /camera/color/image_raw_3hz
rostopic hz /camera/aligned_depth_to_color/image_raw_3hz
```

### 2. 노트북에서 RGB-D + pose 저장

노트북 `192.168.0.100`에서:

```bash
cd ~/catkin_ws
OUTPUT_DIR=/home/harudev/rgb_pose_dataset_01 \
SAVE_DEPTH=True \
./scripts/start_laptop_rgb_pose_collection.sh
```

직접 실행하려면:

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://192.168.0.4:11311
export ROS_IP=192.168.0.100
unset ROS_HOSTNAME

roslaunch uni_navigation collect_rgb_pose.launch \
  image_topic:=/camera/color/image_raw_3hz \
  depth_topic:=/camera/aligned_depth_to_color/image_raw_3hz \
  output_dir:=/home/harudev/rgb_pose_dataset_01 \
  save_depth:=true \
  min_interval:=0.333333 \
  min_translation:=0.0 \
  min_rotation:=0.0
```

이렇게 하면 dataset이 로봇이 아니라 노트북에 저장된다.

```text
/home/harudev/rgb_pose_dataset_01/
  rgb/
  depth/
  poses.csv
  poses_tum.txt
  camera_info.json
  metadata.json
```

## 저장 결과

```text
/home/harudev/rgb_pose_dataset_01/
  rgb/
    000000.png
    000001.png
    000002.png
  depth/
    000000.png
    000001.png
    000002.png
  poses.csv
  poses_tum.txt
  camera_info.json
  metadata.json
```

`poses.csv`:

```text
filename,depth_filename,image_stamp,depth_stamp,tf_stamp,tx,ty,tz,qx,qy,qz,qw
```

`poses_tum.txt`:

```text
# timestamp tx ty tz qx qy qz qw
```

`camera_info.json`:

```text
width, height, fx, fy, cx, cy, distortion_model, distortion_coefficients
```

## 3Hz 수집 설정

기본값이 3Hz지만 명시하려면:

```bash
roslaunch uni_navigation collect_rgb_pose.launch \
  output_dir:=/home/harudev/rgb_pose_dataset_01 \
  min_interval:=0.333333 \
  min_translation:=0.0 \
  min_rotation:=0.0
```

이 설정은 움직임이 거의 없어도 0.333초마다 저장한다.

데이터를 더 줄이고 싶으면 움직임 threshold를 넣는다.

```bash
roslaunch uni_navigation collect_rgb_pose.launch \
  output_dir:=/home/harudev/rgb_pose_dataset_01 \
  min_interval:=0.333333 \
  min_translation:=0.03 \
  min_rotation:=0.03
```

## Nerfstudio / Splatfacto 변환

변환 스크립트:

```text
uni_navigation/scripts/rgb_pose_to_nerfstudio.py
```

실행:

```bash
rosrun uni_navigation rgb_pose_to_nerfstudio.py /home/harudev/rgb_pose_dataset_01
```

생성:

```text
/home/harudev/rgb_pose_dataset_01/transforms.json
```

aligned depth를 사용해 COLMAP 없이 초기 3D point cloud도 같이 만들려면:

```bash
cd ~/catkin_ws
DATA_DIR=/home/harudev/rgb_pose_dataset_01 \
KEEP_EVERY=3 \
INCLUDE_DEPTH=True \
GENERATE_POINT_CLOUD=True \
POINT_STRIDE=6 \
VOXEL_SIZE=0.01 \
./scripts/export_nerfstudio_dataset.sh
```

생성:

```text
/home/harudev/rgb_pose_dataset_01/transforms.json
/home/harudev/rgb_pose_dataset_01/sparse_pc.ply
```

`sparse_pc.ply`는 depth image를 camera intrinsics로 back-project하고, 저장된 camera pose로 world 좌표에 누적한 point cloud다. Nerfstudio `splatfacto`는 `transforms.json` 안의 `ply_file_path`를 보고 이 point cloud를 COLMAP sparse point 대체 초기값으로 사용할 수 있다.

시각화:

```bash
python3 scripts/view_depth_point_cloud.py \
  /home/harudev/rgb_pose_dataset_01/sparse_pc.ply
```

Nerfstudio에서 예시:

```bash
ns-train splatfacto --data /home/harudev/rgb_pose_dataset_01
```

## Online Gaussian Splatting으로 넘길 때의 인터페이스

online GS 쪽에서는 `poses.csv`를 tail하거나, ROS topic에서 직접 subscribe하는 두 가지 방식이 가능하다.

초기 구현 추천:

```text
1. collect_rgb_pose.launch로 3Hz 데이터 저장
2. online trainer가 새 rgb/*.png + poses.csv row를 polling
3. 새 frame이 생기면 Gaussian map update
```

장점:

```text
ROS와 PyTorch 학습 프로세스를 분리
디버깅 쉬움
나중에 rosbag/offline replay와 동일 코드 사용 가능
```

노트북에서 online GS 등록을 바로 실행하려면:

```bash
cd ~/catkin_ws
DATA_DIR=/home/harudev/rgb_pose_dataset_01 \
OUTPUT_DIR=/home/harudev/catkin_ws/outputs/scene01 \
./scripts/start_online_gs_registration.sh
```

추후 직접 ROS subscribe 방식으로 바꿀 수 있도록 online GS의 Frame dataclass는 아래 필드를 갖는 것이 좋다.

```text
rgb image
optional depth
timestamp
intrinsics
initial camera pose
```

## 주의점

현재 `/robot/odom`은 `192.168.0.4`의 `/mot_sbl2360_driver`에서 publish되며 timestamp가 카메라보다 미래로 찍힌 이력이 있다.

따라서 빠른 움직임에서는 image-pose alignment가 밀릴 수 있다.

권장:

```text
천천히 이동/회전
팔 움직임도 급격하게 하지 않기
짧은 sequence로 먼저 테스트
수집 후 transforms.json을 열어 frame 수와 intrinsics 확인
```
