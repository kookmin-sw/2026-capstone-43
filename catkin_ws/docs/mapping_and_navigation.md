# Mapping 저장 후 Navigation 실행 정리

이 문서는 `slam_tools/gmapping.launch`로 지도를 만들고, 저장한 지도를 `uni_navigation/uni_navigation_demo.launch`로 사용하는 흐름을 정리한다.

## 기본 개념

지도 만들 때와 navigation 할 때는 역할이 다르다.

```text
지도 만들 때:
gmapping ON
amcl / navigation OFF

저장된 지도에서 주행할 때:
gmapping OFF
map_server + amcl + move_base ON
```

`gmapping`과 `amcl`은 둘 다 `map -> odom` TF를 만들 수 있으므로 동시에 켜지 않는 것이 좋다.

## 사전 준비

로봇 기본 bringup이 먼저 실행되어 있어야 한다.

필요한 topic/TF:

```text
/robot/scan
/robot/odom
odom -> base_link TF
base_link -> sensor frames
```

시간은 ROS 실행 전에 맞춘다.

```bash
sudo ntpdate -u 192.168.0.100
```

실제 로봇에서는 `/use_sim_time`이 `false`여야 한다.

```bash
rosparam set /use_sim_time false
rosparam get /use_sim_time
```

## 1. SLAM 실행

로봇 bringup이 켜진 상태에서 `gmapping`을 실행한다.

```bash
roslaunch slam_tools gmapping.launch
```

`slam_tools/launch/gmapping.launch`의 기본 topic:

```text
scan: /robot/scan
odom: /robot/odom
map frame: map
odom frame: odom
base frame: base_link
```

RViz에서 `/map`이 만들어지는지 확인한다.

## 2. 로봇을 움직이며 지도 만들기

로봇을 천천히 움직인다.

권장:

```text
천천히 직진/회전
같은 공간을 여러 방향에서 훑기
급회전/급가속 피하기
벽, 모서리, 특징점이 잘 보이게 움직이기
```

지도 품질 확인:

```text
벽이 겹쳐 보이지 않는지
복도가 휘거나 벌어지지 않는지
로봇 위치가 지도 위에서 크게 튀지 않는지
```

## 3. 지도 저장

지도 모양이 괜찮으면 `map_saver`로 저장한다.

주의: 저장할 때 실행하는 것은 `map_server`가 아니라 `map_saver`이다.

```text
맞음: rosrun map_server map_saver -f hd_world
틀림: rosrun map_server map_server -f hd_world
```

`map_server could not open -f`가 뜨면 `map_server`를 실행한 것이다. `map_server`는 `-f` 옵션을 모르기 때문에 `-f`라는 map 파일을 열려고 하다가 실패한다.

현재 navigation launch의 기본 map 파일은:

```text
$(find uni_navigation)/maps/hd_world.yaml
```

따라서 기본 이름으로 덮어 저장하려면:

```bash
cd ~/catkin_ws/src/uni_navigation/maps
rosrun map_server map_saver -f hd_world
```

생성되는 파일:

```text
~/catkin_ws/src/uni_navigation/maps/hd_world.pgm
~/catkin_ws/src/uni_navigation/maps/hd_world.yaml
```

기존 지도를 보존하고 새 이름으로 저장하려면:

```bash
cd ~/catkin_ws/src/uni_navigation/maps
rosrun map_server map_saver -f hd_world_v2
```

이 경우 navigation launch에서 `map_file`을 새 yaml로 넘겨야 한다.

```bash
roslaunch uni_navigation uni_navigation_demo.launch \
  map_file:=~/catkin_ws/src/uni_navigation/maps/hd_world_v2.yaml
```

단, `$(find ...)` 경로를 쓰는 것이 더 안정적이다.

`-f` 오류가 뜨면 먼저 실제 사용법을 확인한다.

```bash
rosrun map_server map_saver -h
```

Noetic 기준 사용법:

```text
map_saver [--occ <threshold_occupied>] [--free <threshold_free>] [-f <mapname>] [ROS remapping args]
```

`-f`는 파일명이 아니라 확장자 없는 map 이름을 받는다. 예를 들어 `-f hd_world`를 주면 `hd_world.pgm`, `hd_world.yaml` 두 파일이 생긴다.

저장 전에 `/map` topic이 실제로 나오는지도 확인한다.

```bash
rostopic echo -n 1 /map/header
```

`/map`이 안 나오면 `gmapping`이 아직 map을 publish하지 않는 상태라서 `map_saver`가 실패하거나 기다릴 수 있다.

또한 `roslaunch` argument와 달리 일반 shell에서는 `~`가 보통 확장되지만, 경로 문제가 헷갈릴 수 있으므로 위처럼 `cd ~/catkin_ws/src/uni_navigation/maps` 후 `-f hd_world`로 저장하는 방식을 권장한다.

## 4. Gmapping 종료

지도 저장 후 `gmapping`은 종료한다.

`gmapping` 터미널에서:

```text
Ctrl + C
```

남아 있으면:

```bash
rosnode kill /slam_gmapping
```

확인:

```bash
rosnode list | grep slam
```

아무것도 나오지 않으면 된다.

## 5. Navigation 실행

SLAM을 끈 뒤 navigation을 실행한다.

```bash
sudo ntpdate -u 192.168.0.100
roslaunch uni_navigation uni_navigation_demo.launch
```

`uni_navigation_demo.launch`가 실행하는 노드:

```text
/map_server
/amcl
/move_base
/rviz
```

실제 로봇에서는 launch 파일에 아래 값이 있어야 한다.

```xml
<param name="/use_sim_time" value="false" />
```

## 6. RViz에서 초기 위치 지정

Navigation을 켠 뒤 RViz에서 `2D Pose Estimate`로 로봇 위치를 찍는다.

순서:

```text
1. 지도 위의 실제 로봇 위치 클릭
2. 로봇이 바라보는 방향으로 드래그
3. AMCL particle cloud가 로봇 주변으로 모이는지 확인
```

초기 위치가 틀리면 global path는 보여도 실제 주행이 이상해질 수 있다.

## 7. 목표점 지정

RViz에서 `2D Nav Goal`로 목표 위치를 찍는다.

정상일 때:

```text
global path가 표시됨
local plan이 표시됨
/cmd_vel로 속도 명령이 나감
로봇이 주행함
```

## Cmd_vel 연결 확인

현재 로봇 드라이버는 `/cmd_vel`을 받는다.

따라서 `move_base.launch`의 remap은 아래처럼 되어 있어야 한다.

```xml
<remap from="cmd_vel" to="/cmd_vel" />
```

확인:

```bash
rostopic info /cmd_vel
```

Navigation goal을 준 상태에서 `/cmd_vel`에 값이 나오는지 확인:

```bash
rostopic echo /cmd_vel
```

예상:

```text
linear.x 또는 angular.z 값이 0이 아닌 값으로 나옴
```

만약 `/robot/move_base/cmd_vel`에는 값이 나오는데 로봇이 안 움직이면 remap이 잘못된 것이다.

## 자주 나는 문제

### RViz에는 경로가 나오는데 로봇이 안 움직임

확인:

```bash
rostopic info /cmd_vel
rostopic info /robot/move_base/cmd_vel
rostopic echo /cmd_vel
```

원인 후보:

```text
move_base cmd_vel remap 오류
base driver가 다른 topic을 구독
software runstop / joystick priority / safety lock
local costmap이 obstacle로 막힘
AMCL 초기 위치가 틀림
```

현재 확인된 수정:

```xml
<remap from="cmd_vel" to="/cmd_vel" />
```

### TF_OLD_DATA 또는 transform too old

확인:

```bash
rosparam get /use_sim_time
rostopic echo -n 1 /robot/odom/header
rostopic echo -n 1 /camera/color/image_raw/header
```

실제 로봇이면:

```bash
rosparam set /use_sim_time false
```

ROS 실행 전에 시간 동기화:

```bash
sudo ntpdate -u 192.168.0.100
```

주의: `/robot/odom` publisher가 `192.168.0.4`의 `/mot_sbl2360_driver`라서, 그 장비 시간이 틀어지면 odom timestamp가 미래로 찍힐 수 있다.

### Navigation 켰는데 AMCL이 못 잡음

확인:

```bash
rostopic echo -n 1 /robot/scan
rostopic echo -n 1 /map
rosrun tf tf_echo map base_link
```

조치:

```text
2D Pose Estimate를 다시 찍기
로봇을 제자리에서 천천히 회전
지도와 실제 위치가 맞는지 확인
```

## 추천 전체 플로우

지도 새로 만들 때:

```bash
sudo ntpdate -u 192.168.0.100
roslaunch slam_tools gmapping.launch
```

로봇을 움직여 지도 작성 후:

```bash
cd ~/catkin_ws/src/uni_navigation/maps
rosrun map_server map_saver -f hd_world
```

`gmapping` 종료:

```text
Ctrl + C
```

Navigation:

```bash
sudo ntpdate -u 192.168.0.100
roslaunch uni_navigation uni_navigation_demo.launch
```

RViz:

```text
2D Pose Estimate
2D Nav Goal
```
