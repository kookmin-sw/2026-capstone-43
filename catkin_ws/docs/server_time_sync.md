# 노트북 시간 서버 / 로봇 동기화 정리

이 문서는 노트북을 로봇 네트워크의 인터넷 공유 서버이자 시간 서버로 쓰는 설정을 정리한다.

## 현재 네트워크 구성

```text
노트북 WiFi: wlp3s0
노트북 유선: eno1 = 192.168.0.100
로봇: 192.168.0.2
ROS master 쪽 장비: 192.168.0.4
```

기본 구조:

```text
인터넷
  -> 노트북 WiFi(wlp3s0)
  -> 노트북 유선(eno1, 192.168.0.100)
  -> 로봇(192.168.0.2)
```

## 노트북 인터넷 공유

노트북에서 IP forwarding을 켠다.

```bash
sudo sysctl -w net.ipv4.ip_forward=1
```

영구 적용하려면:

```bash
echo 'net.ipv4.ip_forward=1' | sudo tee /etc/sysctl.d/99-ip-forward.conf
sudo sysctl --system
```

NAT 설정:

```bash
sudo iptables -t nat -A POSTROUTING -o wlp3s0 -j MASQUERADE
sudo iptables -A FORWARD -i eno1 -o wlp3s0 -j ACCEPT
sudo iptables -A FORWARD -i wlp3s0 -o eno1 -m state --state RELATED,ESTABLISHED -j ACCEPT
```

로봇에서 default gateway가 없으면:

```bash
sudo ip route add default via 192.168.0.100
```

DNS가 안 되면:

```bash
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf
```

확인:

```bash
ping 192.168.0.100
ping 8.8.8.8
ping google.com
```

## 노트북 Chrony 서버

노트북은 `chrony`를 NTP 서버처럼 사용한다.

설치:

```bash
sudo apt update
sudo apt install chrony
```

`/etc/chrony/chrony.conf`에 추가:

```conf
allow 192.168.0.0/24
local stratum 10
```

재시작:

```bash
sudo systemctl restart chrony
sudo systemctl enable chrony
```

상태 확인:

```bash
chronyc tracking
```

## 로봇 시간 맞추기

로봇의 `chrony.service`가 masked 상태라서, 현재는 `ntpdate`로 한 번씩 맞추는 방식이 안전하다.

로봇에서:

```bash
sudo apt update
sudo apt install ntpdate
sudo ntpdate -u 192.168.0.100
date
```

ROS 실행 전에 매번 한 번 실행하는 것을 권장한다.

```bash
sudo ntpdate -u 192.168.0.100
```

주의: `ntpdate`는 시간을 순간적으로 바꾼다. ROS 노드, rosbag, navigation 실행 중간에는 실행하지 말고, 실행 전에 맞춘다.

## SSH 접속

로봇 접속:

```bash
ssh hd@192.168.0.2
```

현재 SSH key 접속은 동작 확인됨.

## ROS 시간 관련 확인

실제 로봇에서는 `/use_sim_time`이 `false`여야 한다.

```bash
rosparam get /use_sim_time
```

필요하면:

```bash
rosparam set /use_sim_time false
```

`uni_navigation_demo.launch`도 실제 로봇에서는 다음 값이어야 한다.

```xml
<param name="/use_sim_time" value="false" />
```

## 확인 명령 모음

노트북 시간:

```bash
date
chronyc tracking
```

로봇 시간:

```bash
ssh hd@192.168.0.2 date
ssh hd@192.168.0.2 "ntpdate -q 192.168.0.100"
```

ROS topic timestamp 확인:

```bash
rostopic echo -n 1 /camera/color/image_raw/header
rostopic echo -n 1 /robot/odom/header
```

## 현재 주의점

`/camera/color/image_raw`는 `192.168.0.2` 시간과 거의 맞는다.

하지만 `/robot/odom` publisher는 `192.168.0.4`의 `/mot_sbl2360_driver`이고, 이 장비의 시간이 약 1350초 정도 미래로 찍히는 현상이 있었다.

```text
/camera/color/image_raw: 현재 시간 근처
/robot/odom: 약 1350초 미래
publisher: /mot_sbl2360_driver on 192.168.0.4
```

`192.168.0.4`에 접속 가능하면 아래를 실행하는 것이 정석이다.

```bash
sudo ntpdate -u 192.168.0.100
```

그 후 `/mot_sbl2360_driver` 또는 해당 bringup을 재시작해야 timestamp가 정상화될 가능성이 높다.

현재는 `192.168.0.4` SSH key/password 문제로 직접 동기화하지 않았다.

## 권장 실행 순서

1. 노트북 유선 IP가 `192.168.0.100`인지 확인한다.

```bash
ip addr show eno1
```

2. 노트북 NAT와 chrony가 켜져 있는지 확인한다.

```bash
sudo sysctl -w net.ipv4.ip_forward=1
chronyc tracking
```

3. 로봇에서 인터넷과 DNS를 확인한다.

```bash
ping 8.8.8.8
ping google.com
```

4. 로봇 시간을 맞춘다.

```bash
sudo ntpdate -u 192.168.0.100
```

5. ROS launch를 실행한다.

```bash
roslaunch uni_navigation uni_navigation_demo.launch
```

