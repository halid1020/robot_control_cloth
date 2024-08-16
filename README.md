
# ROS 2 Usage

## Install
Install `ROS2 Humble`

```
sudo apt install ros-humble-moveit
```

Instlal 

```
sudo apt install ros-humble-ur-robot-driver ros-humble-ur-description ros-humble-ur-moveit-config
```

Install `moveit` python api for humble
```
source /opt/ros/humble/setup.bash
echo "deb [trusted=yes] https://raw.githubusercontent.com/moveit/moveit2_packages/jammy-humble/ ./" | sudo tee /etc/apt/sources.list.d/moveit_moveit2_packages.list
echo "yaml https://raw.githubusercontent.com/moveit/moveit2_packages/jammy-humble/local.yaml humble" | sudo tee /etc/ros/rosdep/sources.list.d/1-moveit_moveit2_packages.list

apt-get update 
Install ros-humble-moveit-py 
apt-get upgrade 
``

## Install Pacakages
```
pip install dotmap
```



## Compile

Inside the workspace
```
source /opt/ros/humble/setup.bash
cd <workspace>/src
colcon build --packages-select robot_control_cloth
source install/setup.sh 
```

## Run

```
ros2 run robot_control_cloth quasi_static_pick_and_place.py
```


1. test ur3e with moveit
```
ros2 launch robot_control_cloth ur3e_robot_moveit_launch.py ur_type:=ur3e > out.txt
```

2. test active gripper
```
ros2 run robot_control_cloth active_gripper_control.py
```

4. test whole system
```
ros2 launch robot_control_cloth ur3e_robot_moveit_launch.py ur_type:=ur3e executable:=test_whole_system.py
```