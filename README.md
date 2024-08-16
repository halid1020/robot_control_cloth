
# Robot Control Cloth
Halid Abdulrahim Kadi and Kasim Terzic.

## I. Installation

1. Install `ROS2 Humble`

```
sudo apt install ros-humble-moveit
```

2. Install relative packages for `Univeral Robots` 

```
sudo apt install ros-humble-ur-robot-driver ros-humble-ur-description ros-humble-ur-moveit-config
```

3. Install `moveit` python api for humble
```
source /opt/ros/humble/setup.bash
echo "deb [trusted=yes] https://raw.githubusercontent.com/moveit/moveit2_packages/jammy-humble/ ./" | sudo tee /etc/apt/sources.list.d/moveit_moveit2_packages.list
echo "yaml https://raw.githubusercontent.com/moveit/moveit2_packages/jammy-humble/local.yaml humble" | sudo tee /etc/ros/rosdep/sources.list.d/1-moveit_moveit2_packages.list

apt-get update 
Install ros-humble-moveit-py 
apt-get upgrade 
```

4. Install Need Python Pacakages
```
pip install dotmap
```

5. Build a workspace
```
mkdir <path-to-ws>/<ws_name>
cd <path-to-ws>/<ws_name>
mkdir src
git clone <this repo>
```

6. Right under `<ws_name>`
```
source /opt/ros/humble/setup.bash
colcon build --packages-select robot_control_cloth
```

## III. Run Quasi-Static Pick-and-Place Robot Control Program

1. Lauch UR driver
```
source /opt/ros/humble/setup.bash

ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur3e robot_ip:=192.168.1.15 launch_rviz:=false
```

2. Run the `URCaps` program in pedant, and set to `remote control`.

3. Right under `<ws_name>`
```
source /opt/ros/humble/setup.bash
source install/setup.sh

ros2 launch robot_control_cloth \
    ur_robot_moveit_executable_launch.py \
    ur_type:=ur3e \
    executable:=quasi_static_pick_and_place.py \
    planner:=ompl
```

4. Run either of the following option to generate pick-and-place policy
    
    a. Human interace
    ```
    source /opt/ros/humble/setup.bash
    source install/setup.sh
    
    ros2 run robot_control_cloth human_interface.py
    ```

    b. Following the [`agent-arena`'s ROS2 `Humble` setup]() for autimatically generate pick-and-place policy.


## IV. Test Individual Components

1. Test ur3e robot with moveit
```
source /opt/ros/humble/setup.bash
source install/setup.sh

ros2 launch robot_control_cloth ur_robot_moveit_executable_launch.py ur_type:=ur3e executable:ur3e_robot_moveit.py
```

2. Test active gripper
```
source /opt/ros/humble/setup.bash
source install/setup.sh

ros2 run robot_control_cloth active_gripper_control.py
```

4. Test whole system
```
source /opt/ros/humble/setup.bash
source install/setup.sh

ros2 launch robot_control_cloth ur_robot_moveit_executable_launch.py ur_type:=ur3e executable:=test_whole_system.py
```