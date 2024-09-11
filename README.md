
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
mkdir rcc_build
cd rcc_build
colcon build --base-paths ../
```

## III. Run Quasi-Static Pick-and-Place Robot Control Program

1. Lauch UR driver
```
source /opt/ros/humble/setup.sh

ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur3e robot_ip:=192.168.1.15 launch_rviz:=false
```

2. Run the `URCaps` program in pedant, and set to `remote control`.

3. Right under `<ws_name>`
```
source /opt/ros/humble/setup.bash
source ./rcc_build/install/setup.sh

ros2 launch rcc_qs_pnp \
    ur_robot_moveit_executable_launch.py \
    ur_type:=ur3e \
    executable:=quasi_static_pick_and_place.py \
    planner:=ompl
```

4. Run either of the following option to generate pick-and-place policy
    
    a. Human interace
    ```
    source /opt/ros/humble/setup.bash
    source <path-to-workspace>rcc_build/install/setup.sh
    
    ros2 run rcc_qs_pnp human_interface.py
    ```

    b. Following the [`agent-arena`'s ROS2 `Humble` setup]() for autimatically generate pick-and-place policy.


## IV. Test Individual Components

1. Test ur3e robot with moveit
```
source /opt/ros/humble/setup.bash
source <path-to-workspace>/rcc_build/install/setup.sh

ros2 launch rcc_qs_pnp ur_robot_moveit_executable_launch.py ur_type:=ur3e executable:ur3e_robot_moveit.py
```

2. Test active gripper
```
source /opt/ros/humble/setup.bash
source <path-to-workspace>rcc_build/install/setup.sh

ros2 run rcc_qs_pnp active_gripper_control.py
```

4. Test whole system
```
source /opt/ros/humble/setup.bash
source ./rcc_build/install/setup.sh

ros2 launch rcc_qs_pnp ur_robot_moveit_executable_launch.py ur_type:=ur3e executable:=test_whole_system.py
```

## V. Build and Run integration with `agent-arena`
1. Build
```
cd <path-to-agent-arena>
. ./setup.sh
source $CONDA_PREFIX/setup.bash
```

```
cd <path-to-workspace>
mkdir agar_build
cd agar_build
colcon build --packages-select rcc_msgs --base-paths ../
source install/setup.sh
```

2. Run `agent-arena` scripts for sending policies to robot

```
cd <path-to-agent-arena>
. ./setup.sh
source $CONDA_PREFIX/setup.bash
source <path-to-workspace>/agar_build/install/setup.sh
```

a. human_interace

```
cd <path-to-workspace>/src/robot_control_cloth/interface/
python human_interface.py
```

## VI. Video Recording

Video Recording Start
```
ffmpeg -f v4l2 -video_size 1920x1080 -framerate 30 -i /dev/video6 -c:v libx264 -crf 23 -preset medium -bufsize 5M output.mp4
```

Video Recoding Stop
```
pkill -SIGINT ffmpeg
```


## V. Run foldsformer

```
python agent_arena_interface.py --agent foldsformer --domain ffmr-square-fabric --adjust_pick --adjust_orien --depth_sim2real v0 --mask_sim2real v2 --task all-corner-inward-folding --config default
```

## VI. Run MJ-TN
```
python agent_arena_interface.py --agent transporter --domain sim2real-square-fabric --adjust_pick --adjust_orien --depth_sim2real v2 --mask_sim2real v2 --task all-corner-inward-folding --config MJ-TN-1000-rgb-maskout-rotation-90

```

## VI. Run Diffusion
```
python agent_arena_interface.py --agent diffusion_policy --domain sim2real-rect-fabric --adjust_pick --adjust_orien --depth_sim2real v2 --mask_sim2real v2 --task flattening --config masked-rgb

```