cmake_minimum_required(VERSION 3.0.2)
project(rcc_qs_pnp)

# Use C++14
set(CMAKE_CXX_STANDARD 14)

# Find dependencies
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  geometry_msgs
  moveit_ros_planning_interface
)

# Define the catkin package
catkin_package(
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs geometry_msgs moveit_ros_planning_interface
)

# Include directories
include_directories(
  ${catkin_INCLUDE_DIRS}
)

# Install Python scripts
catkin_install_python(PROGRAMS
  scripts/quasi_static_pick_and_place.py
  scripts/rcc_utils.py
  scripts/panda_robot_moveit.py
  scripts/panda_gripper_control.py
  scripts/camera_image_retriever.py
  scripts/test_whole_system.py
  scripts/hand_eye_calibrator.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install config files
install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

