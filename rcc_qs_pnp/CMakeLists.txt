cmake_minimum_required(VERSION 3.5)
project(rcc_qs_pnp)

# Use C++14
set(CMAKE_CXX_STANDARD 14)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_py REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
# find_package(rcc_msgs REQUIRED)

# Define message files
# set(msg_files
#   "msg/NormPixelPnP.msg"
#   "msg/Observation.msg"
#   "msg/Reset.msg"
#   "msg/WorldPnP.msg"
# )


# Generate messages
# rosidl_generate_interfaces(${PROJECT_NAME}
#   ${msg_files}
#   DEPENDENCIES builtin_interfaces std_msgs geometry_msgs sensor_msgs
# )

# Install Python scripts
install(PROGRAMS
  scripts/quasi_static_pick_and_place.py
  scripts/utils.py
  scripts/ur3e_robot_moveit.py
  scripts/active_gripper_control.py
  scripts/camera_image_retriever.py
  scripts/test_whole_system.py
  scripts/hand_eye_calibrator.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)


# Export dependencies
ament_export_dependencies(rosidl_default_runtime)
ament_package()
