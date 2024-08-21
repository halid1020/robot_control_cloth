cmake_minimum_required(VERSION 3.5)
project(rcc_msgs)

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

# Define message files
set(msg_files
  "msg/NormPixelPnP.msg"
  "msg/Observation.msg"
  "msg/Reset.msg"
  "msg/WorldPnP.msg"
)


# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES builtin_interfaces std_msgs geometry_msgs sensor_msgs
)

# Export dependencies
ament_export_dependencies(rosidl_default_runtime)
ament_package()
