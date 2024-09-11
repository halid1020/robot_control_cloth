cmake_minimum_required(VERSION 3.0.2)
project(rcc_msgs)

# Use C++14
set(CMAKE_CXX_STANDARD 14)

# Find dependencies
find_package(catkin REQUIRED COMPONENTS
  std_msgs
  sensor_msgs
  geometry_msgs
  message_generation
)

# Define message files
add_message_files(
  FILES
  NormPixelPnP.msg
  Observation.msg
  Reset.msg
  WorldPnP.msg
)

# Generate messages
generate_messages(
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs
)

# Define the catkin package
catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs geometry_msgs sensor_msgs
)

# Include directories
include_directories(
  ${catkin_INCLUDE_DIRS}
)

