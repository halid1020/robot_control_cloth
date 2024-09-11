#!/usr/bin/env python
## robot driver and realsense are launched

import rospy
from active_gripper_control import ActiveGripperControl
from camera_image_retriever import CameraImageRetriever
from panda_robot_moveit import FrankaRobotMoveit  # Changed to FrankaRobotMoveit

def main():
    rospy.init_node('robot_control_node', anonymous=True)  # ROS 1 Node Initialization
    
    gripper_control = ActiveGripperControl()
    camera_retriever = CameraImageRetriever()
    franka_robot = FrankaRobotMoveit()  # Changed to FrankaRobotMoveit

    gripper_control.open()
    gripper_control.grasp()
    franka_robot.go_pose(name='home')
    franka_robot.go_pose(name='up')

    try:
        while not rospy.is_shutdown():  # ROS 1 loop control
            for i in range(3):
                camera_retriever.get_images()
                gripper_control.open()
                gripper_control.grasp()
                franka_robot.go_pose(name='home')
                franka_robot.go_pose(name='up')
            break
    finally:
        # Since ROS 1 doesn't have the destroy_node function, just shut down nodes gracefully
        rospy.loginfo("Shutting down nodes...")
        rospy.signal_shutdown("Robot control node finished")

if __name__ == '__main__':
    main()
