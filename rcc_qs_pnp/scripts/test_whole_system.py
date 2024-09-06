#!/usr/bin/env python3
## robot driver and realsense are launched

import rospy
from active_gripper_control import ActiveGripperControl
from camera_image_retriever import CameraImageRetriever
from panda_robot_moveit import PandaRobotMoveit

def main():
    rospy.init_node('panda_robot_control', anonymous=True)
    
    # Initialize Gripper, Camera, and Robot MoveIt controller
    gripper_control = ActiveGripperControl()
    camera_retriever = CameraImageRetriever(camera_height=0.5)  # Adjust camera height accordingly
    panda_robot = PandaRobotMoveit()

    # Open and grasp using the gripper
    gripper_control.open()
    gripper_control.grasp()
    
    # Move the robot to home and up positions
    panda_robot.go_pose(name='home')
    panda_robot.go_pose(name='up')
    
    try:
        while not rospy.is_shutdown():
            for i in range(3):
                # Capture images from the camera
                camera_retriever.take_rgbd()
                
                # Simulate basic gripper operation and robot movement
                gripper_control.open()
                gripper_control.grasp()
                
                panda_robot.go_pose(name='home')
                panda_robot.go_pose(name='up')
            break  # Stop the loop after 3 iterations
    finally:
        camera_retriever.destroy_node()
        gripper_control.destroy_node()
        rospy.signal_shutdown('Completed robot operations')

if __name__ == '__main__':
    main()
