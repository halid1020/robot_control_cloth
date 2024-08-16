#!/usr/bin/env python3
## robot driver and realsense are launch

import rclpy

from active_gripper_control import ActiveGripperControl
from camera_image_retriever import CameraImageRetriever
from ur3e_robot_moveit import UR3eRobotMoveit


def main():
    #print('pin id', SetIO.PIN_TOOL_DOUT1 )
    rclpy.init()
    gripper_control = ActiveGripperControl()
    camera_retriver = CameraImageRetriever()
    ur3e_robot = UR3eRobotMoveit()

    gripper_control.open()
    gripper_control.grasp()
    ur3e_robot.go_pose(name='home')
    ur3e_robot.go_pose(name='up')
    
    try:
        while rclpy.ok():
            for i in range(3):
                camera_retriver.get_images()
                gripper_control.open()
                gripper_control.grasp()
                ur3e_robot.go_pose(name='home')
                ur3e_robot.go_pose(name='up')
            break
    finally:
        camera_retriver.destroy_node()
        gripper_control.destroy_node()
        rclpy.shutdown()
        
if __name__ == '__main__':
    main()