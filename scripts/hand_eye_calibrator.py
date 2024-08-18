#!/usr/bin/env python3

import os
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
import tkinter as tk
from tkinter import simpledialog
from tkinter import ttk
from scipy.spatial.transform import Rotation as R


ros_version = os.environ.get('ROS_VERSION')
print('ROS version', ros_version)


from utils import  *

from active_gripper_control import ActiveGripperControl
from camera_image_retriever import CameraImageRetriever
from ur3e_robot_moveit import UR3eRobotMoveit


# Assumption:
# 1. Robot base frome is the world frame.
# 2. camera_take_pos is always top-down.

class HandEyeCalibrator(Node):

    def __init__(self, config):
        super().__init__('hand_eye_calibrator')
        #rospy.init_node('quasi_static_pick_and_place', anonymous=True)
        
        # Initialize MoveIt commander
        self.logger = self.get_logger()

        self.config = config

        self._initialize_robot()
       
        self._initialize_camera()

        self.logger.info('Finished Initialisation, and ready for experimetns')


    def update_calibration(self):
        

        def update_values():
            try:
                new_pose = [float(pose_entries[i].get()) for i in range(3)]
                new_euler = [float(euler_entries[i].get()) for i in range(3)]
                new_orien = euler_to_quaternion(new_euler)
                
                self.camera_pos = MyPos(pose=np.array(new_pose), orien=np.array(new_orien))
                self.camera_height = self.camera_pos.pose[2]
                self.logger.info("Camera extrinsics updated successfully.")
                root.destroy()
            except ValueError:
                self.logger.error("Invalid input! Please enter numeric values.")

        root = tk.Tk()
        root.title("Update Camera Extrinsics")

        frame = ttk.Frame(root, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Pose entries
        ttk.Label(frame, text="Camera Pose:").grid(column=0, row=0, sticky=tk.W)
        pose_entries = []
        for i in range(3):
            ttk.Label(frame, text=f"X{i+1}:").grid(column=0, row=i+1, sticky=tk.E)
            entry = ttk.Entry(frame, width=10)
            entry.insert(0, f"{self.camera_pos.pose[i]:.6f}")
            entry.grid(column=1, row=i+1, sticky=(tk.W, tk.E))
            pose_entries.append(entry)

        # Orientation entries (Euler angles)
        ttk.Label(frame, text="Camera Orientation (Euler angles in degrees):").grid(column=2, row=0, sticky=tk.W)
        euler_angles = quaternion_to_euler(self.camera_pos.orien)
        euler_entries = []
        for i, angle in enumerate(['Roll', 'Pitch', 'Yaw']):
            ttk.Label(frame, text=f"{angle}:").grid(column=2, row=i+1, sticky=tk.E)
            entry = ttk.Entry(frame, width=10)
            entry.insert(0, f"{euler_angles[i]:.6f}")
            entry.grid(column=3, row=i+1, sticky=(tk.W, tk.E))
            euler_entries.append(entry)

        # Update button
        update_button = ttk.Button(frame, text="Update", command=update_values)
        update_button.grid(column=0, row=5, columnspan=4, pady=10)

        # Configure grid
        for child in frame.winfo_children(): 
            child.grid_configure(padx=5, pady=5)

        root.mainloop()
            
    def pixel_pick_touch(self, pick_pixel):
        
        
        estimated_depth = self.camera_height

        
        self.logger.info(\
            f"pixel2base pixel {pick_pixel} depth {estimated_depth} intrinsic {self.camera_intrinstic} camera pose {self.camera_pos}")
        base_pick = pixel2base(pick_pixel, self.camera_intrinstic, 
                               self.camera_pos, estimated_depth)
        base_pick = MyPos(pose=base_pick, orien=self.fix_orien)
        

        base_pick.pose[2] = max(0, base_pick.pose[2]) + self.g2e_offset
        
        self.execute_pick_touch(base_pick)
    
    
    def _initialize_gripper(self):
        self.gripper = ActiveGripperControl()
        self.g2e_offset = self.config.g2e_offset

    def _initialize_robot(self):
        self.robot_arm = UR3eRobotMoveit()

        self.ready_joint_states = self.config.ready_joint_states.toDict()
        self.ready_pos = MyPos(
            pose=self.config.eff_ready_pose,
            orien=normalise_quaterion(self.config.eff_ready_orien)
        )
        self.home_joint_states = self.config.home_joint_states.toDict()
        self.fix_orien = normalise_quaterion(self.config.eff_ready_orien)


        self._initialize_gripper()

        self.pick_raise_offset = self.config.pick_raise_offset
        self.place_raise_offset = self.config.place_raise_offset

    def _initialize_camera(self):
        
        orien = euler_to_quaternion(self.config.camera_orien)

        self.camera_pos = MyPos(
            pose=self.config.camera_pose,
            orien=orien
        )
        self.camera_height = self.config.camera_pose[2]

        self.camera = CameraImageRetriever(self.camera_height)
        self.camera_intrinstic = self.camera.get_intrinsic()
        

    def go_home(self):
        self.logger.info('Going Home')
        #self.gripper.open()
        self.robot_arm.go(joint_states=self.home_joint_states)

        self.logger.info('Home position reached !!!')
    
    def go_ready(self):
        self.logger.info('Going ready')
        
        self.robot_arm.go(joint_states=self.ready_joint_states)

        self.logger.info('ready position reached !!!')

    def go_pose(self, pose, straight=False):
        self.robot_arm.go(pose=pose)
        
    def execute_pick_touch(self, pick):
        self.logger.info('Starting Pick {}'.format(pick.pose))

        # Open Gripper
        self.gripper.grasp()

        self.go_ready()

        # Go to Pick Position
        self.logger.info('Going to Pick Position')
        pick_raise_pos = pick.pose + np.asarray([0, 0, 0.01])
        self.go_pose(
            MyPos(pose=pick_raise_pos, orien=pick.orien), straight=True)

        # Get pick depth
        pick_pos = pick.pose.copy()
        self.go_pose(MyPos(pose=pick_pos, orien=pick.orien))

        # Raise after Pick
        self.go_pose(
            MyPos(pose=pick_raise_pos, orien=pick.orien)
        )


    def get_pick_pixel(self):
        rgb, depth = self.camera.take_rgbd()

        clicks = []
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicks.append((x, y))
                cv2.circle(rgb, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow('Click Pick Point', rgb)

        cv2.imshow('Click Touch Point', rgb)
        cv2.setMouseCallback('Click Touch Point', 
                             mouse_callback)

        while len(clicks) < 1:
            cv2.waitKey(1)

        cv2.destroyAllWindows()

        return clicks[0]
        #place_x, place_y = clicks[1]


    def run(self):
        
        self.gripper.grasp()
        self.go_home()
        
        while True:
            self.go_home()
            
            pick_pixel = self.get_pick_pixel()
            self.pixel_pick_touch(pick_pixel)
            self.update_calibration()




if __name__ == '__main__':
    # Check if a config name is provided as a command-line argument
    config_name = "ur3e_active_realsense_standrews"  # Use a default config name if none is provided
    # Load configuration from the YAML file
    config = load_config(config_name)


    rclpy.init()
    
    # Initialize QuasiStaticPickAndPlace with the loaded configuration
    calibrator = HandEyeCalibrator(config=config)

    calibrator.run()
    


    rclpy.shutdown()