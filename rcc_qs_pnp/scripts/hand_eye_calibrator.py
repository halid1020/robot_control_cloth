#!/usr/bin/env python

import os
import numpy as np
import cv2
import rospy
import tkinter as tk
from tkinter import ttk
from scipy.spatial.transform import Rotation as R

ros_version = os.environ.get('ROS_VERSION')
print('ROS version', ros_version)
import traceback

from rcc_utils import *

from panda_gripper_control import PandaGripperControl
from camera_image_retriever import CameraImageRetriever
from panda_robot_moveit import FrankaRobotMoveit  # Changed to FrankaRobotMoveit

# Assumption:
# 1. Robot base frame is the world frame.
# 2. camera_take_pos is always top-down.

class HandEyeCalibrator:

    def __init__(self, config):
        # rospy.init_node('hand_eye_calibrator', anonymous=True)  # ROS 1 Node Initialization
        
        # Initialize MoveIt commander
        self.logger = rospy.loginfo

        self.config = config

        self._initialize_robot()
        self._initialize_camera()

        self.logger('Finished Initialization, and ready for experiments')

    def get_workspace_mask(self):
        rgb, depth = self.camera.take_rgbd()
        height, width = rgb.shape[:2]

        # Get all pixel coordinates
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        pixels = np.stack((x.flatten(), y.flatten()), axis=-1)

        self.logger(f'pixels shape {pixels.shape}')

        # Convert pixels to base coordinates
        base_particles = pixel2base(pixels, self.camera_intrinstic, 
                                    self.camera_pos, [self.camera_height]*pixels.shape[0])
        
        self.logger(f'base particles shape {base_particles.shape}')

        # Check if base_particles are within the workspace region
        distances = np.sqrt(base_particles[:, 0]**2 + base_particles[:, 1]**2)
        self.logger(f'mask distance shape {distances.shape}')
        within_workspace = (distances >= self.work_r) & (distances <= self.work_R)

        # Create workspace mask
        workspace_mask = within_workspace.reshape(height, width)

        save_mask(workspace_mask, filename='workspace_mask')

        alpha = 0.9
        workspace_mask_ = np.repeat(workspace_mask[:,:,np.newaxis], 3, axis=2)
        masked_rgb = alpha*rgb*workspace_mask_ + (1-alpha)*rgb*(1-workspace_mask_)
        save_color(masked_rgb, filename='workmasked_rgb')

        return workspace_mask

    def update_calibration(self):

        def update_values():
            try:
                new_pose = [float(pose_entries[i].get()) for i in range(3)]
                new_euler = [float(euler_entries[i].get()) for i in range(3)]
                new_orien = euler_to_quaternion(new_euler)
                
                # curr_pos = np.asarray([pose.x,pose.y,pose.z])
                curr_pos = self.home_pose
                print(f"Curent end effector pose {curr_pos}")
                self.camera_offset = np.asarray(new_pose)
                print(f"Camera offset {self.camera_offset}")
                self.camera_pose = curr_pos + self.camera_offset
                self.camera_pos = MyPos(pose=self.camera_pose, orien=normalise_quaterion(new_orien))

                
                self.camera_height = self.camera_pos.pose[2]
                self.logger("Camera extrinsics updated successfully.")
                self.get_workspace_mask()
                root.destroy()
            except Exception as e:
                self.logger(f"An unexpected error occurred: {str(e)}")
                self.logger("Stack trace:")
                self.logger(traceback.format_exc())

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
            entry.insert(0, f"{self.camera_offset[i]:.6f}")
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
        self.logger(f"pixel2base pixel {pick_pixel} depth {estimated_depth} intrinsic {self.camera_intrinstic} camera pose {self.camera_pos}")
        base_pick = pixel2base([pick_pixel], self.camera_intrinstic, 
                               self.camera_pos, estimated_depth)[0]
        base_pick = MyPos(pose=base_pick, orien=self.fix_orien)
        base_pick.pose[2] = max(0, base_pick.pose[2]) + self.g2e_offset
        print(f"Base pose value {base_pick.pose}")
        print(f"Camera offser {self.camera_offset}")
        print(f"camera pose {self.camera_pos}")
        print(f"Pick Pixel{pick_pixel}")
        self.execute_pick_touch(base_pick)

    def _initialize_gripper(self):
        self.gripper = PandaGripperControl()
        self.g2e_offset = self.config.g2e_offset

    def _initialize_robot(self):
        self.robot_arm = FrankaRobotMoveit()  # Changed to FrankaRobotMoveit

        self.ready_joint_states = self.config.ready_joint_states.toDict()
        self.ready_pos = list(self.config.ready_pos)
        # self.ready_pos = MyPos(
        #     pose=self.config.eff_ready_pose,
        #     orien=normalise_quaterion(self.config.eff_ready_orien)
        # )
        self.home_joint_states = self.config.home_joint_states.toDict()
        self.fix_orien = normalise_quaterion(self.config.eff_ready_orien)

        self.work_R = self.config.work_R
        self.work_r = self.config.work_r

        self._initialize_gripper()

        self.pick_raise_offset = self.config.pick_raise_offset
        self.place_raise_offset = self.config.place_raise_offset

    def _initialize_camera(self):
        orien = euler_to_quaternion(self.config.camera_orien)
        self.go_home()
        # Retrieve current pose from the robot
        pose_stamped = self.robot_arm.get_current_pose()  # Get structured pose information
        # self.home_posestamp = pose_stamped.copy()

        # rospy.loginfo(f"Current pose_stamped: {pose_stamped}")  # Print debug information

        # Access position and orientation directly
        pose = pose_stamped.position
        orientation = pose_stamped.orientation
        
        curr_pos = np.asarray([pose.x,pose.y,pose.z])
        self.home_pose = curr_pos.copy()
        print(f"Curent end effector pose {curr_pos}")
        self.camera_offset = np.asarray(self.config.camera_pose)
        print(f"Camera offset {self.camera_offset}")
        self.camera_pose = curr_pos + self.camera_offset
        self.camera_pos = MyPos(pose=self.camera_pose, orien=normalise_quaterion(orien))

        # self.camera_pos = MyPos(
        #     pose=self.config.camera_pose,
        #     orien=orien
        # )
        self.camera_height = self.config.camera_pose[2]

        self.camera = CameraImageRetriever(self.camera_height)
        self.camera_intrinstic = self.camera.get_intrinsic()

    def go_home(self):
        self.logger('Going Home')
        self.robot_arm.go(joint_states=self.home_joint_states)
        self.logger('Home position reached !!!')
        self.robot_arm.go(MyPos(pose=self.ready_pos, orien=self.fix_orien),straight=True)

    def go_ready(self):
        self.logger('Going to ready position')
        self.robot_arm.go(joint_states=self.ready_joint_states)
        self.logger('Ready position reached')
        

    def go_pose(self, pose, straight=False):
        self.robot_arm.go(pose=pose,straight=straight)

    def execute_pick_touch(self, pick):
        self.logger('Starting Pick {}'.format(pick.pose))

        self.go_ready()
        self.gripper.grasp()

        # Go to Pick Position
        self.logger('Going to Pick Position')
        pick_raise_pos = pick.pose + np.asarray([0, 0, 0.01])
        self.go_pose(MyPos(pose=pick_raise_pos, orien=pick.orien), straight=True)

        # Get pick depth
        pick_pos = pick.pose.copy()
        self.go_pose(MyPos(pose=pick_pos, orien=pick.orien))

        # Raise after Pick
        self.go_pose(MyPos(pose=pick_raise_pos, orien=pick.orien))

    def get_pick_pixel(self):
        rgb, depth = self.camera.take_rgbd()

        clicks = []
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicks.append((x, y))
                cv2.circle(rgb, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow('Click Touch Point', rgb)

        cv2.imshow('Click Touch Point', rgb)
        cv2.setMouseCallback('Click Touch Point', mouse_callback)

        while len(clicks) < 1:
            cv2.waitKey(1)
        print("Finish clicking")
        cv2.destroyAllWindows()

        return clicks[0]

    def run(self):
        self.go_home()
        self.gripper.grasp()

        while True:
            print("Before update calibration")
            self.update_calibration()
            print("After update calibration")
            self.go_home()
            pick_pixel = self.get_pick_pixel()
            self.pixel_pick_touch(pick_pixel)


if __name__ == '__main__':
    # Check if a config name is provided as a command-line argument
    config_name = "panda_original_realsense_nottingham"  # Use a default config name if none is provided
    # Load configuration from the YAML file
    config = load_config(config_name)

    rospy.init_node('hand_eye_calibrator')

    # Initialize HandEyeCalibrator with the loaded configuration
    calibrator = HandEyeCalibrator(config=config)

    calibrator.run()

    # rospy.spin()  # To keep the node alive