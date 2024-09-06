#!/usr/bin/env python3

import os
import numpy as np
import cv2
import rospy
from geometry_msgs.msg import PoseStamped
from utils import *
from active_gripper_control import ActiveGripperControl
from camera_image_retriever import CameraImageRetriever
from panda_robot_moveit import PandaRobotMoveit  # Updated to PandaRobotMoveit

class HandEyeCalibrator:
    def __init__(self, config):
        rospy.init_node('hand_eye_calibrator', anonymous=True)
        
        self.logger = rospy.loginfo
        self.config = config

        self._initialize_robot()
        self._initialize_camera()

        self.logger("Finished initialization, and ready for experiments.")

    def get_workspace_mask(self):
        rgb, depth = self.camera.take_rgbd()
        height, width = rgb.shape[:2]

        # Get all pixel coordinates
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        pixels = np.stack((x.flatten(), y.flatten()), axis=-1)

        self.logger(f'Pixels shape: {pixels.shape}')

        # Convert pixels to base coordinates
        base_particles = pixel2base(pixels, self.camera_intrinsics, 
                                    self.camera_pos, [self.camera_height] * pixels.shape[0])
        
        self.logger(f'Base particles shape: {base_particles.shape}')

        # Check if base_particles are within the workspace region
        distances = np.sqrt(base_particles[:, 0]**2 + base_particles[:, 1]**2)
        self.logger(f'Mask distance shape: {distances.shape}')
        within_workspace = (distances >= self.work_r) & (distances <= self.work_R)

        # Create workspace mask
        workspace_mask = within_workspace.reshape(height, width)
        save_mask(workspace_mask, filename='workspace_mask')

        alpha = 0.9
        workspace_mask_ = np.repeat(workspace_mask[:, :, np.newaxis], 3, axis=2)
        masked_rgb = alpha * rgb * workspace_mask_ + (1 - alpha) * rgb * (1 - workspace_mask_)
        save_color(masked_rgb, filename='workmasked_rgb')

        return workspace_mask

    def update_calibration(self):
        def update_values():
            try:
                new_pose = [float(pose_entries[i].get()) for i in range(3)]
                new_euler = [float(euler_entries[i].get()) for i in range(3)]
                new_orien = euler_to_quaternion(new_euler)
                
                self.camera_pos = MyPos(pose=np.array(new_pose), orien=np.array(new_orien))
                self.camera_height = self.camera_pos.pose[2]
                self.logger("Camera extrinsics updated successfully.")
                self.get_workspace_mask()
                root.destroy()
            except Exception as e:
                rospy.logerr(f"An unexpected error occurred: {str(e)}")
                rospy.logerr("Stack trace:")
                rospy.logerr(traceback.format_exc())

        import tkinter as tk
        from tkinter import ttk

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
        ttk.Label(frame, text="Camera Orientation (Euler angles):").grid(column=2, row=0, sticky=tk.W)
        euler_angles = quaternion_to_euler(self.camera_pos.orien)
        euler_entries = []
        for i, angle in enumerate(['Roll', 'Pitch', 'Yaw']):
            ttk.Label(frame, text=f"{angle}:").grid(column=2, row=i+1, sticky=tk.E)
            entry = ttk.Entry(frame, width=10)
            entry.insert(0, f"{euler_angles[i]:.6f}")
            entry.grid(column=3, row=i+1, sticky=(tk.W, tk.E))
            euler_entries.append(entry)

        update_button = ttk.Button(frame, text="Update", command=update_values)
        update_button.grid(column=0, row=5, columnspan=4, pady=10)

        root.mainloop()

    def pixel_pick_touch(self, pick_pixel):
        estimated_depth = self.camera_height

        self.logger(f"pixel2base pixel {pick_pixel} depth {estimated_depth} intrinsic {self.camera_intrinsics} camera pose {self.camera_pos}")
        base_pick = pixel2base([pick_pixel], self.camera_intrinsics, 
                               self.camera_pos, estimated_depth)[0]
        base_pick = MyPos(pose=base_pick, orien=self.fix_orien)

        base_pick.pose[2] = max(0, base_pick.pose[2]) + self.g2e_offset
        self.execute_pick_touch(base_pick)

    def _initialize_gripper(self):
        self.gripper = ActiveGripperControl()
        self.g2e_offset = self.config.g2e_offset

    def _initialize_robot(self):
        self.robot_arm = PandaRobotMoveit()

        self.ready_joint_states = self.config.ready_joint_states.toDict()
        self.ready_pos = MyPos(
            pose=self.config.eff_ready_pose,
            orien=normalise_quaternion(self.config.eff_ready_orien)
        )
        self.home_joint_states = self.config.home_joint_states.toDict()
        self.fix_orien = normalise_quaternion(self.config.eff_ready_orien)

        self.work_R = self.config.work_R
        self.work_r = self.config.work_r

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
        self.camera_intrinsics = self.camera.get_intrinsic()

    def go_home(self):
        self.logger('Going Home')
        self.robot_arm.go(joint_states=self.home_joint_states)
        self.logger('Home position reached !!!')

    def go_ready(self):
        self.logger('Going ready')
        self.robot_arm.go(joint_states=self.ready_joint_states)
        self.logger('Ready position reached !!!')

    def go_pose(self, pose, straight=False):
        self.robot_arm.go(pose=pose)

    def execute_pick_touch(self, pick):
        self.logger(f'Starting Pick {pick.pose}')

        self.gripper.grasp()
        self.go_ready()

        pick_raise_pos = pick.pose + np.asarray([0, 0, 0.01])
        self.go_pose(MyPos(pose=pick_raise_pos, orien=pick.orien), straight=True)

        pick_pos = pick.pose.copy()
        self.go_pose(MyPos(pose=pick_pos, orien=pick.orien))

        self.go_pose(MyPos(pose=pick_raise_pos, orien=pick.orien))

    def get_pick_pixel(self):
        rgb, depth = self.camera.take_rgbd()
        clicks = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicks.append((x, y))
                cv2.circle(rgb, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow('Click Pick Point', rgb)

        cv2.imshow('Click Pick Point', rgb)
        cv2.setMouseCallback('Click Pick Point', mouse_callback)

        while len(clicks) < 1:
            cv2.waitKey(1)

        cv2.destroyAllWindows()

        return clicks[0]

    def run(self):
        self.gripper.grasp()

        while True:
            self.update_calibration()
            self.go_home()
            pick_pixel = self.get_pick_pixel()
            self.pixel_pick_touch(pick_pixel)

if __name__ == '__main__':
    config_name = "panda_active_realsense_nottingham"
    config = load_config(config_name)

    rospy.init_node('hand_eye_calibrator')

    calibrator = HandEyeCalibrator(config=config)
    calibrator.run()

    rospy.spin()
