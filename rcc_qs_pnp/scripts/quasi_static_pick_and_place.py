#!/usr/bin/env python

import sys
import os
import rospy
import moveit_commander
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs

ros_version = os.environ.get('ROS_VERSION')
print('ROS version', ros_version)

from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

from rcc_msgs.msg import NormPixelPnP, Observation, Reset, WorldPnP
from utils import *

from active_gripper_control import ActiveGripperControl
from camera_image_retriever import CameraImageRetriever
from panda_robot_moveit import FrankaRobotMoveit
from rospy import Rate

# Assumption:
# 1. Robot base frame is the world frame.
# 2. camera_take_pos is always top-down.

class QuasiStaticPickAndPlace:
    def __init__(self, config, mock=False, estimate_pick_depth=False):
        # Initialize ROS node
        rospy.init_node('quasi_static_pick_and_place', anonymous=True)
        
        # Initialize MoveIt commander
        self.logger = rospy.loginfo

        self.config = config
        self.is_mock = mock
        self.estimate_pick_depth = estimate_pick_depth

        self._initialize_communication()
        self._initialize_robot()
        self._initialize_camera()

        self.logger('Finished Initialisation, and ready for experiments')
        
    def _initialize_communication(self):
        self.pub = rospy.Publisher('/observation', Observation, queue_size=10)
        if self.is_mock:
            self.mock_pnp_pub = rospy.Publisher('/pnp', NormPixelPnP, queue_size=10)
        self.bridge = CvBridge()
        rospy.Subscriber('/norm_pixel_pnp', NormPixelPnP, self.mock_pnp_callback if self.is_mock else self.pnp_callback)
        rospy.Subscriber('/world_pnp', WorldPnP, self.world_pnp_callback)
        rospy.Subscriber('/reset', Header, self.reset_callback)
        self.rate = Rate(10)

    def pnp_callback(self, pnp):
        self.logger("Received pnp")
        orien_degree = pnp.degree
        self.place_raise_offset = pnp.place_height
        pnp = np.asarray(pnp.data)
        pixel_pnp = self.norm2pixel_pnp(pnp)
        self.pixel_pick_and_place(pixel_pnp[:2], pixel_pnp[2:], pick_orien=orien_degree)
        self.go_home()
        self.publish_observation()

    def world_pnp_callback(self, pnp):
        self.logger("Received pnp")
        pnp = np.asarray(pnp.data)
        base_pick = MyPos(pose=pnp[:3], orien=self.eff_default_orien)
        base_place = MyPos(pose=pnp[3:], orien=self.eff_default_orien)
        base_pick.pose[2] += self.g2e_offset
        base_place.pose[2] += self.g2e_offset
        self.execute_pick_and_place(base_pick, base_place)

    def reset_callback(self, reset):
        self.logger("Received reset msg")
        self.go_home()
        self.publish_observation()

    def pixel_pick_and_place(self, pick_pixel, place_pixel, pick_orien=0.0):
        estimated_depth = self.camera_height

        orien = self.fix_orien
        cur_orien_degree = quaternion_to_euler(self.fix_orien)
        cur_orien_degree[2] += pick_orien
        orien = euler_to_quaternion(cur_orien_degree)

        if self.estimate_pick_depth:
            depth_images = [self.camera.take_rgbd()[1] for _ in range(5)]
            for d in depth_images:
                if np.isnan(d).any():
                    self.logger("There is nan input in depth images.")

            x, y = int(pick_pixel[0]), int(pick_pixel[1])
            self.logger(f'Pick point x {x}, y {y}')
            depth_values = [np.median(d[y-2:y+3, x-2:x+3]) for d in depth_images]
            estimated_depth = min(np.median(depth_values) + 0.02, self.camera_height)
            self.logger(f"Estimated pick depth {estimated_depth}")

        self.logger(f"Pixel to base conversion, pixel {pick_pixel} depth {estimated_depth}")
        base_pick = pixel2base([pick_pixel], self.camera_intrinstic, self.camera_pos, [estimated_depth])[0]
        base_pick = MyPos(pose=base_pick, orien=orien)

        base_place = pixel2base([place_pixel], self.camera_intrinstic, self.camera_pos, self.camera_height)[0]
        base_place = MyPos(pose=base_place, orien=orien)
        self.logger(f"Calculated base pick {base_pick}, base place {base_place}")

        base_pick.pose[2] = max(0, base_pick.pose[2]) + self.g2e_offset
        base_place.pose[2] += self.g2e_offset
        self.execute_pick_and_place(base_pick, base_place)
    
    def _initialize_gripper(self):
        self.gripper = ActiveGripperControl()
        self.g2e_offset = self.config.g2e_offset

    def _initialize_robot(self):
        self.robot_arm = FrankaRobotMoveit()

        self.ready_joint_states = self.config.ready_joint_states.toDict()
        self.ready_pos = MyPos(pose=self.config.eff_ready_pose, orien=normalise_quaterion(self.config.eff_ready_orien))
        self.home_joint_states = self.config.home_joint_states.toDict()
        self.fix_orien = normalise_quaterion(self.config.eff_ready_orien)

        self._initialize_gripper()

        self.pick_raise_offset = self.config.pick_raise_offset
        self.place_raise_offset = self.config.place_raise_offset

    def _initialize_camera(self):
        camera_orien = euler_to_quaternion(self.config.camera_orien)
        self.camera_pos = MyPos(pose=self.config.camera_pose, orien=normalise_quaterion(camera_orien))
        self.camera_height = self.config.camera_pose[2]

        self.camera = CameraImageRetriever(self.camera_height)
        self.camera_intrinstic = self.camera.get_intrinsic()

    def go_home(self):
        self.logger('Going Home')
        self.gripper.open()
        self.robot_arm.go(joint_states=self.home_joint_states)
        self.logger('Home position reached')

    def go_ready(self):
        self.logger('Going to ready position')
        self.robot_arm.go(joint_states=self.ready_joint_states)
        self.logger('Ready position reached')

    def go_pose(self, pose, straight=False):
        self.robot_arm.go(pose=pose)
        
    def get_pick_depth(self):
        _, depth_img = self.camera.take_rgbd()
        depth = (depth_img - np.min(depth_img)) / (np.max(depth_img) - np.min(depth_img))
        crop_depth_colormap = cv2.applyColorMap(np.uint8(255 * depth), cv2.COLORMAP_JET)
        cv2.imwrite('tmp/pick_depth_img.png', crop_depth_colormap)
        H = int(depth_img.shape[0] // 2)
        sh = int(3.0 / 4 * H)
        sw = int(1.0 / 4 * H)
        w = 20
        ret_depth = np.mean(depth_img[sh:sh + w, sw:sw + w])
        return ret_depth

    def execute_pick_and_place(self, pick, place):
        self.logger(f'Starting Pick {pick.pose} and Place {place.pose}')
        self.gripper.open()
        self.go_ready()

        pick_raise_pos = pick.pose + np.asarray([0, 0, self.pick_raise_offset])
        self.go_pose(MyPos(pose=pick_raise_pos, orien=pick.orien), straight=True)
        pick_pos = pick.pose.copy()
        self.go_pose(MyPos(pose=pick_pos, orien=pick.orien))
        self.gripper.grasp()
        self.go_pose(MyPos(pose=pick_raise_pos, orien=pick.orien))

        current_pos = pick_raise_pos
        place_raise_pos = place.pose + np.asarray([0, 0, self.place_raise_offset])
        direction = (place_raise_pos - current_pos) / np.linalg.norm(place_raise_pos - current_pos)
        for step in range(int(np.linalg.norm(place_raise_pos - current_pos) / 0.2)):
            intermediate_pos = current_pos + direction * 0.2 * (step + 1)
            self.go_pose(MyPos(pose=intermediate_pos, orien=place.orien), straight=True)
        self.go_pose(MyPos(pose=place_raise_pos, orien=place.orien), straight=True)
        self.gripper.open()

    def run(self):
        self.go_home()
        self.take_cropped_rgbd()
        if self.is_mock:
            self.mock_publish_pnp([0, 0, 0, 0])
        rospy.spin()

if __name__ == '__main__':
    config_name = "ur3e_active_realsense_standrews"
    config = load_config(config_name)

    qspnp = QuasiStaticPickAndPlace(config=config, mock=False)
    qspnp.run()
