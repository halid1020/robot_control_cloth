#!/usr/bin/env python

import sys
import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs

import rospy
from moveit_commander import MoveGroupCommander, RobotCommander, PlanningSceneInterface
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

from rcc_msgs.msg import NormPixelPnP, Observation, Reset, WorldPnP
from utils import *
from active_gripper_control import ActiveGripperControl
from camera_image_retriever import CameraImageRetriever
from panda_robot_moveit import FrankaRobotMoveit  # Updated from UR3eRobotMoveit

class QuasiStaticPickAndPlace:
    def __init__(self, config, mock=False, estimate_pick_depth=False):
        rospy.init_node('quasi_static_pick_and_place', anonymous=True)

        self.logger = rospy.loginfo
        self.config = config
        self.is_mock = mock
        self.estimate_pick_depth = estimate_pick_depth

        self._initialize_communication()
        self._initialize_robot()
        self._initialize_camera()

        self.logger("Finished Initialization, and ready for experiments")

    def _initialize_communication(self):
        self.pub = rospy.Publisher('/observation', Observation, queue_size=10)
        self.bridge = CvBridge()

        rospy.Subscriber('/norm_pixel_pnp', NormPixelPnP, self.pnp_callback if not self.is_mock else self.mock_pnp_callback, queue_size=10)
        rospy.Subscriber('/world_pnp', WorldPnP, self.world_pnp_callback, queue_size=10)
        rospy.Subscriber('/reset', Header, self.reset_callback, queue_size=10)

    def _initialize_robot(self):
        """
        Initialize the Franka Panda robot using MoveIt.
        """
        self.robot_arm = FrankaRobotMoveit()

        self.ready_joint_states = self.config.ready_joint_states
        self.home_joint_states = self.config.home_joint_states

        self.fix_orien = normalise_quaternion(self.config.eff_ready_orien)
        self._initialize_gripper()

        self.pick_raise_offset = self.config.pick_raise_offset
        self.place_raise_offset = self.config.place_raise_offset

    def _initialize_gripper(self):
        self.gripper = ActiveGripperControl()
        self.g2e_offset = self.config.g2e_offset

    def _initialize_camera(self):
        """
        Initialize the camera settings.
        """
        camera_orien = euler_to_quaternion(self.config.camera_orien)
        self.camera_pos = MyPos(pose=self.config.camera_pose, orien=normalise_quaternion(camera_orien))
        self.camera_height = self.config.camera_pose[2]
        self.camera = CameraImageRetriever(self.camera_height)
        self.camera_intrinsic = self.camera.get_intrinsic()

    def pnp_callback(self, pnp):
        """
        Callback for handling pick-and-place requests.
        """
        rospy.loginfo(f"Received PnP: {pnp.data}")
        orien_degree = pnp.degree
        pnp_data = np.asarray(pnp.data)

        # Convert from pixel space to base space
        pixel_pnp = self.norm2pixel_pnp(pnp_data)
        self.pixel_pick_and_place(pixel_pnp[:2], pixel_pnp[2:], pick_orien=orien_degree)
        self.go_home()
        self.publish_observation()

    def world_pnp_callback(self, pnp):
        rospy.loginfo(f"Received world PnP: {pnp.data}")
        pnp_data = np.asarray(pnp.data)

        base_pick = MyPos(pose=pnp_data[:3], orien=self.fix_orien)
        base_place = MyPos(pose=pnp_data[3:], orien=self.fix_orien)
        base_pick.pose[2] += self.g2e_offset
        base_place.pose[2] += self.g2e_offset

        self.execute_pick_and_place(base_pick, base_place)

    def reset_callback(self, reset):
        rospy.loginfo("Received reset signal.")
        self.go_home()
        self.publish_observation()

    def pixel_pick_and_place(self, pick_pixel, place_pixel, pick_orien=0.0):
        """
        Execute pick-and-place motion based on pixel coordinates.
        """
        estimated_depth = self.camera_height
        orien = self.fix_orien

        cur_orien_degree = quaternion_to_euler(self.fix_orien)
        cur_orien_degree[2] += pick_orien
        orien = euler_to_quaternion(cur_orien_degree)

        if self.estimate_pick_depth:
            depth_images = [self.camera.take_rgbd()[1] for _ in range(5)]
            for d in depth_images:
                if np.isnan(d).any():
                    rospy.logerr("There is nan input in depth images.")

            region_size = 5
            x, y = int(pick_pixel[0]), int(pick_pixel[1])
            depth_values = [np.median(d[y - region_size // 2:y + region_size // 2 + 1, x - region_size // 2:x + region_size // 2 + 1]) for d in depth_images]
            estimated_depth = min(np.median(depth_values) + 0.02, self.camera_height)

        # Convert pixel to base coordinates
        base_pick = pixel2base([pick_pixel], self.camera_intrinsic, self.camera_pos, [estimated_depth])[0]
        base_place = pixel2base([place_pixel], self.camera_intrinsic, self.camera_pos, self.camera_height)[0]
        self.execute_pick_and_place(base_pick, base_place)

    def go_home(self):
        self.logger("Going Home")
        self.gripper.open()
        self.robot_arm.go_to_joint_state(self.home_joint_states)
        self.logger("Home position reached")

    def go_ready(self):
        self.logger("Going ready")
        self.robot_arm.go_to_joint_state(self.ready_joint_states)
        self.logger("Ready position reached")

    def execute_pick_and_place(self, pick, place):
        """
        Execute the pick and place routine using robot motion.
        """
        self.logger(f"Starting Pick {pick.pose} and Place {place.pose}")
        self.gripper.open()

        self.go_ready()

        pick_raise_pos = pick.pose + np.array([0, 0, self.pick_raise_offset])
        self.robot_arm.go_to_pose(MyPos(pose=pick_raise_pos, orien=pick.orien))
        self.robot_arm.go_to_pose(pick)

        self.gripper.grasp()

        self.robot_arm.go_to_pose(MyPos(pose=pick_raise_pos, orien=pick.orien))

        place_raise_pos = place.pose + np.array([0, 0, self.place_raise_offset])
        self.robot_arm.go_to_pose(place_raise_pos)
        self.robot_arm.go_to_pose(place)

        self.gripper.open()

    def norm2pixel_pnp(self, pnp):
        """
        Convert normalized coordinates to pixel coordinates.
        """
        pixel_pnp = (pnp + 1) / 2 * self.resolution
        pixel_pnp[0] += self.start_x
        pixel_pnp[2] += self.start_x
        pixel_pnp[1] += self.start_y
        pixel_pnp[3] += self.start_y
        return pixel_pnp

    def take_cropped_rgbd(self):
        """
        Capture cropped RGBD images from the camera.
        """
        color_image, depth_image = self.camera.take_rgbd()
        save_color(color_image, filename='raw_color.png', directory="./tmp")
        save_depth(depth_image, filename='preprocessed_depth.png', directory="./tmp")

        cropped_color, annot_color = self._crop_image(color_image, annotation=True)
        cropped_depth = self._crop_image(depth_image)

        save_color(cropped_color, filename='cropped_color.png', directory="./tmp")
        save_color(annot_color, filename='annotated_color.png', directory="./tmp")
        save_depth(cropped_depth, filename='cropped_depth.png', directory="./tmp")

        return cropped_color, cropped_depth

    def _crop_image(self, image, annotation=False):
        """
        Crop the image and optionally annotate the cropped region.
        """
        org_image = image.copy()
        H, W = image.shape[:2]
        self.resolution = self.config.crop_resol

        self.start_x = self.config.crop_start_x
        self.start_y = self.config.crop_start_y
        self.end_x = self.start_x + self.config.crop_resol + 1
        self.end_y = self.start_y + self.config.crop_resol + 1

        image = image[self.start_y:self.end_y, self.start_x:self.end_x]

        if not annotation:
            return image
        
        # Create a mask for the cropped area
        mask = np.zeros_like(org_image, dtype=np.uint8)
        mask[self.start_y:self.end_y, self.start_x:self.end_x] = (1, 1, 1)

        # Dim the background and highlight the cropped area
        dimmed_image = org_image * 0.5
        highlighted_image = dimmed_image + org_image * mask

        return image, highlighted_image

    def get_observation(self):
        """
        Retrieve RGB, depth, and mask observations from the camera.
        """
        crop_rgb, crop_depth = self.take_cropped_rgbd()
        raw_rgb, raw_depth = self.camera.take_rgbd()
        return {
            'crop_depth': crop_depth,
            'crop_rgb': crop_rgb,
            'raw_rgb': raw_rgb,
            'raw_depth': raw_depth
        }

    def publish(self, observation):
        """
        Publish observations such as RGB and depth images.
        """
        crop_rgb = observation['crop_rgb']
        crop_depth = observation['crop_depth']

        crop_rgb_msg = self.bridge.cv2_to_imgmsg(crop_rgb, encoding="bgr8")
        crop_depth_msg = self.bridge.cv2_to_imgmsg(crop_depth, encoding="64FC1")

        obs_msg = Observation()
        obs_msg.header = Header()
        obs_msg.header.stamp = rospy.Time.now()
        obs_msg.crop_rgb = crop_rgb_msg
        obs_msg.crop_depth = crop_depth_msg
        obs_msg.camera_height = self.camera_height

        self.logger("Publishing RGBD image")
        self.pub.publish(obs_msg)

    def publish_observation(self):
        """
        Publish observations to the ROS topic.
        """
        obs = self.get_observation()
        self.publish(obs)

    def run(self):
        """
        Start the pick-and-place routine.
        """
        self.go_home()
        rospy.spin()

if __name__ == '__main__':
    config_name = "panda_original_realsense_nottingham"  # Use a default config name if none is provided
    config = load_config(config_name)

    node = QuasiStaticPickAndPlace(config=config)
    node.run()
