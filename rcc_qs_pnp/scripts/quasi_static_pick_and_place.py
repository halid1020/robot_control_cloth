#!/usr/bin/env python

import rospy
import sys
import numpy as np
import cv2
from moveit_commander import RobotCommander, MoveGroupCommander, PlanningSceneInterface
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge

from rcc_msgs.msg import NormPixelPnP, Observation, Reset, WorldPnP
from utils import *

from active_gripper_control import ActiveGripperControl
from camera_image_retriever import CameraImageRetriever

class QuasiStaticPickAndPlace:
    def __init__(self, config, mock=False, estimate_pick_depth=False):
        rospy.init_node('quasi_static_pick_and_place', anonymous=True)

        # Initialize MoveIt for Franka Panda
        self.robot_commander = RobotCommander()
        self.move_group = MoveGroupCommander("panda_arm")
        self.scene = PlanningSceneInterface()

        self.config = config
        self.is_mock = mock
        self.estimate_pick_depth = estimate_pick_depth

        self._initialize_communication()
        self._initialize_robot()
        self._initialize_camera()
        rospy.loginfo('Finished Initialization, and ready for experiments')

    def _initialize_communication(self):
        self.pub = rospy.Publisher('/observation', Observation, queue_size=10)
        self.bridge = CvBridge()
        rospy.Subscriber('/norm_pixel_pnp', NormPixelPnP, self.pnp_callback)
        rospy.Subscriber('/world_pnp', WorldPnP, self.world_pnp_callback)
        rospy.Subscriber('/reset', Header, self.reset_callback)

    def pnp_callback(self, pnp):
        rospy.loginfo("Received pnp: {}".format(pnp.data))
        orien_degree = pnp.degree
        pnp = np.asarray(pnp.data)

        pixel_pnp = self.norm2pixel_pnp(pnp)
        self.pixel_pick_and_place(pixel_pnp[:2], pixel_pnp[2:], pick_orien=orien_degree)
        self.go_home()
        self.publish_observation()

    def world_pnp_callback(self, pnp):
        rospy.loginfo("Received pnp: {}".format(pnp.data))
        pnp = np.asarray(pnp.data)
        base_pick = MyPos(pose=pnp[:3], orien=self.eff_default_orien)
        base_place = MyPos(pose=pnp[3:], orien=self.eff_default_orien)
        base_pick.pose[2] += self.g2e_offset
        base_place.pose[2] += self.g2e_offset
        self.execute_pick_and_place(base_pick, base_place)

    def reset_callback(self, reset):
        rospy.loginfo("Received reset msg")
        self.go_home()
        self.publish_observation()

    def pixel_pick_and_place(self, pick_pixel, place_pixel, pick_orien=0.0):
        estimated_depth = self.camera_height
        orien = self.fix_orien

        if self.estimate_pick_depth:
            depth_images = [self.camera.take_rgbd()[1] for _ in range(5)]
            depth_values = [np.median(d) for d in depth_images if not np.isnan(d).any()]
            estimated_depth = min(np.median(depth_values) + 0.02, self.camera_height)
            rospy.loginfo(f"Estimated pick depth {estimated_depth}")

        base_pick = pixel2base([pick_pixel], self.camera_intrinstic, self.camera_pos, [estimated_depth])[0]
        base_pick = MyPos(pose=base_pick, orien=orien)
        base_place = pixel2base([place_pixel], self.camera_intrinstic, self.camera_pos, self.camera_height)[0]
        base_place = MyPos(pose=base_place, orien=orien)

        base_pick.pose[2] = max(0, base_pick.pose[2]) + self.g2e_offset
        base_place.pose[2] += self.g2e_offset

        self.execute_pick_and_place(base_pick, base_place)

    def execute_pick_and_place(self, pick, place):
        rospy.loginfo(f"Starting Pick {pick.pose} and Place {place.pose}")
        self.gripper.open()
        self.go_ready()
        pick_raise_pos = pick.pose + np.asarray([0, 0, self.pick_raise_offset])
        self.go_pose(MyPos(pose=pick_raise_pos, orien=pick.orien), straight=True)

        pick_pos = pick.pose.copy()
        self.go_pose(MyPos(pose=pick_pos, orien=pick.orien))
        self.gripper.grasp()

        self.go_pose(MyPos(pose=pick_raise_pos, orien=pick.orien))
        self.go_pose(MyPos(pose=place.pose, orien=place.orien), straight=True)
        self.gripper.open()

    def go_home(self):
        rospy.loginfo('Going Home')
        self.gripper.open()
        self.move_group.set_joint_value_target(self.home_joint_states)
        self.move_group.go(wait=True)
        self.move_group.stop()

    def go_ready(self):
        rospy.loginfo('Going Ready')
        self.move_group.set_joint_value_target(self.ready_joint_states)
        self.move_group.go(wait=True)
        self.move_group.stop()

    def go_pose(self, pose, straight=False):
        self.move_group.set_pose_target(pose)
        self.move_group.go(wait=True)
        self.move_group.stop()

    def publish_observation(self):
        obs = self.get_observation()
        self.pub.publish(obs)

    def run(self):
        self.go_home()
        self.take_cropped_rgbd()
        if self.is_mock:
            self.mock_publish_pnp([0, 0, 0, 0])
        rospy.spin()

if __name__ == '__main__':
    config_name = "panda_original_realsense_nottingham"  # Update with Franka's config file
    config = load_config(config_name)
    qspnp = QuasiStaticPickAndPlace(config=config, mock=False)
    qspnp.run()
