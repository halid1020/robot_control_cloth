#!/usr/bin/env python3

import sys
import os

import numpy as np

import cv2
from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs

ros_version = os.environ.get('ROS_VERSION')
print('ROS version', ros_version)

import rclpy
from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState
from sensor_msgs_py import point_cloud2

    
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

from robot_control_cloth.msg import NormPixelPnP, Observation, Reset, WorldPnP
from utils import add_quaternions, camera2base, \
    pixel2camera, camera2pixel, MyPos, \
    pixel2base, load_config, prepare_posestamped, \
    normalise_quaterion, save_color, save_depth

from active_gripper_control import ActiveGripperControl
from camera_image_retriever import CameraImageRetriever
from ur3e_robot_moveit import UR3eRobotMoveit
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
# Assumption:
# 1. Robot base frome is the world frame.
# 2. camera_take_pos is always top-down.

class QuasiStaticPickAndPlace(Node):

    def __init__(self, config, mock=False):
        super().__init__('quasi_static_pick_and_place')
        #rospy.init_node('quasi_static_pick_and_place', anonymous=True)
        
        # Initialize MoveIt commander
        self.logger = self.get_logger()

        self.config = config
        self.is_mock = mock

        self._intialize_communication()

        self._initialize_robot()
       
        self._initialize_camera()

        self.logger.info('Finished Initialisation, and ready for experimetns')
        
       
    def _intialize_communication(self):
        
        self.pub = self.create_publisher(Observation, '/observation', 10)
        if self.is_mock:
            self.mock_pnp_pub = self.create_publisher(NormPixelPnP, '/pnp', 10)
        self.bridge = CvBridge()
        self.create_subscription(NormPixelPnP, '/norm_pixel_pnp', (self.mock_pnp_callback if self.is_mock else self.pnp_callback), 10)
        self.create_subscription(WorldPnP, '/world_pnp', self.world_pnp_callback, 10)
        self.create_subscription(Header, '/reset', self.reset_callback, 10)
        self.rate = self.create_rate(10)
            
        

    def pnp_callback(self, pnp):
        print("Received pnp: %s", pnp.data)
        pnp = np.asarray(pnp.data)


        ### Post Process, Convert form pixel space to base space
        pixel_pnp = self.norm2pixel_pnp(pnp)
        self.pixel_pick_and_place(pixel_pnp[:2], pixel_pnp[2:], pick_depth_estimate=True)
        self.go_home()
        self.publish_observation()
    
    def world_pnp_callback(self, pnp):
        print("Received pnp: %s", pnp.data)
        pnp = np.asarray(pnp.data)

        base_pick = MyPos(pose=pnp[:3], orien=self.eff_default_orien)
        base_place = MyPos(pose=pnp[3:], orien=self.eff_default_orien)
        base_pick.pose[2] += self.g2e_offset
        base_place.pose[2] += self.g2e_offset
        self.execute_pick_and_place(base_pick, base_place)

    def reset_callback(self, reset):
        self.logger.info("Received reset msg")
        self.go_home()
        self.publish_observation()


    def pixel_pick_and_place(self, pick_pixel, place_pixel, pick_depth_estimate=False):
        
        
        estimated_depth = self.camera_height
        
        self.logger.info(\
            f"pixel2base pixel {pick_pixel} depth {estimated_depth} intrinsic {self.camera_intrinstic} camera pose {self.camera_pos}")
        base_pick = pixel2base(pick_pixel, self.camera_intrinstic, 
                               self.camera_pos, estimated_depth)
        base_pick = MyPos(pose=base_pick, orien=self.fix_orien)
        
        # print('Coverted pick z', base_pick.pose[2])       
        base_place = pixel2base(place_pixel, self.camera_intrinstic, 
                                self.camera_pos, self.camera_height)
        base_place = MyPos(pose=base_place, orien=self.fix_orien)
        self.logger.info(
            f"Calculated base pick {base_pick}, base place {base_place}"
        )

        base_pick.pose[2] = max(0, base_pick.pose[2]) + self.g2e_offset
        base_place.pose[2] += self.g2e_offset

        self.execute_pick_and_place(base_pick, base_place)
    
    
        


    ######## MOCK PnP  #######
    def mock_publish_pnp(self, pnp):
        pnp_msg = NormPixelPnP()
        pnp_msg.header = Header()
        pnp_msg.header.stamp = rospy.Time.now()
        pnp_msg.data = pnp
        self.mock_pnp_pub.publish(pnp_msg)
        self.rate.sleep()

    def mock_pnp_callback(self, pnp):
        pnp = np.asarray(pnp.data)
        #print("Mock Received pnp: %s", pnp)


        ### Post Process, Convert form pixel space to base space
        pixel_pnp = self.norm2pixel_pnp(pnp)

        self.go_home()
        self.publish_observation()


        ### Publish PnP
        self.mock_publish_pnp(np.random.rand(4))
        
    ######## End #######
    
    
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
        # self.eff_default_orien = self.config.eff_default_orien
        # self.eff_home_pose = self.config.eff_home_pose
        # self.eff_home_z = self.eff_home_pose[-1]
        # self.eff_home_orien = self.eff_default_orien

        self._initialize_gripper()

        self.pick_raise_offset = self.config.pick_raise_offset
        self.place_raise_offset = self.config.place_raise_offset

    def _initialize_camera(self):
        
        self.camera_pos = MyPos(
            pose=self.config.camera_pose,
            orien=normalise_quaterion(self.config.camera_orien)
        )
        self.camera_height = self.config.camera_pose[2]

        self.camera = CameraImageRetriever(self.camera_height)
        self.camera_intrinstic = self.camera.get_intrinsic()
        
        color, depth = self.take_cropped_rgbd()
        

    def go_home(self):
        self.logger.info('Going Home')
        self.gripper.open()
        self.robot_arm.go(joint_states=self.home_joint_states)
        #self.robot_arm.go(pose=self.ready_pos)
        # self.gripper.grasp()
        # self.gripper.open()

        self.logger.info('Home position reached !!!')
    
    def go_ready(self):
        self.logger.info('Going ready')
        
        self.robot_arm.go(joint_states=self.ready_joint_states)
        #self.robot_arm.go(pose=self.ready_pos)


        self.logger.info('ready position reached !!!')

    def go_pose(self, pose, straight=False):
        self.robot_arm.go(pose=pose)
        
    def get_pick_depth(self):
        _, depth_img = self.take_rgbd()
        
        depth = (depth_img - np.min(depth_img))/(np.max(depth_img) - np.min(depth_img))
        crop_depth_colormap = cv2.applyColorMap(np.uint8(255 * depth), cv2.COLORMAP_JET)
        
        cv2.imwrite('tmp/pick_depth_img.png', crop_depth_colormap)
        H = int(depth_img.shape[0]//2)
        sh = int(3.0/4*H)
        
        sw = int(1.0/4*H)
        w = 20
        #mid = int(depth_img.shape[0]//2)
        ret_depth = np.mean(depth_img[sh:sh+w, sw:sw+w])
        # print('ret mean', np.mean(depth_img[sh:sh+w, sw:sw+w]))
        # print('ret max', np.max(depth_img[sh:sh+w, sw:sw+w]))
        # print('ret min', np.min(depth_img[sh:sh+w, sw:sw+w]))
        return ret_depth
        


    def execute_pick_and_place(self, pick, place):
        self.logger.info('Starting Pick {} and Place {}'.format(pick.pose, place.pose))

        # Open Gripper
        self.gripper.open()

        self.go_ready()

        # Go to Pick Position
        self.logger.info('Going to Pick Position')
        pick_raise_pos = pick.pose + np.asarray([0, 0, self.pick_raise_offset])
        self.go_pose(
            MyPos(pose=pick_raise_pos, orien=pick.orien), straight=True)

        # Get pick depth
        pick_pos = pick.pose.copy()
        self.go_pose(MyPos(pose=pick_pos, orien=pick.orien))

        # Close Gripper
        self.gripper.grasp()

        # Raise after Pick
        self.go_pose(
            MyPos(pose=pick_raise_pos, orien=pick.orien)
        )

        # Move to Place Position with 0.3m steps
        self.logger.info('Going to Place Position')
        current_pos = pick_raise_pos
        place_raise_pos = place.pose + np.asarray([0, 0, self.place_raise_offset])
        direction = place_raise_pos - current_pos
        distance = np.linalg.norm(direction)
        direction = direction / distance  # Normalize direction

        step_size = 0.3
        num_steps = int(distance / step_size)
        
        for step in range(num_steps):
            intermediate_pos = current_pos + direction * step_size * (step + 1)
            self.go_pose(MyPos(pose=intermediate_pos, orien=place.orien), straight=True)

        # Ensure final position is reached
        self.go_pose(MyPos(pose=place_raise_pos, orien=place.orien), straight=True)

        # Open Gripper
        self.gripper.open()

    def _crop_image(self, image, annotation=False):
        org_image = image.copy()
        H, W = image.shape[:2]
        min_val = int(min(H*self.config.crop_scale, W*self.config.crop_scale))
        self.px_off = (W - min_val)//2
        self.py_off = (H - min_val)//2
        self.resol = min_val
        mid_x =  W//2
        mid_y = H//2
        # Calculate the coordinates of the cropping rectangle
        start_x = mid_x - min_val // 2
        start_y = mid_y - min_val // 2
        end_x = mid_x + min_val // 2
        end_y = mid_y + min_val // 2

        image = image[start_y:end_y, start_x:end_x]

        if not annotation:
            return image
        
         # Create a mask for the cropped area
        mask = np.zeros_like(org_image, dtype=np.uint8)
        if len(org_image.shape) == 3:  # Color image
            mask[start_y:end_y, start_x:end_x] = (1, 1, 1)
        else:  # Grayscale image (depth)
            mask[start_y:end_y, start_x:end_x] = 1
        
        # Dim the background
        dimmed_image = org_image * 0.5  # Dim the entire image
        highlighted_image = dimmed_image + org_image * mask
        return image, highlighted_image



    def take_cropped_rgbd(self):

        color_image, depth_image = self.camera.take_rgbd()
        save_color(color_image, filename='raw_color.png', directory="./tmp")
        save_depth(depth_image, filename='preprocessed_depth.png', directory="./tmp")


        cropped_color, annot_color = self._crop_image(color_image, annotation=True)
        cropped_depth = self._crop_image(depth_image)
        save_color(cropped_color, filename='cropped_color.png', directory="./tmp")
        save_color(annot_color, filename='annotated_color.png', directory="./tmp")
        save_depth(cropped_depth, filename='cropped_depth.png', directory="./tmp")
        return cropped_color, cropped_depth

    def get_observation(self):
        
        color_image, depth_image = self.take_cropped_rgbd()
        #pointcloud = self.get_pointcloud()

        return {
            'depth': depth_image,
            'color': color_image,
            #'pointcloud': pointcloud
        }


    def publish(self, observation):
        # Simulate an RGB image
        rgb_image = observation['color']
        depth_image = observation['depth']
        
        # Convert OpenCV images to ROS messages
        rgb_msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding="bgr8")
        depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="64FC1")
        

        # Create the custom RGBD message
        obs_msg = Observation()
        obs_msg.header = Header()
        obs_msg.header.stamp = self.get_clock().now().to_msg()
        obs_msg.rgb_image = rgb_msg
        obs_msg.depth_image = depth_msg

        # Publish the message
        self.get_logger().info("Publishing RGBD image")
        self.pub.publish(obs_msg)
        self.get_logger().info("Published")
        #self.rate.sleep()

    def publish_observation(self):
        obs = self.get_observation()
        #self.depth_img = obs['depth']
        self.publish(obs)

    def run(self):
        
        self.go_home()
        self.take_cropped_rgbd()
        #self.publish_observation()
        if self.is_mock:
            self.mock_publish_pnp([0, 0, 0, 0])

        rclpy.spin(self)
        

    def norm2pixel_pnp(self, pnp):
        # print('resol', self.resol)
        # print('py off', self.py_off)
        # print('px off', self.py_off)
        pix_pnp = (pnp+1)/2 * self.resol
        pix_pnp[0] += self.py_off
        pix_pnp[2] += self.py_off
        pix_pnp[1] += self.px_off
        pix_pnp[3] += self.px_off
        print('!!!Pixel Pnp', pix_pnp)
        return pix_pnp

    def test_world_pick_and_place(self):
        self.go_home()
        
        pick = MyPos(
            pose = [-0.5, 0.15, self.config.g2e_offset],
            orien = self.fix_orien)
        
        place = MyPos(
            pose = [-0.5, -0.19, self.config.g2e_offset],
            orien = self.fix_orien)
        
        self.execute_pick_and_place(pick, place)
    
    def test_pixel_pick_and_place(self, pixel_pnp=[-1, -1, -1, -1], random=False):
        self.go_home()
        self.take_cropped_rgbd()
        
        
        
        pixel_pnp_norm = np.asarray(pixel_pnp)

        pixel_pnp = self.norm2pixel_pnp(pixel_pnp_norm)
        self.pixel_pick_and_place(pixel_pnp[:2], pixel_pnp[2:], 
                                  pick_depth_estimate=False)
        
        if random:
            for _ in range(10):
                # Generate random normalized coordinates
                pixel_pnp_norm = np.random.rand(4) * 2 - 1
                
                # Convert normalized coordinates to pixel coordinates
                pixel_pnp = self.norm2pixel_pnp(pixel_pnp_norm)
                
                # Execute pixel pick and place
                self.pixel_pick_and_place(pixel_pnp[:2], pixel_pnp[2:], pick_depth_estimate=False)
        
    def test_camera(self, crop=False):
        self.go_home()
        while True:
            if crop:
                color, depth = self.take_cropped_rgbd()
            else:
                color, depth = self.camera.take_rgbd()

            
            
            depth = (depth - np.min(depth))/(np.max(depth) - np.min(depth))
            depth = np.uint8(255 * depth)
            depth_colormap = cv2.applyColorMap(depth, cv2.COLORMAP_JET)

            self.logger.info(f"depth shape {depth_colormap.shape}")
            self.logger.info(f"color shape {color.shape}")

            # Interpolation factor (adjust to control the blend)
            alpha = 0.5  # Example: 0.5 means equal contribution from both images

            # Interpolate between color and depth images
            interpolated_image = cv2.addWeighted(color, alpha, depth_colormap, 1 - alpha, 0)



            cropped_images = np.hstack((color, depth_colormap, interpolated_image))
            #cv2.imwrite('cropped_rgb.jpg', cropped_images)

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', cropped_images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break




if __name__ == '__main__':
    # Check if a config name is provided as a command-line argument
    config_name = "ur3e_active_realsense_standrews"  # Use a default config name if none is provided
    # Load configuration from the YAML file
    config = load_config(config_name)


    rclpy.init()
    
    # Initialize QuasiStaticPickAndPlace with the loaded configuration
    qspnp = QuasiStaticPickAndPlace(config=config, mock=False)

    #qspnp.run()
    #qspnp.test_world_pick_and_place()
    #qspnp.test_camera(crop=True)
    qspnp.test_pixel_pick_and_place([1, 1, -1, -1])
    # qspnp.test_pixel_pick_and_place()
    


    rclpy.shutdown()