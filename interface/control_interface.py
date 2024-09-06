#!/usr/bin/env python

import os
import rospy
import numpy as np
import cv2
from rcc_msgs.msg import NormPixelPnP, Observation, Reset, WorldPnP
from std_msgs.msg import Header
from cv_bridge import CvBridge
from utils import *
from segment_anything import SamAutomaticMaskGenerator
from scipy.ndimage import distance_transform_edt
from segment_anything import SamAutomaticMaskGenerator


class ControlInterface:
    def __init__(self, task, steps=20, name='control_interface',
                 adjust_pick=False, adjust_orien=False, video_device='/dev/video6'):
        rospy.init_node(f"{name}_interface")
        
        self.task = task
        self.fix_steps = steps
        self.adjust_pick = adjust_pick
        self.adjust_orien = adjust_orien
        self.video_device = video_device
        self.resolution = (256, 256)
        self.step = -1
        self.last_action = None
        self.trj_name = ''
        self.video_process = None

        self.img_sub = rospy.Subscriber('/observation', Observation, self.img_callback)
        self.pnp_pub = rospy.Publisher('/norm_pixel_pnp', NormPixelPnP, queue_size=10)
        self.reset_pub = rospy.Publisher('/reset', Header, queue_size=10)
        
        self.mask_generator = get_mask_generator()

    def publish_action(self, pnp):
        data = pnp['pick-and-place']
        pnp_msg = NormPixelPnP()
        pnp_msg.header.stamp = rospy.Time.now()
        pnp_msg.data = (data[0], data[1], data[2], data[3])
        pnp_msg.degree = pnp['orientation']
        self.pnp_pub.publish(pnp_msg)

    def get_mask(self, rgb):
        results = self.mask_generator.generate(rgb)
        final_mask = None
        max_color_difference = 0

        for result in results:
            segmentation_mask = result['segmentation']
            masked_region = cv2.bitwise_and(rgb, rgb, mask=segmentation_mask.astype(np.uint8) * 255)
            background_region = cv2.bitwise_and(rgb, rgb, mask=cv2.bitwise_not(segmentation_mask.astype(np.uint8) * 255))
            
            masked_pixels = masked_region[segmentation_mask == 255]
            if masked_pixels.size == 0:
                continue
            
            avg_masked_color = np.mean(masked_pixels, axis=0)
            background_pixels = background_region[segmentation_mask == 0]
            avg_background_color = np.mean(background_pixels, axis=0)
            color_difference = np.linalg.norm(avg_masked_color - avg_background_color)
            
            if color_difference > max_color_difference:
                final_mask = (segmentation_mask / 255).astype(np.bool8)
                max_color_difference = color_difference
        
        return final_mask

    def publish_reset(self):
        header = Header()
        header.stamp = rospy.Time.now()
        self.reset_pub.publish(header)
        rospy.loginfo("Published reset!")

    def save_step(self, state):
        rgb = state['observation']['rgb']
        raw_rgb = state['observation']['raw_rgb']
        full_mask = state['observation']['full_mask']
        depth = state['observation']['depth']
        mask = state['observation']['mask']
        
        save_dir = os.path.join(self.save_dir, self.trj_name, f'step_{str(self.step)}')
        os.makedirs(save_dir, exist_ok=True)
        
        save_color(rgb, filename='rgb', directory=save_dir)
        save_color(raw_rgb, filename='raw_rgb', directory=save_dir)
        save_depth(depth, filename='depth', directory=save_dir)
        save_depth(depth, filename='colour_depth', directory=save_dir, colour=True)
        save_mask(mask, filename='mask', directory=save_dir)
        save_mask(full_mask, filename='full_mask', directory=save_dir)

        if 'action' in state:
            action = state['action']
            action_image = state['action_image']
            save_color(action_image, filename='action_image', directory=save_dir)
            with open(f'{save_dir}/action.json', "w") as json_file:
                json.dump(action, json_file, indent=4)

    def evaluate(self, state):
        current_mask = state['observation']['full_mask']
        cur_mask_pixels = int(np.sum(current_mask))
        rospy.loginfo(f"Current mask pixels: {cur_mask_pixels}")

        if self.step == 0:
            self.init_mask_pixels = cur_mask_pixels

        res = {
            'max_coverage': self.max_mask_pixels,
            'init_coverage': self.init_mask_pixels,
            'coverage': cur_mask_pixels,
            'normalised_coverage': 1.0 * cur_mask_pixels / self.max_mask_pixels,
            'normalised_improvement': max(min(1.0 * (cur_mask_pixels - self.init_mask_pixels) / (self.max_mask_pixels - self.init_mask_pixels), 1), 0),
            'success': False
        }

        return res

    def setup_evaluation(self, state):
        current_mask = state['observation']['full_mask']
        self.max_mask_pixels = int(np.sum(current_mask))
        rospy.loginfo(f"Max mask pixels: {self.max_mask_pixels}")
        self.setup_init_state()

    def setup_init_state(self):
        raw_input('[User Attention!] Please set a random initial state, and press any key to continue!')

    def clean_up(self, state):
        self.stop_video()

        while True:
            is_success = raw_input('[User Attention!] Was the task successful? (y/n): ').strip().lower()
            if is_success == 'y':
                state['evaluation']['success'] = True
                break
            elif is_success == 'n':
                state['evaluation']['success'] = False
                break
            else:
                rospy.loginfo('[User Attention!] Invalid input')
                continue

        self.save_step(state)
        self.reset()

    def img_callback(self, data):
        rospy.loginfo("Received observation data")
        crop_rgb = imgmsg_to_cv2_custom(data.crop_rgb, "bgr8")
        crop_depth = imgmsg_to_cv2_custom(data.crop_depth, "64FC1")
        raw_rgb = imgmsg_to_cv2_custom(data.raw_rgb, "bgr8")
        input_state = self.post_process(crop_rgb, crop_depth, raw_rgb)

        if self.step == -1:
            self.step += 1
            self.setup_evaluation(input_state)
            self.start_video()
            self.publish_reset()
            return

        evaluation = self.evaluate(input_state)
        input_state['evaluation'] = evaluation
        done = (self.step >= self.fix_steps)

        if wait_for_user_input():
            rospy.loginfo("User signaled finish.")
            done = True

        if done:
            self.clean_up(input_state)
        elif self.step == 0:
            self.init(input_state)
        else:
            self.update(input_state, self.last_action)

        action = self.act(input_state)
        self.last_action = action
        self.publish_action(action)
        self.step += 1

    def act(self, state):
        raise NotImplementedError("This method should be implemented in a derived class.")

    def update(self, state, action):
        raise NotImplementedError("This method should be implemented in a derived class.")

    def init(self, state):
        raise NotImplementedError("This method should be implemented in a derived class.")

    def get_state(self):
        return {}

    def run(self):
        self.reset()
        rospy.spin()

    def post_process(self, rgb, depth, raw_rgb=None, pointcloud=None):
        raise NotImplementedError("This method should be implemented in a derived class.")

    def stop_video(self):
        if self.video_process is not None:
            rospy.loginfo("Stopping video recording!")
            stop_ffmpeg_recording(self.video_process)

    def start_video(self):
        if self.video_device is not None:
            save_dir = os.path.join(self.save_dir, self.trj_name)
            os.makedirs(save_dir, exist_ok=True)
            output_file = os.path.join(save_dir, 'record.mp4')
            self.video_process = start_ffmpeg_recording(output_file, self.video_device)

    def reset(self):
        self.step = -1
        self.last_action = None

        while True:
            is_continue = raw_input('[User Attention!] Start a new trial? (y/n): ').strip().lower()
            if is_continue == 'y':
                self.trj_name = raw_input('[User Attention!] Enter Trial Name: ').strip()
                break
            elif is_continue == 'n':
                raise rospy.ROSInterruptException("User stopped the process.")
            else:
                rospy.loginfo('[User Attention!] Invalid input')

        self.setup()
