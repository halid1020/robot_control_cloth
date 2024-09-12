#!/usr/bin/env python

import os
import rospy
import numpy as np
import cv2

from rcc_msgs.msg import NormPixelPnP, Observation
from std_msgs.msg import Header
from cv_bridge import CvBridge

from utils import *
from control_interface import ControlInterface


class HumanPickAndPlace(ControlInterface):
    def __init__(self, task, steps=20, adjust_pick=False, adjust_orien=False):
        super(HumanPickAndPlace, self).__init__(task, steps=steps, adjust_pick=adjust_pick, 
                                                name='human', adjust_orien=adjust_orien)
        self.save_dir = f'./human_data/{task}'
        os.makedirs(self.save_dir, exist_ok=True)
        rospy.loginfo('Finish Init Human Interface')

    def act(self, state):
        rgb = state['observation']['rgb']
        mask = state['observation']['mask']
        img = rgb.copy()
        clicks = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicks.append((x, y))
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow('Click Pick and Place Points', img)

        cv2.imshow('Click Pick and Place Points', img)
        cv2.setMouseCallback('Click Pick and Place Points', mouse_callback)

        while len(clicks) < 2:
            cv2.waitKey(1)

        cv2.destroyAllWindows()

        height, width = rgb.shape[:2]

        if self.adjust_pick:
            adjust_pick, errod_mask = adjust_points([clicks[0]], mask.copy())
            clicks[0] = adjust_pick[0]

        orientation = 0.0  
        if self.adjust_orient:
            if self.adjust_pick:
                feed_mask = errod_mask
            else:
                feed_mask = mask
            orientation = get_orientation(clicks[0], feed_mask.copy())
            rospy.loginfo(f'Orientation: {orientation}')

        pick_x, pick_y = clicks[0]
        place_x, place_y = clicks[1]

        normalized_action = [
            (pick_x / width) * 2 - 1,
            (pick_y / height) * 2 - 1,
            (place_x / width) * 2 - 1,
            (place_y / height) * 2 - 1
        ]

        return {
            'pick-and-place': np.array(normalized_action),
            'orientation': orientation
        }

    def post_process(self, rgb, depth, raw_rgb=None, pointcloud=None):
        #mask = self.get_mask(rgb)
        rgb = cv2.resize(rgb, self.resolution)
        depth = cv2.resize(depth, self.resolution)

        mask_v2 = get_mask_v2(self.mask_generator, rgb)
        mask_v1 = get_mask_v1(self.mask_generator, rgb)
        mask_v0 = get_mask_v0(rgb) 
        
        rgb =  cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

       

        norm_depth = (depth - np.min(depth)) / ((np.max(depth) + 0.005) - np.min(depth))

        state = {
            'observation': {
                'rgb': rgb.copy(),
                'depth': norm_depth.copy(),
                'mask': mask_v2.copy(),
                'mask_v1': mask_v1.copy(),
                'mask_v0': mask_v0.copy(),
                'mask_v2': mask_v2.copy()
            }
        }
        
        if raw_rgb is not None:
            raw_rgb = raw_rgb[100:-100, 150:-150]
            full_mask = get_mask_v2(self.mask_generator, raw_rgb)[10:-10, 10:-10]
            state['observation']['full_mask'] = full_mask
            raw_rgb =  cv2.cvtColor(raw_rgb[10:-10, 10:-10], cv2.COLOR_BGR2RGB)
            state['observation']['raw_rgb'] = raw_rgb
        
        return state

    def run(self):
        self.reset()
        rospy.spin()

def main():
    rospy.init_node('human_pick_and_place')
    task = 'flattening'  # Default task, replace with argument parsing if needed
    max_steps = 20  # Default max steps, replace with task-specific logic if needed
    adjust_pick = True
    adjust_orien = True
    sim2real = HumanPickAndPlace(task, max_steps, adjust_pick, adjust_orien)
    sim2real.run()
    rospy.signal_shutdown("Task Complete")

if __name__ == "__main__":
    main()
