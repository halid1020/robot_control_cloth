#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
import numpy as np
import cv2

from rcc_msgs.msg import NormPixelPnP, Observation, Reset, WorldPnP
from std_msgs.msg import Header
from cv_bridge import CvBridge


from utils import *

from control_interface import ControlInterface


class HumanPickAndPlace(ControlInterface):
    def __init__(self, task, steps=20,
                 adjust_pick=False, adjust_orien=False,
                 whole_workspace=False):
        
        super().__init__(task, steps=steps, adjust_pick=adjust_pick, 
                         name='human', adjust_orien=adjust_orien)
        self.save_dir = f'/data/human_data/{task}'
        self.whole_workspace = whole_workspace
        #self.mask_sim2real = mask_sim2real
        os.makedirs(self.save_dir, exist_ok=True)

        self.width_offset = 150
        self.height_offset = 100
        self.mask_offset = 10
        
        print('Finish Init Human Interface')


    def act(self, state):
        rgb = state['observation']['rgb']
        mask = state['observation']['mask']
        workspace = state['observation']['workspace']
        # if self.whole_workspace:
        #     rgb = state['observation']['raw_rgb']
            
        #     # interpolate workspace and rgb
        #     print('workspace', workspace.shape)
        #     alpha = 0.5
        alpha = 0.5
        rgb = alpha * (1.0*rgb/255) + (1-alpha) * workspace
        rgb = (rgb * 255).astype(np.uint8)
            
        img = rgb.copy()
        clicks = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicks.append((x, y))
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow('Click Pick and Place Points', img)

        cv2.imshow('Click Pick and Place Points', img)
        cv2.setMouseCallback('Click Pick and Place Points', 
                             mouse_callback)

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

    def post_process(self, rgb, depth, workspace, raw_rgb=None, pointcloud=None):
        #mask = self.get_mask(rgb)
        org_rgb = rgb.copy()
        # rgb = cv2.resize(rgb, self.resolution)
        # depth = cv2.resize(depth, self.resolution)

        mask_v2 = get_mask_v2(self.mask_generator, org_rgb)
        mask_v1 = get_mask_v1(self.mask_generator, org_rgb)
        mask_v0 = get_mask_v0(org_rgb) 
        
        #rgb =  cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        org_rgb =  cv2.cvtColor(org_rgb, cv2.COLOR_BGR2RGB)

       

        norm_depth = (depth - np.min(depth)) / ((np.max(depth) + 0.005) - np.min(depth))

        state = {
            'observation': {
                #'org_rgb': org_rgb.copy(),
                'rgb': rgb.copy(),
                'depth': norm_depth.copy(),
                'mask': mask_v2.copy(),
                'mask_v1': mask_v1.copy(),
                'mask_v0': mask_v0.copy(),
                'mask_v2': mask_v2.copy(),
                'workspace': workspace.copy()
            }
        }
        
        if raw_rgb is not None:
            
            raw_rgb = raw_rgb[self.height_offset:-self.height_offset, self.width_offset:-self.width_offset]
            #workspace = workspace[self.height_offset:-self.height_offset, self.width_offset:-self.width_offset]
            full_mask = get_mask_v2(self.mask_generator, raw_rgb)[self.mask_offset:-self.mask_offset, self.mask_offset:-self.mask_offset]
            state['observation']['full_mask'] = full_mask
            raw_rgb =  cv2.cvtColor(
                raw_rgb[self.mask_offset:-self.mask_offset, self.mask_offset:-self.mask_offset], 
                cv2.COLOR_BGR2RGB)
            state['observation']['raw_rgb'] = raw_rgb
            #state['observation']['workspace'] = workspace[self.mask_offset:-self.mask_offset, self.mask_offset:-self.mask_offset]

            # print('save_dir', self.save_dir)    
            # save_color(raw_rgb, 'raw_rgb_human', self.save_dir)
            # save_mask(workspace, 'worspace_mask_human', self.save_dir)

        return state

def main():
    rclpy.init()
    task = 'flattening'  # Default task, replace with argument parsing if needed
    max_steps = 20  # Default max steps, replace with task-specific logic if needed
    adjust_pick=False
    adjust_orien=False
    whole_workspace=True
    sim2real = HumanPickAndPlace(task, max_steps,
                                 adjust_pick, adjust_orien,
                                whole_workspace=whole_workspace
                                 )
    sim2real.run()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
