#!/usr/bin/env python

import rospy
import numpy as np
import cv2
from rcc_msgs.msg import NormPixelPnP, Observation, Reset
from std_msgs.msg import Header
from control_interface import ControlInterface
from utils import *


class HumanPickAndPlace(ControlInterface):
    def __init__(self, task, steps=20, 
                 adjust_pick=False, adjust_orien=False):
        super().__init__(task, steps=steps, adjust_pick=adjust_pick, 
                         name='human', adjust_orien=adjust_orien)
        self.save_dir = f'./human_data/{task}'
        os.makedirs(self.save_dir, exist_ok=True)
        rospy.loginfo('Finished initializing HumanPickAndPlace interface')

    def act(self, state):
        """
        Human interface for selecting pick-and-place points using mouse clicks.
        """
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
            adjust_pick, eroded_mask = adjust_points([clicks[0]], mask.copy())
            clicks[0] = adjust_pick[0]

        orientation = 0.0
        if self.adjust_orien:
            feed_mask = eroded_mask if self.adjust_pick else mask
            orientation = get_orientation(clicks[0], feed_mask.copy())
            rospy.loginfo(f'Pick orientation: {orientation}')

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
        """
        Post-process the RGB, depth, and mask to return a state dict.
        """
        rgb = cv2.resize(rgb, self.resolution)
        depth = cv2.resize(depth, self.resolution)
        mask = self.get_mask(rgb)

        norm_depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth) + 0.005)

        state = {
            'observation': {
                'rgb': rgb.copy(),
                'depth': norm_depth.copy(),
                'mask': mask.copy()
            }
        }

        return state


def main():
    rospy.init_node('human_pick_and_place_node')

    task = 'flattening'  # Default task, can be set with argument parsing if needed
    max_steps = 20  # Default max steps, replace with task-specific logic if needed
    adjust_pick = True
    adjust_orien = True

    human_interface = HumanPickAndPlace(task, max_steps, adjust_pick, adjust_orien)
    human_interface.run()

    rospy.signal_shutdown("Task complete")


if __name__ == "__main__":
    main()
