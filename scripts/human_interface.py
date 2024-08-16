#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
import numpy as np
import cv2

from robot_control_cloth.msg import NormPixelPnP, Observation, Reset, WorldPnP
from std_msgs.msg import Header
from cv_bridge import CvBridge

def draw_pick_and_place(image, start, end, color=(143, 201, 58)):
    thickness = max(1, int(image.shape[0] / 100))
    image = cv2.arrowedLine(cv2.UMat(image), start, end, color, thickness)
    return image.get().astype(int).clip(0, 255)

class HumanPickAndPlace(Node):
    def __init__(self, task, steps=20):
        super().__init__('human_interface')
        self.img_sub = self.create_subscription(Observation, '/observation', self.img_callback, 10)
        self.pnp_pub = self.create_publisher(NormPixelPnP, '/norm_pixel_pnp', 10)
        self.reset_pub = self.create_publisher(Header, '/reset', 10)
        self.bridge = CvBridge()
        self.resolution = (256, 256)
        self.fix_steps = steps
        self.save_dir = f'./human_data/{task}'
        os.makedirs(self.save_dir, exist_ok=True)
        self.step = -1
        self.last_action = None
        self.trj_name = ''

    def publish_action(self, pnp):
        pnp_msg = NormPixelPnP()
        pnp_msg.header = Header()
        pnp_msg.header.stamp = self.get_clock().now().to_msg()
        pnp_msg.data = (pnp[0], pnp[1], pnp[2], pnp[3])
        self.pnp_pub.publish(pnp_msg)

    def publish_reset(self):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        self.reset_pub.publish(header)


    def save_step(self, state):
        rgb = state['observation']['rgb']
        depth = state['observation']['depth']
        color_depth = cv2.applyColorMap(np.uint8(255 * depth), cv2.COLORMAP_AUTUMN)
        save_dir = os.path.join(self.save_dir, self.trj_name, f'step_{str(self.step)}')
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(f'{save_dir}/rgb.png', rgb)
        cv2.imwrite(f'{save_dir}/color_depth.png', color_depth)
        cv2.imwrite(f'{save_dir}/depth.png', depth)

        if 'action' in state:
            action = state['action']
            action_image = state['action_image']
            cv2.imwrite(f'{save_dir}/action_image.png', action_image)
            np.save(f'{save_dir}/action.npy', action)

    def img_callback(self, data):
        print('Receive observation data')
        rgb_image = self.bridge.imgmsg_to_cv2(data.rgb_image, "bgr8")
        depth_image = self.bridge.imgmsg_to_cv2(data.depth_image, "64FC1")
        depth_image = depth_image.astype(np.float32)
        input_state = self.post_process(rgb_image, depth_image)

        if self.step >= self.fix_steps - 1:
            self.step += 1
            self.save_step(input_state)
            self.reset()
            return
    
        self.step += 1
        action = self.act(input_state)
        self.last_action = action
        pixel_actions = ((action + 1) / 2 * self.resolution[0]).astype(int).reshape(4)
        action_image = draw_pick_and_place(
            input_state['observation']['rgb'],
            tuple(pixel_actions[:2]),
            tuple(pixel_actions[2:]),
            color=(0, 255, 0)
        )
        input_state['action'] = action.reshape(4)
        input_state['action_image'] = action_image
        self.save_step(input_state)
        self.publish_action(action.reshape(4))

    def act(self, state):
        rgb = state['observation']['rgb']
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
        pick_x, pick_y = clicks[0]
        place_x, place_y = clicks[1]

        normalized_action = [
            (pick_x / width) * 2 - 1,
            (pick_y / height) * 2 - 1,
            (place_x / width) * 2 - 1,
            (place_y / height) * 2 - 1
        ]

        return np.array(normalized_action)

    def reset(self):
        self.step = -1
        self.last_action = None
        while True:
            is_continue = input('Continue for a new trial? (y/n): ')
            if is_continue == 'n':
                rclpy.shutdown()
                exit()
            elif is_continue == 'y':
                self.trj_name = input('Enter Trial Name: ')
                break
            else:
                print('Invalid input')
                continue
        self.publish_reset()
        print('pubslied reset!')

    def run(self):
        self.reset()
        rclpy.spin(self)

    def post_process(self, rgb, depth, pointcloud=None):
        #mask = self.get_mask(rgb)
        rgb = cv2.resize(rgb, self.resolution)
        depth = cv2.resize(depth, self.resolution)
        # mask = cv2.resize(mask.astype(np.float), self.resolution)
        # mask = (mask > 0.9).astype(np.bool8)
        norm_depth = (depth - np.min(depth)) / ((np.max(depth) + 0.005) - np.min(depth))
        # masked_depth = norm_depth * mask
        # new_depth = np.ones_like(masked_depth)
        # new_depth = new_depth * (1 - mask) + masked_depth

        state = {
            'observation': {
                'rgb': rgb.copy(),
                'depth': norm_depth.copy(),
                #'mask': mask.copy()
            }
        }
        return state

def main():
    rclpy.init()
    task = 'flattening'  # Default task, replace with argument parsing if needed
    max_steps = 20  # Default max steps, replace with task-specific logic if needed
    sim2real = HumanPickAndPlace(task, max_steps)
    sim2real.run()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
