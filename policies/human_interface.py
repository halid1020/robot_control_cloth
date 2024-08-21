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


# def draw_pick_and_place(image, start, end, color=(143, 201, 58)):
#     thickness = max(1, int(image.shape[0] / 100))
#     image = cv2.arrowedLine(cv2.UMat(image), start, end, color, thickness)
#     return image.get().astype(int).clip(0, 255)

class HumanPickAndPlace(ControlInterface):
    def __init__(self, task, steps=20, 
                 adjust_pick=False, adjust_orien=False):
        
        super().__init__(task, steps=steps, adjust_pick=adjust_pick, 
                         name='human', adjust_orien=adjust_orien)
        self.save_dir = f'./human_data/{task}'
        os.makedirs(self.save_dir, exist_ok=True)
        # self.img_sub = self.create_subscription(Observation, '/observation', self.img_callback, 10)
        # self.pnp_pub = self.create_publisher(NormPixelPnP, '/norm_pixel_pnp', 10)
        # self.reset_pub = self.create_publisher(Header, '/reset', 10)
        # self.resolution = (256, 256)
        # self.fix_steps = steps
        # self.mask_generator = get_mask_generator()
        # self.save_dir = f'./human_data/{task}'
        # os.makedirs(self.save_dir, exist_ok=True)
        # self.step = -1
        # self.last_action = None
        # self.trj_name = ''
        # self.adjust_pick = adjust_pick
        # self.adjust_orient = adjust_orien

        print('Finish Init Human Interface')

    # def publish_action(self, pnp):
    #     data = pnp['pick-and-place'].reshape(4)
    #     pnp_msg = NormPixelPnP()
    #     pnp_msg.header = Header()
    #     pnp_msg.header.stamp = self.get_clock().now().to_msg()
    #     pnp_msg.data = (data[0], data[1], data[2], data[3])
    #     pnp_msg.degree = pnp['orientation']
    #     self.pnp_pub.publish(pnp_msg)

    # def get_mask(self, rgb):
    #     """
    #     Generate a mask for the given RGB image that is most different from the background.
        
    #     Parameters:
    #     - rgb: A NumPy array representing the RGB image.
        
    #     Returns:
    #     - A binary mask as a NumPy array with the same height and width as the input image.
    #     """
    #     # Generate potential masks from the mask generator
    #     results = self.mask_generator.generate(rgb)
        
    #     final_mask = None
    #     max_color_difference = 0

    #     # Iterate over each generated mask result
    #     for result in results:
    #         segmentation_mask = result['segmentation']
    #         mask_shape = rgb.shape[:2]
            
    #         # Ensure the mask is in the correct format
    #         segmentation_mask = segmentation_mask.astype(np.uint8) * 255
            
    #         # Calculate the masked region and the background region
    #         masked_region = cv2.bitwise_and(rgb, rgb, mask=segmentation_mask)
    #         background_region = cv2.bitwise_and(rgb, rgb, mask=cv2.bitwise_not(segmentation_mask))
            
    #         # Calculate the average color of the masked region
    #         masked_pixels = masked_region[segmentation_mask == 255]
    #         if masked_pixels.size == 0:
    #             continue
    #         avg_masked_color = np.mean(masked_pixels, axis=0)
            
    #         # Calculate the average color of the background region
    #         background_pixels = background_region[segmentation_mask == 0]
    #         if background_pixels.size == 0:
    #             continue
    #         avg_background_color = np.mean(background_pixels, axis=0)
            
    #         # Calculate the Euclidean distance between the average colors
    #         color_difference = np.linalg.norm(avg_masked_color - avg_background_color)
            
    #         # Select the mask with the maximum color difference from the background
    #         if color_difference > max_color_difference:
    #             final_mask = (segmentation_mask/255).astype(np.bool8)
    #             max_color_difference = color_difference
    #     # Ensure final_mask is not None before reshaping
    #     if final_mask is not None:
    #         final_mask = final_mask.reshape(*mask_shape)
    #         #final_mask = filter_small_masks(final_mask, 100)

    #     return final_mask

    # def publish_reset(self):
    #     header = Header()
    #     header.stamp = self.get_clock().now().to_msg()
    #     self.reset_pub.publish(header)


    # def save_step(self, state):
    #     rgb = state['observation']['rgb']
    #     depth = state['observation']['depth']
    #     color_depth = cv2.applyColorMap(np.uint8(255 * depth), cv2.COLORMAP_AUTUMN)
    #     save_dir = os.path.join(self.save_dir, self.trj_name, f'step_{str(self.step)}')
    #     os.makedirs(save_dir, exist_ok=True)
    #     cv2.imwrite(f'{save_dir}/rgb.png', rgb)
    #     cv2.imwrite(f'{save_dir}/color_depth.png', color_depth)
    #     cv2.imwrite(f'{save_dir}/depth.png', depth)

    #     if 'action' in state:
    #         action = state['action']
    #         action_image = state['action_image']
    #         cv2.imwrite(f'{save_dir}/action_image.png', action_image)
    #         np.save(f'{save_dir}/action.npy', action)

    # def img_callback(self, data):
    #     print('Receive observation data')
    #     rgb_image = imgmsg_to_cv2_custom(data.rgb_image, "bgr8")
    #     depth_image = imgmsg_to_cv2_custom(data.depth_image, "64FC1")
    #     input_state = self.post_process(rgb_image, depth_image)

    #     if self.step >= self.fix_steps - 1:
    #         self.step += 1
    #         self.save_step(input_state)
    #         self.reset()
    #         return
    
    #     self.step += 1
    #     action = self.act(input_state)
        
    #     self.last_action = action
    #     pick_and_place = action['pick-and-place']
    #     pixel_actions = ((pick_and_place + 1) / 2 * self.resolution[0]).astype(int).reshape(4)
    #     action_image = draw_pick_and_place(
    #         input_state['observation']['rgb'],
    #         tuple(pixel_actions[:2]),
    #         tuple(pixel_actions[2:]),
    #         color=(0, 255, 0)
    #     )
    #     input_state['action'] = pick_and_place.reshape(4)
    #     input_state['action_image'] = action_image
    #     self.save_step(input_state)
    #     self.publish_action(action)

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
            print('orient', orientation)

            # visualize_points_and_orientations(
            #     mask.copy(), 
            #     [[clicks[0][0], clicks[0][1], orientation]])

        
        
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

    

    # def run(self):
    #     self.reset()
    #     rclpy.spin(self)

    def post_process(self, rgb, depth, pointcloud=None):
        #mask = self.get_mask(rgb)
        rgb = cv2.resize(rgb, self.resolution)
        depth = cv2.resize(depth, self.resolution)
        mask = self.get_mask(rgb)  

        norm_depth = (depth - np.min(depth)) / ((np.max(depth) + 0.005) - np.min(depth))

        state = {
            'observation': {
                'rgb': rgb.copy(),
                'depth': norm_depth.copy(),
                'mask': mask.copy()
            }
        }
        return state

def main():
    rclpy.init()
    task = 'flattening'  # Default task, replace with argument parsing if needed
    max_steps = 20  # Default max steps, replace with task-specific logic if needed
    adjust_pick=True
    adjust_orien=True
    sim2real = HumanPickAndPlace(task, max_steps, 
                                 adjust_pick, adjust_orien)
    sim2real.run()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
