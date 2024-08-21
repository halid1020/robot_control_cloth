#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from matplotlib import pyplot as plt
import api as ag_ar
from gym.spaces import Box

import torch
from segment_anything import sam_model_registry
from utilities.visualisation_utils import draw_pick_and_place, filter_small_masks

from robot_control_cloth.msg import NormPixelPnP, Observation, Reset
from std_msgs.msg import Header



DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device {}'.format(DEVICE))

### Masking Model Macros ###
MODEL_TYPE = "vit_h"
sam = sam_model_registry[MODEL_TYPE](checkpoint='sam_vit_h_4b8939.pth')
sam.to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)

import argparse
from cv_bridge import CvBridge, CvBridgeError
from utils import save_color, save_depth, imgmsg_to_cv2_custom



class AgentArenaInterface(Node):
    def __init__(self, agent, task, config_name, 
                 checkpoint,
                 steps=20):
        super().__init__('agent_arena_interface')

        self.agent = agent
        print('Max steps {}'.format(steps))
         ### Initialise Ros
        self.img_sub = self.create_subscription(Observation, '/observation', self.img_callback, 10)
        self.pnp_pub = self.create_publisher(NormPixelPnP, '/norm_pixel_pnp', 10)
        self.reset_pub = self.create_publisher(Header, '/reset', 10)

        self.task = task

        
        self.bridge = CvBridge()
        self.resolution = (256, 256)
        self.fix_steps = steps

        self.mask_generator = SamAutomaticMaskGenerator(sam)#
        self.save_dir = './agent_data/{}-{}-{}-{}'.\
            format(task, agent.name, config_name, checkpoint)

        os.makedirs(self.save_dir, exist_ok=True)

        print('Finish Init')

    def publish_action(self, pnp):
        print('pick and place action', pnp)
        pnp_msg = NormPixelPnP()
        pnp_msg.header = Header()
        pnp_msg.header.stamp = self.get_clock().now().to_msg()
        pnp_msg.data = [float(pnp[0]), float(pnp[1]), 
                        float(pnp[2]), float(pnp[3])]
        self.pnp_pub.publish(pnp_msg)

    def publish_reset(self):
        
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        self.reset_pub.publish(header)

    def save_step(self, state):
        rgb = state['observation']['rgb']
        depth = state['observation']['depth']
        mask = state['observation']['mask']

        # depth = (depth - np.min(depth))/(np.max(depth) - np.min(depth))
        # depth = cv2.applyColorMap(np.uint8(255 * depth), cv2.COLORMAP_AUTUMN)
        
        save_dir = os.path.join(self.save_dir, self.trj_name, 'step_{}'.format(str(self.step)))
        os.makedirs(save_dir, exist_ok=True)

        save_depth(depth, filename='depth', directory=save_dir)
        save_depth(depth, filename='input_depth', directory='./tmp')
        save_color(rgb, filename='color', directory=save_dir)
        save_color(mask.astype(np.uint8)*255, filename='mask', directory=save_dir)
        


        
        

        if 'action' in state:
            action = state['action']
            action_image = state['action_image'] 
            cv2.imwrite('{}/action_image.png'.format(save_dir,self.step), action_image)
            cv2.imwrite('tmp/action_image.png', action_image)
            np.save('{}/action.npy'.format(save_dir, self.step), action)

        
        if 'masked-pick-heat' in state:
            alpha = 0.5
            pick_heat = state['masked-pick-heat'].copy()
            pick_heat = cv2.resize(pick_heat, rgb.shape[:2])
            
            pick_heat = alpha * pick_heat + (1 - alpha) * rgb
            cv2.imwrite('{}/pick-heat.png'.format(save_dir,self.step), pick_heat)
            cv2.imwrite('tmp/pick-heat.png', pick_heat)

        if 'place-heat-chosen-pick' in state:
            alpha = 0.5
            
            pick_heat = state['place-heat-chosen-pick'].copy()
            pick_heat = cv2.resize(pick_heat, rgb.shape[:2])
            pick_heat = alpha * pick_heat + (1 - alpha) * rgb
            cv2.imwrite('{}/place-heat.png'.format(save_dir,self.step), pick_heat)
            cv2.imwrite('tmp/place-heat.png', pick_heat)

    def img_callback(self, data):

        print('Receive observation data')
        rgb_image = imgmsg_to_cv2_custom(data.rgb_image, "bgr8")

        depth_image = imgmsg_to_cv2_custom(data.depth_image, "64FC1")
        save_depth(depth_image, filename='received_depth', directory='./tmp')
        print('Saved received depth !!!')
        #depth_image = depth_image.astype(np.float32)
        input_state = self.post_process(rgb_image, depth_image)

        if self.step == -1:
            self.agent.init(input_state)#
            self.step += 1
        elif self.step >= self.fix_steps-1:
            self.step += 1
            self.save_step(input_state)#
            self.reset()
            return
        else:
            self.agent.update(input_state, self.last_action)#
            self.step += 1
        print('Step {}'.format(self.step))
              
        action = self.agent.act(input_state)
        internal_state = self.agent.get_state()
        print('inter keys', internal_state.keys())
        self.last_action = action

        ### save action image ###
        pixel_actions = ((action + 1)/2*self.resolution[0]).astype(int).reshape(4)
        action_image = draw_pick_and_place(
            input_state['observation']['rgb'], 
            tuple(pixel_actions[:2]), 
            tuple(pixel_actions[2:]),
            color=(255, 0, 0))
        
        input_state['action'] = action.reshape(4)#
        input_state['action_image'] = action_image
        input_state.update(internal_state)
        self.save_step(input_state)

        ## publish action
        self.publish_action(action.reshape(4))

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


    # def get_mask(self, rgb):
    #     #blur = cv2.blur(rgb, (5,5))
    #     # blur = cv2.blur(blur, (5,5))
    #     result = self.mask_generator.generate(rgb)
            
    #     fmask = None
    #     max_val = 0
    #     ## print all the stablity scores
    #     for r in result:
    #         mask = r['segmentation'].copy()
    #         tmp_mask = mask.copy()
    #         mask = mask.reshape(*rgb.shape[:2], -1)
    #         #print('mask 0 0', mask[0][0])
    #         num_non_mask_corners = (not tmp_mask[5][5]) + (not tmp_mask[5][-5]) +\
    #             (not tmp_mask[-5][5]) + (not tmp_mask[-5][-5])
            
    #         if np.sum(num_non_mask_corners) >= 3 \
    #             and np.sum(num_non_mask_corners) <= 4:
    #             if max_val < np.sum(tmp_mask):
    #                 fmask = mask
    #                 max_val = np.sum(tmp_mask)
                
    #     mask = fmask.reshape(*rgb.shape[:2])
    #     mask = filter_small_masks(mask, 100)

    #     return mask

    def get_mask(self, rgb):
        """
        Generate a mask for the given RGB image that is most different from the background.
        
        Parameters:
        - rgb: A NumPy array representing the RGB image.
        
        Returns:
        - A binary mask as a NumPy array with the same height and width as the input image.
        """
        # Generate potential masks from the mask generator
        results = self.mask_generator.generate(rgb)
        
        final_mask = None
        max_color_difference = 0

        # Iterate over each generated mask result
        for result in results:
            segmentation_mask = result['segmentation']
            mask_shape = rgb.shape[:2]
            
            # Ensure the mask is in the correct format
            segmentation_mask = segmentation_mask.astype(np.uint8) * 255
            
            # Calculate the masked region and the background region
            masked_region = cv2.bitwise_and(rgb, rgb, mask=segmentation_mask)
            background_region = cv2.bitwise_and(rgb, rgb, mask=cv2.bitwise_not(segmentation_mask))
            
            # Calculate the average color of the masked region
            masked_pixels = masked_region[segmentation_mask == 255]
            if masked_pixels.size == 0:
                continue
            avg_masked_color = np.mean(masked_pixels, axis=0)
            
            # Calculate the average color of the background region
            background_pixels = background_region[segmentation_mask == 0]
            if background_pixels.size == 0:
                continue
            avg_background_color = np.mean(background_pixels, axis=0)
            
            # Calculate the Euclidean distance between the average colors
            color_difference = np.linalg.norm(avg_masked_color - avg_background_color)
            
            # Select the mask with the maximum color difference from the background
            if color_difference > max_color_difference:
                final_mask = (segmentation_mask/255).astype(np.bool8)
                max_color_difference = color_difference
        # Ensure final_mask is not None before reshaping
        if final_mask is not None:
            final_mask = final_mask.reshape(*mask_shape)
            #final_mask = filter_small_masks(final_mask, 100)

        return final_mask

    def post_process(self, rgb, depth, pointcloud=None):

        

        rgb = cv2.resize(rgb, self.resolution)
        cv2.imwrite('tmp/rgb.png', rgb)
        depth = cv2.resize(depth, self.resolution) 
        mask = self.get_mask(rgb)  

        rgb =  cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        ### Alert !!! map the depth from 0 (shallowest) to 1 (deepest)
        ### background are 1s.
        ### the agent needs to do the adjustment according! 
        
        norm_depth = (depth - np.min(depth))/((np.max(depth)) - np.min(depth))
        masked_depth = norm_depth * mask
        new_depth = np.ones_like(masked_depth)
        new_depth = new_depth * (1 - mask) + masked_depth

        os.makedirs('tmp', exist_ok=True)
        
        save_depth(new_depth, filename='post_depth.png', directory="./tmp")
        #cv2.imwrite('tmp/depth.png', (depth*255).astype(np.int8))
        cv2.imwrite('tmp/mask.png', mask.astype(np.uint8)*255)

        state = {
            'observation': {
                'rgb': rgb.copy(),
                'depth': new_depth.copy(),
                'mask': mask.copy()
            },
            'action_space': Box(
                -np.ones((1, 4)).astype(np.float32),
                np.ones((1, 4)).astype(np.float32),
                dtype=np.float32),
        }

        return state


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='flattening')
    parser.add_argument('--agent', default='transporter')
    parser.add_argument('--config', default='MJ-TN-2000-rgb-maskout-test')
    parser.add_argument('--log_dir', default='/home/ah390/Data')
    #parser.add_argument('--store_interm', action='store_true', help='store intermediate results')
    parser.add_argument('--eval_checkpoint', default=-1, type=int)


    return parser.parse_args()

if __name__ == "__main__":
        
    args = parse_arguments()

    ### Initialise Agent ####
    print('Initialise Agent ...')
    if args.task == 'flattening':
        domain = 'sim2real-rect-fabric'
        initial = 'crumple'
    elif args.task in ['double-side-folding', 'rectangular-folding']:
        domain = 'sim2real-rect-fabric'
        initial = 'flatten'
    else:
        domain = 'sim2real-square-fabric'
        initial = 'flatten'
    
    if args.task == 'all-corner-inward-folding':
        max_steps = 4
    elif args.task == 'corners-edge-inward-folding':
        max_steps = 6
    elif args.task == 'diagonal-cross-folding':
        max_steps = 2
    elif args.task == 'double-side-folding':
        max_steps = 8
    elif args.task == 'rectangular-folding':
        max_steps = 4
    elif args.task == 'flattening':
        max_steps = 20
 
    arena = 'softgym|domain:{},initial:{},action:pixel-pick-and-place(1),task:{}'.format(domain, initial, args.task)
    agent_config = ag_ar.retrieve_config(
        args.agent, 
        arena, 
        args.config,
        args.log_dir)
    
    agent = ag_ar.build_agent(
        args.agent,
        config=agent_config)

    if args.eval_checkpoint == -1:
        agent.load()
    else:
        agent.load_checkpoint(int(args.eval_checkpoint))
    print('Finsh Initialising Agent ...')

    rclpy.init()

    ### Run Sim2Real ###
    sim2real = AgentArenaInterface(agent, args.task, args.config, 
                                args.eval_checkpoint, max_steps)
    sim2real.run()
    rclpy.shutdown()

if __name__ == "__main__":
    main()