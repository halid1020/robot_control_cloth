#!/usr/bin/env python3

import os
import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry
from segment_anything import \
     sam_model_registry, SamAutomaticMaskGenerator
from gym.spaces import Box
import traceback

import rclpy
from rclpy.node import Node

import agent_arena as agar


from rcc_msgs.msg import NormPixelPnP, Observation
from std_msgs.msg import Header
from agent_arena.utilities.visualisation_utils import *

import argparse
from cv_bridge import CvBridge, CvBridgeError
from utils import *
from control_interface import ControlInterface


class AgentArenaInterface(ControlInterface):
    def __init__(self, agent, task, config_name, 
                 checkpoint,
                 steps=20, 
                 adjust_pick=False, adjust_orien=False):
        super().__init__(task, steps=steps, adjust_pick=adjust_pick, 
                         name='agent', adjust_orien=adjust_orien)

        self.agent = agent
        # print('Max steps {}'.format(steps))
        #  ### Initialise Ros
        # self.img_sub = self.create_subscription(Observation, '/observation', self.img_callback, 10)
        # self.pnp_pub = self.create_publisher(NormPixelPnP, '/norm_pixel_pnp', 10)
        # self.reset_pub = self.create_publisher(Header, '/reset', 10)

        #self.task = task
        # self.resolution = (256, 256)
        # self.fix_steps = steps

        # self.mask_generator = get_mask_generator()
        
       
        self.save_dir = './agent_data/{}-{}-{}-{}'.\
            format(task, agent.name, config_name, checkpoint)

        os.makedirs(self.save_dir, exist_ok=True)
        # self.step = -1
        # self.last_action = None
        # self.trj_name = ''
        # self.adjust_pick = adjust_pick
        # self.adjust_orient = adjust_orien


        print('Finish Init Agent')

    def save_step(self, state):
        super().save_step(state)
        
        rgb = state['observation']['rgb']
        save_dir = os.path.join(self.save_dir, self.trj_name, f'step_{str(self.step)}')
        
        alpha = 0.7
        if 'masked-pick-heat' in state:
            
            pick_heat = state['masked-pick-heat'].copy()
            pick_heat = cv2.resize(pick_heat, rgb.shape[:2])
            
            pick_heat = alpha * pick_heat + (1 - alpha) * rgb
            cv2.imwrite('{}/pick-heat.png'.format(save_dir,self.step), pick_heat)
            cv2.imwrite('tmp/pick-heat.png', pick_heat)

        if 'place-heat-chosen-pick' in state:
            
            pick_heat = state['place-heat-chosen-pick'].copy()
            pick_heat = cv2.resize(pick_heat, rgb.shape[:2])
            pick_heat = alpha * pick_heat + (1 - alpha) * rgb
            cv2.imwrite('{}/place-heat.png'.format(save_dir,self.step), pick_heat)
            cv2.imwrite('tmp/place-heat.png', pick_heat)
    
    def act(self, state):
        mask = state['observation']['mask']
        height, width = mask.shape[:2]
        pnp = self.agent.act(state).reshape(4)
        orientation = 0.0
        pick_pixel = ((pnp[:2] + 1)/2*height).clip(0, height-1).astype(np.int32)
        
        if self.adjust_pick:
           
            adjust_pick, errod_mask = adjust_points([pick_pixel], mask.copy())
            pick_pixel = adjust_pick[0]

        if self.adjust_orient:
            if self.adjust_pick:
                feed_mask = errod_mask
            else:
                feed_mask = mask
            orientation = get_orientation(pick_pixel, feed_mask.copy())
            print('orient', orientation)

        pnp[:2] = np.asarray(pick_pixel, dtype=np.float32)/height * 2 - 1

        action = {
            'pick-and-place': pnp,
            'orientation': orientation
        }

        print('action', action)

        return action
    
    def init(self, state):
        self.agent.init(state)
    
    def update(self, state, action):
        self.agent.update(state, np.asarray(action['pick-and-place']))

    def get_state(self):
        return self.agent.get_state()

    def post_process(self, rgb, depth, raw_rgb=None, pointcloud=None):
        rgb = cv2.resize(rgb, self.resolution)
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

        
        # save_depth(new_depth, filename='post_depth.png', directory="./tmp")
        # #cv2.imwrite('tmp/depth.png', (depth*255).astype(np.int8))
        # cv2.imwrite('tmp/mask.png', mask.astype(np.uint8)*255)

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
            'sim2real': True
        }

        if raw_rgb is not None:
            raw_rgb = raw_rgb[40:-40, 40:-40]
            full_mask = self.get_mask(raw_rgb)
            state['observation']['full_mask'] = full_mask
            raw_rgb =  cv2.cvtColor(raw_rgb, cv2.COLOR_BGR2RGB)
            state['observation']['raw_rgb'] = raw_rgb


        return state


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='flattening')
    parser.add_argument('--domain', default='sim2real-rect-fabric')
    parser.add_argument('--agent', default='transporter')
    parser.add_argument('--config', default='MJ-TN-2000-rgb-maskout-rotation-90')
    parser.add_argument('--log_dir', default='/home/ah390/Data')
    #parser.add_argument('--store_interm', action='store_true', help='store intermediate results')
    parser.add_argument('--eval_checkpoint', default=-1, type=int)


    return parser.parse_args()

if __name__ == "__main__":
        
    args = parse_arguments()

    ### Initialise Agent ####
    print('Initialise Agent ...')
    if args.task == 'flattening':
        #domain = 'sim2real-rect-fabric'
        initial = 'crumple'
    elif args.task in ['double-side-folding', 'rectangular-folding']:
        #domain = 'sim2real-rect-fabric'
        initial = 'flatten'
    else:
        #domain = 'sim2real-square-fabric'
        initial = 'flatten'
    domain = args.domain
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
    agent_config = agar.retrieve_config(
        args.agent, 
        arena, 
        args.config,
        args.log_dir)
    
    agent = agar.build_agent(
        args.agent,
        config=agent_config)

    if args.eval_checkpoint == -1:
        agent.load()
    else:
        agent.load_checkpoint(int(args.eval_checkpoint))
    print('Finsh Initialising Agent ...')

    try:
        rclpy.init()

        adjust_pick=True
        adjust_orien=True

        ### Run Sim2Real ###
        sim2real = AgentArenaInterface(
            agent, args.task, args.config, 
            args.eval_checkpoint, max_steps, 
            adjust_orien=adjust_orien, adjust_pick=adjust_pick)
        
        sim2real.run()
    except Exception as e:
        print(f'Caught exception: {e}')
        print('Stack trace:')
        traceback.print_exc()
        sim2real.stop_video()
        rclpy.try_shutdown()
        sim2real.destroy_node()