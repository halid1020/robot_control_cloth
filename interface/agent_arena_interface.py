#!/usr/bin/env python

import os
import numpy as np
import cv2
import torch
from gym.spaces import Box
import traceback
import rospy
from std_msgs.msg import Header
from rcc_msgs.msg import NormPixelPnP, Observation
from agent_arena.utilities.visualisation_utils import *
from cv_bridge import CvBridge, CvBridgeError
from utils import *
from control_interface import ControlInterface
import agent_arena as agar


class AgentArenaInterface(ControlInterface):
    def __init__(self, agent, task, config_name, checkpoint,
                 steps=20, adjust_pick=False, adjust_orien=False,
                 depth_sim2real='v2', mask_sim2real='v2',
                 sim_camera_height=0.65, save_dir='.'):
        super(AgentArenaInterface, self).__init__(task, steps=steps, adjust_pick=adjust_pick,
                                                  name='agent', adjust_orien=adjust_orien)
        self.agent = agent
        self.save_dir = './agent_data/{}-{}-{}-{}'.format(task, agent.name, config_name, checkpoint)
        os.makedirs(self.save_dir, exist_ok=True)
        self.depth_sim2real = depth_sim2real
        self.mask_sim2real = mask_sim2real
        self.sim_camera_height = sim_camera_height
        rospy.loginfo("Finished Initializing Agent Arena Interface")

    def save_step(self, state):
        super().save_step(state)
        rgb = state['observation']['rgb']
        save_dir = os.path.join(self.save_dir, self.trj_name, f'step_{str(self.step)}')
        alpha = 0.7

        if 'masked-pick-heat' in state:
            pick_heat = state['masked-pick-heat'].copy()
            pick_heat = cv2.resize(pick_heat, rgb.shape[:2])
            pick_heat = alpha * pick_heat + (1 - alpha) * rgb
            cv2.imwrite('{}/pick-heat.png'.format(save_dir), pick_heat)
            cv2.imwrite('tmp/pick-heat.png', pick_heat)

        if 'place-heat-chosen-pick' in state:
            pick_heat = state['place-heat-chosen-pick'].copy()
            pick_heat = cv2.resize(pick_heat, rgb.shape[:2])
            pick_heat = alpha * pick_heat + (1 - alpha) * rgb
            cv2.imwrite('{}/place-heat.png'.format(save_dir), pick_heat)
            cv2.imwrite('tmp/place-heat.png', pick_heat)

    def act(self, state):
        mask = state['observation']['mask']
        height, width = mask.shape[:2]
        pnp = self.agent.act(state).reshape(4)
        orientation = 0.0

        pick_pixel = ((pnp[:2] + 1) / 2 * height).clip(0, height - 1).astype(np.int32)

        if self.adjust_pick:
            adjust_pick, errod_mask = adjust_points([pick_pixel], mask.copy())
            pick_pixel = adjust_pick[0]

        if self.adjust_orien:
            if self.adjust_pick:
                feed_mask = errod_mask
            else:
                feed_mask = mask
            orientation = get_orientation(pick_pixel, feed_mask.copy())
            rospy.loginfo("Orientation: {}".format(orientation))

        pnp[:2] = np.asarray(pick_pixel, dtype=np.float32) / height * 2 - 1

        action = {
            'pick-and-place': pnp,
            'orientation': orientation
        }

        rospy.loginfo("Action: {}".format(action))

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

        norm_depth = (depth - np.min(depth)) / ((np.max(depth)) - np.min(depth))
        masked_depth = norm_depth * mask
        new_depth = np.ones_like(masked_depth)
        new_depth = new_depth * (1 - mask) + masked_depth

        state = {
            'observation': {
                'rgb': rgb.copy(),
                'depth': new_depth.copy(),
                'mask': mask.copy()
            },
            'action_space': Box(
                -np.ones((1, 4)).astype(np.float32),
                np.ones((1, 4)).astype(np.float32),
                dtype=np.float32
            ),
            'sim2real': True
        }

        if raw_rgb is not None:
            raw_rgb = raw_rgb[40:-40, 40:-40]
            full_mask = self.get_mask(raw_rgb)
            state['observation']['full_mask'] = full_mask
            raw_rgb = cv2.cvtColor(raw_rgb, cv2.COLOR_BGR2RGB)
            state['observation']['raw_rgb'] = raw_rgb

        return state


def parse_arguments():
    """Parse command-line arguments."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='flattening')
    parser.add_argument('--domain', default='sim2real-rect-fabric')
    parser.add_argument('--agent', default='transporter')
    parser.add_argument('--config', default='MJ-TN-2000-rgb-maskout-rotation-90')
    parser.add_argument('--log_dir', default='/home/ah390/Data')
    parser.add_argument('--save_dir', default='/data/sim2real')
    parser.add_argument('--eval_checkpoint', default=-1, type=int)

    parser.add_argument('--depth_sim2real', default='v2')
    parser.add_argument('--mask_sim2real', default='v2')
    parser.add_argument('--sim_camera_height', default=0.65, type=float)

    parser.add_argument('--adjust_pick', action='store_true')
    parser.add_argument('--adjust_orien', action='store_true')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    rospy.init_node('agent_arena_interface_node')

    if args.task == 'flattening':
        initial = 'crumple'
    elif args.task in ['double-side-folding', 'rectangular-folding']:
        initial = 'flatten'
    else:
        initial = 'flatten'

    domain = args.domain
    if args.task == 'all-corner-inward-folding':
        max_steps = 6
    elif args.task == 'corners-edge-inward-folding':
        max_steps = 8
    elif args.task == 'diagonal-cross-folding':
        max_steps = 4
    elif args.task == 'double-side-folding':
        max_steps = 8
    elif args.task == 'rectangular-folding':
        max_steps = 4
    elif args.task == 'flattening':
        max_steps = 20

    agent_config = agar.retrieve_config(
        args.agent,
        'softgym|domain:{},initial:{},action:pixel-pick-and-place(1),task:{}'.format(domain, initial, args.task),
        args.config, args.log_dir
    )

    agent = agar.build_agent(
        args.agent,
        config=agent_config
    )

    if args.eval_checkpoint == -1:
        checkpoint = agent.load()
    else:
        agent.load_checkpoint(int(args.eval_checkpoint))

    ### Run Sim2Real ###
    sim2real = AgentArenaInterface(
        agent, args.task, args.config, 
        checkpoint, max_steps, 
        adjust_orien=args.adjust_orien, 
        adjust_pick=args.adjust_pick, 
        save_dir=args.save_dir,
        depth_sim2real=args.depth_sim2real,
        mask_sim2real=args.mask_sim2real,
        sim_camera_height=args.sim_camera_height
    )

    try:
        sim2real.run()
    except Exception as e:
        print(f'Caught exception: {e}')
        print('Stack trace:')
        traceback.print_exc()
        sim2real.stop_video()  # If stop_video is a valid method, otherwise remove this line.
        rospy.signal_shutdown("Exception occurred in Sim2Real node")
        sim2real.destroy_node()
