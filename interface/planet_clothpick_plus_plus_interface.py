#!/usr/bin/env python3

import os
import traceback
import yaml

import rclpy

import agent_arena as agar

import argparse
from utils import *
from agent_arena_interface import AgentArenaInterface
from planet_clothpick_plus_plus.controllers.workspace_sampler_planet_clothpick \
    import WorkspaceSamplerPlaNetClothPick

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='flattening')
    parser.add_argument('--log_dir', default='/data/models')
    parser.add_argument('--save_dir', default='/data/sim2real')
    #parser.add_argument('--store_interm', action='store_true', help='store intermediate results')
    parser.add_argument('--eval_checkpoint', default=-1, type=int)

    parser.add_argument('--agent_config',  default='MD2M-on-mix-200k-sf')
    parser.add_argument('--object', default='longsleeve', type=str)
    parser.add_argument('--sampler_config', default='cloth-mask-centre-multiple', type=str)

    ## Sim2Real Protocol
    parser.add_argument('--depth_sim2real', default='v2')
    parser.add_argument('--mask_sim2real', default='v2')
    parser.add_argument('--sim_camera_height', default=0.65, type=float)

    ## Grasping Protocol
    parser.add_argument('--adjust_pick', action='store_true')
    parser.add_argument('--adjust_orien', action='store_true')




    return parser.parse_args()

if __name__ == "__main__":
        
    args = parse_arguments()

    max_steps = 20

    agent_name = "planet-clothpick"
    trained_arena = f"softgym|domain:clothfunnels-realadapt-{args.object},task:flattening,horizon:30"
    # capitalise the first letter of object
    capitalised_obj = args.object[0].upper() + args.object[1:]
    test_arena = f"softgym|domain:ur3e-small{capitalised_obj},task:flattening,horizon:30"
    

    config = agar.retrieve_config(
        agent_name, 
        trained_arena, 
        args.agent_config)
    
    # TODO: Check if this is necessary
    config.transform.params.depth_clip_min = 0 #0.72
    config.transform.params.depth_clip_max = 1.0 #0.76
    config.transform.params.depth_min = 0 # 0.72
    config.transform.params.depth_max = 1.0 # 0.76
    

    agent = agar.build_agent(agent_name, config)
    agent.set_eval()
    #logger = ag_ar.build_logger(arena.logger_name, config.save_dir)
    save_dir = os.path.join(
        args.log_dir, f'ws_smplr_{agent_name}', f'{args.agent_config}+{args.sampler_config}')
    model_dir = os.path.join(args.log_dir, trained_arena, f'{agent_name}', f'{args.agent_config}')
    agent.set_log_dir(model_dir)

    config_file = os.path.join(
        os.environ['AGENT_ARENA_PATH'], '../planet_clothpick_plus_plus',
        'configs', args.sampler_config + '.yaml')
   
    with open(config_file, 'r') as f:
        worspace_sampler_config = yaml.safe_load(f)

    worspace_sampler_config['agent'] = agent
    worspace_sampler_config['checkpoint'] = int(args.eval_checkpoint)
    worspace_sampler_config['policy'] = {'params': config.policy.params}

    worspace_sampler_config = DotMap(worspace_sampler_config)
    

    workspace_sampler = WorkspaceSamplerPlaNetClothPick(
        worspace_sampler_config)
    workspace_sampler.set_log_dir(save_dir)


    if args.eval_checkpoint == -1:
        checkpoint = agent.load()
    else:
        agent.load_checkpoint(int(args.eval_checkpoint))
        checkpoint = args.eval_checkpoint
    config_name = args.sampler_config + '+' + args.agent_config
    print('Finsh Initialising Agent ...')

    try:
        rclpy.init()
        
        ### Run Sim2Real ###
        sim2real = AgentArenaInterface(
            workspace_sampler, args.task, config_name, 
            checkpoint, max_steps, 
            adjust_orien=args.adjust_orien, 
            adjust_pick=args.adjust_pick, 
            save_dir=args.save_dir,
            depth_sim2real=args.depth_sim2real,
            mask_sim2real=args.mask_sim2real,
            sim_camera_height=args.sim_camera_height)
        
        sim2real.run()
    except Exception as e:
        print(f'Caught exception: {e}')
        print('Stack trace:')
        traceback.print_exc()
        sim2real.stop_video()
        rclpy.try_shutdown()
        sim2real.destroy_node()