# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to run the RL environment for the cartpole balancing task."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")
parser.add_argument("--task", type=str, default="None", help="Task name.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from omni.isaac.lab.envs import ManagerBasedRLEnv

# from omni.isaac.lab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleEnvCfg
import sys
sys.path.append('C:/ML_Projects/IsaacLab/user/tutorial')

from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from quadcopter import QuadcopterEnvCfg, QuadcopterEnv


def main():
    """Main function."""
    # create environment configuration
    env_cfg = QuadcopterEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup RL environment
    # env = ManagerBasedRLEnv(cfg=env_cfg)
    # Instantiate the environment
    env = QuadcopterEnv(env_cfg)

    # Reset the environment
    env.reset()

    # Simulate for a number of steps
    num_steps = 1000
    for _ in range(num_steps):
        # Generate random actions
        actions = torch.rand((env.num_envs, env.cfg.num_actions)) * 2 - 1  # Random actions between -1 and 1

        # Step the environment
        env.step(actions)

        # # Optionally render the scene
        # if simulation_app.is_hardware_renderer:
        #     simulation_app.update()

    # Close the simulation app
    simulation_app.close()

if __name__ == "__main__":
    main()