# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

import wandb
from legged_gym.envs import Go1RecFlatConfig, Go1RecFlatConfigPPO



def log_wandb(args):
    run = wandb.init(project="Legged_Gym_Recover_FFall", entity="pertical", config=args)
    
    # Group configurations
    config_groups = {
        "env": ["num_observations", "num_envs", "episode_length_s"],
        "terrain": ["mesh_type", "measure_heights"],
        "commands": ["heading_command", "resampling_time", "base_height_command", "default_base_height", "num_commands", "ranges.base_height"],
        "init_state": ["pos", "default_joint_angles"],
        "asset": ["penalize_contacts_on"],
        "rewards": [
            "max_contact_force", "scales.lin_vel_z", "scales.ang_vel_xy", "scales.orientation", "scales.torques",
            "scales.dof_vel", "scales.dof_acc", "scales.action_rate", "scales.collision", "scales.termination",
            "scales.dof_pos_limits", "scales.dof_vel_limits", "scales.torque_limits", "scales.feet_stumble",
            "scales.stand_still", "scales.feet_contact_forces", "scales.dof_power", "scales.hip_angle",
            "scales.thigh_angle", "scales.calf_angle", "scales.lin_vel_xy", "scales.tracking_lin_vel",
            "scales.tracking_ang_vel", "scales.feet_air_time", "scales.base_uprightness", "scales.foot_contact",
            "scales.tracking_base_height"
        ]
    }

    # Log configurations
    log_data = {}
    for group, keys in config_groups.items():
        for key in keys:
            # Handle nested attributes
            attr_path = key.split('.')
            value = getattr(getattr(Go1RecFlatConfig, group), attr_path[0]) if len(attr_path) == 1 else getattr(getattr(getattr(Go1RecFlatConfig, group), attr_path[0]), attr_path[1])
            log_data[key] = value

    wandb.log(log_data)

def train(args):


    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    log_wandb(args)
    train(args)
