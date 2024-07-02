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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import pygame
from threading import Thread
import torch

x_vel_cmd, y_vel_cmd, yaw_vel_cmd, base_height_cmd = 0.0, 0.0, 0.0, 0.0
x_vel_cmd_scale, y_vel_cmd_scale, yaw_vel_cmd_scale, base_height_cmd_scale = 0.49, 0.49, 1.0, 1.0
joystick_use = True
joystick_opened = False

if joystick_use:
    pygame.init()
    screen = pygame.display.set_mode((500, 500))

    try:
        # 获取手柄
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        joystick_opened = True
    except Exception as e:
        print(f"无法打开手柄：{e}")

    # 用于控制线程退出的标志
    exit_flag = False


# 处理手柄输入的线程
    def handle_joystick_input():
        global exit_flag, x_vel_cmd, y_vel_cmd, yaw_vel_cmd, base_height_cmd

        while not exit_flag:
            # 获取手柄输入
            # pygame.event.get()
            #
            # # 更新机器人命令
            # x_vel_cmd = -joystick.get_axis(1) * 1.0
            # yaw_vel_cmd = -joystick.get_axis(3) * 3.14
            # # if yaw_vel_cmd >= math.pi:
            # #     yaw_vel_cmd -= 2 * math.pi
            # # if yaw_vel_cmd <= -math.pi:
            # #     yaw_vel_cmd += 2 * math.pi
            #
            # # 等待一小段时间，可以根据实际情况调整

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w:
                        x_vel_cmd += 1
                    if event.key == pygame.K_s:
                        x_vel_cmd -= 1
                    if event.key == pygame.K_a:
                        y_vel_cmd += 1
                    if event.key == pygame.K_d:
                        y_vel_cmd -= 1
                    if event.key == pygame.K_q:
                        yaw_vel_cmd += 1
                    if event.key == pygame.K_e:
                        yaw_vel_cmd -= 1
                    if event.key == pygame.K_r:
                        base_height_cmd += 1

                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_w:
                        x_vel_cmd -= 1
                    if event.key == pygame.K_s:
                        x_vel_cmd += 1
                    if event.key == pygame.K_a:
                        y_vel_cmd -= 1
                    if event.key == pygame.K_d:
                        y_vel_cmd += 1
                    if event.key == pygame.K_q:
                        yaw_vel_cmd -= 1
                    if event.key == pygame.K_e:
                        yaw_vel_cmd += 1
                    if event.key == pygame.K_r:
                        base_height_cmd -= 1

            pygame.time.delay(100)

    # 启动线程
    # if joystick_opened and joystick_use:
    if True:
        joystick_thread = Thread(target=handle_joystick_input)
        joystick_thread.start()

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    for i in range(10*int(env.max_episode_length)):
        if joystick_use:
            env.commands[:, 0] = x_vel_cmd * x_vel_cmd_scale
            env.commands[:, 1]= y_vel_cmd * y_vel_cmd_scale
            env.commands[:, 2] = yaw_vel_cmd * yaw_vel_cmd_scale
            env.commands[:, 4] = base_height_cmd * base_height_cmd_scale
            
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'command_height': env.commands[robot_index, 4].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()
            
        cmd_vx, cmd_vy, cmd_yaw, _, cmd_base_height = env.commands[robot_index].cpu().numpy()
        real_vx, real_vy, _ = env.base_lin_vel[robot_index].cpu().numpy()
        _, _, real_yaw = env.base_ang_vel[robot_index].cpu().numpy()
        real_base_height = torch.mean(env.root_states[:, 2].unsqueeze(1) - env.measured_heights).cpu().numpy()

        print("time: %.2f | cmd  vx %.2f | cmd  vy %.2f | cmd  yaw %.2f | cmd  base  height %.2f" % (
            env.episode_length_buf[robot_index].item() / 50,
            cmd_vx,
            cmd_vy,
            cmd_yaw,
            cmd_base_height
        ))

        print("time: %.2f | real vx %.2f | real vy %.2f | real yaw %.2f | base height %.2f" % (
            env.episode_length_buf[robot_index].item() / 50,
            real_vx,
            real_vy,
            real_yaw,
            real_base_height
        ))

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    
    try:
        play(args)
    except KeyboardInterrupt:
        exit_flag = True

