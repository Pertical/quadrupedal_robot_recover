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
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USEw
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin


import os
import numpy as np
import pygame
from threading import Thread

import isaacgym, torch  # DO NOT CHANGE THIS LINE
from legged_gym.envs import task_list
from legged_gym.utils import get_args, TaskRegistry, webviewer, Logger


x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
x_vel_cmd_scale, y_vel_cmd_scale, yaw_vel_cmd_scale = 0.49, 0.49, 1.0
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
        global exit_flag, x_vel_cmd, y_vel_cmd, yaw_vel_cmd

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

            pygame.time.delay(100)

    # 启动线程
    # if joystick_opened and joystick_use:
    if True:
        joystick_thread = Thread(target=handle_joystick_input)
        joystick_thread.start()


def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if checkpoint == -1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]

    return model, checkpoint


def play(args):
    log_pth = "../../logs/{}/".format(args.proj_name) + args.exptid

    task_registry = TaskRegistry(task_list)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # override some parameters for testing
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0

    env_cfg.env.num_envs = 5
    # env_cfg.domain_rand.add_lag = False
    env_cfg.env.episode_length_s = 60  # 60
    env_cfg.commands.heading_command = False
    # env_cfg.commands.resampling_time = 6

    env_cfg.terrain.terrain_proportions = [1., 0., 0., 0., 0.0]
    # env_cfg.terrain.terrain_proportions = [0.1, 0.1, 0.4, 0.4, 0.0]
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 4
    env_cfg.terrain.height = [0.02, 0.02]
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = False

    env_cfg.depth.angle = [0, 1]
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 3
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False

    stop_state_log = 200

    # prepare environment
    logger = Logger(env_cfg.sim.dt)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    if args.web:
        web_viewer = webviewer.WebViewer()
        web_viewer.setup(env)

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root=log_pth,
                                                                   env=env,
                                                                   name=args.task,
                                                                   args=args,
                                                                   train_cfg=train_cfg,
                                                                   return_log_dir=True)

    if args.use_jit:
        path = os.path.join(log_pth, "traced")
        model, checkpoint = get_load_path(root=path, checkpoint=args.checkpoint)
        path = os.path.join(path, model)
        print("Loading jit for policy: ", path)
        policy_jit = torch.jit.load(path, map_location=env.device)
    else:
        policy = ppo_runner.get_inference_policy(device=env.device)

    estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
    if env.cfg.depth.use_camera:
        depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)

    infos = {"depth": env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_depth else None}

    for i in range(100 * int(stop_state_log)):
        if joystick_use:
            env.commands *= 0
            env.commands[env.lookat_id, 0] = x_vel_cmd * x_vel_cmd_scale
            env.commands[env.lookat_id, 1] = y_vel_cmd * y_vel_cmd_scale
            env.commands[env.lookat_id, 2] = yaw_vel_cmd * yaw_vel_cmd_scale
            # env.commands[:, 0] = 0.5

        if args.use_jit:
            if env.cfg.depth.use_camera:
                if infos["depth"] is not None:
                    depth_latent = torch.ones((env_cfg.env.num_envs, 32), device=env.device)
                    actions, depth_latent = policy_jit(obs.detach(), True, infos["depth"], depth_latent)
                else:
                    depth_buffer = torch.ones((env_cfg.env.num_envs, 58, 87), device=env.device)
                    actions, depth_latent = policy_jit(obs.detach(), False, depth_buffer, depth_latent)
            else:
                obs_jit = torch.cat((obs.detach()[:, :env_cfg.env.n_proprio+env_cfg.env.n_priv], obs.detach()[:, -env_cfg.env.history_len*env_cfg.env.n_proprio:]), dim=1)
                actions = policy(obs_jit)
        else:
            if env.cfg.depth.use_camera:
                if infos["depth"] is not None:
                    obs_student = obs[:, :env.cfg.env.n_proprio].clone()
                    depth_latent = depth_encoder(infos["depth"], obs_student)

            else:
                depth_latent = None

            obs[:, env_cfg.env.n_proprio + env_cfg.env.n_scan:env_cfg.env.n_proprio + env_cfg.env.n_scan + env_cfg.env.n_priv] = estimator(obs[:, :env_cfg.env.n_proprio])

            if hasattr(ppo_runner.alg, "depth_actor"):
                actions = ppo_runner.alg.depth_actor(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
            else:
                actions = policy(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
            
        obs, _, rews, dones, infos = env.step(actions.detach())

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[env.lookat_id, 2].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[env.lookat_id, 2].item(),
                    'dof_vel': env.dof_vel[env.lookat_id, 2].item(),
                    'dof_torque': env.torques[env.lookat_id, 2].item(),
                    'dof_pos_target': actions[env.lookat_id, 0].item() * env.cfg.control.action_scale,
                    'dof_pos_target[0]': actions[env.lookat_id, 0].item() * env.cfg.control.action_scale,
                    'dof_pos_target[1]': actions[env.lookat_id, 1].item() * env.cfg.control.action_scale,
                    'dof_pos_target[2]': actions[env.lookat_id, 2].item() * env.cfg.control.action_scale,
                    'dof_pos_target[3]': actions[env.lookat_id, 3].item() * env.cfg.control.action_scale,
                    'dof_pos_target[4]': actions[env.lookat_id, 4].item() * env.cfg.control.action_scale,
                    'dof_pos_target[5]': actions[env.lookat_id, 5].item() * env.cfg.control.action_scale,
                    'dof_pos_target[6]': actions[env.lookat_id, 6].item() * env.cfg.control.action_scale,
                    'dof_pos_target[7]': actions[env.lookat_id, 7].item() * env.cfg.control.action_scale,
                    'dof_pos_target[8]': actions[env.lookat_id, 8].item() * env.cfg.control.action_scale,
                    'dof_pos_target[9]': actions[env.lookat_id, 9].item() * env.cfg.control.action_scale,
                    'dof_pos_target[10]': actions[env.lookat_id,10].item() * env.cfg.control.action_scale,
                    'dof_pos_target[11]': actions[env.lookat_id, 11].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[env.lookat_id, 0].item(),
                    'dof_pos[0]': env.dof_pos[env.lookat_id, 0].item(),
                    'dof_pos[1]': env.dof_pos[env.lookat_id, 1].item(),
                    'dof_pos[2]': env.dof_pos[env.lookat_id, 2].item(),
                    'dof_pos[3]': env.dof_pos[env.lookat_id, 3].item(),
                    'dof_pos[4]': env.dof_pos[env.lookat_id, 4].item(),
                    'dof_pos[5]': env.dof_pos[env.lookat_id, 5].item(),
                    'dof_pos[6]': env.dof_pos[env.lookat_id, 6].item(),
                    'dof_pos[7]': env.dof_pos[env.lookat_id, 7].item(),
                    'dof_pos[8]': env.dof_pos[env.lookat_id, 8].item(),
                    'dof_pos[9]': env.dof_pos[env.lookat_id, 9].item(),
                    'dof_pos[10]': env.dof_pos[env.lookat_id, 10].item(),
                    'dof_pos[11]': env.dof_pos[env.lookat_id, 11].item(),
                    'dof_torque': env.torques[env.lookat_id, 0].item(),
                    'dof_torque[0]': env.torques[env.lookat_id, 0].item(),
                    'dof_torque[1]': env.torques[env.lookat_id, 1].item(),
                    'dof_torque[2]': env.torques[env.lookat_id, 2].item(),
                    'dof_torque[3]': env.torques[env.lookat_id, 3].item(),
                    'dof_torque[4]': env.torques[env.lookat_id, 4].item(),
                    'dof_torque[5]': env.torques[env.lookat_id, 5].item(),
                    'dof_torque[6]': env.torques[env.lookat_id, 6].item(),
                    'dof_torque[7]': env.torques[env.lookat_id, 7].item(),
                    'dof_torque[8]': env.torques[env.lookat_id, 8].item(),
                    'dof_torque[9]': env.torques[env.lookat_id, 9].item(),
                    'dof_torque[10]': env.torques[env.lookat_id, 10].item(),
                    'dof_torque[11]': env.torques[env.lookat_id, 11].item(),
                    'dof_vel': env.dof_vel[env.lookat_id, 0].item(),
                    'dof_vel[0]': env.dof_vel[env.lookat_id, 0].item(),
                    'dof_vel[1]': env.dof_vel[env.lookat_id, 1].item(),
                    'dof_vel[2]': env.dof_vel[env.lookat_id, 2].item(),
                    'dof_vel[3]': env.dof_vel[env.lookat_id, 3].item(),
                    'dof_vel[4]': env.dof_vel[env.lookat_id, 4].item(),
                    'dof_vel[5]': env.dof_vel[env.lookat_id, 5].item(),
                    'dof_vel[6]': env.dof_vel[env.lookat_id, 6].item(),
                    'dof_vel[7]': env.dof_vel[env.lookat_id, 7].item(),
                    'dof_vel[8]': env.dof_vel[env.lookat_id, 8].item(),
                    'dof_vel[9]': env.dof_vel[env.lookat_id, 9].item(),
                    'dof_vel[10]': env.dof_vel[env.lookat_id, 10].item(),
                    'dof_vel[11]': env.dof_vel[env.lookat_id, 11].item(),
                    'command_x': env.commands[env.lookat_id, 0].item(),
                    'command_y': env.commands[env.lookat_id, 1].item(),
                    'command_yaw': env.commands[env.lookat_id, 2].item(),
                    'base_vel_x': env.base_lin_vel[env.lookat_id, 0].item(),
                    'base_vel_y': env.base_lin_vel[env.lookat_id, 1].item(),
                    'base_vel_z': env.base_lin_vel[env.lookat_id, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[env.lookat_id, 2].item(),
                    'contact_forces_z': env.contact_forces[env.lookat_id, env.feet_indices, 2].cpu().numpy()
                }
            )

        elif i == stop_state_log:
            pass
            # logger.plot_states()

        cmd_vx, cmd_vy, cmd_yaw, _ = env.commands[env.lookat_id].cpu().numpy()
        real_vx, real_vy, _ = env.base_lin_vel[env.lookat_id].cpu().numpy()
        _, _, real_yaw = env.base_ang_vel[env.lookat_id].cpu().numpy()
        real_base_height = torch.mean(env.root_states[:, 2].unsqueeze(1) - env.measured_heights).cpu().numpy()

        print("time: %.2f | cmd  vx %.2f | cmd  vy %.2f | cmd  yaw %.2f" % (
            env.episode_length_buf[env.lookat_id].item() / 50,
            cmd_vx,
            cmd_vy,
            cmd_yaw
        ))

        print("time: %.2f | real vx %.2f | real vy %.2f | real yaw %.2f | base height %.2f" % (
            env.episode_length_buf[env.lookat_id].item() / 50,
            real_vx,
            real_vy,
            real_yaw,
            real_base_height
        ))

        # perc_contact_forces = env.contact_forces_avg[env.lookat_id].cpu().numpy()
        if args.headless:
            perc_contact_forces = torch.mean(env.contact_forces_avg, dim=0).cpu().numpy()
            perc_feet_air_time = torch.mean(env.feet_air_time_avg, dim=0).cpu().numpy()
        else:
            perc_contact_forces = env.contact_forces_avg[env.lookat_id].cpu().numpy()
            perc_feet_air_time = env.feet_air_time_avg[env.lookat_id].cpu().numpy()

        perc_contact_forces = perc_contact_forces / np.sum(perc_contact_forces)
        print("contact forces: FL %.2f | FR %.2f | RL %.2f | RR %.2f" % (
            perc_contact_forces[0],
            perc_contact_forces[1],
            perc_contact_forces[2],
            perc_contact_forces[3]
        ))

        perc_feet_air_time = perc_feet_air_time / np.sum(perc_feet_air_time)
        print("feet air time: FL %.2f | FR %.2f | RL %.2f | RR %.2f \n" % (
            perc_feet_air_time[0],
            perc_feet_air_time[1],
            perc_feet_air_time[2],
            perc_feet_air_time[3]
        ))


if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    FIX_COMMAND = False
    args = get_args()

    try:
        play(args)
    except KeyboardInterrupt:
        exit_flag = True
