

"""
Sim to sim from Isaac gym to mujoco

"""
import argparse
import math
import numpy as np 
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R

from legged_gym.envs.base.legged_robot_rec_config import LeggedRobotRecCfg, LeggedRobotRecCfgPPO
from legged_gym.envs import Go1RecFlatConfig, Go1RecFlatConfigPPO

from legged_gym.envs import * 
from legged_gym.utils import Logger

import torch
import pygame

from threading import Thread

#Command for the robot to follow in the simulation


base_height_cmd = 0.0 
push_robot_cmd = False

joystick_use = True 
joystick_opened = False


if joystick_use:
    pygame.init()
    
    try:
        joystick = pygame.joystick.Joystick(0)
        print("joystick name", joystick)
        joystick.init()
        joystick_opened = True
    
    except Exception as e:
        print(f"Cannot open joystick: {e}")

    exit_flag = False


    def handle_joystick_input():

        global exit_flag, base_height_cmd, push_robot_cmd

        while not exit_flag:

            pygame.event.get()

            base_height_cmd = -joystick.get_button(0) *1.5 

            # print("base_height_cmd", base_height_cmd)

            # for event in pygame.event.get():
            #     if event.type == pygame.KEYDOWN:
            #         if event.key == pygame.K_w:
            #             base_height_cmd += 1
            #         if event.key == pygame.K_s:
            #             base_height_cmd -= 1
            #         if event.key == pygame.K_q:
            #             push_robot_cmd = True
                
            #     if event.type == pygame.KEYUP:
            #         if event.key == pygame.K_q:
            #             push_robot_cmd = False
            
            pygame.time.delay(100)


    if joystick_opened and joystick_use:
        joystick_thread = Thread(target=handle_joystick_input)
        joystick_thread.start()


class cmd:

    base_height = 0.25

def quaternion_to_euler_array(quat):
    """
    Quaternion to Euler angles conversion

    Args: 
        quat: quaternion in the format (x,y,z,w) from mujoco 
    
    Returns:
        np.array([roll_x, pitch_y, yaw_z]): Euler angles in radians
    """

    #Ensure quaternion is in the correct format (x,y,z,w)
    x, y, z, w = quat

    #Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    #Pitch (y-axis rotation)
    t2 = +2.0* (w * y - z * x)
    t2 = np.clip(t1, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    #Yaw (z-axis rotation) 
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    #Return as a numpy array IN RADIANS
    return np.array([roll_x, pitch_y, yaw_z])



def get_obs(data):
    """
    Extracts the observation from the mujoco data structure
    
    Args:
        data: Mujoco data structure
    
    Returns:
        q: joint positions
        dq: joint velocities
        quat: quaternion
        v: velocity
        omega: angular velocity
        gvec: gravity vector

    """

    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('Body_Quat').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double) #velocity in body frame
    omega = data.sensor('Body_Gyro').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double) #gravity vector in body frame

    print("gvec", gvec)
    print("omega", omega)

    return (q, dq, quat, v, omega, gvec) #return as a tuple


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """
    Args:
        target_q: target joint positions
        q: current joint positions
        kp: proportional gain
        target_dq: target joint velocities
        dq: current joint velocities
        kd: derivative gain

    Returns:
        tau: joint torques
    """
    return (target_q - q) * kp + (target_dq - dq) * kd


def run_mujoco(policy, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """

    #Load robot xml model and extract dat from mujoco
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)


    action_startup = np.zeros((cfg.env.num_actions), dtype=np.double)

    #Default joint positions or starting position 
    #action_startup = np.array([0., 0.8, -1.5, 0., 0.8, -1.5, 0., 0.8, -1.5, 0., 0.8, -1.5])
    action_startup = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    #qpos contains the joint configuration, represents a position of each joint in the simulation.
    #qvel denotes velocities of these joints. 
    num_actuated_joints = cfg.env.num_actions 
    # print("qpos shape", data.qpos.shape) #qpos shape (19,)
    print("qpos", data.qpos)

    # data.qpos[-num_actuated_joints:] = action_startup
    data.qpos[7:] = action_startup

    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)
    
    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)

    # hist_obs = deque()
    # for _ in range(cfg.env.frame_stack):
    #     hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))
    count_lowlevel = 0

    # action_startup[:] = action_startup[:] * 10.
    # action[:] = action_startup[:]

    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):

        #Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.env.num_actions:]
        dq = dq[-cfg.env.num_actions:]

        if count_lowlevel % cfg.sim_config.decimation == 0: 

            obs = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi

            # self.obs_buf = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel,
            #                           self.projected_gravity, 
            #                           self.commands[:, :] * self.commands_scale[3], 
            #                           (self.dof_pos - self.default_dof_pos)*self.obs_scales.dof_pos,
            #                           self.dof_vel * self.obs_scales.dof_vel,
            #                           self.actions), dim = -1) 


            obs[0, 0:3] = omega #base angular velocity
            obs[0, 3:6] = gvec #projected gravity

            obs[0, 6:7] = np.array([cmd.base_height]) *cfg.normalization.obs_scales.base_height #commands
            obs[0, 7:19] = (q - cfg.robot_config.default_dof_pos) * cfg.normalization.obs_scales.dof_pos #dof_pos: 12
            #obs[0, 7:19] = (q - action_startup) * cfg.normalization.obs_scales.dof_pos #dof_pos: 12
            
            obs[0, 19:31] = dq * cfg.normalization.obs_scales.dof_vel #angular velocity: 12
            obs[0, 31:43] = action


            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

            action[:] = policy(torch.tensor(obs))[0].detach().numpy()

            clip_actions = cfg.normalization.clip_actions

            action = np.clip(action, -clip_actions, clip_actions)

            if count_lowlevel < 100:

                action[:] = (action_startup[:] / 100 *(100 - count_lowlevel) + action[:] / 100 *count_lowlevel)

    
            target_q = action * cfg.control.action_scale

        #Calculate torques
        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)

        tau = pd_control(target_q, q, cfg.robot_config.kps,
                         target_dq, dq, cfg.robot_config.kds)  # Calc torques
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit) 

        data.ctrl = tau
        applied_tau = data.actuator_force

        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1
    viewer.close()

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Deployment script.')
    # parser.add_argument('--load_model', type=str, required=True, help='Run to load from.')
    args = parser.parse_args()

    class Sim2simCfg(Go1RecFlatConfig):


        class sim_config:
            mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/a1.xml'
            # mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/zq_box_foot.xml'
            sim_duration = 60.0 
            dt = 0.001
            decimation = 10 

        class robot_config:
            kps = np.array([20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20], dtype=np.double)
            kds = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.double)
            tau_limit = 50. * np.ones(12, dtype=np.double)
            default_dof_pos = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    # if args.load_model is None:
    #     args.load_model = f'{LEGGED_GYM_ROOT_DIR}/logs/flat_unitree_go1/exported/policies/policy_1.pt'
    load_model = f'{LEGGED_GYM_ROOT_DIR}/logs/unitree_go1_recover_flat/exported/policies/policy_1.pt'

    try:
        policy = torch.jit.load(load_model)
        print("Model loaded successfully.")
        run_mujoco(policy, Sim2simCfg())
    except RuntimeError as e:
        print(f"Error loading the model: {e}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")













