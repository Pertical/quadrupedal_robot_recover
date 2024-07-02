

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

#Command for the robot to follow in the simulation

class cmd:

    base_height = 0.0 

def quaternion_to_euler_array(quat):

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
    """

    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('Body_Quat').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double) #velocity in body frame
    omega = data.sensor('Body_Gyro').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double) #gravity vector in body frame

    return (q, dq, quat, v, omega, gvec) #return as a tuple




def run_mujoco(policy, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """

    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)

    action_startup = np.array([0., 0.8, -1.5, 0., 0.8, -1.5, 0., 0.8, -1.5, 0., 0.8, -1.5])
    data.qpos[-num_actuated_joints:] = action_startup[:]

    


    num_actuated_joints = cfg.env.num_actions
    data.qpos[-num_actuated_joints:] = cfg.robot_config.default_dof_pos

    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)
    
    
    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)

    hist_obs = deque()

    # for _ in range(cfg.env.frame_stack):
    #     hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))

    count_lowlevel = 1
    logger = Logger(cfg.sim_config.dt)

    stop_state_log = 3000

    np.set_printoptions(formatter={'float': '{:0.4f}'.format})

    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):

        #Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.env.num_actions:]
        dq = dq[-cfg.env.num_actions:]

        if count_lowlevel % cfg.sim_config.decimation == 0: 

            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi

            obs[0, 0:3] = omega #base angular velocity
            obs[0, 3:6] = gvec #projected gravity
            obs[0, 6:7] = np.array([cmd.base_height]) #commands
            obs[0, 7:19] = (q - cfg.robot_config.default_dof_pos) * cfg.obs_scales.dof_pos #dof_pos: 12
            obs[0, 19:31] = dq * cfg.obs_scales.dof_vel #angular velocity: 12
            obs[0, 31:43] = action


            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

            # hist_obs.append(obs)
            # hist_obs.popleft()

            # policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            # for i in range(cfg.env.frame_stack):
            #     policy_input[0, i * cfg.env.num_single_obs:(i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]
            
            action[:] = policy(torch.tensor(obs))[0].detach().numpy()

            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
            target_q = action * cfg.control.action_scale

        #Calculate torques
        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)

        tau = pd_control(target_q, q, cfg.control.stiffness, target_dq, dq, cfg.control.damping, cfg) 
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

        default_dof_pos = np.array([0., 0.8, -1.5, 0., 0.8, -1.5, 0., 0.8, -1.5, 0., 0.8, -1.5])

        class sim_config:
            mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.xml'
            # mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/zq_box_foot.xml'
            sim_duration = 60.0 
            dt = 0.005
            decimation = 10 

        class robot_config:
            kps = np.array([200, 200, 350, 350, 15, 15, 200, 200, 350, 350, 15, 15], dtype=np.double)
            kds = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype=np.double)
            tau_limit = 200. * np.ones(12, dtype=np.double)
            default_dof_pos = np.array([0., 0.8, -1.5, 0., 0.8, -1.5, 0., 0.8, -1.5, 0., 0.8, -1.5])

    # if args.load_model is None:
    #     args.load_model = f'{LEGGED_GYM_ROOT_DIR}/logs/flat_unitree_go1/exported/policies/policy_1.pt'
    load_model = f'{LEGGED_GYM_ROOT_DIR}/logs/flat_unitree_go1/exported/policies/policy_1.pt'

    try:
        policy = torch.jit.load(load_model)
        print("Model loaded successfully.")
        run_mujoco(policy, Sim2simCfg())
    except RuntimeError as e:
        print(f"Error loading the model: {e}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")













