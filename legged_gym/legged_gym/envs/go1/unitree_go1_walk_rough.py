

"""
Config for simulation Unitree go1 recover from fail on flat terrain
Adopted from ETH Zurich "legged_gym example configs"

"""
from legged_gym.envs.base.legged_robot_walk_config import LeggedRobotWalkCfg, LeggedRobotCfgPPO

class Go1WalkConfig(LeggedRobotWalkCfg):

    class env(LeggedRobotWalkCfg.env):
        num_envs = 4096
        num_actions = 12
        episode_length_s = 25

    
    class terrain(LeggedRobotWalkCfg.terrain):

         mesh_type = 'trimesh'



    class init_state(LeggedRobotWalkCfg.init_state):

        pos = [0.0, 0.0, 0.6]

        default_joint_angles = {
            'FL_hip_joint': 0.,
            'RL_hip_joint': 0.,
            'FR_hip_joint': 0.,
            'RR_hip_joint': 0.,

            'FL_thigh_joint': 0.8,
            'RL_thigh_joint': 0.8,
            'FR_thigh_joint': 0.8,
            'RR_thigh_joint': 0.8,

            'FL_calf_joint': -1.5,
            'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'RR_calf_joint': -1.5
        }

    
    class control(LeggedRobotWalkCfg.control):

        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]

        action_scale = 0.25
        decimation = 4

        use_actuator_network = False


    class asset(LeggedRobotWalkCfg.asset):

        self_collision = 1 # 1 to disable, 0 to enable...bitwise filter
        
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1_bridgedp.urdf"

        name = "go1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]

        terminate_after_contacts_on = ["base"]

    class rewards( LeggedRobotWalkCfg.rewards ):
        base_height_target = 0.3
        max_contact_force = 350.
        only_positive_rewards = True
        class scales( LeggedRobotWalkCfg.rewards.scales ):
            pass

    class domain_rand(LeggedRobotWalkCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]

        randomize_base_mass = True
        added_mass_range = [-2., 2.]

        push_robots = True 
        push_interval_s = 5 
        max_push_vel_xy = 2. 


class Go1WalkConfigPPO(LeggedRobotCfgPPO):

    class runner(LeggedRobotCfgPPO.runner):

        
        #load_run = r"/home/bridge/Desktop/legged_gym/logs/flat_unitree_go1/Jun24_12-12-20_go1_flat" #This one can recover, but poor position. 
        #load_run = r"/home/bridge/Desktop/legged_gym/logs/flat_unitree_go1/Jun24_14-28-43_go1_flat"

        max_iterations = 5001
        num_steps_per_env = 24 # 30 steps per env

        #logging
        save_interval = 100

        run_name = 'go1_walk_mixterrain'
        experiment_name = 'mixterrain_go1'
        load_run = -1
        checkpoint = -1 
        resume_path = -1 





        
    


        