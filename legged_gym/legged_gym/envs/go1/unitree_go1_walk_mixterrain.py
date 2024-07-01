

"""
Config for simulation Unitree go1 recover from fail on flat terrain
Adopted from ETH Zurich "legged_gym example configs"

"""

from legged_gym.envs.base.legged_robot_walk_config import LeggedRobotWalkCfg, LeggedRobotWalkCfgPPO


class Go1RecFlatConfig(LeggedRobotWalkCfg):

    class env(LeggedRobotWalkCfg.env):


        # num_observations = 49 #This includes 4 commands
        #num_observations = 46 #This includes 1 command

        num_observations = 43 #This includes 1 command and without base lin vel

        episode_length_s = 20.

        num_envs = 4096

    

    class terrain(LeggedRobotWalkCfg.terrain):

        mesh_type = 'plane'
        measure_heights = False


    class commands(LeggedRobotWalkCfg.commands):

        heading_command = False
        
        resampling_time = 10.

        base_height_command = True
        default_base_height = 0.25

        if base_height_command: 
            num_commands = 1 #Only one command for base height

        else: 
            num_commands = 4

        class ranges:
            lin_vel_x = [0., 0.]
            lin_vel_y = [0., 0.]
            ang_vel_yaw = [0., 0.]
            heading = [0., 0.]
            base_height = [0.2, 0.3]

    class init_state(LeggedRobotWalkCfg.init_state):


        """
        Unitree Go1 Info:
        # Leg0 FR = right front leg
        # Leg1 FL = left front leg
        # Leg2 RR = right rear leg
        # Leg3 RL = left rear leg

        #Joint limitations: 
        #Hip joint: -60 ~ 60 degrees, in radians: -1.0472 ~ 1.0472
        #Thigh joint: -38 ~ 170 degrees, in radians: -0.663225115 ~ 2.96705973
        #Calf joint: -156 ~ -48 degrees, in radians: -2.70526034 ~ -0.837758041
        
        """

        robot_upside_down = True
        pos = [0.0, 0.0, 0.2]

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
        penalize_contacts_on = ["base", "thigh", "calf"]

        terminate_after_contacts_on = []

    class rewards(LeggedRobotWalkCfg.rewards):

        max_contact_force = 350. 
        tracking_sigma  = 0.01

        class scales(LeggedRobotWalkCfg.rewards.scales):

            #Penalization
            lin_vel_z = -0.0 
            ang_vel_xy = -0.0

            orientation = -0.5

            torques = -0.0
            dof_vel = -0.
            dof_acc = -2.5e-7
            action_rate = -0.002

            collision = -0.0
            termination = -0.0

            dof_pos_limits = -0.0
            dof_vel_limits = -0.0
            torque_limits = -0.0

            feet_stumble = -0.0
            stand_still = -0.0
            feet_contact_forces = -0.0

            dof_power = -0.0

            hip_angle = -3.
            thigh_angle = -1.
            calf_angle = -1.
            # target_dof_pos = -2.0

            lin_vel_xy = -0.0

            #Reward 
            tracking_lin_vel = 0.
            tracking_ang_vel = 0.
            feet_air_time = 0.0 

            base_uprightness = 1.0
            foot_contact = 1.0

            tracking_base_height = 1.0

    class domain_rand(LeggedRobotWalkCfg.domain_rand):

        randomize_friction = True 
        randomize_base_mass = True
        push_robots = True
        push_intervel_s = 2. 
        max_push_vel_xy = 5.

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            base_height = 3.0
            height_measurements = 10.0
        clip_observations = 100.
        clip_actions = 100.


    class noise(LeggedRobotWalkCfg.noise):
        add_noise = False
    
class Go1WalkFlatConfigPPO(LeggedRobotWalkCfgPPO):



    class runner(LeggedRobotWalkCfgPPO.runner):

        
        #load_run = r"/home/bridge/Desktop/legged_gym/logs/flat_unitree_go1/Jun24_12-12-20_go1_flat" #This one can recover, but poor position. 
        #load_run = r"/home/bridge/Desktop/legged_gym/logs/flat_unitree_go1/Jun24_14-28-43_go1_flat"

        max_iterations = 2500
        num_steps_per_env = 48 # 30 steps per env

        #logging
        save_interval = 100

        run_name = 'go1_recover_flat'
        experiment_name = 'flat_unitree_go1'
        load_run = -1
        checkpoint = -1 
        resume_path = -1 





        
    


        