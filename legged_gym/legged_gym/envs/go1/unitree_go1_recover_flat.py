

"""
Config for simulation Unitree go1 recover from fail on flat terrain
Adopted from ETH Zurich "legged_gym example configs"

"""


from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Go1RecFlatConfig(LeggedRobotCfg):

    class env(LeggedRobotCfg.env):

        num_observations = 49
        episode_length_s = 20.

        num_envs = 4096

    class terrain(LeggedRobotCfg.terrain):

        mesh_type = 'plane'
        measure_heights = False


    class commands(LeggedRobotCfg.commands):

        heading_command = False
        num_commands = 4
        resampling_time = 5.

        base_height_command = True
        default_base_height = 0.25

        class ranges:
            lin_vel_x = [0., 0.]
            lin_vel_y = [0., 0.]
            ang_vel_yaw = [0., 0.]
            heading = [0., 0.]
            base_height = [0.2, 0.3]

    class init_state(LeggedRobotCfg.init_state):


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

    
    class control(LeggedRobotCfg.control):

        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]

        action_scale = 0.25
        decimation = 4

        use_actuator_network = False


    
    class asset(LeggedRobotCfg.asset):

        self_collision = 1 # 1 to disable, 0 to enable...bitwise filter
        
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/Xinhua_go1.urdf"

        name = "go1"
        foot_name = "foot"
        penalize_contacts_on = ["base", "thigh", "calf"]

        terminate_after_contacts_on = []

    class rewards(LeggedRobotCfg.rewards):

        max_contact_force = 350. 


        class scales(LeggedRobotCfg.rewards.scales):


            #Penalization
            lin_vel_z = -0.0 
            ang_vel_xy = -0.0

            orientation = -0.1

            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            action_rate = -0.002

            collision = -0.1 
            termination = -0.0

            dof_pos_limits = -0.1
            dof_vel_limits = -0.0
            torque_limits = -0.0

            feet_stumble = -0.0
            stand_still = -0.0
            feet_contact_forces = -0.0

            dof_power = -0.0

            # hip_angle = -2.
            # thigh_angle = 2.
            # calf_angle = -2.
            target_dof_pos = -1.5

            lin_vel_xy = -0.5

            #Reward 
            tracking_lin_vel = 0.
            tracking_ang_vel = 0.
            feet_air_time = 0.0 

            base_uprightness = 2.0 
            foot_contact = 1.0

            tracking_base_height = 0.5

    class domain_rand(LeggedRobotCfg.domain_rand):

        randomize_friction = False 
        push_robots = False
        push_intervel_s = 5. 
        max_push_vel_xy = 1.

    
class Go1RecFlatConfigPPO(LeggedRobotCfgPPO):


    class runner(LeggedRobotCfgPPO.runner):

        
        #load_run = r"/home/bridge/Desktop/legged_gym/logs/flat_unitree_go1/Jun24_12-12-20_go1_flat" #This one can recover, but poor position. 
        #load_run = r"/home/bridge/Desktop/legged_gym/logs/flat_unitree_go1/Jun24_14-28-43_go1_flat"

        max_iterations = 1501
        num_steps_per_env = 24 # 30 steps per env

        #logging
        save_interval = 100

        run_name = ''
        experiment_name = 'flat_unitree_go1'
        load_run = -1
        checkpoint = -1 
        resume_path = None 





        
    


        