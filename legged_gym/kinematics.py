

from math import atan, pi, radians, cos, sin
import numpy as np
from numpy.linalg import inv, norm
from numpy import array, asarray, matrix
from math import *
import matplotlib.pyplot as plt


class Inverse_Kinematics():
    
    def __init__(self, robot_type='go1'):

        if robot_type == 'go1':
            
          
         
            #Leg IDs 
            front_left = 0 
            rear_left = 1
            front_right = 2 
            rear_right = 3

            #Note: for go1, only hip angle needs to be adjusted for right legs: mirrored
            self.right_legs = [front_right, rear_right]
            
            #Body dimensions: gathered from the URDF file
            self.hip_offset = 0.08
            self.thigh_len = 0.213
            self.calf_len = 0.213
            
            self.length = 0.3762
            self.width = 0.0935
            self.height = 0.0

            #Leg origins or hip origins w.r.t the center of the robot

            # self.leg_origins = np.array(
            #                 [[self.length/2, self.width/2, self.height],
            #                 [-self.length/2, self.width/2, self.height],
            #                 [-self.length/2, -self.width/2, self.height],
            #                 [self.length/2, -self.width/2, self.height]])
            
            self.leg_origins = np.matrix([[self.length/2, self.width/2, self.height],
                            [-self.length/2, self.width/2, self.height],
                            [-self.length/2, -self.width/2, self.height],
                            [self.length/2, -self.width/2, self.height]])
            

            self.phi = np.pi/2
        
    # this method adjust inputs to the IK calculator by adding rotation and 
    # offset of that rotation from the center of the robot
    def leg_IK(self, xyz, legID=0):

        #Check if the leg is from the right side
        is_right_leg = legID in self.right_legs

        #Convert the target coordinates to numpy array for matrix operations
        target_xyz_np = np.array(xyz)

        #Calculate the position relative to the robot's body frame
        body_frame_pos = np.dot(np.linalg.inv(self.RotMatrix_3D()), 
                                (target_xyz_np + self.leg_origins[legID, :]).T).T
        
        # Leg origin position relative to the body frame
        leg_origin_pos = body_frame_pos - self.leg_origins[legID, :]
        leg_origin_pos = np.array(leg_origin_pos).flatten()

        # calculate the angles and coordinates of the leg relative to the origin of the leg
        return self.leg_IK_calc(leg_origin_pos, is_right_leg)
    

    # IK calculator
    def leg_IK_calc(self, xyz, is_right_leg=False): 

        #Get the foot x, y, z coordinates
        x, y, z = xyz[0], xyz[1], xyz[2] 
        
        # length of vector projected on the YZ plane. equiv. to len_A = sqrt(y**2 + z**2)
        len_A = np.sqrt(y**2 + z**2)

        #Calculate angle between len_A and the leg's projection line on YZ plane
        a_2 = np.arcsin(np.sin(self.phi)*self.hip_offset/len_A)

        #Calculate angle between link_1 and length len_A
        a_3 = np.pi - a_2 - self.phi

        #Calculate the angle from the positive y-axis to the end-effector
        a_1 = self.point_to_rad(y,z)

        #Angle of hip offset about the x-axis in positive y-axis
        
        if is_right_leg: 
            theta_1 = a_1 - a_3
        else: 
            theta_1 = a_1 + a_3

            if theta_1 >= 2*np.pi:
                theta_1 -= 2*np.pi

        #Thight joint coordinates 
        #x, y, z = j2[0], j2[1], j2[2]
        j2 = array([0,self.hip_offset*cos(theta_1),self.hip_offset*sin(theta_1)]) 

        #Foot joint coordinates
        j4 = array(xyz)
        #Vector from j2 to j4, i.e., the vector from the thight to the foot
        j4_2_vec = j4 - j2
    
    
        if is_right_leg:
            R = theta_1 - self.phi - pi/2
        else: 
            R = theta_1 + self.phi - pi/2
        
        # create rotation matrix to work on a new 2D plane (XZ_)
        rot_mtx = self.RotMatrix_3D([-R,0,0])
        j4_2_vec_ = rot_mtx * (np.reshape(j4_2_vec,[3,1]))
        
        # xyz in the rotated coordinate system + offset due to link_1 removed
        x_, y_, z_ = j4_2_vec_[0], j4_2_vec_[1], j4_2_vec_[2]
        
        len_B = norm([x_, z_]) # norm(j4-j2)
        
        # handling mathematically invalid input, i.e., point too far away to reach
        if len_B >= (self.thigh_len + self.calf_len): 
            len_B = (self.thigh_len + self.calf_len) * 0.99999
            # self.node.get_logger().warn('target coordinate: [%f %f %f] too far away' % (x, y, z))
            print('target coordinate: [%f %f %f] too far away' % (x, y, z))
        
        # b_1 : angle between +ve x-axis and len_B (0 <= b_1 < 2pi)
        # b_2 : angle between len_B and link_2
        # b_3 : angle between link_2 and link_3
        b_1 = self.point_to_rad(x_, z_)  
        b_2 = acos((self.thigh_len**2 + len_B**2 - self.calf_len**2) / (2 * self.thigh_len * len_B)) 
        b_3 = acos((self.thigh_len**2 + self.calf_len**2 - len_B**2) / (2 * self.thigh_len * self.calf_len))  
        
        # assuming theta_2 = 0 when the leg is pointing down (i.e., 270 degrees offset from the +ve x-axis)
        theta_2 = b_1 - b_2    
        theta_3 = pi - b_3
        
        # modify angles to match robot's configuration (i.e., adding offsets)
        angles = self.angle_corrector(angles=[theta_1, theta_2, theta_3], is_right_leg=is_right_leg)
        #angles = [theta_1, theta_2, theta_3]
        # print(degrees(angles[0]))
        return [angles[0], angles[1], angles[2]]
    
    def angle_corrector(self, angles=[0,0,0], is_right_leg=True):
        angles[1] -= 1.5*pi; # add offset # 90 degrees
        if is_right_leg:
            theta_1 = angles[0] - pi
            theta_2 = -angles[1]  # 45 degrees initial offset #
        else: 
            if angles[0] > pi:  
                theta_1 = angles[0] - 2*pi
            else: theta_1 = angles[0]
            
            theta_2 = -angles[1] - 0*pi/180
        
        theta_3 = -angles[2] + 0*pi/180
        return [theta_1, theta_2, theta_3]
    
    def point_to_rad(self, p1, p2):
        # Converts 2D cartesian points to polar angles in range -pi to pi
        angle = atan2(p2, p1)
        
        # Adjust angle to be in the range 0 to 2pi
        if angle < 0:
            angle += 2*pi
        
        return angle
    
    def RotMatrix_3D(self, rotation=[0, 0, 0]):
        # rotation matrix about each axis
        roll, pitch, yaw = rotation[0], rotation[1], rotation[2]

        rotX = np.matrix([[1, 0, 0], [0, cos(roll), -sin(roll)], [0, sin(roll), cos(roll)]])
        rotY = np.matrix([[cos(pitch), 0, sin(pitch)], [0, 1, 0], [-sin(pitch), 0, cos(pitch)]])
        rotZ = np.matrix([[cos(yaw), -sin(yaw), 0], [sin(yaw), cos(yaw), 0], [0, 0, 1]])

        return rotZ * rotY * rotX
    

# Example usage:
if __name__ == "__main__":
    kinematics = Inverse_Kinematics()
    xyz_position = [0.0, 0.08, -0.23]  # Example Cartesian position
    legID = 0  # Example leg ID
    angles = kinematics.leg_IK(xyz_position, legID=legID)
    # print("Joint angles:", angles) 

    foot_positions = [
    [0, 0.08, -0.24],  # FL
    [0, 0.08, -0.24],  # RL
    [0, -0.08, -0.25],  # FR
    [0, -0.08, -0.23]] # RR

    for i, foot_positions in enumerate(foot_positions):
        angles = kinematics.leg_IK(foot_positions, legID=i)
        # print("Joint angles:", angles)

        foot_name = ['FL', 'RL', 'FR', 'RR']
        
        print(f"{foot_name[i]}: hip: {angles[0]:.5f}, thigh: {angles[1]:.5f}, calf: {angles[2]:.5f}")


    #About 0, 1, -2
    #For go1 only
    # #FR: hip: 0 thigh: 1, calf: -2
    #FL: hip: 0 thigh: 1, calf: -2
    #RR: hip: 0 thigh: 1, calf: -2
    #RL: hip: 0 thigh: 1, calf: -2 

    """
    Original Output:
    FL: hip: 0.00000, thigh: 1.00047, calf: -2.00094
    FL: hip: 0.00000, thigh: 1.00047, calf: -2.00094
    FL: hip: 0.00000, thigh: -1.00047, calf: -2.00094
    FL: hip: 0.00000, thigh: -1.00047, calf: -2.00094
        
    
    """




