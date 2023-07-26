#!/usr/bin/env python

import rospy
import tf.transformations
import numpy as np

from geometry_msgs.msg import PoseStamped, Pose
from gazebo_msgs.msg import LinkStates
from franka_msgs.msg import FrankaState
from franka_msgs.srv import SetEEFrame

import torch
import torch.nn as nn

PATH = '/home/jinhoo/franka_asl/catkin_ws/src/franka_ros/franka_example_controllers/scripts/checkpoint/FactoryTaskMoveInterlockingTile_PosNoise_RandomizeFriction_ForcePenalty0007_2.pth'

class Actor(nn.Module):
    def __init__(self, observation_num=23, action_num=6, layer_size=[512,256,128]):
        super(Actor, self).__init__()
        self.observation_num = observation_num
        self.action_num = action_num
        self.layer_size = layer_size
        
        self.device = torch.device('cpu')
        
        self.nn_build()
        self.load_checkpoint()


    def nn_build(self):
        class actor_model(nn.Module):
            def __init__(self, observation_num, action_num, layer_size=[512,256,128]):
                super(actor_model, self).__init__()
                self.fc1 = nn.Linear(observation_num, layer_size[0])
                self.fc2 = nn.Linear(layer_size[0], layer_size[1])
                self.fc3 = nn.Linear(layer_size[1], layer_size[2])
                self.mu = nn.Linear(layer_size[2], action_num)
                self.act = nn.ELU()

            def forward(self, x):
                x = self.act(self.fc1(x))
                x = self.act(self.fc2(x))
                x = self.act(self.fc3(x))
                x = self.mu(x)

                return x

        self.model = actor_model(self.observation_num, self.action_num, self.layer_size)
        self.running_mean_std = RunningMeanStd(self.observation_num)


    def load_checkpoint(self):
        checkpoint = torch.load(PATH, map_location=self.device)
        model_dict = checkpoint['model']
        with torch.no_grad():
            self.model.fc1.weight.copy_(model_dict['a2c_network.actor_mlp.0.weight'])
            self.model.fc1.bias.copy_(model_dict['a2c_network.actor_mlp.0.bias'])
            self.model.fc2.weight.copy_(model_dict['a2c_network.actor_mlp.2.weight'])
            self.model.fc2.bias.copy_(model_dict['a2c_network.actor_mlp.2.bias'])
            self.model.fc3.weight.copy_(model_dict['a2c_network.actor_mlp.4.weight'])
            self.model.fc3.bias.copy_(model_dict['a2c_network.actor_mlp.4.bias'])
            self.model.mu.weight.copy_(model_dict['a2c_network.mu.weight'])
            self.model.mu.bias.copy_(model_dict['a2c_network.mu.bias'])
            self.running_mean_std.running_mean.copy_(model_dict['running_mean_std.running_mean'])
            self.running_mean_std.running_var.copy_(model_dict['running_mean_std.running_var'])
            self.running_mean_std.count.copy_(model_dict['running_mean_std.count'])

    def normalize_input(self, input):
        normalized_input = self.running_mean_std(input)
        return normalized_input
    
    def preprocess_action(self, action):
        clamped_action = torch.clamp(action, -1.0, 1.0)

        return clamped_action


    def forward(self, input):
        normalized_input = self.normalize_input(input)
        action = self.model(normalized_input)
        action = self.preprocess_action(action)

        return action


class RunningMeanStd(nn.Module):
    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False):
        super(RunningMeanStd, self).__init__()
        print('RunningMeanStd: ', insize)
        self.insize = insize
        self.epsilon = epsilon

        self.norm_only = norm_only
        self.per_channel = per_channel
        if per_channel:
            if len(self.insize) == 3:
                self.axis = [0,2,3]
            if len(self.insize) == 2:
                self.axis = [0,2]
            if len(self.insize) == 1:
                self.axis = [0]
            in_size = self.insize[0] 
        else:
            self.axis = [0]
            in_size = insize

        self.register_buffer("running_mean", torch.zeros(in_size, dtype = torch.float64))
        self.register_buffer("running_var", torch.ones(in_size, dtype = torch.float64))
        self.register_buffer("count", torch.ones((), dtype = torch.float64))

    def _update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count

    def forward(self, input, unnorm=False, mask=None):
        if self.training:
            mean = input.mean(self.axis) # along channel axis
            var = input.var(self.axis)
            self.running_mean, self.running_var, self.count = self._update_mean_var_count_from_moments(self.running_mean, self.running_var, self.count, 
                                                    mean, var, input.size()[0] )

        # change shape
        if self.per_channel:
            if len(self.insize) == 3:
                current_mean = self.running_mean.view([1, self.insize[0], 1, 1]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0], 1, 1]).expand_as(input)
            if len(self.insize) == 2:
                current_mean = self.running_mean.view([1, self.insize[0], 1]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0], 1]).expand_as(input)
            if len(self.insize) == 1:
                current_mean = self.running_mean.view([1, self.insize[0]]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0]]).expand_as(input)        
        else:
            current_mean = self.running_mean
            current_var = self.running_var
        # get output


        if unnorm:
            y = torch.clamp(input, min=-5.0, max=5.0)
            y = torch.sqrt(current_var.float() + self.epsilon)*y + current_mean.float()
        else:
            if self.norm_only:
                y = input/ torch.sqrt(current_var.float() + self.epsilon)
            else:
                y = (input - current_mean.float()) / torch.sqrt(current_var.float() + self.epsilon)
                y = torch.clamp(y, min=-5.0, max=5.0)
        return y


class ManipulationPolicy:
    def __init__(self):
        self.link_name = rospy.get_param("~link_name")
        self.target_pose = PoseStamped()
        self.target_pose.header.frame_id = self.link_name
        self.target_pose.header.stamp = rospy.Time(0)
        self.object_states = LinkStates()
        
        self.initialize_states()

        self.observation = torch.zeros(23)
        self.actor = Actor(observation_num=23, action_num=6)

        self.listener = tf.TransformListener()
        self.setEEframe([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0.095, 1])
        self.pose_publisher = rospy.Publisher(
            "equilibrium_pose", PoseStamped, queue_size=10)
        
        self.test_publisher = rospy.Publisher(
            "test", PoseStamped, queue_size=10)
        
        self.franka_state_subscriber = rospy.Subscriber("franka_state_controller/franka_states", FrankaState, self.franka_state_subscriber_callback)
        self.link_state_subscriber = rospy.Subscriber("gazebo/link_states", LinkStates, self.passive_tile_pose_subscriber_callback)

        self.use_pre_defined_trajectory = True

    def initialize_states(self):
        self.passive_tile_pose = Pose()
        self.passive_tile_quat = np.zeros(4)
        self.passive_tile_rotMat = np.eye(4)
        self.passive_tile_pos = np.zeros(3)
        self.refreshed_passive_tile_pose_rotMat = np.eye(4)
        self.refreshed_passive_tile_quat = np.zeros(4)
        self.refreshed_passive_tile_euler = np.zeros(3)

        self.ee_pos_worldFrame = np.zeros(3)
        self.ee_quat_worldFrame = np.zeros(4)

        self.phase = 0

        self.robot_T_world = tf.transformations.rotation_matrix(np.pi,(0,0,1))
        self.robot_T_world[0,3] = 0.5
        self.world_T_robot = np.linalg.inv(self.robot_T_world)
        self.world_T_robot_vel = self.get_world_T_robot_vel()

        self.pos_action_scale = 0.25
        self.rot_action_scale = 0.75
        self.rot_thresh = 0.001

    def get_world_T_robot_vel(self):
        Tmat_vel = np.zeros((6,6))
        world_T_robot = np.linalg.inv(self.robot_T_world)
        world_R_robot = world_T_robot[0:3,0:3]
        P_world_R = world_T_robot[0:3,3]
        skew_P_world_R = skew_symmetrix_matrix(P_world_R)
        Tmat_vel[0:3,0:3] = world_R_robot
        Tmat_vel[0:3,3:6] = np.matmul(skew_P_world_R, world_R_robot)
        Tmat_vel[3:6,3:6] = world_R_robot
        return Tmat_vel

    def franka_state_subscriber_callback(self,msg):

        # All franka_states are in the robot_base_frame (Not the World Frame - ex. IsaacGym)
        O_T_EE = msg.O_T_EE
        ee_vel = msg.ee_vel
        force = msg.O_F_ext_hat_K

        ee_pose_mat_robotFrame =  np.array(O_T_EE).reshape(4,4).transpose(1,0)

        self.ee_pos_worldFrame, self.ee_quat_worldFrame = self.robot_base_pose_to_world_pose(ee_pose_mat_robotFrame)
        self.ee_vel_worldFrame = self.robot_base_vel_to_world_vel(ee_vel)
        self.force_worldFrame = np.matmul(self.world_T_robot[0:3,0:3], np.array(force[0:3]))


        # self.ee_pos_worldFrame, self.ee_quat_worldFrame = self.trans_matrix_to_pos_quat(ee_pose_mat_worldFrame)

        #observation:tile pose(7), tile vel(6), force(3), passive tile pose(7)
        self.observation[0:3] = torch.from_numpy(self.ee_pos_worldFrame)
        self.observation[3:7] = torch.from_numpy(self.ee_quat_worldFrame)
        self.observation[7:13] = torch.tensor(self.ee_vel_worldFrame)
        self.observation[13:16] = torch.from_numpy(self.force_worldFrame)
        self.observation[16:19] = torch.from_numpy(self.passive_tile_pos)
        self.observation[19:23] = torch.from_numpy(self.passive_tile_quat)

    def trans_matrix_to_pos_quat(self, mat):
        mat_np = np.array(mat).reshape(4,4).transpose(1,0)
        euler = np.array(tf.transformations.euler_from_matrix(mat_np, 'rxyz'))
        quat = quat_from_euler_xyz(euler)
        pos = mat_np[0:3,3]

        return pos, quat
    
    def passive_tile_pose_subscriber_callback(self,msg):
        self.passive_tile_pose = msg.pose[2]
        self.passive_tile_quat[0] = self.passive_tile_pose.orientation.x
        self.passive_tile_quat[1] = self.passive_tile_pose.orientation.y
        self.passive_tile_quat[2] = self.passive_tile_pose.orientation.z
        self.passive_tile_quat[3] = self.passive_tile_pose.orientation.w

        self.passive_tile_pos[0] = self.passive_tile_pose.position.x 
        self.passive_tile_pos[1] = self.passive_tile_pose.position.y
        self.passive_tile_pos[2] = self.passive_tile_pose.position.z

        passive_tile_euler = np.array(tf.transformations.euler_from_quaternion(self.passive_tile_quat, 'rxyz'))
        self.passive_tile_rotMat = rotMat_from_euler_xyz(passive_tile_euler,self.passive_tile_pos)

        self.refresh_passive_tile_pose()

    def command_pose(self):
        if self.use_pre_defined_trajectory:
            action = self.get_pre_defined_trajectory()
        else:
            action = self.get_manipulation_policy_target()

        self.apply_actions_as_ctrl_targets(action)

        self.pose_publisher.publish(self.target_pose)
        # self.target_pose.pose.position.x = self.ee_pos_worldFrame[0]
        # self.target_pose.pose.position.y = self.ee_pos_worldFrame[1]
        # self.target_pose.pose.position.z = self.ee_pos_worldFrame[2]

        # self.target_pose.pose.orientation.x = self.ee_quat_worldFrame[0]
        # self.target_pose.pose.orientation.y = self.ee_quat_worldFrame[1]
        # self.target_pose.pose.orientation.z = self.ee_quat_worldFrame[2]
        # self.target_pose.pose.orientation.w = self.ee_quat_worldFrame[3]

        self.test_publisher.publish(self.target_pose)


    def get_manipulation_policy_target(self):
        with torch.no_grad():
            action = self.actor(self.observation).numpy()
        return action
    
    def find_d_pose(self, target_pos, target_quat):
        pos_error = target_pos - self.ee_pos_worldFrame
        axis_angle_error = get_orientation_error(self.ee_quat_worldFrame, target_quat)

        return pos_error, axis_angle_error
    
    def refresh_passive_tile_pose(self):
        self.refreshed_passive_tile_euler = np.array(
            tf.transformations.euler_from_quaternion(self.passive_tile_quat, 'rxyz')
        )
        self.refreshed_passive_tile_euler[1] = self.refreshed_passive_tile_euler[1] + np.pi
        self.refreshed_passive_tile_euler[2] = self.find_target_orientation(self.refreshed_passive_tile_euler[2])


        self.refreshed_passive_tile_quat = quat_from_euler_xyz(self.refreshed_passive_tile_euler)
        self.refreshed_passive_tile_pose_rotMat = rotMat_from_euler_xyz(self.refreshed_passive_tile_euler,self.passive_tile_pos)

    def get_pre_defined_trajectory(self):
        
        # Z-axis of the tile should face downard, while Z-axis from the pre-assembled tile face upward. 
        # -> 180 degree on y axis turn
        # Assumes there is no roll, so pitch can be simply added (the pre-assembled tile only turns in yaw direction)
        

        dist_from_mid_target = np.zeros((3,1))
        dist_from_mid_target[1] = 0.1
        mid_target_d = np.matmul(self.passive_tile_rotMat[0:3,0:3],dist_from_mid_target)
        
        final_target = self.refreshed_passive_tile_pose_rotMat[0:3, 3]
        mid_target = final_target + mid_target_d.flatten()
        phase_change_dist = 0.01
        dist_to_mid_target = np.linalg.norm(self.ee_pos_worldFrame - mid_target)
        self.phase = max((dist_to_mid_target<phase_change_dist), self.phase)

        target_pos = mid_target * (1-self.phase) + final_target * self.phase
        target_quat = self.refreshed_passive_tile_quat
        # if self.phase:
        #     target_pos[0] = target_pos[0] - 0.03

        d_position, d_orientation_angle_axis = self.find_d_pose(target_pos, target_quat)
        scale = 1
        action = np.concatenate((d_position, d_orientation_angle_axis)) * scale

        # target_rotMat = np.eye(4)
        # target_rotMat[0:3,0:3] = self.refreshed_passive_tile_pose_rotMat[0:3,0:3]
        # target_rotMat[0:3,3] = target_pos


        # passive_tile_pos_robotBaseFrame, passive_tile_quat_robotBaseFrame = self.world_pose_to_robot_base_pose(target_rotMat)

        # self.target_pose.pose.position.x = passive_tile_pos_robotBaseFrame[0]
        # self.target_pose.pose.position.y = passive_tile_pos_robotBaseFrame[1]
        # self.target_pose.pose.position.z = passive_tile_pos_robotBaseFrame[2]

        # self.target_pose.pose.orientation.x = passive_tile_quat_robotBaseFrame[0]
        # self.target_pose.pose.orientation.y = passive_tile_quat_robotBaseFrame[1]
        # self.target_pose.pose.orientation.z = passive_tile_quat_robotBaseFrame[2]
        # self.target_pose.pose.orientation.w = passive_tile_quat_robotBaseFrame[3]
        return action
    
    def apply_actions_as_ctrl_targets(self, action):
        pos_action = self.pos_action_scale * action[0:3]
        ctrl_target_pos = self.ee_pos_worldFrame + pos_action

        rot_action = self.rot_action_scale * action[3:6]
        angle = np.array([np.linalg.norm(rot_action)])
        axis = rot_action / angle
        if np.linalg.norm(rot_action) > self.rot_thresh:
            rot_action_quat = quat_from_angle_axis(angle,axis)
        else:
            rot_action_quat = np.array([0., 0., 0., 1.])
        ctrl_target_quat = tf.transformations.quaternion_multiply(rot_action_quat, self.ee_quat_worldFrame)

        self.generate_pose_command(ctrl_target_pos,ctrl_target_quat)
    
    def generate_pose_command(self, target_pos, target_quat):
        target_euler = np.array(tf.transformations.euler_from_quaternion(target_quat, 'rxyz'))
        target_rotMat = rotMat_from_euler_xyz(target_euler)
        target_rotMat[0:3,3] = target_pos


        passive_tile_pos_robotBaseFrame, passive_tile_quat_robotBaseFrame = self.world_pose_to_robot_base_pose(target_rotMat)

        self.target_pose.pose.position.x = passive_tile_pos_robotBaseFrame[0]
        self.target_pose.pose.position.y = passive_tile_pos_robotBaseFrame[1]
        self.target_pose.pose.position.z = passive_tile_pos_robotBaseFrame[2]

        self.target_pose.pose.orientation.x = passive_tile_quat_robotBaseFrame[0]
        self.target_pose.pose.orientation.y = passive_tile_quat_robotBaseFrame[1]
        self.target_pose.pose.orientation.z = passive_tile_quat_robotBaseFrame[2]
        self.target_pose.pose.orientation.w = passive_tile_quat_robotBaseFrame[3]

    def world_pose_to_robot_base_pose(self, rotMat):
        rotMat_robotBaseFrame = np.matmul(self.robot_T_world, rotMat)
        euler_robotBaseFrame = np.array(tf.transformations.euler_from_matrix(rotMat_robotBaseFrame))
        quat_robotBaseFrame = quat_from_euler_xyz(euler_robotBaseFrame)
        pos_robotBaseFrame = rotMat_robotBaseFrame[0:3,3]

        return pos_robotBaseFrame, quat_robotBaseFrame
    
    def robot_base_pose_to_world_pose(self, rotMat):
        rotMat_worldFrame = np.matmul(self.world_T_robot, rotMat)
        euler_worldFrame = np.array(tf.transformations.euler_from_matrix(rotMat_worldFrame))
        quat_worldFrame = quat_from_euler_xyz(euler_worldFrame)
        pos_worldFrame= rotMat_worldFrame[0:3,3]

        return pos_worldFrame, quat_worldFrame


    #https://physics.stackexchange.com/questions/197009/transform-velocities-from-one-frame-to-an-other-within-a-rigid-body
    def robot_base_vel_to_world_vel(self, vel):
        world_vel = np.matmul(self.world_T_robot_vel, vel)

        return world_vel

        
    def find_target_orientation(self, yaw):
        yaw_remainder = np.remainder(yaw,np.pi/3*2)
        if yaw_remainder<=np.pi/3:
            target_yaw = yaw_remainder
        else:
            target_yaw = yaw_remainder-np.pi/3*2
        return target_yaw

    def setEEframe(self, NE_T_EE):
        rospy.wait_for_service('/franka_control/set_EE_frame')
        try:
            resp = rospy.ServiceProxy('/franka_control/set_EE_frame', SetEEFrame)(NE_T_EE)
            print ("setEEframe: ", resp)
        except rospy.ServiceException as e:
            print ("Service call failed: %s"%e)

def skew_symmetrix_matrix(xyz):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]

    skew_mat = np.matrix([[0, -z, y],
                          [z, 0, -x],
                          [-y, x, 0]])
    
    return skew_mat


def quat_from_euler_xyz(rpy):

    roll = rpy[0]
    pitch = rpy[1]
    yaw = rpy[2]

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return np.array([qx, qy, qz, qw])

def rotMat_from_euler_xyz(rpy, pos=np.zeros(3)):
    roll = rpy[0]
    pitch = rpy[1]
    yaw = rpy[2]

    Rx = tf.transformations.rotation_matrix(roll,(1,0,0))
    Ry = tf.transformations.rotation_matrix(pitch,(0,1,0))
    Rz = tf.transformations.rotation_matrix(yaw,(0,0,1))

    rotMat = tf.transformations.concatenate_matrices(Rx, Ry, Rz)
    rotMat[0:3,3] = pos

    return rotMat

def get_orientation_error(current_quat, target_quat):
        # quat_inv = tf.transformations.quaternion_inverse(current_quat)
        quat_conj = tf.transformations.quaternion_conjugate(current_quat)
        quat_norm = tf.transformations.quaternion_multiply(
            current_quat, quat_conj
        )[3]
        quat_inv = quat_conj / quat_norm
        quat_error = tf.transformations.quaternion_multiply(target_quat, quat_inv)
        axis_angle_error = axis_angle_from_quat(quat_error)

        return axis_angle_error
    
def axis_angle_from_quat(quat, eps=1.0e-6):
    mag = np.linalg.norm(quat[0:3])
    half_angle = np.arctan2(mag, quat[3])
    angle = 2.0 * half_angle
    if np.abs(angle) > eps:
        sin_half_angle_over_angle = np.sin(half_angle) / angle
    else:
        sin_half_angle_over_angle = 1 / 2 - angle ** 2.0 / 48
    axis_angle = quat[0:3] / sin_half_angle_over_angle
    
    return axis_angle

def quat_from_angle_axis(angle: np.array, axis: np.array):
    theta = (angle / 2)
    xyz = normalize(axis) * np.sin(theta)
    w = np.cos(theta)

    return normalize(np.concatenate((xyz, w)))

def normalize(x, eps: float = 1e-9):
    return x / np.linalg.norm(x)

# target_pose = PoseStamped()
# object_states = LinkStates()
# pose_pub = None
# # [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
# position_limits = [[-0.6, 0.6], [-0.6, 0.6], [0.05, 0.9]]

# #change the following transformation matrix if the configuration of the franka_interlocking_tile changes
# # tile_T_j7 = tf.transformations.rotation_matrix(np.pi/4,(0,0,1))
# # tile_T_j7[2,3] = -0.107 - 0.065
# # j7_T_tile = np.linalg.inv(tile_T_j7)
# robot_T_world = tf.transformations.rotation_matrix(np.pi,(0,0,1))
# robot_T_world[0,3] = 0.5


# def publisher_callback(msg, link_name):
#     target_pose.header.frame_id = link_name
#     target_pose.header.stamp = rospy.Time(0)
#     pose_pub.publish(target_pose)

# def passive_tile_pose_subscriber_callback(link_states):
#     target_pose.header.frame_id = link_name
#     target_pose.header.stamp = rospy.Time(0)
#     passive_tile_pose = link_states.pose[2]
    
#     passive_tile_quat = np.zeros(4)
#     passive_tile_quat[0] = passive_tile_pose.orientation.x
#     passive_tile_quat[1] = passive_tile_pose.orientation.y
#     passive_tile_quat[2] = passive_tile_pose.orientation.z
#     passive_tile_quat[3] = passive_tile_pose.orientation.w

#     passive_tile_euler = np.array(
#         tf.transformations.euler_from_quaternion(passive_tile_quat, 'rxyz')
#     )



#     # Z-axis of the tile should face downard, while Z-axis from the pre-assembled tile face upward. 
#     # -> 180 degree on y axis turn
#     # Assumes there is no roll, so pitch can be simply added (the pre-assembled tile only turns in yaw direction)
#     passive_tile_euler[1] = passive_tile_euler[0] + np.pi
#     passive_tile_euler[2] = find_target_orientation(passive_tile_euler[2])
#     # passive_tile_euler[2] = find_target_orientation(passive_tile_euler[2])
#     Rx = tf.transformations.rotation_matrix(passive_tile_euler[0],(1,0,0))
#     Ry = tf.transformations.rotation_matrix(passive_tile_euler[1],(0,1,0))
#     Rz = tf.transformations.rotation_matrix(passive_tile_euler[2],(0,0,1))
#     passive_tile_rotMat = tf.transformations.concatenate_matrices(Rx, Ry, Rz)

#     passive_tile_rotMat[0,3] = passive_tile_pose.position.x 
#     passive_tile_rotMat[1,3] = passive_tile_pose.position.y
#     passive_tile_rotMat[2,3] = passive_tile_pose.position.z

#     dist_from_mid_target = np.zeros((3,1))
#     dist_from_mid_target[0] = -0.3
#     mid_target = np.matmul(passive_tile_rotMat[0:3,0:3],dist_from_mid_target)
#     mid_target = np.matmul(robot_T_world[0:3,0:3],mid_target)

#     passive_tile_rotMat_robotBaseFrame = np.matmul(robot_T_world, passive_tile_rotMat)

#     target_pose.pose.position.x = passive_tile_rotMat_robotBaseFrame[0,3] -0.3
#     target_pose.pose.position.y = passive_tile_rotMat_robotBaseFrame[1,3]
#     target_pose.pose.position.z = passive_tile_rotMat_robotBaseFrame[2,3]

#     passive_tile_quat_robotBaseFrame = tf.transformations.quaternion_from_matrix(passive_tile_rotMat_robotBaseFrame)

#     target_pose.pose.orientation.x = passive_tile_quat_robotBaseFrame[0]
#     target_pose.pose.orientation.y = passive_tile_quat_robotBaseFrame[1]
#     target_pose.pose.orientation.z = passive_tile_quat_robotBaseFrame[2]
#     target_pose.pose.orientation.w = passive_tile_quat_robotBaseFrame[3]


#     #The command frame is in link7, which is 0.095m above(in z-direction) from the base of the tile


#     pose_pub.publish(target_pose)

# def find_target_orientation(yaw):
#     yaw_remainder = np.remainder(yaw+np.pi/3,np.pi/3*2)
#     if yaw_remainder<=np.pi/3:
#         target_yaw = yaw_remainder
#     else:
#         target_yaw = yaw_remainder-np.pi/3*2
#     return target_yaw

# def wait_for_initial_pose():
#     msg = rospy.wait_for_message("franka_state_controller/franka_states",
#                                  FrankaState)  # type: FrankaState

#     initial_quaternion = \
#         tf.transformations.quaternion_from_matrix(
#             np.transpose(np.reshape(msg.O_T_EE,
#                                     (4, 4))))
#     initial_quaternion = initial_quaternion / \
#         np.linalg.norm(initial_quaternion)
    
# def rotMat_from_quat(quat):
#     euler = np.array(
#         tf.transformations.euler_from_quaternion(quat, 'rzyx'))
#     Rx = tf.transformations.rotation_matrix(euler[0],(1,0,0))
#     Ry = tf.transformations.rotation_matrix(euler[1],(0,1,0))
#     Rz = tf.transformations.rotation_matrix(euler[2],(0,0,1))
#     rotMat = tf.transformations.concatenate_matrices(Rx, Ry, Rz)
#     return rotMat

# # def get_tile_pose(joint7_pose):
# #     j7_quaternion = np.zeros(4)
# #     j7_position = np.ones(4)

# #     j7_quaternion[0] = joint7_pose.orientation.x
# #     j7_quaternion[1] = joint7_pose.orientation.y
# #     j7_quaternion[2] = joint7_pose.orientation.z
# #     j7_quaternion[3] = joint7_pose.orientation.w
    
# #     j7_pose = rotMat_from_quat(j7_quaternion)

# #     j7_pose[0,3] = joint7_pose.position.x
# #     j7_pose[1,3] = joint7_pose.position.y
# #     j7_pose[2,3] = joint7_pose.position.z
    
# #     tile_pose = np.matmul(tile_T_j7,j7_pose)

#     # return tile_pose

# def phase_check():
#     pass

#     #the command must be in robot frame
# def world_to_robot_frame():
#     pass


# def setEEframe(NE_T_EE):
#     rospy.wait_for_service('/franka_control/set_EE_frame')
#     try:
#         resp = rospy.ServiceProxy('/franka_control/set_EE_frame', SetEEFrame)(NE_T_EE)
#     except rospy.ServiceException as e:
#         print ("Service call failed: %s"%e)


# def franka_state_subscriber_callback(franka_state):
#     O_T_EE = franka_state.O_T_EE
#     ee_vel = franka_state.ee_vel
#     force = franka_state.O_F_ext_hat_K

#     #observation:tile pose(7), tile vel(6), force(3), passive tile pose(7)

#     target_pose = O_T_EE
#     pose_pub.publish(target_pose)

if __name__ == "__main__":
    # rospy.init_node("manipulation_policy_node")
    # listener = tf.TransformListener()
    # link_name = rospy.get_param("~link_name")
    # setEEframe([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0.095, 1])
    # pose_pub = rospy.Publisher(
    #     "equilibrium_pose", PoseStamped, queue_size=10)

    # rate = rospy.Rate(10)
    # # # wait_for_initial_pose()
    # while not rospy.is_shutdown():
    #     rospy.Subscriber("gazebo/link_states", LinkStates, passive_tile_pose_subscriber_callback)
    #     # rospy.Subscriber("franka_state_controller/franka_states", FrankaState, franka_state_subscriber_callback)

    #     rate.sleep()
    
    # # run pose publisher

    # rospy.spin()
    rospy.init_node("manipulation_policy_node")
    rate = rospy.Rate(200)
    mp = ManipulationPolicy()
    # while not rospy.is_shutdown():

    #     mp.command_pose()
    #     rate.sleep()
