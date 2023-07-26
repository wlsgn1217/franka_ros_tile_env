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

PATH = '/home/jinhoo/franka_asl/catkin_ws/src/franka_ros/franka_example_controllers/scripts/checkpoint/FactoryTaskMoveInterlockingTileNew.pth'

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

        
        self.setEEframe([0.7071068, 0.7071068, 0, 0, -0.7071068, 0.7071068, 0, 0, 0, 0, 1, 0, 0, 0, 0.074867, 1])
        #self.setEEframe([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0.074867, 1])
        self.pose_publisher = rospy.Publisher(
            "equilibrium_pose", PoseStamped, queue_size=10)
        
        self.test_publisher = rospy.Publisher(
            "test", PoseStamped, queue_size=10)
        
        self.franka_state_subscriber = rospy.Subscriber("franka_state_controller/franka_states", FrankaState, self.franka_state_subscriber_callback)
        self.link_state_subscriber = rospy.Subscriber("gazebo/link_states", LinkStates, self.passive_tile_pose_subscriber_callback)

        self.use_pre_defined_trajectory = True
        self.policy_action_scale = 0.1

    def resetEEframe(self):
        inv_init_rotMat = np.linalg.inv(self.ee_rotMat)
        inv_init_rotMat[0:3,3] = np.array([0.0, 0.0, 0.074867])
        EEframe_change = inv_init_rotMat.transpose().flatten().tolist()
        self.setEEframe(EEframe_change)

    def initialize_states(self):
        self.passive_tile_pose = Pose()
        self.passive_tile_quat = np.zeros(4)
        self.passive_tile_rotMat = np.eye(4)
        self.passive_tile_pos = np.zeros(3)
        self.refreshed_passive_tile_pose_rotMat = np.eye(4)
        self.refreshed_passive_tile_quat = np.zeros(4)
        self.refreshed_passive_tile_euler = np.zeros(3)

        self.ee_pos = np.zeros(3)
        self.ee_quat = np.zeros(4)
        self.ee_rotMat = np.eye(4)
        self.ee_vel = np.zeros(6)
        self.force = np.zeros(3)

        self.phase = 0

        self.pos_action_scale = 0.25
        self.rot_action_scale = 1.0
        self.rot_thresh = 0.001

    def franka_state_subscriber_callback(self,msg):

        # All franka_states are in the robot_base_frame (Not the World Frame - ex. IsaacGym)
        O_T_EE = msg.O_T_EE
        ee_vel = msg.ee_vel
        force = msg.O_F_ext_hat_K

        self.ee_rotMat = np.array(O_T_EE).reshape(4,4).transpose()
        ee_euler = np.array(tf.transformations.euler_from_matrix(self.ee_rotMat))
        self.ee_quat = quat_from_euler_xyz(ee_euler)
        self.ee_pos = self.ee_rotMat[0:3,3]

        self.ee_vel = np.array(ee_vel)
        self.force = np.array(force)[0:3]


        #observation:tile pose(7), tile vel(6), force(3), passive tile pose(7)
        self.observation[0:3] = torch.from_numpy(self.ee_pos)
        self.observation[3:7] = torch.from_numpy(self.ee_quat)
        self.observation[7:13] = torch.tensor(self.ee_vel)
        self.observation[13:16] = torch.from_numpy(self.force)
        self.observation[16:19] = torch.from_numpy(self.passive_tile_pos)
        self.observation[19:23] = torch.from_numpy(self.passive_tile_quat)

    
    def passive_tile_pose_subscriber_callback(self,msg):
        self.passive_tile_pose = msg.pose[1]
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
            action = self.policy_action_scale * action

        self.apply_actions_as_ctrl_targets(action)

        self.pose_publisher.publish(self.target_pose)

        self.test_publisher.publish(self.target_pose)


    def get_manipulation_policy_target(self):
        with torch.no_grad():
            action = self.actor(self.observation).numpy()
        
        d_pos = action[0:3]
        target_pos = self.ee_pos +d_pos
        target_depth_limit = self.passive_tile_pos[2]-0.1
        if target_pos[2] < target_depth_limit:
            target_pos[2] = target_depth_limit

        action = target_pos-self.ee_pos
        return action
    

    def limit_depth_target(self):
        pass

    def find_d_pose(self, target_pos, target_quat):
        pos_error = target_pos - self.ee_pos
        axis_angle_error = get_orientation_error(self.ee_quat, target_quat)

        return pos_error, axis_angle_error
    
    def refresh_passive_tile_pose(self):
        self.refreshed_passive_tile_euler = np.array(
            tf.transformations.euler_from_quaternion(self.passive_tile_quat, 'rxyz')
        )

        self.refreshed_passive_tile_euler[0] = self.refreshed_passive_tile_euler[0] + np.pi
        self.refreshed_passive_tile_euler[2] = self.find_target_orientation(self.refreshed_passive_tile_euler[2])

        self.refreshed_passive_tile_quat = quat_from_euler_xyz(self.refreshed_passive_tile_euler)
        self.refreshed_passive_tile_pose_rotMat = rotMat_from_euler_xyz(self.refreshed_passive_tile_euler,self.passive_tile_pos)

    def get_pre_defined_trajectory(self):
        
        # Z-axis of the tile should face downard, while Z-axis from the pre-assembled tile face upward. 
        # -> 180 degree on y axis turn
        # Assumes there is no roll, so pitch can be simply added (the pre-assembled tile only turns in yaw direction)
        

        dist_from_mid_target = np.zeros((3,1))
        dist_from_mid_target[0] = -0.2
        mid_target_d = np.matmul(self.passive_tile_rotMat[0:3,0:3],dist_from_mid_target)
        
        final_target = self.passive_tile_pos #self.refreshed_passive_tile_pose_rotMat[0:3, 3]
        mid_target = final_target + mid_target_d.flatten()
        phase_change_dist = 0.1
        dist_to_mid_target = np.linalg.norm(self.ee_pos - mid_target)
        self.phase = max((dist_to_mid_target<phase_change_dist), self.phase)

        target_pos = mid_target * (1-self.phase) + final_target * self.phase
        target_quat = self.refreshed_passive_tile_quat
        # if self.phase:
        #     target_pos[0] = target_pos[0] - 0.03

        d_position, d_orientation_angle_axis = self.find_d_pose(target_pos, target_quat)
        scale = 1
        action = np.concatenate((d_position, d_orientation_angle_axis)) * scale

        return action
    
 
    
    
    def apply_actions_as_ctrl_targets(self, action):
        pos_action = self.pos_action_scale * action[0:3]
        ctrl_target_pos = self.ee_pos + pos_action

        rot_action = self.rot_action_scale * action[3:6]
        angle = np.array([np.linalg.norm(rot_action)])
        axis = rot_action / angle
        if np.linalg.norm(rot_action) > self.rot_thresh:
            rot_action_quat = quat_from_angle_axis(angle,axis)
        else:
            rot_action_quat = np.array([0., 0., 0., 1.])
        ctrl_target_quat = tf.transformations.quaternion_multiply(rot_action_quat, self.ee_quat)

        self.generate_pose_command(ctrl_target_pos,ctrl_target_quat)
    
    def generate_pose_command(self, target_pos, target_quat):
        self.target_pose.pose.position.x = target_pos[0]
        self.target_pose.pose.position.y = target_pos[1]
        self.target_pose.pose.position.z = target_pos[2]

        self.target_pose.pose.orientation.x = target_quat[0]
        self.target_pose.pose.orientation.y = target_quat[1]
        self.target_pose.pose.orientation.z = target_quat[2]
        self.target_pose.pose.orientation.w = target_quat[3]
        
    def find_target_orientation(self, yaw):
        yaw_remainder = np.remainder(yaw+np.pi/3,np.pi/3*2)
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


if __name__ == "__main__":
    rospy.init_node("manipulation_policy_node")
    rate = rospy.Rate(200)
    mp = ManipulationPolicy()
    while not rospy.is_shutdown():

        mp.command_pose()
        rate.sleep()
