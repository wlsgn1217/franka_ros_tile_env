#!/usr/bin/env python

import torch
import torch.nn as nn

PATH = '/home/jinhoo/franka_asl/catkin_ws/src/franka_ros/franka_example_controllers/scripts/checkpoint/FactoryTaskMoveInterlockingTile_PosNoise_RandomizeFriction_ForcePenalty0007_2.pth'

class actor(nn.Module):
    def __init__(self, observation_num, action_num, layer_size=[512,256,128]):
        super(actor, self).__init__()
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

if __name__ == "__main__":
    a = actor(observation_num=23, action_num=6)
    obs1 = torch.tensor([-1.6389e-02, -3.7289e-02,  5.7867e-01,  4.4269e-02,  9.9897e-01,
         9.7118e-03, -6.4554e-04, -1.4711e-01, -3.8183e-01, -3.1135e-01,
         1.5905e-01, -7.3863e-02, -3.7792e-01,  0.0000e+00,  0.0000e+00,
         0.0000e+00, -1.5265e-01, -5.6620e-02,  4.0311e-01,  0.0000e+00,
         0.0000e+00, -8.7141e-01,  4.9056e-01])
    act1 = a(obs1)

    print(act1)
