#!/usr/bin/env python
# coding=utf-8
'''
Author:Tai Lei
Date:Thu 16 Mar 2017 09:55:00 PM WAT
Info:
'''

import gym
import scipy.signal
import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.optim as optim
from environment import AtariEnv
import torch.multiprocessing as mp
import multiprocessing

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

class A3CLSTMNet(nn.Module):

    def __init__(self, state_shape, action_dim):
        super(A3CLSTMNet, self).__init__()
        self.state_shape = state_shape 
        self.action_dim = action_dim
        self.conv1 = nn.Conv2d(self.state_shape[0],32,3,stride=2)
        self.conv2 = nn.Conv2d(32,32,3,stride=2, padding = 1)
        self.conv3 = nn.Conv2d(32,32,3,stride=2, padding = 1)
        self.conv4 = nn.Conv2d(32,32,3,stride=2, padding = 1)
        self.lstm = nn.LSTMCell(3*3*32,256,1)
        # hang policy output
        self.linear_policy_1 = nn.Linear(256,self.action_dim)
        self.softmax_policy = nn.Softmax()
        # hang value output
        self.linear_value_1 = nn.Linear(256,1)
        
        self.apply(weights_init)
        self.linear_policy_1.weight.data = normalized_columns_initializer(
            self.linear_policy_1.weight.data, 0.01)
        self.linear_policy_1.bias.data.fill_(0)
        self.linear_value_1.weight.data = normalized_columns_initializer(
            self.linear_value_1.weight.data, 1.0)
        self.linear_value_1.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
    
    def forward(self, x, hidden):
        x = x.view(-1, self.state_shape[0], 
                self.state_shape[1],self.state_shape[2]) 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 3*3*32) 
        x,c = self.lstm(x, (hidden[0],hidden[1]))
        pl = self.linear_policy_1(x)
        pl = self.softmax_policy(pl)
        v = self.linear_value_1(x)
        return pl,v,(x,c)

class A3CSingleProcess(mp.Process):
    
    def __init__(self, process_id, master, logger_):
        super(A3CSingleProcess, self).__init__(name="process_%d" % process_id)
        self.process_id = process_id
        self.logger_ = logger_
        self.master = master
        self.args = master.args
        self.env = AtariEnv(gym.make(self.args.game), self.args.frame_seq,self.args.frame_skip)
        self.local_model = A3CLSTMNet(self.env.state_shape, self.env.action_dim)
        # sync the weights at the beginning
        self.sync_network() 
        self.loss_history = []
        self.win = None
        self.state_final = None
        self.Image = None
    def sync_network(self): 
        self.local_model.load_state_dict(self.master.shared_model.state_dict()) 
    
    def forward_explore(self, hidden):
        terminal = False
        t_start = 0
        rollout_path = {"state": [], "action": [], "rewards": [], "done": []}
        pl_roll = []
        v_roll = []
        while not terminal and (t_start <= self.args.t_max):
            t_start += 1
            state_ = self.env.state
            state_tensor = Variable(
                    torch.from_numpy(state_).float())
            pl, v, hidden = self.local_model(state_tensor,hidden)
            pl_roll.append(pl)
            v_roll.append(v)
            
            action = pl.multinomial().data.numpy()[0]
            self.state_final, reward, terminal = self.env.forward_action(action)
            
            rollout_path["state"].append(state_)
            rollout_path["action"].append(action)
            rollout_path["rewards"].append(reward)
            rollout_path["done"].append(terminal) 
        
        return rollout_path, hidden, pl_roll, v_roll
        
    def discount(self, x):
        return scipy.signal.lfilter([1], [1, -self.args.gamma], x[::-1], axis=0)[::-1][:-1]

    def run(self):
        self.env.reset_env()
        loop = 0
        lstm_h = Variable(torch.zeros(1,256))
        lstm_c = Variable(torch.zeros(1,256))
        while True:
            loop += 1
            
            rollout_path, (lstm_h,lstm_c), p_roll, v_roll= self.forward_explore((lstm_h,lstm_c))
            if rollout_path["done"][-1]:
                rollout_path["rewards"].append(0) 
                self.env.reset_env()
                lstm_h = Variable(torch.zeros(1,256))
                lstm_c = Variable(torch.zeros(1,256))
            else:
                state_tensor = Variable(torch.from_numpy(
                    self.state_final).float()) 
                _, v_t, _ = self.local_model(state_tensor,(lstm_h,lstm_c))
                lstm_h = Variable(lstm_h.data) 
                lstm_c = Variable(lstm_c.data) 
                rollout_path["rewards"].append(v_t.data.numpy())
            
            # calculate rewards 
            rollout_path["returns"] = self.discount(rollout_path["rewards"])
            
            loss = self.PathBackProp(rollout_path, p_roll, v_roll)
            self.loss_visual(loss, loop)
            self.master.main_update_step.value += 1
            self.sync_network()

    def loss_visual(self,loss_, loop_):
        self.loss_history.append(loss_) 
        if loop_>2:
            Y_ = np.array(self.loss_history).reshape(-1,1)
            self.win = self.master.vis.line(Y = Y_, X = np.arange(len(self.loss_history)), win=self.win)
            #self.Image = self.master.vis.image(np.resize(self.state_final,(160,160)), win=self.Image)
    
    def ensure_shared_grads(self, model, shared_model):
        for param, shared_param in zip(model.parameters(), shared_model.parameters()):
            if shared_param.grad is not None:
                return 
            shared_param._grad = param.grad 
    
    def PathBackProp(self, rollout_path_, p_roll, v_roll):
        # backprop of the network both policy and value
        state = np.array(rollout_path_['state'])
        target_q = np.array(rollout_path_['returns'])
        action = np.array(rollout_path_['action'])
        rewards = np.array(rollout_path_['rewards'])
        #tensor_target_q = torch.from_numpy(target_q).float()
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1,1)
        for i in reversed(range(len(p_roll))):
            log_prob = torch.log(p_roll[i])
            entropy = - torch.dot(log_prob, p_roll[i])
            log_prob_ = log_prob.gather(1, Variable(torch.from_numpy(action[i].reshape(1,1))))
            advantage = Variable(torch.from_numpy(np.array(target_q[i]).reshape(1,1)).float())-v_roll[i] 
            value_loss = value_loss + 0.5 * advantage.pow(2)
            if i != (len(p_roll)-1):
                delta_t = rewards[i] + self.args.gamma * v_roll[i+1].data - v_roll[i].data 
            else:
                delta_t = rewards[i] + self.args.gamma * rewards[i+1] - v_roll[i].data 
            
            gae = gae * self.args.gamma + delta_t
            policy_loss = policy_loss - log_prob_ * Variable(gae) - 0.01 * entropy 

        self.master.optim.zero_grad()
        loss_all = 0.5* value_loss + policy_loss
        loss_all.backward()
        torch.nn.utils.clip_grad_norm(self.local_model.parameters(), 40)
        self.ensure_shared_grads(self.local_model,self.master.shared_model)
        self.master.optim.step()
        self.logger_.info("pl_loss %f, v_loss %f", 
                policy_loss.data.numpy()[0], 
                value_loss.data.numpy()[0])
        return  loss_all.data.numpy()
