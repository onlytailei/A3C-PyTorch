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
import torch.autograd.Variable as Variable
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
        x,hidden = self.lstm(x, hidden)
        pl = self.linear_policy_1(x)
        pl = self.softmax_policy(pl)
        v = self.linear_value_1(x)
        return pl,v,hidden

class A3CModel(object):
    """
    add PathBackProp compared with A3CNet
    """
    def __init__(self, state_shape,action_dim, args_, logger_):
        
        self.net = A3CLSTMNet(state_shape,action_dim)
        self.action_dim = action_dim 
        self.v_criterion = nn.MSELoss() 
        self.args = args_ 
        self.logger_ = logger_ 
    
    def PathBackProp(self,rollout_path_, lstm_hidden=None):
        # backprop of the network both policy and value
        state = np.array(rollout_path_['state'])
        target_q = np.array(rollout_path_['returns'])
        action = np.array(rollout_path_['action'])
        tensor_target_q = torch.from_numpy(target_q).float().cuda()
        
        batch_size = state.shape[0]
        
        batch_state =  Variable(torch.from_numpy(state).float().cuda())
        batch_action = Variable(torch.from_numpy(action).float().view(-1,self.action_dim,1).cuda())
        batch_target_q = Variable(tensor_target_q)
        
        hidden = (Variable(lstm_hidden[0]),
                        Variable(lstm_hidden[1]))
        pl, v, hidden = self.net(batch_state,hidden)
        pl = pl.view(-1,1,self.action_dim)
        pl_prob = torch.squeeze(torch.bmm(pl,batch_action))
        pl_log = torch.log(pl_prob) 
        diff = tensor_target_q-v.data
        entropy = -torch.dot(pl_prob, torch.log(pl_prob))
        pl_loss = -(torch.dot(pl_log, Variable(diff)) + entropy * self.args.entropy_beta )
        v_loss = self.v_criterion(v, batch_target_q) * batch_size 
        loss_all = 0.5* v_loss + pl_loss
        loss_all.backward()
        self.logger_.info("pl_loss %f, v_loss %f, entropy_loss %f", pl_loss.cpu().data.numpy()[0], v_loss.cpu().data.numpy()[0], entropy.cpu().data.numpy()[0])
        return  loss_all.cpu().data.numpy()
        
        # another way for val loss
        #v_prime = torch.sum((target_q_torch-v)*(target_q_torch-v),0)
        #assert v_loss.data.numpy() == v_prime.data.numpy()


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
    
    def sync_network(self): 
        self.local_model.net.load_state_dict(self.master.shared_model.state_dict()) 
    
    def apply_gadients(self):
        for share_i,local_i in zip(
                self.master.shared_model.parameters(),
                self.local_model.net.parameters()):
            share_i._grad = local_i.grad
            #assert np.array_equal(share_i.grad.data.numpy(), local_i.grad.data.numpy())

    def weighted_choose_action(self, pi_probs):
        r = random.uniform(0, sum(pi_probs))
        upto = 0
        for idx, prob in enumerate(pi_probs):
            if upto + prob >= r:
                return idx
            upto += prob
        return len(pi_probs) - 1
    
    def forward_explore(self, hidden):
        terminal = False
        t_start = 0
        rollout_path = {"state": [], "action": [], "rewards": [], "done": []}
        pl_roll = []
        v_roll = []
        while not terminal and (t_start <= self.args.t_max):
            t_start += 1
            state_tensor = Variable(
                    torch.from_numpy(self.env.state).float())
            pl, v, hidden = self.local_model.net(state_tensor,hidden)
            pl_roll.append(pl)
            v_roll.append(v)
            
            action = prob.multinomial().data
            _, reward, terminal = self.env.forward_action(action)
            
            rollout_path["state"].append(self.env.state)
            one_hot_action = np.zeros(self.env.action_dim)
            one_hot_action[action] = 1
            rollout_path["action"].append(one_hot_action.reshape(self.env.action_dim,1))
            rollout_path["rewards"].append(reward)
            rollout_path["done"].append(terminal) 
        
        return rollout_path, hidden, p_roll, v_roll
        
    def discount(self, x):
        return scipy.signal.lfilter([1], [1, -self.args.gamma], x[::-1], axis=0)[::-1]

    def run(self):
        self.env.reset_env()
        loop = 0
        lstm_h = Variable(torch.zeros(1,1,256))
        lstm_c = Variable(torch.zeros(1,1,256))
        for _step in range(self.args.t_train):
            loop += 1
            rollout_path, (lstm_h,lstm_c), p_roll, v_roll= self.forward_explore((lstm_h,lstm_c))
            
            if rollout_path["done"][-1]:
                rollout_path["rewards"][-1] = 0
                self.env.reset_env()
                lstm_h = Variable(torch.zeros(1,1,256))
                lstm_c = Variable(torch.zeros(1,1,256))
            else:
                state_tensor = Variable(torch.from_numpy(
                    rollout_path["state"][-1]).float()) 
                _, v_t, _ = self.local_model.net(state_tensor,(lstm_h,lstm_c))
                lstm_h = Variable(lstm_h.data) 
                lstm_c = Variable(lstm_c.data) 
                rollout_path["rewards"][-1] = v_t.data.numpy()
            
            # calculate rewards 
            rollout_path["returns"] = self.discount(rollout_path["rewards"])
            
            self.PathBackProp(rollout_path, p_roll, v_roll)

            self.logger_.info("process %d, step %d, loss %f", self.process_id, loop, loss)
            self.loss_visual(loss, loop)

            self.optimizer.zero_grad()

            (policy_loss + 0.5 * value_loss).backward()
            torch.nn.utils.clip_grad_norm(self.local_model.parameters(), 40)
            self.ensure_shared_grads(self.local_model, self.shared_model)
            self.optimizer.step()
            
            self.master.main_update_step += 1
            self.sync_network()

    def loss_visual(self,loss_, loop_):
        self.loss_history.append(loss_) 
        if loop_>2:
            Y_ = np.array(self.loss_history).reshape(-1,1)
            self.win = self.master.vis.line(Y = Y_, X = np.arange(len(self.loss_history)), win=self.win)
    
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
        tensor_target_q = torch.from_numpy(target_q).float()
       

        for (i, item) in enumerate(p_roll):
            item



        batch_size = state.shape[0]
        
        batch_state =  Variable(torch.from_numpy(state).float())
        batch_action = Variable(torch.from_numpy(action).float().view(-1,self.action_dim,1))
        batch_target_q = Variable(tensor_target_q)
        
        pl = pl.view(-1,1,self.action_dim)
        pl_prob = torch.squeeze(torch.bmm(pl,batch_action))
        pl_log = torch.log(pl_prob) 
        diff = tensor_target_q-v.data
        entropy = -torch.dot(pl_prob, torch.log(pl_prob))
        pl_loss = -(torch.dot(pl_log, Variable(diff)) + entropy * self.args.entropy_beta )
        v_loss = self.v_criterion(v, batch_target_q) * batch_size 
        loss_all = 0.5* v_loss + pl_loss
        loss_all.backward()
        self.logger_.info("pl_loss %f, v_loss %f, entropy_loss %f", pl_loss.cpu().data.numpy()[0], v_loss.cpu().data.numpy()[0], entropy.cpu().data.numpy()[0])
        return  loss_all.cpu().data.numpy()
