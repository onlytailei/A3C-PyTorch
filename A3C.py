#!/usr/bin/env python
# coding=utf-8
'''
Author:Tai Lei
Date:Thu 16 Mar 2017 09:55:00 PM WAT
Info:
'''

import gym
import scipy.signal
import threading
import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.autograd as autograd
import torch.optim as optim
from environment import AtariEnv
import torch.multiprocessing as mp
import multiprocessing


class A3CLSTMNet(nn.Module):

    def __init__(self, state_shape, action_dim):
        super(A3CLSTMNet, self).__init__()
        self.state_shape = state_shape 
        self.action_dim = action_dim
        self.conv1 = nn.Conv2d(self.state_shape[0],32,3,stride=2)
        self.conv2 = nn.Conv2d(32,32,3,stride=2, padding = 1)
        self.conv3 = nn.Conv2d(32,32,3,stride=2, padding = 1)
        self.conv4 = nn.Conv2d(32,32,3,stride=2, padding = 1)
        self.lstm = nn.LSTM(3*3*32,256,1)
        # hang policy output
        self.linear_policy_1 = nn.Linear(256,self.action_dim)
        self.softmax_policy = nn.Softmax()
        # hang value output
        self.linear_value_1 = nn.Linear(256,1)

    def forward(self, x, hidden):
        x = x.view(-1, self.state_shape[0], 
                self.state_shape[1],self.state_shape[2]) 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 1,3*3*32) 
        x,hidden = self.lstm(x, hidden)
        x = x.view(-1,256)
        pl = self.linear_policy_1(x)
        pl = self.softmax_policy(pl)
        v = self.linear_value_1(x)
        return pl,v,hidden
    
     #def init_hidden(self):
        #return (Variable(torch.randn(1, 1, 256)),
                #Variable(torch.randn(1, 1, 256)))
    
    #def weights_init(m):
        #classname = m.__class__.__name__
        #if classname.find('LSTMCell') != -1:
            #weight_shape = list(m.weight.data.size())
            #fan_in = np.prod(weight_shape[1:4])
            #fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
            #w_bound = np.sqrt(6. / (fan_in + fan_out))
            #m.weight.data.uniform_(-w_bound, w_bound)
            #m.bias.data.fill_(0)

class A3CNet(nn.Module):

    def __init__(self, state_shape, action_dim):
        super(A3CNet, self).__init__()
        self.state_shape = state_shape 
        self.action_dim = action_dim
        self.conv1 = nn.Conv2d(self.state_shape[0],32,3,stride=2)
        self.conv2 = nn.Conv2d(32,32,3,stride=2,padding = 1)
        self.conv3 = nn.Conv2d(32,32,3,stride=2,padding = 1)
        self.conv4 = nn.Conv2d(32,32,3,stride=2,padding = 1)
        # hang policy output
        self.linear_policy_1 = nn.Linear(3*3*32,256)
        self.linear_policy_2 = nn.Linear(256,self.action_dim)
        self.softmax_policy = nn.Softmax()
        # hang value output
        self.linear_value_1 = nn.Linear(3*3*32,256)
        self.linear_value_2 = nn.Linear(256,1)

    def forward(self,x):
        x = x.view(-1, self.state_shape[0], 
                self.state_shape[1],self.state_shape[2]) 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 3*3*32) 
        pl = F.relu(self.linear_policy_1(x))
        pl = self.linear_policy_2(pl)
        pl = self.softmax_policy(pl)
        v = F.relu(self.linear_value_1(x))
        v = self.linear_value_2(v)
        return pl,v


class A3CModel(object):
    """
    add PathBackProp compared with A3CNet
    """
    def __init__(self, state_shape,action_dim, args_, logger_):
        
        if args_.use_lstm:
            self.net = A3CLSTMNet(state_shape,action_dim)
        else:
            self.net = A3CNet(state_shape,action_dim)
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
        
        batch_state =  autograd.Variable(torch.from_numpy(state).float().cuda())
        batch_action = autograd.Variable(torch.from_numpy(action).float().view(-1,self.action_dim,1).cuda())
        batch_target_q = autograd.Variable(tensor_target_q)
        if self.args.use_lstm:
            hidden = (autograd.Variable(lstm_hidden[0]),
                        autograd.Variable(lstm_hidden[1]))
            pl, v, hidden = self.net(batch_state,hidden)
        else:
            pl,v = self.net(batch_state)
        pl = pl.view(-1,1,self.action_dim)
        pl_prob = torch.squeeze(torch.bmm(pl,batch_action))
        pl_log = torch.log(pl_prob) 
        diff = tensor_target_q-v.data
        entropy = -torch.dot(pl_prob, torch.log(pl_prob))
        pl_loss = -(torch.dot(pl_log, autograd.Variable(diff)) + entropy * self.args.entropy_beta )
        v_loss = self.v_criterion(v, batch_target_q) * batch_size 
        loss_all = 0.5* v_loss + pl_loss
        loss_all.backward()
        self.logger_.info("pl_loss %f, v_loss %f, entropy_loss %f", pl_loss.cpu().data.numpy()[0], v_loss.cpu().data.numpy()[0], entropy.cpu().data.numpy()[0])
        return  loss_all.cpu().data.numpy()
        
        # another way for val loss
        #v_prime = torch.sum((target_q_torch-v)*(target_q_torch-v),0)
        #assert v_loss.data.numpy() == v_prime.data.numpy()


class A3CSingleThread(multiprocessing.Process):
    
    def __init__(self, thread_id, master, logger_):
        super(A3CSingleThread, self).__init__(self, name="process_%d" % thread_id)
        self.thread_id = thread_id
        self.logger_ = logger_
        
        #multiprocessing.Process.__init__(self, name = "thread_%d" % thread_id) 
        self.master = master
        self.args = master.args
        self.env = AtariEnv(gym.make(self.args.game), self.args.frame_seq,self.args.frame_skip,self.master.lock)
        self.local_model = A3CModel(self.env.state_shape, self.env.action_dim, master.args, logger_)
        # sync the weights at the beginning
        self.sync_network() 
        # optimizer is used to zero the old grad
        self.optim = optim.RMSprop(self.local_model.net.parameters())
        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.local_model.net.type(dtype)
        self.loss_history = []
        self.win = None
    
    def sync_network(self): 
        self.local_model.net.load_state_dict(self.master.shared_net.state_dict()) 
    
    def apply_gadients(self):
        for share_i,local_i in zip(self.master.shared_net.parameters(),
                self.local_model.net.parameters()):
            #share_i.grad.data = local_i.grad.data.clone()
            
            # another way to update
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
    
    def forward_explore(self, train_step):
        terminal = False
        t_start = train_step
        rollout_path = {"state": [], "action": [], "rewards": [], "done": []}
        if self.args.use_lstm:
            hidden = (autograd.Variable(self.lstm_h_init), 
                    autograd.Variable(self.lstm_c_init))
        while not terminal and (train_step - t_start <= self.args.t_max):
            state_tensor = autograd.Variable(
                    torch.from_numpy(self.env.state).float().cuda())
            if self.args.use_lstm:
                pl, v, hidden = self.local_model.net(state_tensor,hidden)
            else:
                pl, v = self.local_model.net(state_tensor)
            
            if random.random() < 0.8:
                action = self.weighted_choose_action(pl.cpu().data.numpy()[0])
            else:
                action = random.randint(0, self.env.action_dim - 1)
            
            _, reward, terminal = self.env.forward_action(action)
            train_step += 1
            rollout_path["state"].append(self.env.state)
            one_hot_action = np.zeros(self.env.action_dim)
            one_hot_action[action] = 1
            rollout_path["action"].append(one_hot_action.reshape(self.env.action_dim,1))
            rollout_path["rewards"].append(reward)
            rollout_path["done"].append(terminal) 
        if self.args.use_lstm:
            return train_step, rollout_path, hidden
        else:
            return train_step, rollout_path
        
    def discount(self, x):
        return scipy.signal.lfilter([1], [1, -self.args.gamma], x[::-1], axis=0)[::-1]

    def run(self):
        self.env.reset_env()
        loop = 0
        self.lstm_h_init = torch.randn(1,1,256).cuda()
        self.lstm_c_init = torch.randn(1,1,256).cuda()
        
        while self.args.train_step <= self.args.t_train:
            train_step = 0 
            loop += 1
            with self.master.lock:
                self.sync_network()
            self.optim.zero_grad()    
            
            if self.args.use_lstm:
                train_step, rollout_path, hidden= self.forward_explore(train_step)
                
                if rollout_path["done"][-1]:
                    rollout_path["rewards"][-1] = 0
                    self.env.reset_env()
                    self.lstm_h_init = torch.randn(1,1,256).cuda()
                    self.lstm_c_init = torch.randn(1,1,256).cuda()
                else:
                    state_tensor = autograd.Variable(torch.from_numpy(
                        rollout_path["state"][-1]).float().cuda()) 
                    _, v_t, hidden = self.local_model.net(state_tensor,hidden)
                    self.lstm_h_init
                    self.lstm_c_init
                    rollout_path["rewards"][-1] = v_t.cpu().data.numpy()
            
            else: 
                train_step, rollout_path = self.forward_explore(train_step)
            
            if rollout_path["done"][-1]:
                rollout_path["rewards"][-1] = 0
                self.env.reset_env()
            
            elif self.args.use_lstm:
                state_tensor = autograd.Variable(torch.from_numpy(
                        rollout_path["state"][-1]).float().cuda()) 
                _, v_t, _ = self.local_model.net(state_tensor,hidden)
                rollout_path["rewards"][-1] = v_t.cpu().data.numpy()
            else:
                state_tensor = autograd.Variable(torch.from_numpy(
                        rollout_path["state"][-1]).float().cuda()) 
                _, v_t = self.local_model.net(state_tensor)
                rollout_path["rewards"][-1] = v_t.cpu().data.numpy()
            # calculate rewards 
            rollout_path["returns"] = self.discount(rollout_path["rewards"])
            
            if self.args.use_lstm: 
                loss = self.local_model.PathBackProp(rollout_path, lstm_hidden=(self.lstm_h_init,self.lstm_c_init))
            else:
                loss = self.local_model.PathBackProp(rollout_path)

            self.logger_.info("thread %d, step %d, loss %f", self.thread_id, loop, loss)
            self.loss_visual(loss, loop)
            with self.master.lock:
                self.apply_gadients()
                self.master.optim_shared_net()
                self.master.main_update_step += 1
    def loss_visual(self,loss_, loop_):
        self.loss_history.append(loss_) 
        if loop_>2:
            Y_ = np.array(self.loss_history).reshape(-1,1)
            self.win = self.master.vis.line(Y = Y_, X = np.arange(len(self.loss_history)), win=self.win)
