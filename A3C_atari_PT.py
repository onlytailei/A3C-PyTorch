#!/usr/bin/env python
# coding=utf-8
'''
Author:Tai Lei
Date:Tue 14 Mar 2017 09:26:14 AM WAT
Info: A3C continuous control in PyTorch
'''

import numpy as np
import torch
import argparse
import cv2
import gym
import signal
import scipy.signal
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.autograd as autograd
import threading
import torch.optim as optim
import random


class AtariEnv(object):
    def __init__(self, env, frame_seq,frame_skip,screen_size=(84, 84)):
        self.env = env
        self.screen_size = screen_size
        self.frame_skip = frame_skip
        self.frame_seq = frame_seq
        self.state = np.zeros(self.state_shape, dtype=np.float)

    @property
    def state_shape(self):
        return [self.frame_seq, self.screen_size[0], self.screen_size[1]]

    @property
    def action_dim(self):
        return self.env.action_space.n

    def precess_image(self, image):
        image = cv2.cvtColor(cv2.resize(image, self.screen_size), cv2.COLOR_BGR2GRAY)
        image = np.divide(image, 256.0)
        return image

    def reset_env(self):
        obs = self.env.reset()
        self.state[:-1, :, :] = 0
        self.state[-1, :, :] = self.precess_image(obs)
        return self.state

    def forward_action(self, action):
        obs, reward, done = None, None, None
        for _ in xrange(self.frame_skip):
            obs, reward, done, _ = self.env.step(action)
            if done:
                break
        obs = self.precess_image(obs)
        obs = np.reshape(obs, newshape=[1] + list(self.screen_size))
        self.state = np.append(self.state[1:, :, :], obs, axis=0)
        # clip reward in range(-1, 1)
        reward = np.clip(reward, -1, 1)
        return self.state, reward, done

class A3CNet(nn.Module):
    def __init__(self, state_shape, action_dim):
        super(A3CNet, self).__init__()
        self.state_shape = state_shape 
        self.action_dim = action_dim
        self.conv1 = nn.Conv2d(self.state_shape[0],16,8,stride=4)
        self.conv2 = nn.Conv2d(16,32,4,stride=2)
        self.conv2 = nn.Conv2d(16,32,4,stride=2)
        self.linear0 = nn.Linear(9*9*32, 256)
        
        # policy
        self.linear_policy_1 = nn.Linear(256,256)
        self.linear_policy_2 = nn.Linear(256,self.action_dim)
        self.softmax_policy = nn.Softmax()
        # value
        self.linear_value_1 = nn.Linear(256,256)
        self.linear_value_2 = nn.Linear(256,1)

    def forward(self,x):
        x = autograd.Variable(x)
        x = x.view(-1, self.state_shape[0], 
                self.state_shape[1],self.state_shape[2]) 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 9*9*32) 
        x = F.relu(self.linear0(x)) 
        pl = F.relu(self.linear_policy_1(x))
        pl = F.relu(self.linear_policy_2(pl))
        pl = self.softmax_policy(pl)
        v = F.relu(self.linear_value_1(x))
        v = F.relu(self.linear_value_2(v))
        return pl,v

class A3CModel(object):
    def __init__(self, state_shape,action_dim):
        
        self.net = A3CNet(state_shape,action_dim)
        self.optim = optim.RMSprop(self.net.parameters())
        self.action_dim = action_dim 
    
    def PathBackProp(self,rollout_path_):
        self.optim.zero_grad()
        state = rollout_path_['state']
        target_q = np.array(rollout_path_['returns'])
        action = rollout_path_['action']
        
        batch_state =  torch.from_numpy(np.array(state)).float()
        batch_action = torch.from_numpy(np.array(action)).float() 
        pl,v = self.net(batch_state)
        pl = pl.view(-1,1,self.action_dim)
        
        batch_action = autograd.Variable(batch_action.view(-1,self.action_dim,1))
        pl_prob = torch.log(torch.bmm(pl,batch_action))
        pl_prob = torch.squeeze(pl_prob)
        diff = (target_q-v.data.numpy()[0])
        pl_loss = -torch.dot(pl_prob,autograd.Variable(torch.from_numpy(diff).float()))
        pl_loss.backward(retain_variables=True)
        target_q_torch = autograd.Variable(torch.from_numpy(target_q).float())
        v_loss = torch.sum((target_q_torch-v)*(target_q_torch-v),0)
        v_loss.backward()
        return self.net.parameters()

class A3CSingleThread(threading.Thread):
    def __init__(self, thread_id, master):
        
        self.thread_id = thread_id
        threading.Thread.__init__(self, name = "thread_%d" % thread_id) 
        self.master = master
        self.args = master.args
        self.env = AtariEnv(gym.make(self.args.game), self.args.frame_seq,self.args.frame_skip)
        # TODO lstm layer 
        #if flags.use_lstm:
            #self.local_net = A3CLSTMNet(self.env.state_shape, self.env.action_dim, scope="local_net_%d" % thread_id)
        #else:
        self.local_net = A3CModel(self.env.state_shape, self.env.action_dim)
        # sync the weights of all the network
        self.sync = self.sync_network(master.shared_net) 
        self.optim = optim.RMSprop(self.local_net.net.parameters())
        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.local_net.net.type(dtype)

    def sync_network(self, net): 
        self.local_net.net.load_state_dict(net.state_dict()) 
    
    def apply_gadients(self):
        local_param = [_ for _ in self.local_net.net.parameters()]
        master_param = self.master.shared_net.parameters()
        for i,item in enumerate(master_param):
            item += local_param[i].grad
            assert (item.data.numpy().shape == local_param[i].data.numpy().shape)

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
        while not terminal and (train_step - t_start <= self.args.t_max):
            state_tensor = torch.from_numpy(self.env.state).float() 
            pl, v = self.local_net.net(state_tensor)
            if random.random() < 0.8:
                action = self.weighted_choose_action(pl.data.numpy()[0])
            else:
                action = random.randint(0, self.env.action_dim - 1)
            
            _, reward, terminal = self.env.forward_action(action)
            train_step += 1
            rollout_path["state"].append(self.env.state)
            one_hot_action = np.zeros(self.env.action_dim)
            one_hot_action[action] = 1
            rollout_path["action"].append(one_hot_action)
            rollout_path["rewards"].append(reward)
            rollout_path["done"].append(terminal) 
        return train_step, rollout_path
        
    def discount(self, x):
        return scipy.signal.lfilter([1], [1, self.args.gamma], x[::-1], axis=0)[::-1]

    def run(self):
        self.env.reset_env()
        loop = 0
        while self.args.train_step <= self.args.t_train:
            train_step = 0 
            loop += 1
            self.sync_network(self.master.shared_net)
            self.optim.zero_grad()    
            train_step, rollout_path = self.forward_explore(train_step)
            if rollout_path["done"][-1]:
                rollout_path["rewards"][-1] = 0
                self.env.reset_env()
                #if args.use_lstm:
                    #self.local_net.reset_lstm_state()
            else:
                state_tensor = torch.from_numpy(
                        rollout_path["state"][-1]).float() 
                _, v_t = self.local_net.net(state_tensor)
                rollout_path["rewards"][-1] = v_t.data.numpy()
            # calculate rewards 
            rollout_path["returns"] = self.discount(rollout_path["rewards"])
            
            self.local_net.PathBackProp(rollout_path)
            self.apply_gadients()             


class A3CAtari(object):
    def __init__(self, args_):
        self.args = args_
        self.env = AtariEnv(gym.make(self.args.game),args_.frame_seq,args_.frame_skip)
        # TODO lstm layer
        #if args.use_lstm:
            #self.shared_net = A3CLSTMNet(self.env.state_shape, self.env.action_dim)
        #else:
        self.shared_net = A3CNet(self.env.state_shape, self.env.action_dim)
        self.optim = optim.RMSprop(self.shared_net.parameters(),self.args.lr) 
        # training threads
        self.jobs = []
        for thread_id in xrange(self.args.jobs):
            job = A3CSingleThread(thread_id, self)
            self.jobs.append(job)
    
    def train(self):
        self.args.train_step = 0  
        signal.signal(signal.SIGINT, signal_handler)
        for job in self.jobs:
            job.start()
        for job in self.jobs:
            job.join()
    
    def test_sync(self):
        # TODO
        pass


def signal_handler():
    sys.exit(0)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type = str, 
            default = 'Breakout-v0',
            help = "gym environment name")
    parser.add_argument("--train_dir", type = str, 
            default = './models/experiment0',
            help = "save environment")
    parser.add_argument("--gpu", type = int, 
            default = 0,
            help = "gpu id")
    parser.add_argument("--use_lstm", type = bool,
            default = False, 
            help = "use LSTM layer")
    parser.add_argument("--t_max", type = int, 
            default = 6,
            help = "episode max time step")
    parser.add_argument("--t_train", type = int,
            default = 1e9, 
            help = "train max time step")
    parser.add_argument("--t_test", type = int,
            default = 1e4, 
            help = "test max time step")
    parser.add_argument("--jobs", type = int, 
            default = 2, 
            help = "parallel running thread number")
    parser.add_argument("--frame_skip", type = int,
            default = 1, 
            help = "number of frame skip")
    parser.add_argument("--frame_seq", type = int,
            default = 4, 
            help = "number of frame sequence")
    parser.add_argument("--opt", type = str,
            default = "rms", 
            help = "choice in [rms, adam, sgd]")
    parser.add_argument("--lr", type = float,
            default = 7e-4, 
            help = "param of smooth")
    parser.add_argument("--grad_clip", type = float,
            default = 40.0, 
            help = "gradient clipping cut-off")
    parser.add_argument("--eps", type = float,
            default = 1e-8, 
            help = "param of smooth")
    parser.add_argument("--entropy_beta", type = float,
            default = 1e-4, 
            help = "param of policy entropy weight")
    parser.add_argument("--gamma", type = float, 
            default = 0.95, 
            help = "discounted ratio")
    parser.add_argument("--train_step", type = int, 
            default = 0, 
            help = "train step. unchanged")
    args_ = parser.parse_args()

    model = A3CAtari(args_)
    model.train()

