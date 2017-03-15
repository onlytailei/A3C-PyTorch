#!/usr/bin/env python
# coding=utf-8
'''
Author:Tai Lei
Date:Tue 14 Mar 2017 09:26:14 AM WAT
Info: A3C continuous control in PyTorch
'''

import torch
import argparse
import gym
import scipy.signal
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.autograd as autograd

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
parser.add_argument("t_train", type = int,
        default = 1e9, 
        help = "train max time step")
parser.add_argument("t_test", type = int,
        default = 1e4, 
        help = "test max time step")
parser.add_argument("jobs", type = int, 
        default = 8, 
        help = "parallel running thread number")

parser.add_argument("frame_skip", type = int,
        default = 1, 
        help"number of frame skip")
parser.add_argument("frame_seq", type = int,
        default = 4, 
        help = "number of frame sequence")
parser.add_argument("opt", type = str,
        default = "rms", 
        help = "choice in [rms, adam, sgd]")
parser.add_argument("learn_rate", type = float,
        default = 7e-4, 
        help = "param of smooth")
parser.add_argument("grad_clip", type = float,
        default = 40.0, 
        help = "gradient clipping cut-off")
parser.add_argument("eps", type = float,
        default = 1e-8, 
        help = "param of smooth")
parser.add_argument("entropy_beta", type = float,
        default = 1e-4, 
        help = "param of policy entropy weight")
parser.add_argument("gamma", type = float, 
        default = 0.95, 
        help = "discounted ratio")
parser.add_argument("train_step", type = int, 
        default = 0, 
        help = "train step. unchanged")

args = parser.parse_args()


class AtariEnv(object):
    def __init__(self, env, screen_size=(84, 84)):
        self.env = env
        # constants
        self.screen_size = screen_size
        self.frame_skip = args.frame_skip
        self.frame_seq = args.frame_seq
        # local variables
        self.state = np.zeros(self.state_shape, dtype=np.float32)

    @property
    def state_shape(self):
        return [self.screen_size[0], self.screen_size[1], self.frame_seq]

    @property
    def action_dim(self):
        return self.env.action_space.n

    def precess_image(self, image):
        image = cv2.cvtColor(cv2.resize(image, self.screen_size), cv2.COLOR_BGR2GRAY)
        image = np.divide(image, 256.0)
        return image

    def reset_env(self):
        obs = self.env.reset()
        self.state[:, :, :-1] = 0
        self.state[:, :, -1] = self.precess_image(obs)
        return self.state

    def forward_action(self, action):
        obs, reward, done = None, None, None
        for _ in xrange(self.frame_skip):
            obs, reward, done, _ = self.env.step(action)
            if done:
                break
        obs = self.precess_image(obs)
        obs = np.reshape(obs, newshape=list(self.screen_size) + [1]) / 256.0
        self.state = np.append(self.state[:, :, 1:], obs, axis=2)
        # clip reward in range(-1, 1)
        reward = np.clip(reward, -1, 1)
        return self.state, reward, done

class A3CNet(nn.Module):
    def __init__(self, state_shape, action_dim):
        self.state_shape = state_shape 
        self.action_dim = action_dim
        self.conv1 = nn.Conv2d(self.state_shape[-1],16,8,stride=4)
        self.conv2 = nn.Conv2d(16,32,4,stride=2)
        self.conv2 = nn.Conv2d(16,32,4,stride=2)
        self.linear0 = nn.Linear(9*9*32, 256)
        
        # policy
        self.linear_policy_1 = nn.Linear(256,256)
        self.linear_policy_2 = nn.Linear(256,self.action_dim)
        self.softmax_policy = nn.Softmax()
        # value
        self.linear_value_1 = nn.linear(256,256)
        self.linear_value_2 = nn.linear(256,1)

    def forward(x):
        x = autograd.Variable(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 9*9*32) 
        x = F.relu(self.linear0(x)) 
        pl = F.relu(self.linear_policy_1(x))
        pl = F.relu(self.linear_policy_2(pl))
        pl = self.softmax_policy(pl)
        v = F.relu(self.linear_value_1(x))
        v = F.relu(self.linear_value_2(v))


class A3CSingleThread(object):
    def __init__(self):
        pass
   

class A3CAtari(object):
    def __init__(self):
        self.env = AtariEnv(gym.make(args.game))
        # TODO lstm layer
        #if args.use_lstm:
            #self.shared_net = A3CLSTMNet(self.env.state_shape, self.env.action_dim)
        #else:
        self.shared_net = A3CNet(self.env.state_shape, self.env.action_dim)
        
        # training threads
        self.jobs = []
        for thread_id in xrange(flags.jobs):
            job = A3CSingleThread(thread_id, self)
            self.jobs.append(job)
        # TODO
        # save model
        # restore model
    
    # TODO
    def shared_optimizer(self):
        pass

    def train(self):
        args.train_step = 0  
        signal.signal(signal.SIGINT, signal_handler)
        for job in self.jobs:
            job.start()
        for job in self.jobs:
            job.join()
    
    def test_sync(self):
        pass

def main():
    model = A3CAtari()
    model.train()

def signal_handler():
    sys.exit(0)

if __name__ == "__main__":
    main()
