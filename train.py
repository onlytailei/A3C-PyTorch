#!/usr/bin/env python
# coding=utf-8
'''
Author:Tai Lei
Date:Tue 14 Mar 2017 09:26:14 AM WAT
Info: A3C continuous control in PyTorch
'''

import numpy as np
import argparse
import signal
import threading
import torch.optim as optim
import sys
import os
from environment import AtariEnv
from A3C import *


class A3CAtari(object):
    
    def __init__(self, args_):
        self.args = args_
        self.env = AtariEnv(gym.make(self.args.game),args_.frame_seq,args_.frame_skip)
        # TODO lstm layer
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

    def save_model(self):
        torch.save(self.shared_net.state_dict(), './net.pth')
    
    def load_model(self):
        self.shared_net.load_state_dict(torch.load('./net.pth'))

def signal_handler():
    sys.exit(0)

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
        default = 8, 
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

if __name__=="__main__":
    args_ = parser.parse_args()
    model = A3CAtari(args_)
    model.train()

