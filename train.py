#!/usr/bin/env python
# coding=utf-8
'''
Author:Tai Lei
Date:Tue 14 Mar 2017 09:26:14 AM WAT
Info: A3C continuous control in PyTorch
'''

import numpy as np
import argparse
import torch.optim as optim
import sys
import os
import logging
import time
from environment import AtariEnv
from A3C import *
import visdom
import torch.multiprocessing as mp
import my_optim
from multiprocessing import Value
import cv2
class A3CAtari(object):
    
    def __init__(self, args_,logger_):
        self.args = args_
        self.logger = logger_
        self.env = AtariEnv(gym.make(self.args.game),args_.frame_seq,args_.frame_skip,render = True)
        self.shared_model = A3CLSTMNet(self.env.state_shape, self.env.action_dim)
        self.shared_model.share_memory()
        self.optim = my_optim.SharedAdam(self.shared_model.parameters(),lr=self.args.lr)  
        self.optim.share_memory() 
        # visdom
        self.vis = visdom.Visdom()
        self.main_update_step = Value('d', 0)
        # load model
        if self.args.load_weight !=0 :
            self.load_model(self.args.load_weight)
        
        self.jobs = []
        if self.args.t_flag:
            for process_id in xrange(self.args.jobs):
                job = A3CSingleProcess(process_id, self, logger_)
                self.jobs.append(job)
        self.test_win = None
    def train(self):
        test_p = mp.Process(target=self.test)
        self.jobs.append(test_p)
        self.args.train_step = 0  
        for job in self.jobs:
            job.start()
        for job in self.jobs:
            job.join()
    
    def test_sync(self):
        pass
    
    def test(self, render_=False):
        test_env = AtariEnv(gym.make(self.args.game),self.args.frame_seq,self.args.frame_skip,render = render_)
        test_model = A3CLSTMNet(self.env.state_shape, self.env.action_dim)
        while True:
            terminal = False
            reward_ = 0
            lstm_h = Variable(torch.zeros(1,256), volatile=True)
            lstm_c = Variable(torch.zeros(1,256), volatile=True)
            test_env.reset_env()
            if (int(self.main_update_step.value)) % 500 == 0:
                print "step: ", int(self.main_update_step.value)
                episode_length = 0
                self.save_model(int(self.main_update_step.value))
                test_model.load_state_dict(self.shared_model.state_dict())
                while not terminal:
                    state_tensor = Variable(
                            torch.from_numpy(test_env.state).float())
                    pl, v, (lstm_h,lstm_c) = test_model(state_tensor,(lstm_h,lstm_c))
                    #print pl.data.numpy()[0]
                    action = pl.max(1)[1].data.numpy()[0]
                    _, reward, terminal = test_env.forward_action(action)
                    reward_ += reward
                    episode_length += 1
                    #img_ = (test_env.state.copy().reshape(42,42)*256)
                    #img_ = cv2.resize(img_, (160,160))
                    #img_ = np.stack((img_,)*3)
                    #self.test_win = self.vis.image(img_, 
                            #win = self.test_win)
                print "Reward: ", reward_
                print "episode_length", episode_length
    
    def save_model(self,name):
        torch.save(self.shared_model.state_dict(), self.args.train_dir + str(name) + '_weight')
    
    def load_model(self, name):
        self.shared_model.load_state_dict(torch.load(self.args.train_dir + str(name) + '_weight'))


def loggerConfig():
    ts = str(time.strftime('%Y-%m-%d-%H-%M-%S'))
    logger = logging.getLogger()
    formatter = logging.Formatter(
            '%(asctime)s %(levelname)-2s %(message)s')
    #streamhandler_ = logging.StreamHandler()
    #streamhandler_.setFormatter(formatter)
    #logger.addHandler(streamhandler_)
    fileHandler_ = logging.FileHandler("log/a3c_training_log_"+ts)
    fileHandler_.setFormatter(formatter)
    logger.addHandler(fileHandler_)
    logger.setLevel(logging.WARNING)
    return logger

parser = argparse.ArgumentParser()
parser.add_argument("--game", type = str, 
        default = 'PongDeterministic-v3',
        help = "gym environment name")
parser.add_argument("--train_dir", type = str, 
        default = './models/',
        help = "save environment")
parser.add_argument("--gpu", type = int, 
        default = 0,
        help = "gpu id")
parser.add_argument("--use_lstm", type = int,
        default = 0, 
        help = "use LSTM layer")
parser.add_argument("--t_max", type = int, 
        default = 20,
        help = "episode max time step")
parser.add_argument("--t_train", type = int,
        default = 1e9, 
        help = "train max time step")
parser.add_argument("--t_test", type = int,
        default = 1e4, 
        help = "test max time step")
parser.add_argument("--t_flag", type = int,
        default = 1, 
        help = "training flag")
parser.add_argument("--jobs", type = int, 
        default = 16, 
        help = "parallel running thread number")
parser.add_argument("--frame_skip", type = int,
        default = 1, 
        help = "number of frame skip")
parser.add_argument("--frame_seq", type = int,
        default = 1, 
        help = "number of frame sequence")
parser.add_argument("--opt", type = str,
        default = "rms", 
        help = "choice in [rms, adam, sgd]")
parser.add_argument("--lr", type = float,
        default = 1e-4, 
        help = "learning rate")
parser.add_argument("--grad_clip", type = float,
        default = 40.0, 
        help = "gradient clipping cut-off")
parser.add_argument("--eps", type = float,
        default = 1e-8, 
        help = "param of smooth")
parser.add_argument("--entropy_beta", type = float,
        default = 1e-5, 
        help = "param of policy entropy weight")
parser.add_argument("--gamma", type = float, 
        default = 0.99, 
        help = "discounted ratio")
parser.add_argument("--load_weight", type = int, 
        default = 0, 
        help = "train step. unchanged")


if __name__=="__main__":
    args_ = parser.parse_args()
    logger = loggerConfig() 
    model = A3CAtari(args_, logger)
    if args_.t_flag:
        print "======training=====" 
        model.train()
    else:
        print "=====testing====="
        model.test(True)

