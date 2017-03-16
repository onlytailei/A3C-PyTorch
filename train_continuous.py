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
parser.add_argument("--t_max", type = int, 
        default = 32,
        help = "episode max time step")
parser.add_argument("t_train", type = int,
        default = 1e4, 
        help = "train max time step")
parser.add_argument("t_test", type = int,
        default = 1e4, 
        help = "test max time step")
parser.add_argument("jobs", type = int, 
        default = 8, 
        help = "parallel running thread number")
parser.add_argument("learn_rate", type = float,
        default = 5e-4, 
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
