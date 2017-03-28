#!/usr/bin/env python
# coding=utf-8
'''
Author:Tai Lei
Date:Thu 16 Mar 2017 09:52:48 PM WAT
Info:
'''
import numpy as np
import cv2
from threading import Lock

class AtariEnv(object):
    """
    a wrapper of the origin gym env class
    """
    def __init__(self, env, frame_seq,frame_skip,lock_,screen_size=(84, 84),render=False):
        self.env = env
        self.screen_size = screen_size
        self.frame_skip = frame_skip
        self.frame_seq = frame_seq
        self.state = np.zeros(self.state_shape, dtype=np.float)
        self.lock = lock_
        self.count_ = 0
        self.render=render
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
            if self.render: 
                #with self.lock:
                self.env.render()
            obs, reward, done, _ = self.env.step(action)
            self.count_+=1
            if done:
                break
        obs = self.precess_image(obs)
        obs = np.reshape(obs, newshape=[1] + list(self.screen_size))
        self.state = np.append(self.state[1:, :, :], obs, axis=0)
        # clip reward in range(-1, 1)
        reward = np.clip(reward, -1, 1)
        return self.state, reward, done
