#!/usr/bin/env python
# coding=utf-8
'''
Author:Tai Lei
Date:Thu 16 Mar 2017 09:52:48 PM WAT
Info:
'''
import numpy as np
import cv2

class AtariEnv(object):
    """
    a wrapper of the origin gym env class
    """
    def __init__(self, env, frame_seq,frame_skip, screen_size=(42, 42),render=False):
        self.env = env
        self.screen_size = screen_size
        self.frame_skip = frame_skip
        self.frame_seq = frame_seq
        self.state = np.zeros(self.state_shape, dtype=np.float)
        self.count_ = 0
        self.render=render
    
    @property
    def state_shape(self):
        return [self.frame_seq, self.screen_size[0], self.screen_size[1]]

    @property
    def action_dim(self):
        return self.env.action_space.n

    def precess_image(self, frame):
        frame = frame[34:34 + 160, :160]
        frame = cv2.resize(frame, (80, 80))
        frame = cv2.resize(frame, (42, 42))
        frame = frame.mean(2)
        frame = frame.astype(np.float32)
        frame *= (1.0 / 255.0)
        frame = np.reshape(frame, [1, 42, 42])
        return frame

    def reset_env(self):
        obs = self.env.reset()
        self.state[:-1, :, :] = 0
        self.state[-1, :, :] = self.precess_image(obs)
        return self.state

    def forward_action(self, action):
        obs, reward, done = None, None, None
        for _ in xrange(self.frame_skip):
            if self.render: 
                self.env.render()
            obs, reward, done, _ = self.env.step(action)
            self.count_+=1
            if done:
                break
        obs = self.precess_image(obs)
        self.state = np.append(self.state[1:, :, :], obs, axis=0)
        # clip reward in range(-1, 1)
        reward = np.clip(reward, -1, 1)
        return self.state, reward, done
