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
import threading

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
        default = 2, 
        help = "parallel running thread number")
parser.add_argument("frame_skip", type = int,
        default = 1, 
        help = "number of frame skip")
parser.add_argument("frame_seq", type = int,
        default = 4, 
        help = "number of frame sequence")
parser.add_argument("opt", type = str,
        default = "rms", 
        help = "choice in [rms, adam, sgd]")
parser.add_argument("lr", type = float,
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

class A3CModel(object):
    def __init__(self, state_shape,action_dim):
        class A3CNet(nn.Module):
            def __init__(self, state_shape, action_dim):
                self.state_shape = state_shape 
                self.action_dim = action_dim
                self.conv1 = nn.Conv2d(self.state_shape[-1],
                        16,8,stride=4)
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
                x = x.view(-1, state_shape[0], state_shape[1],state_shape[2]) 
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
        
        self.net = A3CNet(state_shape,action_dim)
        self.v_criterion = nn.CrossEntropyLoss()
        self.optim = optim.RMSprop(self.net.parameters(), args.lr)
    
    def get_policy(self,x):
        return self.net(x)[0]
    
    def get_value(self,x):
        return self.net(x)[1] 
    
    def PathBackProp(self,rollout_path_):
        optim.zero_grad()
        state = rollout_path_['state']
        target_q = rollout_path_['returns']
        action = rollout_path_['action']
         
        pl,v = self.net(state)
        pl_prob = torch.log(torch.bmm(pl,action))
        # change to value
        value = (target_q-v.data.numpy())
        pl_loss = - torch.bmm(pl_prob,value)
        pl_loss.backward()
        v_loss = (target_q-v)*(target_q-v)
        v_loss.backward()
        return self.net.parameters()

class A3CSingleThread(threading.Thread):
    def __init__(self, thread_id, master):
        self.thread_id = thread_id
        threading.Thread.__init__(self, name = "thread_%d" % thread_id) 
        self.env = AtariEnv(gym.make(args.game))
        self.master = master
        # TODO lstm layer 
        #if flags.use_lstm:
            #self.local_net = A3CLSTMNet(self.env.state_shape, self.env.action_dim, scope="local_net_%d" % thread_id)
        #else:
        self.local_net = A3CModel(self.env.state_shape, self.env.action_dim)
        # sync the weights of all the network
        self.sync = self.sync_network(master.shared_net) 
        self.optim = optim.RMSprop(model.parameters())
        # self.criterion = nn.CrossEntropyLoss()
        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.local_net.type(dtype)

    def sync_network(self, net): 
        self.local_net.load_state_dict(net.state_dict()) 
    
    def apply_gadients(self):
        #for param in model.parameters():
            #print(type(param.data), param.size())
        self.master.shared_net.optim.zero_grad()
        local_param = [_ for _ in self.local_net.parameters()]
        for i in xrange(len(local_net)):
            self.master.shared_net.parameters().data.grad = local_param[i].data.grad
        self.master.shared_net.optim.step()

    def weighted_choose_action(self, pi_probs):
        r = random.uniform(0, sum(pi_probs))
        upto = 0
        for idx, prob in enumerate(pi_probs):
            if upto + prob >= r:
                return idx
            upto += prob
        return len(pi_probs) - 1
    
    def forward_explore(self, train_step):
        termial = False
        t_start = train_step
        rollout_path = {"state": [], "action": [], "rewards": [], "done": []}
        while not terminal and (train_step - t_start <= args.t_max):
            
            pl, v = self.local_net.net(self.env.state)
            
            if random.random() < 0.8:
                action = self.weighted_choose_action(pl.numpy())
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
        return scipy.signal.lfilter([1], [1, args.gamma], x[::-1], axis=0)[::-1]

    def run(self):
        self.env.reset_env()
        loop = 0
        while args.train_step <= args.t_train:
            train_step = 0 
            loo += 1
            self.sync_network(self.master.shared_net)
            self.optim.zero_grad()    
            train_step, rollout_path = self.forward_explore(train_step)
            if rollout_path["done"][-1]:
                rollout_path["rewards"][-1] = 0
                self.env.reset_env()
            #if args.use_lstm:
                #self.local_net.reset_lstm_state()
            else:
                _, rollout_path["rewards"][-1] = self.local_net.net(rollout_path["state"][-1])
            # calculate rewards 
            rollout_path["returns"] = self.discount(rollout_path["rewards"])
            
            self.local_net.PathBackProp(rollout_path)
             
            # apply gradients to main net
            # main optimizer 0 gradients
            # copy gradients from subnet to mainnet


class A3CAtari(object):
    def __init__(self):
        self.env = AtariEnv(gym.make(args.game))
        # TODO lstm layer
        #if args.use_lstm:
            #self.shared_net = A3CLSTMNet(self.env.state_shape, self.env.action_dim)
        #else:
        self.shared_net = A3CNet(self.env.state_shape, self.env.action_dim)
        self.optim = RMSprop(self.shared_net.parameters(),args.rl) 
        # training threads
        self.jobs = []
        for thread_id in xrange(args.jobs):
            job = A3CSingleThread(thread_id, self)
            self.jobs.append(job)
    

    def train(self):
        args.train_step = 0  
        signal.signal(signal.SIGINT, signal_handler)
        for job in self.jobs:
            job.start()
        for job in self.jobs:
            job.join()
    
    def test_sync(self):
        # TODO
        pass

def main():
    model = A3CAtari()
    model.train()

def signal_handler():
    sys.exit(0)

if __name__ == "__main__":
    main()
