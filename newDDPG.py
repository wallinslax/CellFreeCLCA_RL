# proprietary design 
from toolUtil import *
from newENV import BS
# Public Lib
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn
import random
#environment
#from newENV import BS,plot_UE_SBS_association,UE_SBS_location_distribution,plot_UE_SBS_distribution
import os
import time
from math import sqrt
import copy
#pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#####################  hyper parameters  ####################
# DDPG Parameter
RESET_CHANNEL = False
LR_DECAY = False
MAX_EPISODES = 20
#MAX_EPISODES = 1
MAX_EP_STEPS = 100 

LR_A = 0.0001    # learning rate for actor
LR_C = 0.0002    # learning rate for critic
GAMMA = 0.99     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 128
n_realizations = 100
n_LR_update = 1000
MEM_SIZE = 60000
SEED = 0 # random seed
Var = 1 # control exploration
#np.random.seed(SEED)
#####################################


logging_interval = 40
animate_interval = logging_interval * 5
logdir='./DDPG/'
VISUALIZE = False
SEED = 0
MAX_PATH_LENGTH = 500
NUM_EPISODES = 12000
GAMMA=0.9
BATCH_SIZE = 128
LR_A = 1e-4
LR_C = 1e-3

# make variable types for automatic setting to GPU or CPU, depending on GPU availability
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
        
class OrnsteinUhlenbeckProcess: # the noise added to the action space
    def __init__(self, mu = None, sigma=0.05, theta=.25, dimension=1e-2, x0=None,num_steps=12000):
        self.theta = theta
        self.mu = mu 
        self.sigma = sigma
        self.dt = dimension
        self.x0 = x0
        self.reset()

    def step(self): # sample
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

class AdaptiveParamNoiseSpec(object):
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.2, adaptation_coefficient=1.01):
        """
        Note that initial_stddev and current_stddev refer to std of parameter noise, 
        but desired_action_stddev refers to (as name notes) desired std in action space
        """
        self.initial_stddev = initial_stddev # sigma
        self.desired_action_stddev = desired_action_stddev # delta
        self.adaptation_coefficient = adaptation_coefficient # alpha

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adaptation_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adaptation_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adaptation_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adaptation_coefficient)

def ddpg_distance_metric(actions1, actions2):
    """
    Compute "distance" between actions taken by two policies at the same states
    Expects numpy arrays
    """
    diff = actions1-actions2
    mean_diff = np.mean(np.square(diff), axis=0)
    dist = sqrt(np.mean(mean_diff))
    return dist

def hard_update(target, source): # copy parameters directly
    for target_param, param in zip(target.parameters(), source.parameters()):
           target_param.data.copy_(param.data)

class actor(nn.Module):
    def __init__(self, input_size, output_size):
        super(actor, self).__init__()

        self.state_dim = input_size
        self.action_dim = output_size
        self.h1_dim = 2*input_size###
        self.h2_dim = 2*input_size

        self.fc1 = nn.Linear(self.state_dim, self.h1_dim)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.bn1 = nn.BatchNorm1d(self.h1_dim)###
        self.ln1 = nn.LayerNorm(self.h1_dim)

        self.fc2 = nn.Linear(self.h1_dim, self.h2_dim)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.bn2 = nn.BatchNorm1d(self.h2_dim)
        self.ln2 = nn.LayerNorm(self.h2_dim)

        self.fc3 = nn.Linear(self.h2_dim, self.action_dim)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, state):        
        #x = F.relu(self.bn1(self.fc1(state)))
        #x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc1(state)
        x = self.bn1(x)
        x = self.ln1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.ln2(x)
        x = F.relu(x)

        x = self.fc3(x)
        #action = F.relu(x)
        action = F.tanh(x)
        #action = F.softmax(x)
        #action = F.sigmoid(x)
        '''
        x = F.relu(self.ln1(self.bn1(self.fc1(state))))###
        x = F.relu(self.ln2(self.bn2(self.fc2(x))))
        action = F.relu(self.fc3(x))
        '''
        return action

class critic(nn.Module):
    def __init__(self, state_size, action_size, output_size = 1):
        super(critic, self).__init__()

        self.state_dim = state_size
        self.action_dim = action_size
        self.h1_dim = 2*state_size
        self.h2_dim = 2*action_size
        self.fc1 = nn.Linear(self.state_dim, self.h1_dim)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.bn1 = nn.BatchNorm1d(self.h1_dim)
        self.ln1 = nn.LayerNorm(self.h1_dim)
        self.fc2 = nn.Linear(self.h1_dim + self.action_dim, self.h2_dim)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(self.h2_dim, output_size)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, state, action):
        #s1 = F.relu(self.fc1(state))
        s1 = F.relu(self.ln1(self.bn1(self.fc1(state))))
        x = torch.cat((s1,action), dim=1)###
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DDPG:
    def __init__(self, obs_dim, act_dim, critic_lr = LR_C, actor_lr = LR_A, gamma = GAMMA, batch_size = BATCH_SIZE, memMaxSize = 60000):
        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE
        self.actor_lr = LR_A
        self.critic_lr = LR_C

        #Memory
        self.memMaxSize = memMaxSize
        self.BATCH_SIZE = BATCH_SIZE
        self.memory = [] #zeros(self.memMaxSize)
        self.position=0
        self.memorySize=0
        
        # Dimension
        self.act_dim = act_dim
        self.obs_dim = obs_dim

        # actor
        self.actor =           actor(input_size = obs_dim, output_size = act_dim).type(FloatTensor)
        self.actor_target =    actor(input_size = obs_dim, output_size = act_dim).type(FloatTensor)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # critic
        self.critic = critic(state_size = obs_dim, action_size = act_dim, output_size = 1).type(FloatTensor)
        self.critic_target = critic(state_size = obs_dim, action_size = act_dim, output_size = 1).type(FloatTensor)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        if use_cuda:
            self.actor.cuda()
            self.actor_target.cuda()
            self.critic.cuda()
            self.critic_target.cuda()
            
        # optimizers
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr = self.actor_lr,weight_decay=1e-2)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr = self.critic_lr, weight_decay=1e-2)
        
        # critic loss
        self.critic_loss = nn.MSELoss()
        
        # noise
        self.noise = OrnsteinUhlenbeckProcess(mu = np.zeros(act_dim),dimension = act_dim, num_steps = NUM_EPISODES) # OU noise
        self.noise.reset() # reset actor OU noise
        self.Var = Var
    
    def random_action(self):
        action = np.random.uniform(-1.0,1.0,self.act_dim)
        return action
    
    def action(self, s, noise = 0): # choose action
        obs = torch.from_numpy(s).unsqueeze(0)
        inp = Variable(obs,requires_grad=False).type(FloatTensor)

        self.actor.eval()# switch to evaluation mode
        action = self.actor(inp).data[0].cpu().numpy()### .
        self.actor.train()# switch to trainning mode
        
        # add action space noise
        #action = np.random.normal(action, self.Var)
        action += noise
        '''
        self.noise.reset() # reset actor OU noise
        noise = self.noise.step()
        a += noise
        ''' 
        # clipping --> around
        #a = np.around(np.clip(a, 0, 1))            
        return action

    def addMemory(self, ep):
        if self.memorySize < self.memMaxSize:
            self.memory.append(ep)
            self.memorySize+=1 
        else:
            self.memory[self.position] = ep
        self.position = (self.position + 1) % self.memMaxSize 

        '''
        self.memory[self.position] = ep   
        self.position = (self.position + 1) % self.memMaxSize 
        self.memorySize+=1
        '''
        
    def sampleMemory(self, batch_size):     
        return random.sample(self.memory, batch_size)

    def train(self):
        training_data = np.array(self.sampleMemory(BATCH_SIZE))
        #batch_s,batch_a,batch_r,batch_s1= training_data
        batch_s,batch_a,batch_r,batch_s1=zip(*training_data)
        s1 = Variable(FloatTensor(batch_s))
        a1 = Variable(FloatTensor(batch_a))
        r1 = Variable(FloatTensor(np.array(batch_r).reshape(-1,1)))
        s2 = Variable(FloatTensor(batch_s1))
        
        
        # ---------------------- optimize critic ----------------------###
        a2 = self.actor_target(s2)
        next_val = self.critic_target(s2, a2).detach()
        q_expected = r1 + self.gamma*next_val
        q_predicted = self.critic(s1, a1)
        
        # compute critic loss, and update the critic
        loss_critic = self.critic_loss(q_predicted, q_expected)
        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()

        # ---------------------- optimize actor ----------------------
        pred_a1 = self.actor.forward(s1)
        #print(self.critic(s1, pred_a1))
        loss_actor = -1*self.critic(s1, pred_a1)#############
        loss_actor = loss_actor.mean()
        
        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()

        # sychronize target network with fast moving one
        self.weightSync(self.critic_target, self.critic)
        self.weightSync(self.actor_target, self.actor)

        return loss_actor, loss_critic

    def weightSync(self,target_model, source_model, tau = 0.001): # Update the target networks (soft update)
        for parameter_target, parameter_source in zip(target_model.parameters(), source_model.parameters()):
            #print('parameter_target:',parameter_target)
            #print('parameter_source:',parameter_source)
            parameter_target.data.copy_((1 - tau) * parameter_target.data + tau * parameter_source.data)

if __name__ == "__main__":
    #env = BS(nBS=4,nUE=4,nMaxLink=2,nFile=5,nMaxCache=2,loadENV = True)
    env = BS(nBS=40,nUE=10,nMaxLink=2,nFile=50,nMaxCache=2,loadENV = True)
    obs_dim = len(env.s_)
    cluster_act_dim = (env.U*env.B)
    RL_s = env.s_
    Mddpg_cl = DDPG(obs_dim = obs_dim, act_dim = cluster_act_dim)
    a_cl = Mddpg_cl.action(RL_s)
        