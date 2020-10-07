import matplotlib.pyplot as plt
import numpy as np
import random
import torch.nn.functional as F
from numpy.random import randn
#environment
#from newENV import BS,plot_UE_SBS_association,UE_SBS_location_distribution,plot_UE_SBS_distribution
import os
import time
#pytorch
import torch
from torch.autograd import Variable
import torch.nn as nn
from math import sqrt
import copy
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
#np.random.seed(SEED)
#####################################


logging_interval = 40
animate_interval = logging_interval * 5
logdir='./DDPG/'
VISUALIZE = False
SEED = 0
MAX_PATH_LENGTH = 500
NUM_EPISODES = 12000
GAMMA=0.99
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
        self.h1_dim = 300###
        self.h2_dim = 300

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
        action = F.relu(x)
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
        self.h1_dim = 400
        self.h2_dim = 400
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
        self.memory=[]
        self.position=0
        self.memMaxSize = memMaxSize
        self.BATCH_SIZE = BATCH_SIZE
        
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
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr = self.actor_lr)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr = self.critic_lr, weight_decay=1e-2)
        
        # critic loss
        self.critic_loss = nn.MSELoss()
        
        # noise
        self.noise = OrnsteinUhlenbeckProcess(mu = np.zeros(act_dim),dimension = act_dim, num_steps = NUM_EPISODES) # OU noise
        self.noise.reset() # reset actor OU noise
    
    def action(self, s): # choose action
        obs = torch.from_numpy(s).unsqueeze(0)
        inp = Variable(obs,requires_grad=False).type(FloatTensor)

        self.actor.eval()# switch to evaluation mode
        a = self.actor(inp).data[0].cpu().numpy()### 
        self.actor.train()# switch to trainning mode
        # add action space noise
        self.noise.reset() # reset actor OU noise
        noise = self.noise.step()
        a += noise
        #print('noise=',noise)
        '''
        if noise.any():
            #a += noise # OU noise
            a += self.noise.step()
        '''
            
        # clipping --> around
        #a = np.around(np.clip(a, 0, 1))            
        return a

    def addMemory(self, ep):
        self.memory.append(ep)
        self.position = (self.position + 1) % self.memMaxSize       
        #self.memory[self.position] = tuple(ep)
        
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
        
        a2 = self.actor_target(s2)
        # ---------------------- optimize critic ----------------------###
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
        loss_actor = -1*self.critic(s1, pred_a1)###
        loss_actor = loss_actor.mean()
        
        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()

        # sychronize target network with fast moving one
        self.weightSync(self.critic_target, self.critic)
        self.weightSync(self.actor_target, self.actor)

    def weightSync(self,target_model, source_model, tau = 0.001): # Update the target networks (soft update)
        for parameter_target, parameter_source in zip(target_model.parameters(), source_model.parameters()):
            parameter_target.data.copy_((1 - tau) * parameter_target.data + tau * parameter_source.data)

if __name__ == "__main__":
    def channel_reset(U,B,pl):
        h = (np.sqrt(h_var/2)*(randn(1,U*B)+1j*randn(1,U*B))).reshape((U,B)) # h~CN(0,1); |h|~Rayleigh fading
        G = pl*np.power(abs(h),2)    
        '''[6] Received power of UE from BS'''
        P = np.zeros((U,B))
        P_bs = np.zeros((U,B))
        P_bs[:,0]=P_MBS
        P_bs[:,1:]=P_SBS
        P = G*P_bs  
        return P   
    '''[1] SBS/ UE distribution'''  
    u_coordinate = UE_SBS_location_distribution(lambda0 = lambda_u) # UE
    U = len(u_coordinate)
    #
    B = num_bs+1    
    bs_coordinate = np.array([ [ 0. ,  0.  ],
                               [-0.2,  0.1],
                               [ 0.3, -0.05 ],
                               [0.2,  0.2]])
    '''[2] Pair-wise distance'''
    D = np.zeros((U,B))
    for u in  range(U):
        for b in range(B):
            D[u][b] = 1000*np.sqrt(sum((u_coordinate[u]-bs_coordinate[b])**2)) # km -> m
    
    '''[3] Channel gain'''
    pl = k*np.power(D, -alpha) # Path-loss
    P = channel_reset(U,B,pl)

    '''
    xx_sbs, yy_sbs = UE_SBS_location_distribution(numbPoints = num_bs) # SBS
    sbs_coordinate = np.concatenate((xx_sbs,yy_sbs),axis=1)
    bs_coordinate = np.concatenate((np.array([[0,0]]), sbs_coordinate),axis=0) # MBS+SBS
    B = len(bs_coordinate)
    
    U = num_u
    u_coordinate = np.array([[ 0.14171213,  -0.1],
                               [ 0.0137101 ,  0.05360341],
                               [ 0.05249104,  0.19713417],
                               [-0.15, -0.05],
                               [-0.05418348, 0.15],
                               [-0.25, 0.15],
                               [0.3, -0.1],
                               [ 0.22401997,  0.04174466],
                               [ 0.22792172,  0.20416831],
                               [ 0.03664668, -0.03173442]])
    B = num_bs+1    
    bs_coordinate = np.array([ [ 0. ,  0.  ],
                               [-0.2,  0.1],
                               [ 0.3, -0.05 ],
                               [0.2,  0.2]])
    '''
    ############################################
    #bs_coordinate,u_coordinate,B,U,P,up,EE_mean,EE_std,CS_mean,CS_std,pl = load_env_distribution()
    #env = BS(bs_coordinate,u_coordinate,B,U,P,up,REQUEST_DUPLICATE,False,EE_mean,EE_std,CS_mean,CS_std)
    env1 = BS(bs_coordinate,u_coordinate,B,U,P)
    obs_dim = len(env.SINR)+len(env.s_)+len(env.Prof_state)
    act_dim = (env.U*env.B)+(env.B*env.F)
    ddpg1 = DDPG(obs_dim = obs_dim, act_dim = act_dim)
    memory1 = Replay(env=env,maxlen=MEM_SIZE)
    memory1.initialize(init_length= 150)
    param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.05,desired_action_stddev=0.3, adaptation_coefficient=1.05)
    step = 0
    for i in range(1):
        '''
        [1]multi DDPG + parameter noise ->3*DDPG
        [2]multi DDPG ->3*DDPG
        [3]single DDPG ->1*DDPG
        [4]Popular cache & RL clustering
        [5]Popular cache & random clustering
        '''
        total_reward = [0, 0, 0, 0, 0, 0,0,0,0]
        step_counter = [0, 0, 0, 0, 0, 0,0,0,0] # distributor agent's episode length (related to convergence rate)
        done = [0, 0, 0, 0, 0, 0,0,0,0]
        total_ee = [0, 0, 0, 0, 0, 0,0,0,0]
        total_cs = [0, 0, 0, 0, 0, 0,0,0,0]
        total_hit = [0, 0, 0, 0, 0, 0,0,0,0]
        total_itf = [0, 0, 0, 0, 0, 0,0,0,0]
        
        
        noise_counter = 0 # G: sum of N running agents' episode length
        
        # get initial state
        RL1_s = env1.reset() 
        
        
        # add parameter noise onto N running agents. 
        ddpg1.perturb_actor_parameters(param_noise) 
        
        
        for j in range(10):
            
            ddpg1.noise.reset() # reset actor OU noise
            RL1_a = ddpg1.action(RL1_s, ddpg1.noise.step(), param_noise).astype(int)# choose action
            RL1_clustering_policy_num,RL1_caching_policy_num = env1.action_vec_2_num(RL1_a)
            RL1_r, done[0], RL1_EE, RL1_CS, RL1_SINR, RL1_Hit_rate, RL1_s_, RL1_Itf = env1.step(clustering_policy_num = RL1_clustering_policy_num,
                                                                                       caching_policy_num = RL1_caching_policy_num, 
                                                                                       update_Req = 1, 
                                                                                       normalize = 1)
            memory1.add((RL1_s,RL1_a,RL1_r,RL1_s_))
            total_reward[0] += RL1_r
            total_ee[0] += RL1_EE
            total_cs[0] += RL1_CS
            total_hit[0] += RL1_Hit_rate
            total_itf[0] += RL1_Itf
            step_counter[0] += 1
            training_data1 = np.array(memory1.sample(BATCH_SIZE))
            ddpg1.train(training_data1)
            noise_counter += 1
            RL1_s = RL1_s 
   
    myddpg_1 =  ddpg1
    # calculate ddpg_distance_metric and update parameter noise
    if memory1.position-noise_counter > 0:
        noise_data=memory1.data[memory1.position-noise_counter:memory1.position]
    else:
        noise_data=memory1.data[memory1.position-noise_counter+MEM_SIZE:MEM_SIZE] \
        + memory1.data[0:memory1.position]   
    noise_data=np.array(noise_data)
    noise_s, noise_a, _,_ = zip(*noise_data)
    perturbed_actions = noise_a### this one is strange
    noise_s = np.array(noise_s)
    unperturbed_actions = myddpg_1.action(noise_s, None, None).astype(int)### noise_counter != 300
    ddpg_dist = ddpg_distance_metric(perturbed_actions, unperturbed_actions)
    param_noise.adapt(ddpg_dist) # update parameter noise
    
        