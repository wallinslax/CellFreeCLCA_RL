from newDDPG import DDPG
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import time
import matplotlib.pyplot as plt
from datetime import date
today = date.today()
#####################  hyper parameters  ####################

MAX_EPISODES = 1000
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
var = 3  # control exploration
###############################  training  ####################################
ENV_NAME = 'Pendulum-v0'
#ENV_NAME = 'LunarLander-v2'
env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)
s_dim = env.observation_space.shape[0]
#a_dim = env.action_space.n
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

ddpg_gym = DDPG(obs_dim = s_dim, act_dim = a_dim)
mu = 0
noiseSigma = 3 # control exploration

t1 = time.time()
flag_render = False
poolLossActor=[]
poolLossCritic=[]
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if flag_render:
            env.render()

        # Add exploration noise
        if(j%10 ==0):
                noiseSigma*=0.995# decay the action randomness
        noise = np.random.normal(mu, noiseSigma,size=a_dim)
        a = ddpg_gym.action(s,noise)
        s_, r, done, info = env.step(a) #in Pendulum-v0 case, done is always False
        ddpg_gym.addMemory([s,a,r,s_])

        #if len(ddpg_gym.memory) > ddpg_gym.BATCH_SIZE:
        if len(ddpg_gym.memory) > 1000:
            lossActor, lossCritic = ddpg_gym.train()
            poolLossActor.append(lossActor)
            poolLossCritic.append(lossCritic)
            
        s = s_
        ep_reward += r

        if j == MAX_EP_STEPS-1:
            #print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            print('Episode:{} Reward:{} Explore:{}'.format(i,ep_reward,noiseSigma))
            
            if ep_reward>-300:
                flag_render = True
            break
    
print('Running time: ', time.time() - t1)

# plot Brute Force V.S. RL------------------------------------------------------------------
plt.cla()
plt.plot(range(len(poolLossActor)),poolLossActor,'r-',label='Loss of actor')
plt.plot(range(len(poolLossCritic)),poolLossCritic,'c-',label='Loss of critic')

plt.title('Loss of DDPG') # title
plt.ylabel("Bits/J") # y label
plt.xlabel("Iteration") # x label
#plt.xlim([0, len(poolEE)])
plt.grid()
plt.legend()
fig = plt.gcf()
filename = 'data/DDPG_of_pytorch_'+ENV_NAME+'_'+str(today)
fig.savefig(filename + '.eps', format='eps',dpi=1200)
fig.savefig(filename + '.png', format='png',dpi=1200)
fig.show()