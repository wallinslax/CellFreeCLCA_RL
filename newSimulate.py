#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Proprietary Design
#import robot_ghliu
from newDDPG import DDPG
from newENV import BS
# Public Lib
from torch.autograd import Variable
import torch
import numpy as np
import time,copy,os,csv,random,pickle
import matplotlib.pyplot as plt
from numpy.random import randn
#from random import randint
from tqdm import tqdm
from datetime import date
today = date.today()
os.environ['CUDA_VISIBLE_DEVICES']='1'
#####################  hyper parameters  ####################
# Simulation Parameter
LOAD_EVN = True
RESET_CHANNEL = True
REQUEST_DUPLICATE = False
          
MAX_EPISODES = 10**3
MAX_EP_STEPS = 10**2
warmup = -1
epsilon = 0.2
#####################################
def plotMetric(poolEE,poolBestEE):
    xScale = 100
    x = range( len(poolEE[-xScale:]) )
    plt.cla()
    plt.plot(x,poolEE[-xScale:],'bo-',label='EE RL')
    plt.plot(x,poolBestEE[-xScale:],'r^-',label='EE BF')
    plt.title("Metric Visualization") # title
    plt.ylabel("Bits/J") # y label
    plt.xlabel("Iteration") # x label
    plt.xlim([0, xScale])
    #plt.ylim([0, 40])
    plt.grid()
    plt.legend()
    #handles, labels = plt.gca().get_legend_handles_labels()
    #by_label = dict(zip(labels, handles))
    #plt.legend(by_label.values(), by_label.keys())
    fig = plt.gcf()
    fig.show()
    fig.canvas.draw()
    plt.pause(0.001)

def train_1act(env):
    poolEE=[]
    poolHR=[]
    poolLossActor = []
    poolLossCritic = []
    # new ACT 
    obs_dim = len(env.s_)
    cluster_act_dim = (env.U*env.B)
    cache_act_dim = (env.B*env.F)
    actDim = cluster_act_dim + cache_act_dim
    #ddpg_s = DDPG(obs_dim = obs_dim, act_dim = actDim)###
    ddpg_cl = DDPG(obs_dim = obs_dim, act_dim = cluster_act_dim)
    ddpg_ca = DDPG(obs_dim = obs_dim, act_dim = cache_act_dim)
    
    #ddpg_cl.actor = torch.load('CellFreeCLCA_RL/data/cl_mddpg_actor.pt')
    #ddpg_ca.actor = torch.load('CellFreeCLCA_RL/data/ca_mddpg_actor.pt')
    mu = 0
    noiseSigma = 1 # control exploration
    
    #---------------------------------
    # Load Optimal clustering and caching Policy
    filenameBF = 'data/Result_BruteForce_4AP_4UE_2020-10-12'
    with open(filenameBF+'.pkl','rb') as f: 
        bs_coordinate, u_coordinate , g, userPreference, Req, bestEE, opt_clustering_policy_UE, opt_caching_policy_BS = pickle.load(f)
    # Load opt_caching_policy_BS and convert to opt_caching_state
    opt_caching_state = np.zeros([env.B,env.F])
    for b in range(env.B):
        opt_caching_state[b][ list(opt_caching_policy_BS[b]) ] = 1
    #--------------------------------- 
    for ep in tqdm(range(MAX_EPISODES)):
        ep_reward = 0
        obs = env.reset()# Get initial state
        for step in range(MAX_EP_STEPS):
            #Epsilon-Greedy Algorithm
            # https://medium.com/analytics-vidhya/the-epsilon-greedy-algorithm-for-reinforcement-learning-5fe6f96dc870
            '''
            if np.random.rand() > epsilon: 
                noise = np.zeros(cluster_act_dim)
                a_cl = ddpg_cl.action(obs,noise)# choose action [ env.U*env.B x 1 ]
                
                a_ca = opt_caching_state.flatten()
                action = np.concatenate((a_cl, a_ca), axis=0)
            
            else:
                action = env.action_space.sample()
                action[-cache_act_dim:] = opt_caching_state.flatten()
            '''
            if(step%30 ==0):
                noiseSigma*=0.995
            noise = np.random.normal(mu, noiseSigma,size=cluster_act_dim)
            a_cl = ddpg_cl.action(obs,noise)# choose action [ env.U*env.B x 1 ]
            a_ca = opt_caching_state.flatten()
            action = np.concatenate((a_cl, a_ca), axis=0)
            
            # take action to ENV
            obs2, reward, done, info = env.step(action)
            EE = reward
            poolEE.append(EE)
            HR = info["HR"]

            # RL update
            ddpg_cl.addMemory([obs,a_cl,reward,obs2])
            ddpg_ca.addMemory([obs,a_ca,reward,obs2])
            if len(ddpg_cl.memory) > ddpg_cl.BATCH_SIZE:
                lossActor, lossCritic = ddpg_cl.train()
                poolLossActor.append(lossActor)
                poolLossCritic.append(lossCritic)
            obs = obs2
            ep_reward += reward
        if ep_reward>8000:
            print('Episode:{} Reward:{} Explore:{}'.format(ep,ep_reward,noiseSigma))
    #---------------------------------------------------------------------------------------------    
    '''
    # save actor parameter
    path = "data/"
    filenameSDDPG = path + "SDDPG_Model_" + str(env.B)+'AP_'+str(env.U)+'UE_' + str(today) + '.pt'
    torch.save(ddpg_s.actor, filenameSDDPG)
    ddpg_s.actor = torch.load(filenameSDDPG)
    '''
    # save actor parameter
    path = "data/"
    filenameDDPG_CL = path + "DDPG_CL_Model_" + str(env.B)+'AP_'+str(env.U)+'UE_' + str(today) + '.pt'
    filenameDDPG_CA = path + "DDPG_CA_Model_" + str(env.B)+'AP_'+str(env.U)+'UE_' + str(today) + '.pt'
    torch.save(ddpg_cl.actor, filenameDDPG_CL)
    torch.save(ddpg_ca.actor, filenameDDPG_CA)
    ddpg_cl.actor = torch.load(filenameDDPG_CL)
    ddpg_ca.actor = torch.load(filenameDDPG_CA)
    
    #---------------------------------------------------------------------------------------------
    # plot Brute Force V.S. RL
    plt.cla()
    plt.plot(range(len(poolLossActor)),poolLossActor,'r-',label='Loss of actor')
    plt.plot(range(len(poolLossCritic)),poolLossCritic,'c-',label='Loss of critic')

    nXpt=len(poolEE)
    plt.plot(range(nXpt),poolEE,'b-',label='EE of 2 Actors: DDPG_Cluster + DDPG_Cache')
    plt.plot(range(nXpt),bestEE*np.ones(nXpt),'k-',label='EE of Brute Force')

    titleNmae = 'Energy Efficiency \n nBS='+str(env.B)+ \
                                    ',nUE='+str(env.U)+\
                                    ',nMaxLink='+str(env.L)+\
                                    ',nFile='+str(env.F)+\
                                    ',nMaxCache='+str(env.N)
    plt.title(titleNmae) # title
    plt.ylabel("Bits/J") # y label
    plt.xlabel("Iteration") # x label
    #plt.xlim([0, len(poolEE)])
    plt.grid()
    plt.legend()
    fig = plt.gcf()
    filename = 'data/BF_vs_RL'+ str(env.B)+'AP_'+str(env.U)+'UE_' +str(MAX_EPISODES)+'_'+str(today)
    #filename = 'data/1DDPG'+ str(env.B)+'AP_'+str(env.U)+'UE_' + str(MAX_EPISODES)+'_'+str(today)
    fig.savefig(filename + '.eps', format='eps',dpi=1200)
    fig.savefig(filename + '.png', format='png',dpi=1200)
    fig.show()
    # Save the plot point
    with open(filename+'.pkl', 'wb') as f:  
        pickle.dump([env, poolEE,poolLossActor,poolLossCritic], f)
    # Load the plot point 
    with open(filename+'.pkl','rb') as f: 
        env, poolEE,poolLossActor,poolLossCritic = pickle.load(f)
    #---------------------------------------------------------------------------------------------
    # plot Hit Rate
    # plt.cla()
    # plt.plot(range(len(poolEE)),poolEE,'bo-',label='EE of 2 Actors: DDPG_Cluster + DDPG_Cache')

if __name__ == '__main__':
    # new ENV
    #env1 = BS(nBS=40,nUE=10,nMaxLink=2,nFile=50,nMaxCache=10,loadENV = True)
    env3 = BS(nBS=4,nUE=4,nMaxLink=2,nFile=5,nMaxCache=2,loadENV = True)
    train_1act(env3)
    #env2 = BS(nBS=40,nUE=10,nMaxLink=2,nFile=50,nMaxCache=2,loadENV = True)
    #train_2act(env2)
    