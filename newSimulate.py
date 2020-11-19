#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Proprietary Design
#import robot_ghliu
from newDDPG import DDPG
from newENV import BS
from newENV import plot_UE_BS_distribution_Cache
# Public Lib
from torch.autograd import Variable
import torch
import numpy as np
import time,copy,os,csv,random,pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
from numpy.random import randn
#from random import randint
from tqdm import tqdm
from datetime import date
today = date.today()
os.environ['CUDA_VISIBLE_DEVICES']='1'
import concurrent.futures
import multiprocessing
num_cores = multiprocessing.cpu_count()
#####################  hyper parameters  ####################
# Simulation Parameter
LOAD_EVN = True
RESET_CHANNEL = True
REQUEST_DUPLICATE = False
          
MAX_EPISODES = 10**2*10
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

def trainModel(env,actMode,changeReq,changeChannel):
    poolEE=[]
    poolHR=[]
    poolLossActor = []
    poolLossCritic = []
    # new ACT 
    ddpg_s = DDPG(obs_dim = env.dimObs, act_dim = env.dimAct)###
    ddpg_cl = DDPG(obs_dim = env.dimObs, act_dim = env.dimActCL)
    ddpg_ca = DDPG(obs_dim = env.dimObs, act_dim = env.dimActCA)
    #ddpg_cl.actor = torch.load('CellFreeCLCA_RL/data/cl_mddpg_actor.pt')
    #ddpg_ca.actor = torch.load('CellFreeCLCA_RL/data/ca_mddpg_actor.pt')
    #---------------------------------------------------------------------------------------------
    '''
    # Load Optimal clustering and caching Policy
    filenameBF = 'data/Result_BruteForce_4AP_4UE_2020-10-28'
    with open(filenameBF+'.pkl','rb') as f: 
        bs_coordinate, u_coordinate , g, userPreference, Req, bestEE, opt_clustering_policy_UE, opt_caching_policy_BS = pickle.load(f)
    
    # Load opt_caching_policy_BS and convert to opt_caching_state
    opt_caching_state = np.zeros([env.B,env.F])
    for b in range(env.B):
        opt_caching_state[b][ list(opt_caching_policy_BS[b]) ] = 1
    '''
    #---------------------------------------------------------------------------------------------
    mu = 0
    noiseSigma = 1 # control exploration
    for ep in tqdm(range(MAX_EPISODES)):
        ep_reward = 0
        obs = env.reset()# Get initial state
        for step in range(MAX_EP_STEPS):
            #Epsilon-Greedy Algorithm
            # https://medium.com/analytics-vidhya/the-epsilon-greedy-algorithm-for-reinforcement-learning-5fe6f96dc870
            '''
            if np.random.rand() > epsilon: 
                noise = np.zeros(env.dimActCL)
                a_cl = ddpg_cl.action(obs,noise)# choose action [ env.U*env.B x 1 ]
                
                a_ca = opt_caching_state.flatten()
                action = np.concatenate((a_cl, a_ca), axis=0)
            
            else:
                action = env.action_space.sample()
                action[-env.dimActCA:] = opt_caching_state.flatten()
            '''
            if(step%30 ==0):
                noiseSigma*=0.995
                
                if changeReq:
                    env.resetReq()

                if changeChannel:
                    env.resetChannel()
                
                    

            if actMode == '2act':
                noise = np.random.normal(mu, noiseSigma,size=env.dimActCL)
                a_cl = ddpg_cl.action(obs,noise)# choose action [ env.U*env.B x 1 ]
                #a_ca = opt_caching_state.flatten()
                noise = np.random.normal(mu, noiseSigma,size=env.dimActCA)
                a_ca = ddpg_ca.action(obs,noise)# choose action [ env.U*env.B x 1 ]
                action = np.concatenate((a_cl, a_ca), axis=0)
            elif actMode == '1act':
                noise = np.random.normal(mu, noiseSigma,size=env.dimAct)
                action = ddpg_s.action(obs,noise)
            
            # take action to ENV
            obs2, reward, done, info = env.step(action)
            EE = reward
            poolEE.append(EE)
            HR = info["HR"]

            # RL Add Memory
            if actMode == '2act':
                ddpg_cl.addMemory([obs,a_cl,reward,obs2])
                ddpg_ca.addMemory([obs,a_ca,reward,obs2])
            elif actMode == '1act':
                ddpg_s.addMemory([obs,action,reward,obs2])

            # RL Upadate
            if actMode == '2act':
                if len(ddpg_cl.memory) > ddpg_cl.BATCH_SIZE:
                    lossActor, lossCritic = ddpg_cl.train()
                    poolLossActor.append(lossActor)
                    poolLossCritic.append(lossCritic)
                    lossActor, lossCritic = ddpg_ca.train()
            elif actMode == '1act':
                if len(ddpg_s.memory) > ddpg_s.BATCH_SIZE:
                    lossActor, lossCritic = ddpg_s.train()
                    poolLossActor.append(lossActor)
                    poolLossCritic.append(lossCritic)

            obs = obs2
            ep_reward += reward

        if ep_reward>10000:
            print('\nEpisode:{} Reward:{} Explore:{}'.format(ep,ep_reward,noiseSigma))
        
        
        
    #---------------------------------------------------------------------------------------------  
    #return ddpg_s.actor,env,poolEE,poolLossActor,poolLossCritic
    #---------------------------------------------------------------------------------------------     
    # Save actor SDDPG
    if actMode == '2act':
        filenameDDPG_CL = 'data/2ACT_' + "DDPG_CL_" + str(env.B)+'AP_'+str(env.U)+'UE_' + str(env.F) + 'File_'+ str(env.N) +'Cache_'+ str(today) + '.pt'
        filenameDDPG_CA = 'data/2ACT_' + "DDPG_CA_" + str(env.B)+'AP_'+str(env.U)+'UE_' + str(env.F) + 'File_'+ str(env.N) +'Cache_'+ str(today) + '.pt'
        torch.save(ddpg_cl.actor, filenameDDPG_CL)
        torch.save(ddpg_ca.actor, filenameDDPG_CA)
        ddpg_cl.actor = torch.load(filenameDDPG_CL)
        ddpg_ca.actor = torch.load(filenameDDPG_CA)
    # Save actor MDDPG
    elif actMode == '1act': 
        filenameSDDPG = 'data/1ACT_' + "DDPG_ALL_" + str(env.B)+'AP_'+str(env.U)+'UE_' + str(env.F) + 'File_'+ str(env.N) +'Cache_' + str(today) + '.pt'
        torch.save(ddpg_s.actor, filenameSDDPG)
    
    # Save the plot point
    if actMode == '2act':
        filename = 'data/2ACT_'+ str(env.B)+'AP_'+str(env.U)+'UE_' + str(env.F) + 'File_'+ str(env.N) +'Cache_' +str(MAX_EPISODES*MAX_EP_STEPS)+'_'+str(today)
    elif actMode == '1act':
        filename = 'data/1ACT_'+ str(env.B)+'AP_'+str(env.U)+'UE_' + str(env.F) + 'File_'+ str(env.N) +'Cache_' +str(MAX_EPISODES*MAX_EP_STEPS)+'_'+str(today)

    with open(filename+'.pkl', 'wb') as f:  
        pickle.dump([env, poolEE,poolLossActor,poolLossCritic,], f)

if __name__ == '__main__':
    # new ENV
    env = BS(nBS=4,nUE=4,nMaxLink=2,nFile=5,nMaxCache=2,loadENV = True)
    '''
    with concurrent.futures.ProcessPoolExecutor(max_workers= (num_cores-2) ) as executor:
        futures = []
        for i in range(num_cores-3):

            #subBestEE,subOpt_clustering_policy_UE,subOpt_caching_policy_BS = self.smallPeice(universe_clustering_policy_UE,caching_policy_BS)
            future = executor.submit(trainModel, env,'1act',True,False) 
            futures.append(future)
        bestEE=0

        for future in tqdm(concurrent.futures.as_completed(futures),total=len(futures)):
            #print(future.result())
            ddpg_s = DDPG(obs_dim = env.dimObs, act_dim = env.dimAct)
            ddpg_s.actor,env,poolEE,poolLossActor,poolLossCritic = future.result()
            obs = env.reset()
            subBestEE = 0
            for i in range(1000):
                noise = np.random.normal(0,0,size=env.dimAct)
                action = ddpg_s.action(obs,noise)
                obs, reward, done, info = env.step(action)
                if reward>subBestEE:
                    subBestEE = reward
            if subBestEE>bestEE:
                bestEE = subBestEE
                kingActor = ddpg_s.actor
                kingPoolEE = poolEE
                kingPoolLossActor = poolLossActor
                kingPoolLossCritic = poolLossCritic
    '''
    #trainModel(env,actMode='1act',changeReq=True, changeChannel=False)
    #---------------------------------------------------------------------------------------------
    
    # Load Optimal clustering and caching Policy
    filenameBF = 'data/4.4.2.5.2/BF_4AP_4UE_5File_2Cache_2020-11-10'
    with open(filenameBF+'.pkl','rb') as f: 
        bs_coordinate, u_coordinate , g, userPreference, Req, bestEE, opt_clustering_policy_UE, opt_caching_policy_BS = pickle.load(f)
    

    #---------------------------------------------------------------------------------------------
    # Load the plot point 
    #filename = 'data/BF_vs_RL4AP_4UE_1000_2020-11-02'
    filename = 'data/1ACT_'+ str(env.B)+'AP_'+str(env.U)+'UE_' + str(env.F) + 'File_'+ str(env.N) +'Cache_' +str(MAX_EPISODES*MAX_EP_STEPS)+'_'+str(today)
    with open(filename+'.pkl','rb') as f: 
        env, poolEE,poolLossActor,poolLossCritic = pickle.load(f)
    #---------------------------------------------------------------------------------------------
    # plot Brute Force V.S. RL
    print(max(poolLossCritic))
    print(min(poolLossActor))
    print(max(poolEE))
    plt.cla()
    plt.plot(range(len(poolLossActor)),poolLossActor,'r-',label='Loss of actor')
    #plt.plot(range(len(poolLossCritic)),poolLossCritic,'c-',label='Loss of critic')

    nXpt=len(poolEE)
    plt.plot(range(nXpt),poolEE,'b-',label='EE of 1 Actors')
    #plt.plot(range(nXpt),poolEE,'b-',label='EE of 2 Actors: DDPG_Cluster + DDPG_Cache')
    finalValue = "{:.2f}".format(max(poolEE))
    plt.annotate(finalValue, (nXpt,poolEE[-1]),textcoords="offset points",xytext=(0,-20),ha='center',color='b')
    
    plt.plot(range(nXpt),bestEE*np.ones(nXpt),'k-',label='EE of Brute Force')
    finalValue = "{:.2f}".format(bestEE)
    plt.annotate(finalValue, (nXpt,bestEE),textcoords="offset points",xytext=(0,10),ha='center',color='k')
    

    titleNmae = 'Energy Efficiency \n'+filename
    plt.title(titleNmae) # title
    plt.ylabel("Bits/J") # y label
    plt.xlabel("Iteration") # x label
    #plt.xlim([0, len(poolEE)])
    plt.grid()
    plt.legend()
    fig = plt.gcf()
    #fig.savefig(filename + '.eps', format='eps',dpi=1200)
    fig.savefig(filename + '.png', format='png',dpi=1200)
    fig.show()
    
    #---------------------------------------------------------------------------------------------
    # plot CL/CA Policy
    ddpg_s = DDPG(obs_dim = env.dimObs, act_dim = env.dimAct)###
    filenameSDDPG = 'data/1ACT_' + "DDPG_ALL_" + str(env.B)+'AP_'+str(env.U)+'UE_' + str(env.F) + 'File_'+ str(env.N) +'Cache_' + str(today) + '.pt'
    ddpg_s.actor = torch.load(filenameSDDPG)
    
    rlBestEE = 0
    rlBestCLPolicy_UE=[]
    rlBestCAPolicy_BS=[]
    obs = env.reset()# Get initial state
    for i in range(1000):
        noise = np.random.normal(0,0,size=env.dimAct)
        action = ddpg_s.action(obs,noise)
        obs, reward, done, info = env.step(action)
        if reward>rlBestEE:
            rlBestEE = reward
            rlBestCLPolicy_UE, rlBestCAPolicy_BS = env.action2Policy(action)
    print('rlBestCLPolicy_UE:',rlBestCLPolicy_UE)
    print('rlBestCAPolicy_BS:',rlBestCAPolicy_BS)
    print('rlBestEE:',rlBestEE)
    '''
    print('opt_clustering_policy_UE:',opt_clustering_policy_UE)
    print('opt_caching_policy_BS:',opt_caching_policy_BS)
    print('bestEE:',bestEE)
    '''
    displayrlBestEE = "{:.2f}".format(rlBestEE)
    filenamePV = 'data/Visualization_1ACT_'+str(env.B)+'AP_'+str(env.U)+'UE_' + str(env.F) + 'File_'+ str(env.N) +'Cache_' +str(today)+'_'+displayrlBestEE
    plot_UE_BS_distribution_Cache(env.bs_coordinate,env.u_coordinate,env.Req,rlBestCLPolicy_UE,rlBestCAPolicy_BS,displayrlBestEE,filenamePV)

    #filenameBF = 'data/Visualization_BF_'+str(env.B)+'AP_'+str(env.U)+'UE_' + str(env.F) + 'File_'+ str(env.N) +'Cache_' +str(today)
    #plot_UE_BS_distribution_Cache(bs_coordinate,u_coordinate,Req,opt_clustering_policy_UE,opt_caching_policy_BS,bestEE,filenameBF)
    #---------------------------------------------------------------------------------------------
    