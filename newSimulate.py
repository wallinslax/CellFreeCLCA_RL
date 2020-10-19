#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Proprietary Design
import robot_ghliu
from newDDPG import OrnsteinUhlenbeckProcess,AdaptiveParamNoiseSpec, ddpg_distance_metric, hard_update, critic, actor, DDPG
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
          
MAX_EPISODES = 10**4
MAX_EP_STEPS = 10**4
warmup = 100
#####################################

def plotMetric3(poolEE,poolHR,poolCS):
    xScale = 100
    x = range( len(poolEE[-xScale:]) )
    plt.cla()
    plt.plot(x,poolEE[-xScale:],'bo-',label='Energy Efficiency (Bits/J)')
    plt.plot(x,poolHR[-xScale:],'r+-',label='Hite Rate (0~1)')
    plt.plot(x,poolCS[-xScale:],'yx-',label='Cluster Similarity (0~1)')
    plt.title("Metric Visualization") # title
    plt.ylabel("Ratio") # y label
    plt.xlabel("Time frame") # x label
    plt.xlim([0, xScale])
    plt.ylim([0, 2])
    plt.grid()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    fig = plt.gcf()
    fig.show()
    fig.canvas.draw()
    plt.pause(0.001)
def plotMetricBF(poolEE,poolBestEE):
    xScale = 100
    x = range( len(poolEE[-xScale:]) )
    plt.cla()
    plt.plot(x,poolEE[-xScale:],'bo-',label='Energy Efficiency (Bits/J)')
    plt.plot(x,poolBestEE[-xScale:],'r^-',label='Best Energy Efficiency')
    plt.title("Metric Visualization") # title
    plt.ylabel("Ratio") # y label
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

def train_1act(env, bestEE=0, bestHR=0):
    Addpg = DDPG(obs_dim = obs_dim, act_dim = cluster_act_dim+cache_act_dim) 
    a_all = Addpg.random_action()
    a_all = Addpg.action(RL_s)
    a_cl = a_all[0:cluster_act_dim]
    a_ca = a_all[cluster_act_dim:]


def train_2act(env, bestEE=0, bestHR=0):
    # While loop setup
    nGame = 1
    done = False
    poolMaxEE=[]
    while(1):
        poolEE=[]
        poolHR=[]
        poolBestEE = []
        poolLossActorCL = []
        poolLossCriticCL = []
        # new ACT 
        obs_dim = len(env.s_)
        cluster_act_dim = (env.U*env.B)
        cache_act_dim = (env.B*env.F)
        
        Mddpg_cl = DDPG(obs_dim = obs_dim, act_dim = cluster_act_dim)###
        Mddpg_ca = DDPG(obs_dim = obs_dim, act_dim = cache_act_dim)###
        #Mddpg_cl.actor = torch.load('CellFreeCLCA_RL/data/cl_mddpg_actor.pt')
        #Mddpg_ca.actor = torch.load('CellFreeCLCA_RL/data/ca_mddpg_actor.pt')

        # Get initial state
        RL_s = env.s_
        for ep in tqdm(range(MAX_EPISODES)):
            if ep <= warmup:
                a_cl = Mddpg_cl.random_action()
                a_ca = Mddpg_ca.random_action()
            else:
                a_cl = Mddpg_cl.action(RL_s)# choose action [ env.U*env.B x 1 ]
                a_ca = Mddpg_ca.action(RL_s)# choose action [ env.B*env.F x 1 ]

            # Convert action value to policy //Clustering Part
            connectionScore = np.reshape(a_cl, (env.U,env.B) ) #[env.U x env.B]
            clustering_policy_UE = []
            for u in range(env.U): 
                #print(connectionScore[u])
                #print(connectionScore[u].argsort())
                #print(connectionScore[u].argsort()[::-1][:env.L])
                bestBS = connectionScore[u].argsort()[::-1][:env.L]
                clustering_policy_UE.append(bestBS)
            
            # Convert action value to policy //Caching Part
            cacheScore = np.reshape(a_ca, (env.B,env.F) )
            caching_policy_BS = []
            for b in range(env.B):
                top_N_idx = np.sort(cacheScore[b].argsort()[-env.N:])# pick up N file with highest score, N is capacity of BSs
                caching_policy_BS.append(top_N_idx)
            # take action to ENV
            EE, HR, RL_s_, done  = env.step(clustering_policy_UE,caching_policy_BS)
            poolEE.append(EE)
            poolHR.append(HR)
            #bestEE, bestHR, RL_s_BF, done = env.step(opt_clustering_policy_UE,opt_caching_policy_BS)
            poolBestEE.append(bestEE)
            #plotMetricBF(poolEE,poolBestEE)
            #plotMetric3(poolEE,poolHR,poolCS)
            #with np.printoptions(precision=3, suppress=True):                                                     
            #    print(ep,'EE=',EE,'\tCS=',CS,'\tHR=',HR)
            
            # RL update
            r_cl = EE
            r_ca = HR
            Mddpg_cl.addMemory([RL_s,a_cl,r_cl,RL_s_])
            Mddpg_ca.addMemory([RL_s,a_ca,r_ca,RL_s_])
            RL_s = RL_s_
            
            if len(Mddpg_cl.memory) > Mddpg_cl.BATCH_SIZE:
                lossActorCL, lossCriticCL = Mddpg_cl.train()
                poolLossActorCL.append(lossActorCL)
                poolLossCriticCL.append(lossCriticCL)
                lossActorCA, lossCriticCA = Mddpg_ca.train()
        
        maxEE = max(poolEE)
        poolMaxEE.append(maxEE)
        #maxEE = bestEE
        if maxEE > bestEE/3: 
            break
        else:
            print(nGame, 'th game failed with maxEE: ',maxEE)
            nGame += 1

        
    # save actor parameter
    
    path = "data/"
    filenameDDPG_CL = path + "DDPG_CL" + str(today) + '.pt'
    filenameDDPG_CA = path + "DDPG_CA" + str(today) + '.pt'
    torch.save(Mddpg_cl.actor, filenameDDPG_CL)
    torch.save(Mddpg_ca.actor, filenameDDPG_CA)
    Mddpg_cl.actor = torch.load(filenameDDPG_CL)
    Mddpg_ca.actor = torch.load(filenameDDPG_CA)
    # plot Brute Force V.S. RL------------------------------------------------------------------
    plt.cla()
    plt.plot(range(len(poolEE)),poolEE,'b-',label='EE of 2 Actors: DDPG_Cluster + DDPG_Cache')
    plt.plot(range(len(poolBestEE)),poolBestEE,'k-',label='EE of Brute Force')
    plt.plot(range(len(poolLossActorCL)),poolLossActorCL,'r-',label='Loss of actorCL')
    plt.plot(range(len(poolLossCriticCL)),poolLossCriticCL,'c-',label='Loss of criticCL')
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
    filename = 'data/BF_vs_RL'+str(MAX_EPISODES)+'_'+str(today)
    fig.savefig(filename + '.eps', format='eps',dpi=1200)
    fig.savefig(filename + '.png', format='png',dpi=1200)
    fig.show()
    # plot Hit Rate------------------------------------------------------------------
    # plt.cla()
    # plt.plot(range(len(poolEE)),poolEE,'bo-',label='EE of 2 Actors: DDPG_Cluster + DDPG_Cache')

if __name__ == '__main__':
    # new ENV
    env = BS(nBS=4,nUE=4,nMaxLink=2,nFile=5,nMaxCache=2,loadENV = True)
    # Load Optimal clustering and caching Policy
    filenameBF = 'data/Result_BruteForce_'+str(env.B)+'AP_'+str(env.U)+'UE_'+str(today)
    filenameBF = 'data/Result_BruteForce_4AP_4UE_2020-10-12'
    with open(filenameBF+'.pkl','rb') as f: 
        bs_coordinate, u_coordinate , g, userPreference, Req, bestEE, opt_clustering_policy_UE, opt_caching_policy_BS = pickle.load(f)
    
    train_2act(env, bestEE)