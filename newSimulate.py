#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Proprietary Design
#import robot_ghliu
from newDDPG import actor
from newDDPG import DDPG
from newENV import BS
from newENV import plot_UE_BS_distribution_Cache
# Public Lib
from torch.autograd import Variable
import torch
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter('runs/fashion_mnist_experiment_1')
import numpy as np
import time,copy,os,csv,random,pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 90000
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
#def drewNetwork(model,input):
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

def plotEEFamily(poolEE,poolThroughput,poolPsys,plotName):
    plt.cla()
    plt.plot(poolEE,label='EE')
    plt.plot(poolThroughput,label='Throughput',alpha=0.7)
    plt.plot(poolPsys,label='Psys',alpha=0.7)
    '''
    plt.plot(poolEE,'b-',label='poolEE')
    plt.plot(poolThroughput,'r-',label='poolThroughput',alpha=0.7)
    plt.plot(poolPsys,'g-',label='poolPsys',alpha=0.7)
    '''
    titleNmae = 'Training History: Energy Efficiency(8a)\n'
    plt.title(titleNmae) # title
    plt.ylabel("Bits/J") # y label
    plt.xlabel("Iteration") # x label
    plt.grid()
    plt.legend()
    fig = plt.gcf()
    fig.savefig('data/'+ 'EE_Family_' + plotName +'.png', format='png',dpi=200)

def plotPsysFamily(poolPsys,poolmissCounterAP,poolmissCounterCPU,plotName):
    plt.cla()
    plt.plot(poolPsys,'b-',label='poolPsys')
    plt.plot(poolmissCounterAP,'r-',label='poolmissCounterAP',alpha=0.7)
    plt.plot(poolmissCounterCPU,'g-',label='poolmissCounterCPU',alpha=0.7)
    plt.grid()
    plt.legend()
    fig = plt.gcf()
    fig.savefig('data/'+ 'Psys_Family_'+ plotName +'.png', format='png',dpi=200)

def plotlist(listA,listName): # for debugging.
    plt.cla()
    plt.plot(listA)
    plt.grid()
    fig = plt.gcf()
    fig.savefig('data/'+ listName +'.png', format='png',dpi=200)

def plotTrainingHistory(filename,isPlotLoss=False,isPlotEE=False,isPlotHR=False,isEPS=False):
    with open(filename+'.pkl','rb') as f: 
        env, poolEE,poolThroughput,poolPsys,poolHR,poolLossActor,poolLossCritic = pickle.load(f)
    #---------------------------------------------------------------------------------------------
    if isPlotLoss:
        # plot RL: poolLossCritic/poolLossActor
        plt.cla()
        plt.plot(range(len(poolLossCritic)),poolLossCritic,'c-',label='Loss of critic')
        plt.plot(range(len(poolLossActor)),poolLossActor,'r-',label='Loss of actor')
        
        titleNmae = 'Training History: Critic and Actor Loss\n'+filename
        plt.title(titleNmae) # title
        plt.ylabel("Q") # y label
        plt.xlabel("Iteration") # x label
        plt.grid()
        plt.legend()
        fig = plt.gcf()
        fig.savefig(filename + '_Loss.png', format='png',dpi=600)
        if isEPS:
            fig.savefig(filename + '_Loss.eps', format='eps',dpi=600)
    #---------------------------------------------------------------------------------------------
    if isPlotEE: #poolEE/poolThroughput/poolPsys/env
        # plot RL: poolEE
        plt.cla()
        nXpt=len(poolEE)
        '''
        plt.plot(range(nXpt),poolEE,'b-',label='EE')
        plt.plot(range(nXpt),poolThroughput,'r-',label='Throughput')
        plt.plot(range(nXpt),poolPsys,'g-',label='Psys')
        '''
        plt.plot(range(nXpt),poolEE,label='EE')
        plt.plot(range(nXpt),poolThroughput,label='Throughput')
        plt.plot(range(nXpt),poolPsys,label='Psys')
        #plt.plot(range(nXpt),poolEE,'b-',label='EE of 2 Actors: DDPG_Cluster + DDPG_Cache')
        finalValue = "{:.2f}".format(max(poolEE))
        plt.annotate(finalValue, (nXpt,poolEE[-1]),textcoords="offset points",xytext=(0,-20),ha='center',color='b')
        #---------------------------------------------------------------------------------------------
        if env.B==4 and env.U ==4 and env.F==5 and env.N==2:
            # Load Optimal clustering and caching Policy
            filenameBF = 'data/4.4.5.2/BF_4AP_4UE_5File_2Cache_2020-11-24'
            with open(filenameBF+'.pkl','rb') as f: 
                env, bestEE, opt_clustering_policy_UE, opt_caching_policy_BS = pickle.load(f)

            plt.plot(range(nXpt),bestEE*np.ones(nXpt),'k-',label='Brute Force')
            finalValue = "{:.2f}".format(bestEE)
            plt.annotate(finalValue, (nXpt,bestEE),textcoords="offset points",xytext=(0,10),ha='center',color='k')
        #---------------------------------------------------------------------------------------------
        titleNmae = 'Training History: Energy Efficiency(8a)\n'+filename
        plt.title(titleNmae) # title
        plt.ylabel("Bits/J") # y label
        plt.xlabel("Iteration") # x label
        plt.grid()
        plt.legend()
        fig = plt.gcf()
        fig.savefig(filename + '_EE.png', format='png',dpi=120)
        if isEPS:
            fig.savefig(filename + '_EE.eps', format='eps',dpi=600)
    #---------------------------------------------------------------------------------------------
    if isPlotHR:
        # plot RL: poolHR
        plt.cla()
        plt.plot(range(len(poolHR)),poolHR,'y-',label='1Act')
        titleNmae = 'Training History: Hit Rate(5) \n'+filename
        plt.title(titleNmae) # title
        plt.ylabel("Ratio") # y label
        plt.xlabel("Iteration") # x label
        plt.grid()
        plt.legend()
        fig = plt.gcf()
        fig.savefig(filename + '_HR.png', format='png',dpi=600)
        if isEPS:
            fig.savefig(filename + '_HR.eps', format='eps',dpi=600)
        #fig.show()

def trainModel(env,actMode,changeReq,changeChannel,loadActor):
    TopologyName = str(env.B)+'AP_'+str(env.U)+'UE_' + str(env.F) + 'File_'+ str(env.N) +'Cache_'
    # new ACT 
    if actMode == '2act':
        ddpg_cl = DDPG(obs_dim = env.dimObs, act_dim = env.dimActCL)
        ddpg_ca = DDPG(obs_dim = env.dimObs, act_dim = env.dimActCA)
        if(loadActor):
            modelPath = 'D:\\/Model/' + TopologyName+'/' + actMode + '_CL_Actor'+'.pt'
            ddpg_cl.actor = torch.load(modelPath)
            ddpg_cl.actor_target.load_state_dict(ddpg_s.actor.state_dict())

            modelPath = 'D:\\/Model/' + TopologyName+'/' + actMode + '_CL_Critic'+'.pt'
            ddpg_cl.critic = torch.load(modelPath)
            ddpg_cl.critic_target.load_state_dict(ddpg_s.critic.state_dict())

            modelPath = 'D:\\/Model/' + TopologyName+'/' + actMode + '_CA_Actor'+'.pt'
            ddpg_ca.actor = torch.load(modelPath)
            ddpg_ca.actor_target.load_state_dict(ddpg_s.actor.state_dict())

            modelPath = 'D:\\/Model/' + TopologyName+'/' + actMode + '_CA_Critic'+'.pt'
            ddpg_ca.critic = torch.load(modelPath)
            ddpg_ca.critic_target.load_state_dict(ddpg_s.critic.state_dict())


    elif actMode == '1act':
        ddpg_s = DDPG(obs_dim = env.dimObs, act_dim = env.dimAct,memMaxSize=20000)###
        if(loadActor):
            modelPath = 'D:\\/Model/' + TopologyName+'/' + actMode + '_Actor'+'.pt'
            ddpg_s.actor = torch.load(modelPath)
            ddpg_s.actor_target.load_state_dict(ddpg_s.actor.state_dict())

            modelPath = 'D:\\/Model/' + TopologyName+'/' + actMode + '_Critic'+'.pt'
            ddpg_s.critic = torch.load(modelPath)
            ddpg_s.critic_target.load_state_dict(ddpg_s.critic.state_dict())
        '''
        print('Actor Network')
        print(ddpg_s.actor)
        print('Critic Network')
        print(ddpg_s.critic)
        '''
    #---------------------------------------------------------------------------------------------
    mu = 0
    noiseSigma = 1 # control exploration
    iteraion = 0
    countChangeReq = 0
    countChangeChannel = 0

    poolEE=[]
    poolThroughput = []
    poolPsys = []
    poolmissCounterAP = []
    poolmissCounterCPU = []

    poolSP_EE=[]
    poolSP_Throughput=[]
    poolSP_Psys = []
    poolSP_missCounterAP = []
    poolSP_missCounterCPU = []

    poolHR=[]
    poolLossActor = []
    poolLossCritic = []
    poolVarLossCritic = []
    poolVarEE = []

    for ep in tqdm(range(MAX_EPISODES),bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
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
            poolThroughput.append(sum(env.Throughput))
            poolPsys.append(env.P_sys/1000)
            poolmissCounterAP.append(env.missCounterAP)
            poolmissCounterCPU.append(env.missCounterCPU)
            
            HR = info["HR"]
            poolHR.append(HR)

            # RL Add Memory
            if actMode == '2act':
                ddpg_cl.addMemory([obs,a_cl,reward,obs2])
                ddpg_ca.addMemory([obs,a_ca,-env.P_sys,obs2])# remind the reward of CA agent
            elif actMode == '1act':
                ddpg_s.addMemory([obs,action,reward,obs2])

            # RL Upadate
            if actMode == '2act':
                if len(ddpg_cl.memory) > ddpg_cl.batch_size:
                    lossActor, lossCritic = ddpg_cl.train()
                    poolLossActor.append(lossActor.item())
                    poolLossCritic.append(lossCritic.item())
                    lossActor, lossCritic = ddpg_ca.train()
                    
            elif actMode == '1act':
                if len(ddpg_s.memory) > ddpg_s.batch_size:
                    lossActor, lossCritic = ddpg_s.train()
                    poolLossActor.append(lossActor.item())
                    poolLossCritic.append(lossCritic.item())
            obs = obs2
            ep_reward += reward

            # iteration update
            iteraion +=1
            if (iteraion % 1000) == 0:
                #print(list(poolLossCritic[-100:]))
                poolVarLossCritic.append( np.var(poolLossCritic[-1000:]) )
                poolVarEE.append( np.var(poolEE[-1000:]) )

                if (poolVarLossCritic[-1] < 10) and (poolVarEE[-1] < 10) and changeReq:
                    env.resetReq()
                    print('**Change Request: ',env.Req)
                    countChangeReq+=1
                    noiseSigma = 1 # reset explore

            if iteraion % 30 == 0:
                noiseSigma*=0.995

            if iteraion % 300 == 0 and changeChannel:    
                env.timeVariantChannel()   
                countChangeChannel+=1

            if (iteraion % 1000) == 0: # Mectric Snapshot
                Best_snrCL_popCA_EE, snrCL_policy_UE, popCA_policy_BS = env.getBestEE_snrCL_popCA(cacheMode='pref',isSave=False,isPlot=False,isEPS=False)
                if poolEE[-1]>Best_snrCL_popCA_EE:
                    print('poolEE win!',poolEE[-1], 'Best_snrCL_popCA_EE loss QQ', Best_snrCL_popCA_EE)
                '''
                else:
                    if (iteraion % 50000) == 0:
                        noiseSigma = 1 # reset explore
                '''
                '''
                poolSP_EE = poolSP_EE + np.ones(1000)*Best_snrCL_popCA_EE
                poolSP_Throughput = poolSP_Throughput + np.ones(1000)*sum(env.Throughput)
                poolSP_Psys = poolSP_Psys + np.ones(1000)*env.P_sys/1000
                poolSP_missCounterAP = poolSP_missCounterAP + np.ones(1000)*env.missCounterAP
                poolSP_missCounterCPU = poolSP_missCounterCPU + np.ones(1000)*env.missCounterCPU
                '''
                poolSP_EE.extend(np.ones(1000)*Best_snrCL_popCA_EE)
                poolSP_Throughput.extend(np.ones(1000)*sum(env.Throughput))
                poolSP_Psys.extend(np.ones(1000)*env.P_sys/1000)
                poolSP_missCounterAP.extend(np.ones(1000)*env.missCounterAP)
                poolSP_missCounterCPU.extend(np.ones(1000)*env.missCounterCPU)
                
                plotEEFamily(poolEE,poolThroughput,poolPsys,'RL')
                plotEEFamily(poolSP_EE,poolSP_Throughput,poolSP_Psys,'SP')
                plotPsysFamily(poolPsys,poolmissCounterAP,poolmissCounterCPU,'RL')
                plotPsysFamily(poolSP_Psys,poolSP_missCounterAP,poolSP_missCounterCPU,'SP')

                plotlist(poolLossCritic,'LossCritic')
                plotlist(poolLossActor,'LossActor')
                plotlist(poolHR,'poolHR')
                
        #if ep_reward>28500:
        #    print('\nEpisode:{} Reward:{} Explore:{}'.format(ep,ep_reward,noiseSigma))
        
    #---------------------------------------------------------------------------------------------  
    #return ddpg_s.actor,env,poolEE,poolLossActor,poolLossCritic
    #---------------------------------------------------------------------------------------------   
    TopologyName = str(env.B)+'AP_'+str(env.U)+'UE_' + str(env.F) + 'File_'+ str(env.N) +'Cache_'
    # Save Model MDDPG
    if actMode == '2act':
        modelPath = 'D:\\/Model/' + TopologyName+'/' + actMode + '_CL_Actor'+'.pt'
        torch.save(ddpg_cl.actor, modelPath)
        modelPath = 'D:\\/Model/' + TopologyName+'/' + actMode + '_CL_Critic'+'.pt'
        torch.save(ddpg_cl.critic, modelPath)
        modelPath = 'D:\\/Model/' + TopologyName+'/' + actMode + '_CA_Actor'+'.pt'
        torch.save(ddpg_ca.actor, modelPath)
        modelPath = 'D:\\/Model/' + TopologyName+'/' + actMode + '_CA_Critic'+'.pt'
        torch.save(ddpg_ca.critic, modelPath)
    # Save Model SDDPG
    elif actMode == '1act': 
        #filenameSDDPG = 'data/1ACT_' + "DDPG_ALL_" + TopologyName + str(today) + '.pt'
        modelPath = 'D:\\/Model/' + TopologyName+'/' + actMode + '_Actor'+'.pt'
        torch.save(ddpg_s.actor, modelPath)
        modelPath = 'D:\\/Model/' + TopologyName+'/' + actMode + '_Critic'+'.pt'
        torch.save(ddpg_s.critic, modelPath)
    
    # Save the plot point
    if actMode == '2act':
        filename = 'data/2ACT_'+ TopologyName +str(MAX_EPISODES*MAX_EP_STEPS)+'_Train_'+str(today)
    elif actMode == '1act':
        filename = 'data/1ACT_'+ TopologyName +str(MAX_EPISODES*MAX_EP_STEPS)+'_Train_'+str(today)

    with open(filename+'.pkl', 'wb') as f:  
        pickle.dump([env, poolEE,poolThroughput,poolPsys,poolHR,poolLossActor,poolLossCritic], f)

def getEE_RL(env,ddpg,isPlot=False,isEPS=False):

    rlBestEE = 0
    rlBestCLPolicy_UE=[]
    rlBestCAPolicy_BS=[]
    obs = env.reset()# Get initial state
    for i in range(1000):
        noise = np.random.normal(0,0,size=env.dimAct)
        action = ddpg.action(obs,noise)
        obs, reward, done, info = env.step(action)
        if reward>rlBestEE:
            rlBestEE = reward
            rlBestCLPolicy_UE, rlBestCAPolicy_BS = env.action2Policy(action)
    #print('rlBestCLPolicy_UE:',rlBestCLPolicy_UE)
    #print('rlBestCAPolicy_BS:',rlBestCAPolicy_BS)
    #print('rlBestEE:',rlBestEE)
    if isPlot:
        displayrlBestEE = "{:.2f}".format(rlBestEE)
        filenamePV = 'data/Visualization_1ACT_'+str(env.B)+'AP_'+str(env.U)+'UE_' + str(env.F) + 'File_'+ str(env.N) +'Cache_' +str(today)+'_'+displayrlBestEE
        plot_UE_BS_distribution_Cache(env.bs_coordinate,env.u_coordinate,env.Req,rlBestCLPolicy_UE,rlBestCAPolicy_BS,displayrlBestEE,filenamePV,isEPS)
    return rlBestEE,rlBestCLPolicy_UE,rlBestCAPolicy_BS

if __name__ == '__main__':
    # new ENV
    #env = BS(nBS=100,nUE=10,nMaxLink=2,nFile=5,nMaxCache=2,loadENV = True)
    env = BS(nBS=40,nUE=10,nMaxLink=2,nFile=50,nMaxCache=5,loadENV = True)
    #env = BS(nBS=10,nUE=4,nMaxLink=2,nFile=5,nMaxCache=2,loadENV = True)
    actMode = '2act'
    TopologyName = str(env.B)+'AP_'+str(env.U)+'UE_' + str(env.F) + 'File_'+ str(env.N) +'Cache_'
    # Training Phase
    rlBestEE = 0
    Best_snrCL_popCA_EE, snrCL_policy_UE, popCA_policy_BS = env.getBestEE_snrCL_popCA(cacheMode='pref',isSave=False,isPlot=False,isEPS=False)
    while(rlBestEE<Best_snrCL_popCA_EE):
        trainModel(env,actMode=actMode,changeReq=False, changeChannel=False, loadActor = False)
        filename = 'data/1ACT_'+ TopologyName +str(MAX_EPISODES*MAX_EP_STEPS)+'_Train_'+str(today)
        plotTrainingHistory(filename,isPlotLoss=True,isPlotEE=True,isPlotHR=True,isEPS=False)

        filenameSDDPG = 'data/1ACT_' + "DDPG_ALL_" + TopologyName + str(today) + '.pt'
        # evaluate performance
        ddpg_s = DDPG(obs_dim = env.dimObs, act_dim = env.dimAct)###
        modelPath = 'D:\\/Model/' + TopologyName+'/' + actMode + '_Actor'+'.pt'
        ddpg_s.actor = torch.load(modelPath)
        ddpg_s.actor_target.load_state_dict(ddpg_s.actor.state_dict())
        modelPath = 'D:\\/Model/' + TopologyName+'/' + actMode + '_Critic'+'.pt'
        ddpg_s.critic = torch.load(modelPath)
        ddpg_s.critic_target.load_state_dict(ddpg_s.critic.state_dict())

        rlBestEE, rlBestCLPolicy_UE, rlBestCAPolicy_BS = getEE_RL(env,ddpg_s,isPlot=False,isEPS=False)

    #---------------------------------------------------------------------------------------------
    # Show Training Phase 
    filename = 'data/1ACT_'+ TopologyName +str(MAX_EPISODES*MAX_EP_STEPS)+'_Train_'+str(today)
    #filename = 'data/1ACT_40AP_10UE_50File_5Cache_100000_Train_2020-12-25'
    plotTrainingHistory(filename,isPlotLoss=True,isPlotEE=True,isPlotHR=True,isEPS=False)
    #==============================================================================================
    # Show Testing Phase (RL vs Benchmark1)
    ddpg_s = DDPG(obs_dim = env.dimObs, act_dim = env.dimAct)###
    # Load Actor
    modelPath = 'D:\\/Model/' + TopologyName+'/' + actMode + '_Actor'+'.pt'
    ddpg_s.actor = torch.load(modelPath)
    ddpg_s.actor_target.load_state_dict(ddpg_s.actor.state_dict())
    # Load Critic
    modelPath = 'D:\\/Model/' + TopologyName+'/' + actMode + '_Critic'+'.pt'
    ddpg_s.critic = torch.load(modelPath)
    ddpg_s.critic_target.load_state_dict(ddpg_s.critic.state_dict())

    poolRL=[]
    poolsnrCL_popCA=[]
    
    for itr in tqdm(range(100)):
        rlBestEE, rlBestCLPolicy_UE, rlBestCAPolicy_BS = getEE_RL(env,ddpg_s,isPlot=False,isEPS=False)
        poolRL.append(rlBestEE)
        Best_snrCL_popCA_EE, snrCL_policy_UE, popCA_policy_BS = env.getBestEE_snrCL_popCA(cacheMode='pref',isSave=False,isPlot=False,isEPS=False)
        poolsnrCL_popCA.append(Best_snrCL_popCA_EE)
        if itr == 50:
            rlBestEE, rlBestCLPolicy_UE, rlBestCAPolicy_BS = getEE_RL(env,ddpg_s,isPlot=True,isEPS=True)
            Best_snrCL_popCA_EE, snrCL_policy_UE, popCA_policy_BS = env.getBestEE_snrCL_popCA(cacheMode='pref',isSave=False,isPlot=True,isEPS=False)

        env.timeVariantChannel()
        #env.resetReq()
    filename = 'data/1ACT_'+ TopologyName +str(MAX_EPISODES*MAX_EP_STEPS)+'_Eval_'+str(today)    
    with open(filename+'.pkl', 'wb') as f:  
        pickle.dump([env, poolRL,poolsnrCL_popCA], f)
    with open(filename+'.pkl','rb') as f: 
        env, poolRL, poolsnrCL_popCA = pickle.load(f)

    # plot RL vs Benchmark1 in Evaluation Phase
    plt.cla()
    plt.plot(range(len(poolRL)),poolRL,'b-',label='1Act')
    plt.plot(range(len(poolsnrCL_popCA)),poolsnrCL_popCA,'g-',label='snrCL_popCA')
    titleNmae = 'Evaluation Phase: Energy Efficiency(8a) \n'+filename
    plt.title(titleNmae) # title
    plt.ylabel("Bits/J") # y label
    plt.xlabel("Iteration(t)") # x label
    plt.grid()
    plt.legend()
    fig = plt.gcf()
    fig.savefig(filename + '.png', format='png',dpi=1200)
    fig.savefig(filename + '.eps', format='eps',dpi=1200)
    #---------------------------------------------------------------------------------------------
    