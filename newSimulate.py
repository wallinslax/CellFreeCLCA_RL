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
          
MAX_EPISODES = 10**2*5
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

def trainModel(env,actMode,changeReq,changeChannel,loadActor):
    # new ACT 
    modelPath = 'D:\\/Model/' + env.TopologyName+'/'
    if actMode == '2act':
        ddpg_cl = DDPG(obs_dim = env.dimObs, act_dim = env.dimActCL,memMaxSize=20000)
        ddpg_ca = DDPG(obs_dim = env.dimObs, act_dim = env.dimActCA,memMaxSize=20000)
        if(loadActor):
           ddpg_cl.loadModel(modelPath = modelPath, modelName= actMode+'_cl') 
           ddpg_ca.loadModel(modelPath = modelPath, modelName= actMode+'_ca') 
    elif actMode == '1act':
        ddpg_s = DDPG(obs_dim = env.dimObs, act_dim = env.dimAct,memMaxSize=20000)###
        if(loadActor):
            ddpg_s.loadModel(modelPath = modelPath, modelName= actMode)
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
    # RL
    poolEE_RL=[]
    poolTP_RL = []
    poolPsys_RL = []
    poolHR_RL=[]
    poolmissCounterAP_RL = [] # MCAP = miss file count at APs
    poolmissCounterCPU_RL = [] # MCCPU = miss file count at CPU
    poolLossActor = []
    poolLossCritic = []
    poolVarLossCritic = []
    poolVarEE = []

    # BM1 snrCL_popCA
    poolEE_BM1=[]
    poolTP_BM1=[]
    poolPsys_BM1 = []
    poolHR_BM1 = []
    poolmissCounterAP_BM1 = []
    poolmissCounterCPU_BM1 = []

    # BM2
    poolEE_BM2=[]
    poolTP_BM2=[]
    poolPsys_BM2 = []
    poolHR_BM2 = []
    poolmissCounterAP_BM2 = []
    poolmissCounterCPU_BM2 = []

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
            poolEE_RL.append(EE)
            poolTP_RL.append(sum(env.Throughput))
            poolPsys_RL.append(env.P_sys/1000)
            poolHR_RL.append(env.HR)
            poolmissCounterAP_RL.append(env.missCounterAP)
            poolmissCounterCPU_RL.append(env.missCounterCPU)
            # RL Add Memory
            if actMode == '2act':
                ddpg_cl.addMemory([obs,a_cl,reward,obs2])
                ddpg_ca.addMemory([obs,a_ca,-env.P_sys,obs2])# remind the reward of CA agent
            elif actMode == '1act':
                ddpg_s.addMemory([obs,action,reward,obs2])
            #---------------------------------------------------------------------------------------------  
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

            if iteraion % 30 == 0:
                noiseSigma*=0.995

            if (iteraion % 1000) == 0:
                #print(list(poolLossCritic[-100:]))
                poolVarLossCritic.append( np.var(poolLossCritic[-1000:]) )
                poolVarEE.append( np.var(poolEE_RL[-1000:]) )

                if (poolVarLossCritic[-1] < 10) and (poolVarEE[-1] < 10) and changeReq:
                    env.resetReq()
                    print('**Change Request: ',env.Req)
                    countChangeReq+=1
                    noiseSigma = 1 # reset explore     

            if (iteraion % 1000) == 0: # Mectric Snapshot
                # BM1
                EE_BM1, SNR_CL_Policy_UE, POP_CA_Policy_BS = env.getBestEE_snrCL_popCA(cacheMode='pref')
                TP_BM1 = sum(env.Throughput)
                Psys_BM1 = env.P_sys/1000 # mW->W
                HR_BM1 = env.calHR(SNR_CL_Policy_UE,POP_CA_Policy_BS)
                poolEE_BM1.extend(np.ones(1000)*EE_BM1)
                poolTP_BM1.extend(np.ones(1000)*TP_BM1)
                poolPsys_BM1.extend(np.ones(1000)*Psys_BM1)
                poolHR_BM1.extend(np.ones(1000)*HR_BM1)
                poolmissCounterAP_BM1.extend(np.ones(1000)*env.missCounterAP)
                poolmissCounterCPU_BM1.extend(np.ones(1000)*env.missCounterCPU)
                #BM2
                SNR_CL_Policy_UE = env.getSNR_CL_Policy()
                POP_CA_Policy_BS = env.getPOP_CA_Policy()
                EE_BM2 = env.calEE(SNR_CL_Policy_UE,POP_CA_Policy_BS)
                TP_BM2 = sum(env.Throughput)
                Psys_BM2 = env.P_sys/1000 # mW->W
                HR_BM2 = env.calHR(SNR_CL_Policy_UE,POP_CA_Policy_BS)
                poolEE_BM2.extend(np.ones(1000)*EE_BM2)
                poolTP_BM2.extend(np.ones(1000)*TP_BM2)
                poolPsys_BM2.extend(np.ones(1000)*Psys_BM2)
                poolHR_BM2.extend(np.ones(1000)*HR_BM2)
                poolmissCounterAP_BM2.extend(np.ones(1000)*env.missCounterAP)
                poolmissCounterCPU_BM2.extend(np.ones(1000)*env.missCounterCPU)

                # Preview
                plotEEFamily(poolEE_RL,poolTP_RL,poolPsys_RL,'RL')
                plotEEFamily(poolEE_BM1,poolTP_BM1,poolPsys_BM1,'BM1')
                plotEEFamily(poolEE_BM2,poolTP_BM2,poolPsys_BM2,'BM2')
                plotPsysFamily(poolPsys_RL,poolmissCounterAP_RL,poolmissCounterCPU_RL,'RL')
                plotPsysFamily(poolPsys_BM1,poolmissCounterAP_BM1,poolmissCounterCPU_BM1,'BM1')
                plotPsysFamily(poolPsys_BM2,poolmissCounterAP_BM2,poolmissCounterCPU_BM2,'BM2')

                plotlist(poolLossCritic,'LossCritic')
                plotlist(poolLossActor,'LossActor')

                if poolEE_RL[-1]>EE_BM1:
                    print('poolEE_RL win!',poolEE_RL[-1], 'EE_BM1 loss QQ', EE_BM1)
                '''
                else:
                    if (iteraion % 50000) == 0:
                        noiseSigma = 1 # reset explore
                '''
                if changeChannel:
                    env.timeVariantChannel()   
                    countChangeChannel+=1
                
        #if ep_reward>28500:
        #    print('\nEpisode:{} Reward:{} Explore:{}'.format(ep,ep_reward,noiseSigma))
    #---------------------------------------------------------------------------------------------   
    # Save Model
    if actMode == '2act':
        ddpg_cl.saveModel(modelPath = modelPath,modelName=actMode+'_cl')
        ddpg_ca.saveModel(modelPath = modelPath,modelName=actMode+'_ca')
    elif actMode == '1act':
        ddpg_s.saveModel(modelPath = modelPath,modelName=actMode)
    
    # Save Line
    filename = 'data/'+env.TopologyCode+'/TrainingPhase/'+ env.TopologyName +str(MAX_EPISODES*MAX_EP_STEPS)+'_Train_'
    with open(filename+ actMode +'RL.pkl', 'wb') as f:  
        pickle.dump([env, poolEE_RL,poolTP_RL,poolPsys_RL,poolHR_RL,poolmissCounterAP_RL,poolmissCounterCPU_RL,poolLossActor,poolLossCritic], f)
    with open(filename+'BM1.pkl', 'wb') as f:  
        pickle.dump([env, poolEE_BM1,poolTP_BM1,poolPsys_BM1,poolHR_BM1,poolmissCounterAP_BM1,poolmissCounterCPU_BM1], f)
    with open(filename+'BM2.pkl', 'wb') as f:  
        pickle.dump([env, poolEE_BM2,poolTP_BM2,poolPsys_BM2,poolHR_BM2,poolmissCounterAP_BM2,poolmissCounterCPU_BM2], f)

def plotTrainingHistory(filename,isPlotLoss=False,isPlotEE=False,isPlotTP=False,isPlotPsys=False,isPlotHR=False,isEPS=False):
    with open(filename+ actMode +'RL.pkl','rb') as f: 
        env, poolEE_RL,poolTP_RL,poolPsys_RL,poolHR_RL,poolmissCounterAP_RL,poolmissCounterCPU_RL,poolLossActor,poolLossCritic = pickle.load(f)
    with open(filename+'BM1.pkl','rb') as f: 
        env, poolEE_BM1,poolTP_BM1,poolPsys_BM1,poolHR_BM1,poolmissCounterAP_BM1,poolmissCounterCPU_BM1 = pickle.load(f)
    with open(filename+'BM2.pkl','rb') as f: 
        env, poolEE_BM2,poolTP_BM2,poolPsys_BM2,poolHR_BM2,poolmissCounterAP_BM2,poolmissCounterCPU_BM2 = pickle.load(f)
    #---------------------------------------------------------------------------------------------
    # Load Brute Force Policy
    if env.B==4 and env.U ==4 and env.F==5 and env.N==2:
        # Load Optimal clustering and caching Policy
        filenameBF = 'data/4.4.5.2/BF/BF_4AP_4UE_5File_2Cache_2021-01-06'
        with open(filenameBF+'.pkl','rb') as f: 
            envBF, EE_BF, opt_clustering_policy_UE, opt_caching_policy_BS = pickle.load(f)
        envBF.calEE(opt_clustering_policy_UE, opt_caching_policy_BS)
        EE_BF = envBF.EE
        Throughput_BF= sum(envBF.Throughput)
        Psys_BF = envBF.P_sys/1000 # mW->W
        HR_BF = envBF.calHR(opt_clustering_policy_UE, opt_caching_policy_BS)
    #---------------------------------------------------------------------------------------------
    if isPlotLoss:
        # plot RL: poolLossCritic/poolLossActor
        plt.cla()
        plt.plot(range(len(poolLossCritic)),poolLossCritic,'c-',label='Loss of critic')
        plt.plot(range(len(poolLossActor)),poolLossActor,'r-',label='Loss of actor')
        
        plt.title('Training History: Critic and Actor Loss\n' + env.TopologyName) # title
        plt.ylabel("Q") # y label
        plt.xlabel("Iteration") # x label
        plt.grid()
        plt.legend()
        fig = plt.gcf()
        fig.savefig(filename + '_Loss.png', format='png',dpi=120)
        if isEPS:
            fig.savefig(filename + '_Loss.eps', format='eps',dpi=120)
    #---------------------------------------------------------------------------------------------
    if isPlotEE:
        plt.cla()
        nXpt=len(poolEE_RL)
        # plot DDPG
        plt.plot(range(nXpt),poolEE_RL,'b-',label='DDPG 1act')
        finalValue = "{:.2f}".format(poolEE_RL[-1])
        plt.annotate(finalValue, (nXpt,poolEE_RL[-1]),textcoords="offset points",xytext=(0,-10),ha='center',color='b')
        # plot BM1
        plt.plot(range(nXpt),poolEE_BM1,'g-',label='BM1')
        finalValue = "{:.2f}".format(poolEE_BM1[-1])
        plt.annotate(finalValue, (nXpt,poolEE_BM1[-1]),textcoords="offset points",xytext=(0,-20),ha='center',color='g')
        # plot BM2
        plt.plot(range(nXpt),poolEE_BM2,'y-',label='BM2')
        finalValue = "{:.2f}".format(poolEE_BM2[-1])
        plt.annotate(finalValue, (nXpt,poolEE_BM2[-1]),textcoords="offset points",xytext=(0,-30),ha='center',color='y')
        #---------------------------------------------------------------------------------------------
        # plot Brute Force
        if env.B==4 and env.U ==4 and env.F==5 and env.N==2:
            plt.plot(range(nXpt),EE_BF*np.ones(nXpt),'k-',label='Brute Force')
            finalValue = "{:.2f}".format(EE_BF)
            plt.annotate(finalValue, (nXpt,EE_BF),textcoords="offset points",xytext=(0,10),ha='center',color='k')
        #---------------------------------------------------------------------------------------------
        plt.title('Training Phase: Energy Efficiency (EE)\n'+env.TopologyName) # title
        plt.ylabel("Bits/J") # y label
        plt.xlabel("Iteration") # x label
        plt.grid()
        plt.legend()
        fig = plt.gcf()
        fig.savefig(filename + '_EE.png', format='png',dpi=120)
        if isEPS:
            fig.savefig(filename + '_EE.eps', format='eps',dpi=120)
    #---------------------------------------------------------------------------------------------
    if isPlotTP:
        plt.cla()
        nXpt=len(poolTP_RL)
        # plot DDPG
        plt.plot(range(nXpt),poolTP_RL,'b-',label='DDPG 1act')
        finalValue = "{:.2f}".format(poolTP_RL[-1])
        plt.annotate(finalValue, (nXpt,poolTP_RL[-1]),textcoords="offset points",xytext=(0,-10),ha='center',color='b')
        # plot BM1
        plt.plot(range(nXpt),poolTP_BM1,'g-',label='BM1')
        finalValue = "{:.2f}".format(poolTP_BM1[-1])
        plt.annotate(finalValue, (nXpt,poolTP_BM1[-1]),textcoords="offset points",xytext=(0,-20),ha='center',color='g')
        # plot BM2
        plt.plot(range(nXpt),poolTP_BM2,'y-',label='BM2')
        finalValue = "{:.2f}".format(poolTP_BM2[-1])
        plt.annotate(finalValue, (nXpt,poolTP_BM2[-1]),textcoords="offset points",xytext=(0,-30),ha='center',color='y')
        #---------------------------------------------------------------------------------------------
        # plot Brute Force
        if env.B==4 and env.U ==4 and env.F==5 and env.N==2:
            plt.plot(range(nXpt),Throughput_BF*np.ones(nXpt),'k-',label='Brute Force')
            finalValue = "{:.2f}".format(Throughput_BF)
            plt.annotate(finalValue, (nXpt,Throughput_BF),textcoords="offset points",xytext=(0,10),ha='center',color='k')
        #---------------------------------------------------------------------------------------------
        plt.title('Training Phase: Throughput\n'+env.TopologyName) # title
        plt.ylabel("Bits/s") # y label
        plt.xlabel("Iteration") # x label
        plt.grid()
        plt.legend()
        fig = plt.gcf()
        fig.savefig(filename + '_Throughput.png', format='png',dpi=120)
        if isEPS:
            fig.savefig(filename + '_Throughput.eps', format='eps',dpi=120)
    #---------------------------------------------------------------------------------------------
    if isPlotPsys:
        plt.cla()
        nXpt=len(poolPsys_RL)
        # plot DDPG
        plt.plot(range(nXpt),poolPsys_RL,'b-',label='DDPG 1act')
        finalValue = "{:.2f}".format(poolPsys_RL[-1]) 
        plt.annotate(finalValue, (nXpt,poolPsys_RL[-1]),textcoords="offset points",xytext=(0,-10),ha='center',color='b')
        # plot BM1
        plt.plot(range(nXpt),poolPsys_BM1,'g-',label='BM1')
        finalValue = "{:.2f}".format(poolPsys_BM1[-1])
        plt.annotate(finalValue, (nXpt,poolPsys_BM1[-1]),textcoords="offset points",xytext=(0,-20),ha='center',color='g')
        # plot BM2
        plt.plot(range(nXpt),poolPsys_BM2,'y-',label='BM2')
        finalValue = "{:.2f}".format(poolPsys_BM2[-1])
        plt.annotate(finalValue, (nXpt,poolPsys_BM2[-1]),textcoords="offset points",xytext=(0,-30),ha='center',color='y')
        #---------------------------------------------------------------------------------------------
        # plot Brute Force
        if env.B==4 and env.U ==4 and env.F==5 and env.N==2:
            plt.plot(range(nXpt),Psys_BF*np.ones(nXpt),'k-',label='Brute Force')
            finalValue = "{:.2f}".format(Psys_BF)
            plt.annotate(finalValue, (nXpt,Psys_BF),textcoords="offset points",xytext=(0,10),ha='center',color='k')
        #---------------------------------------------------------------------------------------------
        plt.title('Training Phase: System Power Consumption\n'+env.TopologyName) # title
        plt.ylabel("W") # y label
        plt.xlabel("Iteration") # x label
        plt.grid()
        plt.legend()
        fig = plt.gcf()
        fig.savefig(filename + '_Psys.png', format='png',dpi=120)
        if isEPS:
            fig.savefig(filename + '_Psys.eps', format='eps',dpi=120)
    #---------------------------------------------------------------------------------------------
    if isPlotHR:
        plt.cla()
        nXpt=len(poolHR_RL)
        # plot DDPG
        plt.plot(range(nXpt),poolHR_RL,'b-',label='DDPG 1act')
        # plot BM1
        plt.plot(range(nXpt),poolHR_BM1,'g-',label='BM1')
        # plot BM2
        plt.plot(range(nXpt),poolHR_BM2,'y-',label='BM2')   
        #---------------------------------------------------------------------------------------------
        # plot Brute Force
        if env.B==4 and env.U ==4 and env.F==5 and env.N==2:
            plt.plot(range(nXpt),HR_BF*np.ones(nXpt),'k-',label='Brute Force')
        #---------------------------------------------------------------------------------------------
        plt.title('Training Phase: Hit Rate (HR) \n'+env.TopologyName) # title
        plt.ylabel("Ratio") # y label
        plt.xlabel("Iteration") # x label
        plt.grid()
        plt.legend()
        fig = plt.gcf()
        fig.savefig(filename + '_HR.png', format='png',dpi=120)
        if isEPS:
            fig.savefig(filename + '_HR.eps', format='eps',dpi=120)
        #fig.show()

def EvaluateModel(env,actMode, nItr=100):
    # new ACT 
    modelPath = 'D:\\/Model/' + env.TopologyName+'/'
    if actMode == '2act':
        ddpg_cl = DDPG(obs_dim = env.dimObs, act_dim = env.dimActCL,memMaxSize=20000)
        ddpg_ca = DDPG(obs_dim = env.dimObs, act_dim = env.dimActCA,memMaxSize=20000)
        # load Model
        ddpg_cl.loadModel(modelPath = modelPath, modelName= actMode+'_cl') 
        ddpg_ca.loadModel(modelPath = modelPath, modelName= actMode+'_ca') 
    elif actMode == '1act':
        ddpg_s = DDPG(obs_dim = env.dimObs, act_dim = env.dimAct,memMaxSize=20000)###
        # load Model
        ddpg_s.loadModel(modelPath = modelPath, modelName= actMode)
    
    poolEE_RL=[]
    poolEE_BM=[]
    poolThroughput_RL=[]
    poolThroughput_BM=[]
    poolPsys_RL=[]
    poolPsys_BM=[]
    poolHR_RL=[]
    poolHR_BM=[]
    poolMissCounterAP_RL = []
    poolMissCounterAP_BM = []
    poolMissCounterCPU_RL = []
    poolMissCounterCPU_BM = []
    for ep in tqdm(range(nItr),bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        # Calcualte Benchmark (BM): EE/Throughput/Psys/HR/missCounterAP/missCounterCPU
        EE_BM, Throughput_BM, Psys_BM, HR_BM, snrCL_policy_UE, popCA_policy_BS = env.getBestEE_snrCL_popCA(cacheMode='pref')
        poolEE_BM.append(EE_BM)
        poolThroughput_BM.append(Throughput_BM)
        poolPsys_BM.append(Psys_BM)
        poolHR_BM.append(HR_BM)
        poolMissCounterAP_BM.append(env.missCounterAP)
        poolMissCounterCPU_BM.append(env.missCounterCPU)
        ## Calcualte RL
        EE_RL, Throughput_RL, Psys_RL, HR_RL, RL_CLPolicy_UE, RL_CAPolicy_BS = getEE_RL(env,actMode = actMode,ddpg_s=ddpg_s,isPlot=False,isEPS=False)
        poolEE_RL.append(EE_RL)
        poolThroughput_RL.append(Throughput_RL)
        poolPsys_RL.append(Psys_RL)
        poolHR_RL.append(HR_RL)
        poolMissCounterAP_RL.append(env.missCounterAP)
        poolMissCounterCPU_RL.append(env.missCounterCPU)
        # Sample CL/CA Policy Visualization
        if ep == nItr/2:
            filenameRL = 'data/'+env.TopologyCode+'/EvaluationPhase/'+actMode+'_'   +env.TopologyName+'_Evaluation'
            plot_UE_BS_distribution_Cache(env.bs_coordinate,env.u_coordinate,env.Req,RL_CLPolicy_UE,RL_CAPolicy_BS,EE_RL,filenameRL,isEPS=False)
            filenameBM = 'data/'+env.TopologyCode+'/EvaluationPhase/'+'snrCL_popCA_'+env.TopologyName + str(env.L) +'L_' +'_Evaluation'
            plot_UE_BS_distribution_Cache(env.bs_coordinate,env.u_coordinate,env.Req,snrCL_policy_UE,popCA_policy_BS,EE_BM,filenameBM,isEPS=False)
        # Change Environment
        env.timeVariantChannel()
        #env.resetReq()
    # Save Line
    filename = 'data/'+env.TopologyCode+'/EvaluationPhase/'+actMode+'_'+ env.TopologyName +'_Evaluation'
    with open(filename+'.pkl', 'wb') as f:  
        pickle.dump([env, poolEE_BM,poolThroughput_BM,poolPsys_BM,poolHR_BM,poolMissCounterAP_BM,poolMissCounterCPU_BM, \
                          poolEE_RL,poolThroughput_RL,poolPsys_RL,poolHR_RL,poolMissCounterAP_RL,poolMissCounterCPU_RL], f)
    with open(filename+'.pkl','rb') as f: 
        env, poolEE_BM,poolThroughput_BM,poolPsys_BM,poolHR_BM,poolMissCounterAP_BM,poolMissCounterCPU_BM, \
             poolEE_RL,poolThroughput_RL,poolPsys_RL,poolHR_RL,poolMissCounterAP_RL,poolMissCounterCPU_RL= pickle.load(f)

def plotEvalutionHistory(filename,isEPS=False):
    # plot EE/Throughput/Psys/HR
    with open(filename+'.pkl','rb') as f: 
        env, poolEE_BM,poolThroughput_BM,poolPsys_BM,poolHR_BM,poolMissCounterAP_BM,poolMissCounterCPU_BM, \
             poolEE_RL,poolThroughput_RL,poolPsys_RL,poolHR_RL,poolMissCounterAP_RL,poolMissCounterCPU_RL= pickle.load(f)
    
    #---------------------------------------------------------------------------------------------
    # EE
    plt.cla()
    plt.plot(poolEE_RL,'b-',label='DDPG 1act')
    plt.plot(poolEE_BM,'g-',label='SNR-based Clustering + Popularity-based Caching')
    plt.title('Evaluation Phase: Energy Efficiency (EE)\n'+env.TopologyName) # title
    plt.ylabel("Bits/J") # y label
    plt.xlabel("Iteration") # x label
    plt.grid()
    plt.legend()
    fig = plt.gcf()
    fig.savefig(filename + '_EE.png', format='png',dpi=120)
    if isEPS:
        fig.savefig(filename + '_EE.eps', format='eps',dpi=120)
    #---------------------------------------------------------------------------------------------
    # Throughput
    plt.cla()
    plt.plot(poolThroughput_RL,'b-',label='DDPG 1act')
    plt.plot(poolThroughput_BM,'g-',label='SNR-based Clustering + Popularity-based Caching')
    plt.title('Evaluation Phase: Throughput\n'+env.TopologyName) # title
    plt.ylabel("Bits/s") # y label
    plt.xlabel("Iteration") # x label
    plt.grid()
    plt.legend()
    fig = plt.gcf()
    fig.savefig(filename + '_Throughput.png', format='png',dpi=120)
    if isEPS:
        fig.savefig(filename + '_Throughput.eps', format='eps',dpi=120)
    #---------------------------------------------------------------------------------------------
    # Psys
    plt.cla()
    plt.plot(poolPsys_RL,'b-',label='DDPG 1act')
    plt.plot(poolPsys_BM,'g-',label='SNR-based Clustering + Popularity-based Caching')
    plt.title('Evaluation Phase: System Power Consumption\n'+env.TopologyName) # title
    plt.ylabel("W") # y label
    plt.xlabel("Iteration") # x label
    plt.grid()
    plt.legend()
    fig = plt.gcf()
    fig.savefig(filename + '_Psys.png', format='png',dpi=120)
    if isEPS:
        fig.savefig(filename + '_Psys.eps', format='eps',dpi=120)
    #---------------------------------------------------------------------------------------------
    # HR
    plt.cla()
    plt.plot(poolHR_RL,'b-',label='DDPG 1act')
    plt.plot(poolHR_BM,'g-',label='SNR-based Clustering + Popularity-based Caching')
    plt.title('Evaluation Phase: Hit Rate (HR) \n'+env.TopologyName) # title
    plt.ylabel("Ratio") # y label
    plt.xlabel("Iteration") # x label
    plt.grid()
    plt.legend()
    fig = plt.gcf()
    fig.savefig(filename + '_HR.png', format='png',dpi=120)
    if isEPS:
        fig.savefig(filename + '_HR.eps', format='eps',dpi=120)

def getEE_RL(env,actMode,ddpg_s=None,ddpg_cl=None,ddpg_ca=None,isPlot=False,isEPS=False):
    EE_RL = 0
    RL_CLPolicy_UE=[]
    RL_CAPolicy_BS=[]
    obs = env.reset()# Get initial state
    for i in range(1000):
        if actMode == '2act':
            noise = np.random.normal(0, 0,size=env.dimActCL)
            a_cl = ddpg_cl.action(obs,noise)# choose action [ env.U*env.B x 1 ]
            #a_ca = opt_caching_state.flatten()
            noise = np.random.normal(0, 0,size=env.dimActCA)
            a_ca = ddpg_ca.action(obs,noise)# choose action [ env.U*env.B x 1 ]
            action = np.concatenate((a_cl, a_ca), axis=0)
        elif actMode == '1act':
            noise = np.random.normal(0, 0,size=env.dimAct)
            action = ddpg_s.action(obs,noise)

        obs, reward, done, info = env.step(action)
        if reward>EE_RL:
            EE_RL = reward
            RL_CLPolicy_UE, RL_CAPolicy_BS = env.action2Policy(action)
    #print('rlBestCLPolicy_UE:',rlBestCLPolicy_UE)
    #print('rlBestCAPolicy_BS:',rlBestCAPolicy_BS)
    #print('rlBestEE:',rlBestEE)
    EE_RL = env.calEE(RL_CLPolicy_UE,RL_CAPolicy_BS)
    return EE_RL, sum(env.Throughput),env.P_sys/1000, env.HR, RL_CLPolicy_UE,RL_CAPolicy_BS

if __name__ == '__main__':
    # new ENV
    env = BS(nBS=4,nUE=4,nMaxLink=2,nFile=5,nMaxCache=2,loadENV = True)
    actMode = '1act'
    # Derive Policy: BM
    EE_BM1, snrCL_policy_UE, popCA_policy_BS = env.getBestEE_snrCL_popCA(cacheMode='pref')
    #==============================================================================================
    # Training Phase
    EE_RLBest = 0
    while(EE_RLBest<EE_BM1):
        #trainModel(env,actMode=actMode,changeReq=False, changeChannel=False, loadActor = False)  
        trainModel(env,actMode=actMode,changeReq=False, changeChannel=True, loadActor = False)  
        filename = 'data/'+env.TopologyCode+'/TrainingPhase/'+ env.TopologyName +str(MAX_EPISODES*MAX_EP_STEPS)+'_Train_'
        plotTrainingHistory(filename,isPlotLoss=True,isPlotEE=True,isPlotTP=True,isPlotPsys=True,isPlotHR=True,isEPS=False)

        # evaluate performance
        if actMode == '2act':
            ddpg_cl = DDPG(obs_dim = env.dimObs, act_dim = env.dimActCL,memMaxSize=20000)
            ddpg_ca = DDPG(obs_dim = env.dimObs, act_dim = env.dimActCA,memMaxSize=20000)
            ddpg_cl.loadModel(modelPath = modelPath, modelName= actMode+'_cl') 
            ddpg_ca.loadModel(modelPath = modelPath, modelName= actMode+'_ca') 
        elif actMode == '1act':
            ddpg_s = DDPG(obs_dim = env.dimObs, act_dim = env.dimAct,memMaxSize=20000)###
            ddpg_s.loadModel(modelPath = modelPath, modelName= actMode)
            
            EE_RLBest, RLCLPolicy_UE, RLCAPolicy_BS = getEE_RL(env,ddpg_s,isPlot=False,isEPS=False)
            plot_UE_BS_distribution_Cache(env.bs_coordinate, env.u_coordinate, env.Req, RLCLPolicy_UE, RLCAPolicy_BS, EE_RLBest,filename,isEPS=False)

    #---------------------------------------------------------------------------------------------
    # Show Training Phase 
    filename = 'data/'+env.TopologyCode+'/TrainingPhase/'+actMode+'_'+ env.TopologyName +str(MAX_EPISODES*MAX_EP_STEPS)+'_Train'
    plotTrainingHistory(filename,isPlotLoss=True,isPlotEE=True,isPlotTP=True,isPlotPsys=True,isPlotHR=True,isEPS=False)
    
    #==============================================================================================
    # Evaluation Phase
    EvaluateModel(env,actMode=actMode, nItr=100)
    # Load Line
    filename = 'data/'+env.TopologyCode+'/EvaluationPhase/'+actMode+'_'+ env.TopologyName +'_Evaluation'
    plotEvalutionHistory(filename,isEPS=False)
    