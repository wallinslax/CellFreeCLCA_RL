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
import matplotlib
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
MAX_EPISODES=1000
MAX_EP_STEPS=100
warmup = -1
epsilon = 0.2
#####################################
# plot style
'''
font = {'family' : 'Verdana',
        'weight' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)
markerSize = 20*4**1
'''
linestyles = ['-.', '-', '--', ':']
#####################################
def plotMetric(poolEE,poolBestEE):
    xScale = 100
    x = range( len(poolEE[-xScale:]) )
    plt.cla()
    plt.plot(x,poolEE[-xScale:],'bo-',label='EE RL')
    plt.plot(x,poolBestEE[-xScale:],'r^-',label='EE BF')
    plt.title("Metric Visualization") # title
    plt.ylabel("Bits/J") # y label
    plt.xlabel("t") # x label
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

def plotlist(listA,listName): # for debugging.
    plt.cla()
    plt.plot(listA)
    plt.grid()
    fig = plt.gcf()
    fig.savefig( listName +'.png', format='png',dpi=200)
#1 EE
def plotEE(env,filename,poolEE_RL1act=None,poolEE_RL2act=None,poolEE_BM1=None,poolEE_BM2=None,poolEE_BM3=None,poolEE_BF=None,isEPS=False):
    if 'Training' in filename:
        phaseName = 'Training Phase'
    elif 'Evaluation' in filename:
        phaseName = 'Evaluation Phase'
    elif 'Preview' in filename:
        phaseName = 'Preview Phase'
    
    plt.cla()
    plt.clf()
    nXpt=len(poolEE_BM1[0])
    #---------------------------------------------------------------------------------------------
    # plot Brute Force
    if poolEE_BF!=None:
        plt.plot(range(nXpt),poolEE_BF,'k:',label='Brute Force')
        finalValue = "{:.2f}".format(poolEE_BF[-1])
        #plt.annotate(finalValue, (nXpt,poolEE_BF[-1]),textcoords="offset points",xytext=(20,10),ha='center',color='k')
    #---------------------------------------------------------------------------------------------
    # plot DDPG 1act
    if poolEE_RL1act != None:
        plt.plot(range(nXpt),poolEE_RL1act,'b-',label='DDPG 1act')
        finalValue = "{:.2f}".format(poolEE_RL1act[-1])
        #plt.annotate(finalValue, (nXpt,poolEE_RL1act[-1]),textcoords="offset points",xytext=(20,-10),ha='center',color='b')
    #---------------------------------------------------------------------------------------------
    # plot DDPG 2act
    if poolEE_RL2act != None:
        plt.plot(range(nXpt),poolEE_RL2act,'r-',label='DDPG 2act')
        finalValue = "{:.2f}".format(poolEE_RL2act[-1])
        #plt.annotate(finalValue, (nXpt,poolEE_RL2act[-1]),textcoords="offset points",xytext=(20,-10),ha='center',color='r')
    #---------------------------------------------------------------------------------------------
    for l in range(1,env.L+1):
        # plot BM1
        plt.plot(range(nXpt),poolEE_BM1[l],color='g',linestyle=linestyles[l],label='BM1'+'_L='+str(l))
        finalValue = "{:.2f}".format(poolEE_BM1[l][-1])
        #plt.annotate(finalValue, (nXpt,poolEE_BM1[-1]),textcoords="offset points",xytext=(20,10),ha='center',color='g')
        # plot BM2
        plt.plot(range(nXpt),poolEE_BM2[l],color='y',linestyle=linestyles[l],label='BM2'+'_L='+str(l))
        finalValue = "{:.2f}".format(poolEE_BM2[l][-1])
        #plt.annotate(finalValue, (nXpt,poolEE_BM2[-1]),textcoords="offset points",xytext=(20,10),ha='center',color='y')
        # plot BM3
        plt.plot(range(nXpt),poolEE_BM3[l],color='c',linestyle=linestyles[l],label='BM3'+'_L='+str(l))
        finalValue = "{:.2f}".format(poolEE_BM2[l][-1])
        #plt.annotate(finalValue, (nXpt,poolEE_BM2[-1]),textcoords="offset points",xytext=(20,10),ha='center',color='y')

    #---------------------------------------------------------------------------------------------
    #plt.title(phaseName+': Energy Efficiency (EE)\n Topology:'+env.TopologyCode) # title
    plt.ylabel("Bits/J") # y label
    plt.xlabel("t") # x label
    plt.grid()
    plt.legend()
    plt.xlim(0,nXpt-1)
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(filename + '_EE.png', format='png',dpi=120)
    if isEPS:
        fig.savefig(filename + '_EE.eps', format='eps',dpi=120)
#2 TP
def plotTP(env,filename,poolTP_RL1act=None,poolTP_RL2act=None,poolTP_BM1=None,poolTP_BM2=None,poolTP_BM3=None,poolTP_BF=None,isEPS=False):
    if 'Training' in filename:
        phaseName = 'Training Phase'
    elif 'Evaluation' in filename:
        phaseName = 'Evaluation Phase'
    elif 'Preview' in filename:
        phaseName = 'Preview Phase'
    plt.cla()
    nXpt=len(poolTP_BM1[0])
    #---------------------------------------------------------------------------------------------
    if poolTP_BF != None:
        plt.plot(range(nXpt),poolTP_BF,'k-',label='Brute Force')
        finalValue = "{:.2f}".format(poolTP_BF[-1])
        #plt.annotate(finalValue, (nXpt,poolTP_BF[-1]),textcoords="offset points",xytext=(20,10),ha='center',color='k')
    #---------------------------------------------------------------------------------------------
    # plot DDPG 1act
    if poolTP_RL1act != None:
        plt.plot(range(nXpt),poolTP_RL1act,'b-',label='DDPG 1act')
        finalValue = "{:.2f}".format(poolTP_RL1act[-1])
        #plt.annotate(finalValue, (nXpt,poolTP_RL1act[-1]),textcoords="offset points",xytext=(20,-10),ha='center',color='b')
    #---------------------------------------------------------------------------------------------
    # plot DDPG 2act
    if poolTP_RL2act != None:
        plt.plot(range(nXpt),poolTP_RL2act,'r-',label='DDPG 2act')
        finalValue = "{:.2f}".format(poolTP_RL2act[-1])
        #plt.annotate(finalValue, (nXpt,poolTP_RL2act[-1]),textcoords="offset points",xytext=(20,-10),ha='center',color='r')
    #---------------------------------------------------------------------------------------------
    for l in range(1,env.L+1):
        # plot BM1
        plt.plot(range(nXpt),poolTP_BM1[l],color='g',linestyle=linestyles[l],label='BM1'+'_L='+str(l))
        finalValue = "{:.2f}".format(poolTP_BM1[l][-1])
        #plt.annotate(finalValue, (nXpt,poolTP_BM1[-1]),textcoords="offset points",xytext=(20,10),ha='center',color='g')
        # plot BM2
        plt.plot(range(nXpt),poolTP_BM2[l],color='y',linestyle=linestyles[l],label='BM2'+'_L='+str(l))
        finalValue = "{:.2f}".format(poolTP_BM2[l][-1])
        #plt.annotate(finalValue, (nXpt,poolTP_BM2[-1]),textcoords="offset points",xytext=(20,10),ha='center',color='y')
        # plot BM3
        plt.plot(range(nXpt),poolTP_BM3[l],color='c',linestyle=linestyles[l],label='BM3'+'_L='+str(l))
        finalValue = "{:.2f}".format(poolTP_BM2[l][-1])
        #plt.annotate(finalValue, (nXpt,poolTP_BM2[-1]),textcoords="offset points",xytext=(20,10),ha='center',color='y')
    #---------------------------------------------------------------------------------------------
    #plt.title(phaseName+': Throughput\n Topology:'+env.TopologyCode) # title
    plt.ylabel("Bits/s") # y label
    plt.xlabel("t") # x label
    plt.grid()
    plt.legend()
    plt.xlim(0,nXpt-1)
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(filename + '_Throughput.png', format='png',dpi=120)
    if isEPS:
        fig.savefig(filename + '_Throughput.eps', format='eps',dpi=120)
#3 Psys
def plotPsys(env,filename,poolPsys_RL1act=None,poolPsys_RL2act=None,poolPsys_BM1=None,poolPsys_BM2=None,poolPsys_BM3=None,poolPsys_BF=None,isEPS=False):
    if 'Training' in filename:
        phaseName = 'Training Phase'
    elif 'Evaluation' in filename:
        phaseName = 'Evaluation Phase'
    elif 'Preview' in filename:
        phaseName = 'Preview Phase'
    plt.cla()
    nXpt=len(poolPsys_BM1[0])
    #---------------------------------------------------------------------------------------------
    # plot Brute Force
    if poolPsys_BF != None:
        plt.plot(range(nXpt),poolPsys_BF,'ko-',label='Brute Force')
        finalValue = "{:.2f}".format(poolPsys_BF[-1])
        #plt.annotate(finalValue, (nXpt,poolPsys_BF[-1]),textcoords="offset points",xytext=(20,10),ha='center',color='k')
    #---------------------------------------------------------------------------------------------
    # plot DDPG 1act
    if poolPsys_RL1act != None:
        plt.plot(range(nXpt),poolPsys_RL1act,'b-',label='DDPG 1act')
        finalValue = "{:.2f}".format(poolPsys_RL1act[-1]) 
        #plt.annotate(finalValue, (nXpt,poolPsys_RL1act[-1]),textcoords="offset points",xytext=(20,-10),ha='center',color='b')
    #---------------------------------------------------------------------------------------------
    # plot DDPG 2act
    if poolPsys_RL2act != None:
        plt.plot(range(nXpt),poolPsys_RL2act,'r-',label='DDPG 2act')
        finalValue = "{:.2f}".format(poolPsys_RL2act[-1]) 
        #plt.annotate(finalValue, (nXpt,poolPsys_RL2act[-1]),textcoords="offset points",xytext=(20,-10),ha='center',color='r')
    #---------------------------------------------------------------------------------------------
    for l in range(1,env.L+1):
        # plot BM1
        plt.plot(range(nXpt),poolPsys_BM1[l],color='g',linestyle=linestyles[l],label='BM1'+'_L='+str(l))
        finalValue = "{:.2f}".format(poolPsys_BM1[l][-1])
        #plt.annotate(finalValue, (nXpt,poolPsys_BM1[-1]),textcoords="offset points",xytext=(20,10),ha='center',color='g')
        # plot BM2
        plt.plot(range(nXpt),poolPsys_BM2[l],color='y',linestyle=linestyles[l],label='BM2'+'_L='+str(l))
        finalValue = "{:.2f}".format(poolPsys_BM2[l][-1])
        #plt.annotate(finalValue, (nXpt,poolPsys_BM2[-1]),textcoords="offset points",xytext=(20,10),ha='center',color='y')
        # plot BM3
        plt.plot(range(nXpt),poolPsys_BM3[l],color='c',linestyle=linestyles[l],label='BM3'+'_L='+str(l))
        finalValue = "{:.2f}".format(poolPsys_BM2[l][-1])
        #plt.annotate(finalValue, (nXpt,poolPsys_BM2[-1]),textcoords="offset points",xytext=(20,10),ha='center',color='y')
    #---------------------------------------------------------------------------------------------
    #plt.title(phaseName+': System Power Consumption\n Topology:'+env.TopologyCode) # title
    plt.ylabel("W") # y label
    plt.xlabel("t") # x label
    plt.grid()
    plt.legend()
    plt.xlim(0,nXpt-1)
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(filename + '_Psys.png', format='png',dpi=120)
    if isEPS:
        fig.savefig(filename + '_Psys.eps', format='eps',dpi=120)
#4 MCAP
def plotMCAP(env,filename,poolMCAP_RL1act=None,poolMCAP_RL2act=None,poolMCAP_BM1=None,poolMCAP_BM2=None,poolMCAP_BM3=None,poolMCAP_BF=None,isEPS=False):
    if 'Training' in filename:
        phaseName = 'Training Phase'
    elif 'Evaluation' in filename:
        phaseName = 'Evaluation Phase'
    elif 'Preview' in filename:
        phaseName = 'Preview Phase'
    plt.cla()
    nXpt=len(poolMCAP_BM1[0])
    #---------------------------------------------------------------------------------------------
    # plot Brute Force
    if poolMCAP_BF != None:
        plt.plot(range(nXpt),poolMCAP_BF,'k-',label='Brute Force')
        finalValue = "{:.2f}".format(poolMCAP_BF[-1])
        #plt.annotate(finalValue, (nXpt,poolMCAP_BF[-1]),textcoords="offset points",xytext=(20,10),ha='center',color='k')
    #---------------------------------------------------------------------------------------------
    # plot DDPG 1act
    if poolMCAP_RL1act != None:
        plt.plot(range(nXpt),poolMCAP_RL1act,'b-',label='DDPG 1act')
        finalValue = "{:.2f}".format(poolMCAP_RL1act[-1]) 
        #plt.annotate(finalValue, (nXpt,poolMCAP_RL1act[-1]),textcoords="offset points",xytext=(20,-10),ha='center',color='b')
    #---------------------------------------------------------------------------------------------
    # plot DDPG 2act
    if poolMCAP_RL2act != None:
        plt.plot(range(nXpt),poolMCAP_RL2act,'r-',label='DDPG 2act')
        finalValue = "{:.2f}".format(poolMCAP_RL2act[-1]) 
        #plt.annotate(finalValue, (nXpt,poolMCAP_RL2act[-1]),textcoords="offset points",xytext=(20,-10),ha='center',color='r')
    #---------------------------------------------------------------------------------------------
    for l in range(1,env.L+1):
        # plot BM1
        plt.plot(range(nXpt),poolMCAP_BM1[l],color='g',linestyle=linestyles[l],label='BM1'+'_L='+str(l))
        finalValue = "{:.2f}".format(poolMCAP_BM1[l][-1])
        #plt.annotate(finalValue, (nXpt,poolMCAP_BM1[-1]),textcoords="offset points",xytext=(20,10),ha='center',color='g')
        # plot BM2
        plt.plot(range(nXpt),poolMCAP_BM2[l],color='y',linestyle=linestyles[l],label='BM2'+'_L='+str(l))
        finalValue = "{:.2f}".format(poolMCAP_BM2[l][-1])
        #plt.annotate(finalValue, (nXpt,poolMCAP_BM2[-1]),textcoords="offset points",xytext=(20,10),ha='center',color='y')
        # plot BM3
        plt.plot(range(nXpt),poolMCAP_BM3[l],color='c',linestyle=linestyles[l],label='BM3'+'_L='+str(l))
        finalValue = "{:.2f}".format(poolMCAP_BM3[l][-1])
        #plt.annotate(finalValue, (nXpt,poolMCAP_BM2[-1]),textcoords="offset points",xytext=(20,10),ha='center',color='y')
    #---------------------------------------------------------------------------------------------
    #plt.title(phaseName+': Miss Count of AP\n Topology:'+env.TopologyCode) # title
    plt.ylabel("Counts") # y label
    plt.xlabel("t") # x label
    plt.grid()
    plt.legend()
    plt.xlim(0,nXpt-1)
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(filename + '_MCAP.png', format='png',dpi=120)
    if isEPS:
        fig.savefig(filename + '_MCAP.eps', format='eps',dpi=120)
#5 MCCPU
def plotMCCPU(env,filename,poolMCCPU_RL1act=None,poolMCCPU_RL2act=None,poolMCCPU_BM1=None,poolMCCPU_BM2=None,poolMCCPU_BM3=None,poolMCCPU_BF=None,isEPS=False):
    if 'Training' in filename:
        phaseName = 'Training Phase'
    elif 'Evaluation' in filename:
        phaseName = 'Evaluation Phase'
    elif 'Preview' in filename:
        phaseName = 'Preview Phase'
    plt.cla()
    nXpt=len(poolMCCPU_BM1[0])
    #---------------------------------------------------------------------------------------------
    # plot Brute Force
    if poolMCCPU_BF != None:
        plt.plot(range(nXpt),poolMCCPU_BF,'k-',label='Brute Force')
        finalValue = "{:.2f}".format(poolMCCPU_BF[-1])
        #plt.annotate(finalValue, (nXpt,poolMCCPU_BF[-1]),textcoords="offset points",xytext=(20,10),ha='center',color='k')
    #---------------------------------------------------------------------------------------------
    # plot DDPG 1act
    if poolMCCPU_RL1act != None:
        plt.plot(range(nXpt),poolMCCPU_RL1act,'b-',label='DDPG 1act')
        finalValue = "{:.2f}".format(poolMCCPU_RL1act[-1]) 
        #plt.annotate(finalValue, (nXpt,poolMCCPU_RL1act[-1]),textcoords="offset points",xytext=(20,-10),ha='center',color='b')
    #---------------------------------------------------------------------------------------------
    # plot DDPG 2act
    if poolMCCPU_RL2act != None:
        plt.plot(range(nXpt),poolMCCPU_RL2act,'r-',label='DDPG 2act')
        finalValue = "{:.2f}".format(poolMCCPU_RL2act[-1]) 
        #plt.annotate(finalValue, (nXpt,poolMCCPU_RL2act[-1]),textcoords="offset points",xytext=(20,-10),ha='center',color='r')
    #---------------------------------------------------------------------------------------------
    for l in range(1,env.L+1):
        # plot BM1
        plt.plot(range(nXpt),poolMCCPU_BM1[l],color='g',linestyle=linestyles[l],label='BM1'+'_L='+str(l))
        finalValue = "{:.2f}".format(poolMCCPU_BM1[l][-1])
        #plt.annotate(finalValue, (nXpt,poolMCCPU_BM1_BM1[-1]),textcoords="offset points",xytext=(20,10),ha='center',color='g')
        # plot BM2
        plt.plot(range(nXpt),poolMCCPU_BM2[l],color='y',linestyle=linestyles[l],label='BM2'+'_L='+str(l))
        finalValue = "{:.2f}".format(poolMCCPU_BM2[l][-1])
        #plt.annotate(finalValue, (nXpt,poolMCCPU_BM1_BM2[-1]),textcoords="offset points",xytext=(20,10),ha='center',color='y')
        # plot BM3
        plt.plot(range(nXpt),poolMCCPU_BM3[l],color='c',linestyle=linestyles[l],label='BM3'+'_L='+str(l))
        finalValue = "{:.2f}".format(poolMCCPU_BM3[l][-1])
        #plt.annotate(finalValue, (nXpt,poolMCCPU_BM1_BM2[-1]),textcoords="offset points",xytext=(20,10),ha='center',color='y')
    #---------------------------------------------------------------------------------------------
    #plt.title(phaseName+': Miss Count of CPU\n Topology:'+env.TopologyCode) # title
    plt.ylabel("Counts") # y label
    plt.xlabel("t") # x label
    plt.grid()
    plt.legend()
    plt.xlim(0,nXpt-1)
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(filename + '_MCCPU.png', format='png',dpi=120)
    if isEPS:
        fig.savefig(filename + '_MCCPU.eps', format='eps',dpi=120)
#6 HR
def plotHR(env,filename,poolHR_RL1act=None,poolHR_RL2act=None,poolHR_BM1=None,poolHR_BM2=None,poolHR_BM3=None,poolHR_BF=None,isEPS=False):
    if 'Training' in filename:
        phaseName = 'Training Phase'
    elif 'Evaluation' in filename:
        phaseName = 'Evaluation Phase'
    elif 'Preview' in filename:
        phaseName = 'Preview Phase'
    plt.cla()
    nXpt=len(poolHR_BM1[0])
    #---------------------------------------------------------------------------------------------
    # plot Brute Force
    if poolHR_BF != None:
        plt.plot(range(nXpt),poolHR_BF,'k-',label='Brute Force')
    # plot DDPG 1act
    if poolHR_RL1act != None:
        plt.plot(range(nXpt),poolHR_RL1act,'b-',label='DDPG 1act')
    # plot DDPG 2act
    if poolHR_RL2act != None:
        plt.plot(range(nXpt),poolHR_RL2act,'r-',label='DDPG 2act')
    # plot BM1-BM3
    for l in range(1,env.L+1):
        plt.plot(range(nXpt),poolHR_BM1[l],color='g',linestyle=linestyles[l],label='BM1'+'_L='+str(l))
        plt.plot(range(nXpt),poolHR_BM2[l],color='y',linestyle=linestyles[l],label='BM2'+'_L='+str(l))
        plt.plot(range(nXpt),poolHR_BM3[l],color='c',linestyle=linestyles[l],label='BM3'+'_L='+str(l))
    #---------------------------------------------------------------------------------------------
    #plt.title(phaseName+': Hit Rate (HR) \n Topology:'+env.TopologyCode) # title
    plt.ylabel("Ratio") # y label
    plt.xlabel("t") # x label
    plt.grid()
    plt.legend()
    plt.axis([10, nXpt-1, 0, 1.1])
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(filename + '_HR.png', format='png',dpi=120)
    if isEPS:
        fig.savefig(filename + '_HR.eps', format='eps',dpi=120)

def plotHistory(env,filename,isPlotEE=False,isPlotTP=False,isPlotPsys=False,isPlotHR=False,isEPS=False,loadBF=False):
    if 'Training' in filename:
        phaseName = 'Training Phase'
    elif 'Evaluation' in filename:
        phaseName = 'Evaluation Phase'
    elif 'Preview' in filename:
        phaseName = 'Preview'
    
    # Initialization
    poolEE_BF = None
    poolTP_BF = None
    poolPsys_BF = None
    poolHR_BF = None
    poolMCAP_BF = None
    poolMCCPU_BF = None

    poolEE_RL1act=None
    poolTP_RL1act=None
    poolPsys_RL1act=None
    poolHR_RL1act=None
    poolMCAP_RL1act=None
    poolMCCPU_RL1act=None
    poolLossActor1act=None
    poolLossCritic1act=None

    poolEE_RL2act=None
    poolTP_RL2act=None
    poolPsys_RL2act=None
    poolHR_RL2act=None
    poolMCAP_RL2act=None
    poolMCCPU_RL2act=None
    poolLossActor2act=None
    poolLossCritic2act=None
    #---------------------------------------------------------------------------------------------
    # Load Brute Force Policy
    if env.B==4 and env.U ==4 and env.F==5 and env.N==2 and loadBF:
        with open(filename+ 'BF.pkl', 'rb') as f:
            env, poolEE_BF,poolTP_BF,poolPsys_BF,poolHR_BF,poolMCAP_BF,poolMCCPU_BF = pickle.load(f)
    #---------------------------------------------------------------------------------------------
    file_1act = filename+ '1act' +'RL.pkl'
    file_2act = filename+ '2act' +'RL.pkl'
    if os.path.isfile(file_1act):
        with open(file_1act,'rb') as f:
            env, poolEE_RL1act,poolTP_RL1act,poolPsys_RL1act,poolHR_RL1act,poolMCAP_RL1act,poolMCCPU_RL1act\
                ,poolLossActor1act,poolLossCritic1act = pickle.load(f)
    if os.path.isfile(file_2act):
        with open(file_2act,'rb') as f:
            env, poolEE_RL2act,poolTP_RL2act,poolPsys_RL2act,poolHR_RL2act,poolMCAP_RL2act,poolMCCPU_RL2act\
                ,poolLossActor2act,poolLossCritic2act = pickle.load(f)
    with open(filename+'BM1.pkl','rb') as f: 
        env, poolEE_BM1,poolTP_BM1,poolPsys_BM1,poolHR_BM1,poolMCAP_BM1,poolMCCPU_BM1 = pickle.load(f)
    with open(filename+'BM2.pkl','rb') as f:
        env, poolEE_BM2,poolTP_BM2,poolPsys_BM2,poolHR_BM2,poolMCAP_BM2,poolMCCPU_BM2 = pickle.load(f)
    with open(filename+'BM3.pkl','rb') as f:
        env, poolEE_BM3,poolTP_BM3,poolPsys_BM3,poolHR_BM3,poolMCAP_BM3,poolMCCPU_BM3 = pickle.load(f)

    #---------------------------------------------------------------------------------------------
    if poolLossActor1act != None or poolLossActor2act != None:
        # plot RL: poolLossCritic/poolLossActor
        plt.cla()
        plt.clf()
        if os.path.isfile(file_1act):
            plt.plot(range(len(poolLossCritic1act)),poolLossCritic1act,'b-',label='Loss of critic 1act')
            plt.plot(range(len(poolLossActor1act)),poolLossActor1act,'c-',label='Loss of actor 1act')
        if os.path.isfile(file_2act):
            plt.plot(range(len(poolLossCritic2act)),poolLossCritic2act,'r-',label='Loss of critic 2act')
            plt.plot(range(len(poolLossActor2act)),poolLossActor2act,'m-',label='Loss of actor 2act')
        
        plt.title(phaseName+': Critic and Actor Loss\n' + env.TopologyName) # title
        plt.ylabel("Q") # y label
        plt.xlabel("t") # x label
        plt.grid()
        plt.legend()
        fig = plt.gcf()
        fig.savefig(filename + '_Loss.png', format='png',dpi=120)
        if isEPS:
            fig.savefig(filename + '_Loss.eps', format='eps',dpi=120)
    #---------------------------------------------------------------------------------------------
    if isPlotEE:
        plotEE(env,filename,poolEE_RL1act=poolEE_RL1act,poolEE_RL2act=poolEE_RL2act\
            ,poolEE_BM1=poolEE_BM1,poolEE_BM2=poolEE_BM2,poolEE_BM3=poolEE_BM3,poolEE_BF=poolEE_BF,isEPS=isEPS)
        plotMCAP(env,filename,poolMCAP_RL1act=poolMCAP_RL1act,poolMCAP_RL2act=poolMCAP_RL2act\
            ,poolMCAP_BM1=poolMCAP_BM1,poolMCAP_BM2=poolMCAP_BM2,poolMCAP_BM3=poolMCAP_BM3,poolMCAP_BF=poolMCAP_BF,isEPS=isEPS)
        plotMCCPU(env,filename,poolMCCPU_RL1act=poolMCCPU_RL1act,poolMCCPU_RL2act=poolMCCPU_RL2act\
            ,poolMCCPU_BM1=poolMCCPU_BM1,poolMCCPU_BM2=poolMCCPU_BM2,poolMCCPU_BM3=poolMCCPU_BM3,poolMCCPU_BF=poolMCCPU_BF,isEPS=isEPS)
    #---------------------------------------------------------------------------------------------
    if isPlotTP:
        plotTP(env,filename,poolTP_RL1act=poolTP_RL1act,poolTP_RL2act=poolTP_RL2act\
            ,poolTP_BM1=poolTP_BM1,poolTP_BM2=poolTP_BM2,poolTP_BM3=poolTP_BM3,poolTP_BF=poolTP_BF,isEPS=isEPS)
    #---------------------------------------------------------------------------------------------
    if isPlotPsys:
        plotPsys(env,filename,poolPsys_RL1act=poolPsys_RL1act,poolPsys_RL2act=poolPsys_RL2act\
            ,poolPsys_BM1=poolPsys_BM1,poolPsys_BM2=poolPsys_BM2,poolPsys_BM3=poolPsys_BM3,poolPsys_BF=poolPsys_BF,isEPS=isEPS)  
    #---------------------------------------------------------------------------------------------
    if isPlotHR:
        plotHR(env,filename,poolHR_RL1act=poolHR_RL1act,poolHR_RL2act=poolHR_RL2act\
            ,poolHR_BM1=poolHR_BM1,poolHR_BM2=poolHR_BM2,poolHR_BM3=poolHR_BM3,poolHR_BF=poolHR_BF,isEPS=isEPS)

def trainModel(env,actMode,changeReq,changeChannel,loadActor,randSEED=0):
    # new ACT 
    #modelPath = 'D:\\/Model/' + env.TopologyName+'/'
    modelPath = 'data/'+env.TopologyCode+'/Model/'
    if actMode == '2act':
        ddpg_cl = DDPG(obs_dim = env.dimObs, act_dim = env.dimActCL,memMaxSize=25000)
        ddpg_ca = DDPG(obs_dim = env.dimObs, act_dim = env.dimActCA,memMaxSize=25000)
        if(loadActor):
           ddpg_cl.loadModel(modelPath = modelPath, modelName= actMode+'_cl') 
           ddpg_ca.loadModel(modelPath = modelPath, modelName= actMode+'_ca') 
    elif actMode == '1act':
        ddpg_s = DDPG(obs_dim = env.dimObs, act_dim = env.dimAct,memMaxSize=25000)###
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
    countChangeReq = 0
    countChangeChannel = 0
    
    nItr = MAX_EPISODES*MAX_EP_STEPS
    # RL
    poolEE_RL   = [0]*nItr
    poolTP_RL   = [0]*nItr
    poolPsys_RL = [0]*nItr
    poolHR_RL   = [0]*nItr
    poolMCAP_RL = [0]*nItr # MCAP = miss file count at APs
    poolMCCPU_RL= [0]*nItr # MCCPU = miss file count at CPU
    poolCL_RL   = [0]*nItr
    poolCA_RL   = [0]*nItr

    poolLossActor = [0]*nItr
    poolLossCritic= [0]*nItr
    poolVarLossCritic = []
    poolVarEE = []

    # BM1 Initialization [SNR-based] 
    poolEE_BM1 = [[0]*nItr  for i in range(env.L+1)]
    poolTP_BM1 = [[0]*nItr  for i in range(env.L+1)]
    poolPsys_BM1=[[0]*nItr  for i in range(env.L+1)]
    poolHR_BM1 = [[0]*nItr  for i in range(env.L+1)]
    poolMCAP_BM1=[[0]*nItr  for i in range(env.L+1)]
    poolMCCPU_BM1=[[0]*nItr  for i in range(env.L+1)]
    poolCL_BM1 = [[0]*nItr  for i in range(env.L+1)]
    poolCA_BM1 = [[0]*nItr  for i in range(env.L+1)]
    
    # BM2 Initialization [SNR-based]
    poolEE_BM2 = [[0]*nItr  for i in range(env.L+1)]
    poolTP_BM2 = [[0]*nItr  for i in range(env.L+1)]
    poolPsys_BM2=[[0]*nItr  for i in range(env.L+1)]
    poolHR_BM2 = [[0]*nItr  for i in range(env.L+1)]
    poolMCAP_BM2=[[0]*nItr  for i in range(env.L+1)]
    poolMCCPU_BM2=[[0]*nItr  for i in range(env.L+1)]
    poolCL_BM2 = [[0]*nItr  for i in range(env.L+1)]
    poolCA_BM2 = [[0]*nItr  for i in range(env.L+1)]

    # BM3 Initialization [File-based] 
    poolEE_BM3 = [[0]*nItr  for i in range(env.L+1)]
    poolTP_BM3 = [[0]*nItr  for i in range(env.L+1)]
    poolPsys_BM3=[[0]*nItr  for i in range(env.L+1)]
    poolHR_BM3 = [[0]*nItr  for i in range(env.L+1)]
    poolMCAP_BM3=[[0]*nItr  for i in range(env.L+1)]
    poolMCCPU_BM3=[[0]*nItr  for i in range(env.L+1)]
    poolCL_BM3 = [[0]*nItr  for i in range(env.L+1)]
    poolCA_BM3 = [[0]*nItr  for i in range(env.L+1)]

    for epsilon in tqdm(range(MAX_EPISODES),bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        ep_reward = 0
        obs = env.reset()# Get initial state
        for step in range(MAX_EP_STEPS):
            ep = epsilon*MAX_EP_STEPS+step
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
            EE_RL = reward
            TP_RL = sum(env.Throughput)
            Psys_RL = env.P_sys/1000
            HR_RL = env.HR
            CL_Policy_UE_RL,CA_Policy_BS_RL =  env.action2Policy(action)

            poolEE_RL[ep]   = EE_RL
            poolTP_RL[ep]   = TP_RL
            poolPsys_RL[ep] = Psys_RL
            poolHR_RL[ep]   = HR_RL
            poolMCAP_RL[ep] = env.missCounterAP
            poolMCCPU_RL[ep]= env.missCounterCPU
            poolCL_RL[ep]   = CL_Policy_UE_RL
            poolCA_RL[ep]   = CA_Policy_BS_RL
            #===========================================================================
            # Experience Injection
            #if ep < 1000:
            #    EE_BM1, SNR_CL_Policy_UE_BM1, POP_CA_Policy_BS_BM1,bestL = env.getBestEE_snrCL_popCA(cacheMode='pref')

            # RL Add Memory
            if actMode == '2act':
                ddpg_cl.addMemory([obs,a_cl,reward,obs2])
                ddpg_ca.addMemory([obs,a_ca,-env.P_sys,obs2])# remind the reward of CA agent
            elif actMode == '1act':
                ddpg_s.addMemory([obs,action,reward,obs2])
            #===========================================================================
            # RL Upadate
            if actMode == '2act':
                if len(ddpg_cl.memory) > ddpg_cl.batch_size:
                    lossActor, lossCritic = ddpg_cl.train()
                    poolLossActor[ep] = lossActor.item()
                    poolLossCritic[ep] = lossCritic.item()
                    lossActor, lossCritic = ddpg_ca.train()
                    
            elif actMode == '1act':
                if len(ddpg_s.memory) > ddpg_s.batch_size:
                    lossActor, lossCritic = ddpg_s.train()
                    poolLossActor[ep] = lossActor.item()
                    poolLossCritic[ep] = lossCritic.item()
            obs = obs2
            ep_reward += reward
            #---------------------------------------------------------------------------------------------  
            # iteration update
            if ep % 30 == 0:
                noiseSigma*=0.995
            if (ep % 1000) == 0:
                #print(list(poolLossCritic[-100:]))
                poolVarLossCritic.append( np.var(poolLossCritic[-1000:]) )
                poolVarEE.append( np.var(poolEE_RL[-1000:]) )
                if (poolVarLossCritic[-1] < 10) and (poolVarEE[-1] < 10) and changeReq:
                    env.resetReq()
                    print('**Change Request: ',env.Req)
                    countChangeReq+=1
                    noiseSigma = 1 # reset explore     
            fixInterval=1000
            if (ep % fixInterval) == 0: # Mectric Snapshot
                for l in range(1,env.L+1):
                    # Benchmark 1    
                    EE_BM1, CL_Policy_UE_BM1, CA_Policy_BS_BM1 = env.getPolicy_BM1(cacheMode='pref',nLink=l)
                    EE_BM1 = env.calEE(CL_Policy_UE_BM1,CA_Policy_BS_BM1)
                    TP_BM1 = sum(env.Throughput)
                    Psys_BM1 = env.P_sys/1000 # mW->W
                    HR_BM1 = env.calHR(CL_Policy_UE_BM1,CA_Policy_BS_BM1)
                    poolEE_BM1[l][ep:ep+fixInterval]    =[EE_BM1]*fixInterval
                    poolTP_BM1[l][ep:ep+fixInterval]    =[TP_BM1]*fixInterval
                    poolPsys_BM1[l][ep:ep+fixInterval]  =[Psys_BM1]*fixInterval
                    poolHR_BM1[l][ep:ep+fixInterval]    =[HR_BM1]*fixInterval
                    poolMCAP_BM1[l][ep:ep+fixInterval]  =[env.missCounterAP]*fixInterval
                    poolMCCPU_BM1[l][ep:ep+fixInterval] =[env.missCounterCPU]*fixInterval
                    poolCL_BM1[l][ep:ep+fixInterval]    =[CL_Policy_UE_BM1]*fixInterval
                    poolCA_BM1[l][ep:ep+fixInterval]    =[CA_Policy_BS_BM1]*fixInterval
                    #print('EE_BM1'+'_L'+str(l),'=', EE_BM1)
                    # Benchmark 2  
                    EE_BM2, CL_Policy_UE_BM2, CA_Policy_BS_BM2 = env.getPolicy_BM2(nLink=l)
                    EE_BM2 = env.calEE(CL_Policy_UE_BM2,CA_Policy_BS_BM2)
                    TP_BM2 = sum(env.Throughput)
                    Psys_BM2 = env.P_sys/1000 # mW->W
                    HR_BM2 = env.calHR(CL_Policy_UE_BM2,CA_Policy_BS_BM2)
                    poolEE_BM2[l][ep:ep+fixInterval]    =[EE_BM2]*fixInterval
                    poolTP_BM2[l][ep:ep+fixInterval]    =[TP_BM2]*fixInterval
                    poolPsys_BM2[l][ep:ep+fixInterval]  =[Psys_BM2]*fixInterval
                    poolHR_BM2[l][ep:ep+fixInterval]    =[HR_BM2]*fixInterval
                    poolMCAP_BM2[l][ep:ep+fixInterval]  =[env.missCounterAP]*fixInterval
                    poolMCCPU_BM2[l][ep:ep+fixInterval] =[env.missCounterCPU]*fixInterval
                    poolCL_BM2[l][ep:ep+fixInterval]    =[CL_Policy_UE_BM2]*fixInterval
                    poolCA_BM2[l][ep:ep+fixInterval]    =[CA_Policy_BS_BM2]*fixInterval
                    #print('EE_BM2'+'_L'+str(l),'=', EE_BM2)
                    # Benchmark 3 # DEBUG
                    EE_BM3, CL_Policy_UE_BM3, CA_Policy_BS_BM3 = env.getPolicy_BM3(nLink=l)
                    EE_BM3 = env.calEE(CL_Policy_UE_BM3,CA_Policy_BS_BM3)
                    TP_BM3 = sum(env.Throughput)
                    Psys_BM3 = env.P_sys/1000 # mW->W
                    HR_BM3 = env.calHR(CL_Policy_UE_BM3,CA_Policy_BS_BM3)
                    poolEE_BM3[l][ep:ep+fixInterval]    =[EE_BM3]*fixInterval
                    poolTP_BM3[l][ep:ep+fixInterval]    =[TP_BM3]*fixInterval
                    poolPsys_BM3[l][ep:ep+fixInterval]  =[Psys_BM3]*fixInterval
                    poolHR_BM3[l][ep:ep+fixInterval]    =[HR_BM3]*fixInterval
                    poolMCAP_BM3[l][ep:ep+fixInterval]  =[env.missCounterAP]*fixInterval
                    poolMCCPU_BM3[l][ep:ep+fixInterval] =[env.missCounterCPU]*fixInterval
                    poolCL_BM3[l][ep:ep+fixInterval]    =[CL_Policy_UE_BM3]*fixInterval
                    poolCA_BM3[l][ep:ep+fixInterval]    =[CA_Policy_BS_BM3]*fixInterval
                    #print('EE_BM3'+'_L'+str(l),'=', EE_BM3)
                
                # Preview / Save Line
                filename = 'data/'+env.TopologyCode+'/Preview/'+'['+ str(randSEED) +']'+ env.TopologyName +str(MAX_EPISODES*MAX_EP_STEPS)+'_Train_'
                with open(filename+ actMode +'RL.pkl', 'wb') as f:  
                    pickle.dump([env, poolEE_RL,poolTP_RL,poolPsys_RL,poolHR_RL,poolMCAP_RL,poolMCCPU_RL,poolLossActor,poolLossCritic], f)
                with open(filename+'BM1.pkl', 'wb') as f:  
                    pickle.dump([env, poolEE_BM1,poolTP_BM1,poolPsys_BM1,poolHR_BM1,poolMCAP_BM1,poolMCCPU_BM1], f)
                with open(filename+'BM2.pkl', 'wb') as f:  
                    pickle.dump([env, poolEE_BM2,poolTP_BM2,poolPsys_BM2,poolHR_BM2,poolMCAP_BM2,poolMCCPU_BM2], f)
                with open(filename+'BM3.pkl', 'wb') as f:  
                    pickle.dump([env, poolEE_BM3,poolTP_BM3,poolPsys_BM3,poolHR_BM3,poolMCAP_BM3,poolMCCPU_BM3], f)
                plotHistory(env,filename,isPlotEE=True,isPlotTP=True,isPlotPsys=True,isPlotHR=True,isEPS=False,loadBF=False)
                
                '''
                if poolEE_RL[-1]>EE_BM1:
                    print('poolEE_RL win!',poolEE_RL[-1], 'EE_BM1 loss QQ', EE_BM1)
                elif (ep % 30000) == 0:
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
        ddpg_cl.saveModel(modelPath = modelPath,modelName= '['+ str(randSEED) +']'+ actMode+'_cl')
        ddpg_ca.saveModel(modelPath = modelPath,modelName= '['+ str(randSEED) +']'+ actMode+'_ca')
    elif actMode == '1act':
        ddpg_s.saveModel(modelPath = modelPath,modelName= '['+ str(randSEED) +']' + actMode)
    
    # Save Line
    filename = 'data/'+env.TopologyCode+'/TrainingPhase/'+'['+ str(randSEED) +']'+ env.TopologyName +str(MAX_EPISODES*MAX_EP_STEPS)+'_Train_'
    with open(filename+ actMode +'RL.pkl', 'wb') as f:  
        pickle.dump([env, poolEE_RL,poolTP_RL,poolPsys_RL,poolHR_RL,poolMCAP_RL,poolMCCPU_RL,poolLossActor,poolLossCritic], f)
    with open(filename+'BM1.pkl', 'wb') as f:  
        pickle.dump([env, poolEE_BM1,poolTP_BM1,poolPsys_BM1,poolHR_BM1,poolMCAP_BM1,poolMCCPU_BM1], f)
    with open(filename+'BM2.pkl', 'wb') as f:  
        pickle.dump([env, poolEE_BM2,poolTP_BM2,poolPsys_BM2,poolHR_BM2,poolMCAP_BM2,poolMCCPU_BM2], f)
    with open(filename+'BM3.pkl', 'wb') as f:  
        pickle.dump([env, poolEE_BM3,poolTP_BM3,poolPsys_BM3,poolHR_BM3,poolMCAP_BM3,poolMCCPU_BM3], f)
    

def evaluateModel(env,actMode, nItr=100, randSEED=0):
    # new ACT 
    modelPath = 'data/'+env.TopologyCode+'/Model/'
    if actMode == '2act':
        ddpg_cl = DDPG(obs_dim = env.dimObs, act_dim = env.dimActCL,memMaxSize=20000)
        ddpg_ca = DDPG(obs_dim = env.dimObs, act_dim = env.dimActCA,memMaxSize=20000)
        # load Model
        ddpg_cl.loadModel(modelPath = modelPath, modelName= '['+ str(randSEED) +']' + actMode+'_cl') 
        ddpg_ca.loadModel(modelPath = modelPath, modelName= '['+ str(randSEED) +']' + actMode+'_ca') 
    elif actMode == '1act':
        ddpg_s = DDPG(obs_dim = env.dimObs, act_dim = env.dimAct,memMaxSize=20000)###
        # load Model
        ddpg_s.loadModel(modelPath = modelPath, modelName= '['+ str(randSEED) +']'+ actMode)
    # BF
    if env.B==4 and env.U ==4 and env.F==5 and env.N==2:
        poolEE_BF   = [0]*nItr
        poolTP_BF   = [0]*nItr
        poolPsys_BF = [0]*nItr
        poolHR_BF   = [0]*nItr
        poolMCAP_BF = [0]*nItr # MCAP = miss file count at APs
        poolMCCPU_BF= [0]*nItr # MCCPU = miss file count at CPU
        poolCL_BF   = [0]*nItr
        poolCA_BF   = [0]*nItr
    # RL
    poolEE_RL   = [0]*nItr
    poolTP_RL   = [0]*nItr
    poolPsys_RL = [0]*nItr
    poolHR_RL   = [0]*nItr
    poolMCAP_RL = [0]*nItr # MCAP = miss file count at APs
    poolMCCPU_RL= [0]*nItr # MCCPU = miss file count at CPU
    poolCL_RL   = [0]*nItr
    poolCA_RL   = [0]*nItr

    poolLossActor = [0]*nItr
    poolLossCritic = [0]*nItr

    # BM1 Initialization [SNR-based] 
    poolEE_BM1 = [[0]*nItr  for i in range(env.L+1)]
    poolTP_BM1 = [[0]*nItr  for i in range(env.L+1)]
    poolPsys_BM1=[[0]*nItr  for i in range(env.L+1)]
    poolHR_BM1 = [[0]*nItr  for i in range(env.L+1)]
    poolMCAP_BM1=[[0]*nItr  for i in range(env.L+1)]
    poolMCCPU_BM1=[[0]*nItr  for i in range(env.L+1)]
    poolCL_BM1 = [[0]*nItr  for i in range(env.L+1)]
    poolCA_BM1 = [[0]*nItr  for i in range(env.L+1)]
    
    # BM2 Initialization [SNR-based]
    poolEE_BM2 = [[0]*nItr  for i in range(env.L+1)]
    poolTP_BM2 = [[0]*nItr  for i in range(env.L+1)]
    poolPsys_BM2=[[0]*nItr  for i in range(env.L+1)]
    poolHR_BM2 = [[0]*nItr  for i in range(env.L+1)]
    poolMCAP_BM2=[[0]*nItr  for i in range(env.L+1)]
    poolMCCPU_BM2=[[0]*nItr  for i in range(env.L+1)]
    poolCL_BM2 = [[0]*nItr  for i in range(env.L+1)]
    poolCA_BM2 = [[0]*nItr  for i in range(env.L+1)]

    # BM3 Initialization [File-based] 
    poolEE_BM3 = [[0]*nItr  for i in range(env.L+1)]
    poolTP_BM3 = [[0]*nItr  for i in range(env.L+1)]
    poolPsys_BM3=[[0]*nItr  for i in range(env.L+1)]
    poolHR_BM3 = [[0]*nItr  for i in range(env.L+1)]
    poolMCAP_BM3=[[0]*nItr  for i in range(env.L+1)]
    poolMCCPU_BM3=[[0]*nItr  for i in range(env.L+1)]
    poolCL_BM3 = [[0]*nItr  for i in range(env.L+1)]
    poolCA_BM3 = [[0]*nItr  for i in range(env.L+1)]

    for ep in tqdm(range(nItr),bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        # iterate each L
        #===============================
        for l in range(1,env.L+1):
            # Benchmark 1    
            EE_BM1, CL_Policy_UE_BM1, CA_Policy_BS_BM1 = env.getPolicy_BM1(cacheMode='pref',nLink=l)
            EE_BM1 = env.calEE(CL_Policy_UE_BM1,CA_Policy_BS_BM1)
            TP_BM1 = sum(env.Throughput)
            Psys_BM1 = env.P_sys/1000 # mW->W
            HR_BM1 = env.calHR(CL_Policy_UE_BM1,CA_Policy_BS_BM1)
            poolEE_BM1[l][ep]    =EE_BM1
            poolTP_BM1[l][ep]    =TP_BM1
            poolPsys_BM1[l][ep]  =Psys_BM1
            poolHR_BM1[l][ep]    =HR_BM1
            poolMCAP_BM1[l][ep]  =env.missCounterAP
            poolMCCPU_BM1[l][ep] =env.missCounterCPU
            poolCL_BM1[l][ep]    =CL_Policy_UE_BM1
            poolCA_BM1[l][ep]    =CA_Policy_BS_BM1
            #print('EE_BM1'+'_L'+str(l),'=', EE_BM1)
            # Benchmark 2  
            EE_BM2, CL_Policy_UE_BM2, CA_Policy_BS_BM2 = env.getPolicy_BM2(nLink=l)
            EE_BM2 = env.calEE(CL_Policy_UE_BM2,CA_Policy_BS_BM2)
            TP_BM2 = sum(env.Throughput)
            Psys_BM2 = env.P_sys/1000 # mW->W
            HR_BM2 = env.calHR(CL_Policy_UE_BM2,CA_Policy_BS_BM2)
            poolEE_BM2[l][ep]    =EE_BM2
            poolTP_BM2[l][ep]    =TP_BM2
            poolPsys_BM2[l][ep]  =Psys_BM2
            poolHR_BM2[l][ep]    =HR_BM2
            poolMCAP_BM2[l][ep]  =env.missCounterAP
            poolMCCPU_BM2[l][ep] =env.missCounterCPU
            poolCL_BM2[l][ep]    =CL_Policy_UE_BM2
            poolCA_BM2[l][ep]    =CA_Policy_BS_BM2
            #print('EE_BM2'+'_L'+str(l),'=', EE_BM2)
            # Benchmark 3 # DEBUG
            EE_BM3, CL_Policy_UE_BM3, CA_Policy_BS_BM3 = env.getPolicy_BM3(nLink=l)
            EE_BM3 = env.calEE(CL_Policy_UE_BM3,CA_Policy_BS_BM3)
            TP_BM3 = sum(env.Throughput)
            Psys_BM3 = env.P_sys/1000 # mW->W
            HR_BM3 = env.calHR(CL_Policy_UE_BM3,CA_Policy_BS_BM3)
            poolEE_BM3[l][ep]    =EE_BM3
            poolTP_BM3[l][ep]    =TP_BM3
            poolPsys_BM3[l][ep]  =Psys_BM3
            poolHR_BM3[l][ep]    =HR_BM3
            poolMCAP_BM3[l][ep]  =env.missCounterAP
            poolMCCPU_BM3[l][ep] =env.missCounterCPU
            poolCL_BM3[l][ep]    =CL_Policy_UE_BM3
            poolCA_BM3[l][ep]    =CA_Policy_BS_BM3
            #print('EE_BM3'+'_L'+str(l),'=', EE_BM3)
        #===============================
        
        # DDPG
        if actMode == '2act':
            EE_RL, CL_Policy_UE_RL, CA_Policy_BS_RL = getEE_RL(env,actMode = actMode,ddpg_cl=ddpg_cl,ddpg_ca=ddpg_ca)
        elif actMode == '1act':
            EE_RL, CL_Policy_UE_RL, CA_Policy_BS_RL = getEE_RL(env,actMode = actMode,ddpg_s=ddpg_s)
        TP_RL = sum(env.Throughput)
        Psys_RL = env.P_sys/1000
        HR_RL = env.calHR(CL_Policy_UE_RL,CA_Policy_BS_RL)
        poolEE_RL[ep]   = EE_RL
        poolTP_RL[ep]   = TP_RL
        poolPsys_RL[ep] = Psys_RL
        poolHR_RL[ep]   = HR_RL
        poolMCAP_RL[ep] = env.missCounterAP
        poolMCCPU_RL[ep]= env.missCounterCPU
        poolCL_RL[ep]   = CL_Policy_UE_RL
        poolCA_RL[ep]   = CA_Policy_BS_RL

        # BF
        if env.B==4 and env.U ==4 and env.F==5 and env.N==2:
            print('Calculate BF itr index:',ep)
            EE_BF, CL_Policy_UE_BF, CA_Policy_BS_BF = env.getOptEE_BF(isSave=True)
            EE_BF = env.calEE(CL_Policy_UE_BF,CA_Policy_BS_BF)
            TP_BF = sum(env.Throughput)
            Psys_BF = env.P_sys/1000 # mW->W
            HR_BF = env.calHR(CL_Policy_UE_BF,CA_Policy_BS_BF)

            poolEE_BF[ep] = EE_BF
            poolTP_BF[ep] = TP_BF
            poolPsys_BF[ep] = Psys_BF
            poolHR_BF[ep] = HR_BF
            poolMCAP_BF[ep] = env.missCounterAP
            poolMCCPU_BF[ep] = env.missCounterCPU
            poolCL_BF[ep] = CL_Policy_UE_BF
            poolCA_BF[ep] = CA_Policy_BS_BF
        
        # Change Environment
        env.timeVariantChannel()
        #env.resetReq()

    # Sample CL/CA Policy Visualization
    sampled = int(nItr/2)
    filename = 'data/'+env.TopologyCode+'/EVSampledPolicy/'+'['+ str(randSEED) +']'+ env.TopologyName +'_EVSampledPolicy_'
    # BF PV
    if env.B==4 and env.U ==4 and env.F==5 and env.N==2:
        plot_UE_BS_distribution_Cache(env, poolCL_BF[sampled], poolCA_BF[sampled], poolEE_BF[sampled],filename+'BF',isEPS=True)
    # RL PV
    plot_UE_BS_distribution_Cache(env, poolCL_RL[sampled], poolCA_RL[sampled], poolEE_RL[sampled],filename+actMode+'RL',isEPS=False)
    # BM PV
    for l in range(1,env.L+1):
        plot_UE_BS_distribution_Cache(env, poolCL_BM1[l][sampled], poolCA_BM1[l][sampled], poolEE_BM1[l][sampled],filename+'BM1_L'+str(l),isEPS=False)
        plot_UE_BS_distribution_Cache(env, poolCL_BM2[l][sampled], poolCA_BM2[l][sampled], poolEE_BM2[l][sampled],filename+'BM2_L'+str(l),isEPS=False)
        plot_UE_BS_distribution_Cache(env, poolCL_BM3[l][sampled], poolCA_BM3[l][sampled], poolEE_BM3[l][sampled],filename+'BM3_L'+str(l),isEPS=False)
    
    # Save Line
    filename = 'data/'+env.TopologyCode+'/EvaluationPhase/'+'['+ str(randSEED) +']'+ env.TopologyName +'_Evaluation_'
    if env.B==4 and env.U ==4 and env.F==5 and env.N==2:
        with open(filename+ 'BF.pkl', 'wb') as f:  
            pickle.dump([env, poolEE_BF,poolTP_BF,poolPsys_BF,poolHR_BF,poolMCAP_BF,poolMCCPU_BF], f)
    with open(filename+ actMode +'RL.pkl', 'wb') as f:  
        pickle.dump([env, poolEE_RL,poolTP_RL,poolPsys_RL,poolHR_RL,poolMCAP_RL,poolMCCPU_RL,poolLossActor,poolLossCritic], f)
    with open(filename+'BM1.pkl', 'wb') as f:
        pickle.dump([env, poolEE_BM1,poolTP_BM1,poolPsys_BM1,poolHR_BM1,poolMCAP_BM1,poolMCCPU_BM1], f)
    with open(filename+'BM2.pkl', 'wb') as f:
        pickle.dump([env, poolEE_BM2,poolTP_BM2,poolPsys_BM2,poolHR_BM2,poolMCAP_BM2,poolMCCPU_BM2], f)
    with open(filename+'BM3.pkl', 'wb') as f:
        pickle.dump([env, poolEE_BM3,poolTP_BM3,poolPsys_BM3,poolHR_BM3,poolMCAP_BM3,poolMCCPU_BM3], f)

    # Calculate Loss Count
    lossCount = 0
    for i in range(len(poolEE_RL)):
        if poolEE_RL[i] < poolEE_BM1[1][i]:
            lossCount+=1
    return lossCount

def getEE_RL(env,actMode,ddpg_s=None,ddpg_cl=None,ddpg_ca=None):
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
    return EE_RL, RL_CLPolicy_UE,RL_CAPolicy_BS

if __name__ == '__main__':
    
    actMode = '1act'
    tryCount = 20
    lossCountVec = [99]*tryCount
    # Good case: 4.4.5.2 [0,2,5,12,19,25,26,36,39,43] / 10.5.20.2 [0,28]
    #for randSEED in [0,2,5,12,19,25,26,36,39,43]
    #for randSEED in [2,19,25]:
    envSeed = 11
    for randSEED in range(0,tryCount):
        print('randSEED:',randSEED)
        #####################  hyper parameters  ####################
        # Random Seed
        np.random.seed(randSEED)
        torch.manual_seed(randSEED)
        torch.cuda.manual_seed_all(randSEED)
        
        # Training Phase--------
        # new ENV
        #env = BS(nBS=4,nUE=4,nMaxLink=2,nFile=5,nMaxCache=2,loadENV = True,SEED=0)
        env = BS(nBS=10,nUE=5,nMaxLink=3,nFile=20,nMaxCache=2,loadENV=True,SEED=envSeed)
        trainModel(env,actMode=actMode,changeReq=False, changeChannel=True, loadActor = False,randSEED=randSEED) 
        filename = 'data/'+env.TopologyCode+'/TrainingPhase/'+'['+ str(randSEED) +']'+ env.TopologyName +str(MAX_EPISODES*MAX_EP_STEPS)+'_Train_'
        plotHistory(env,filename,isPlotEE=True,isPlotTP=True,isPlotPsys=True,isPlotHR=True,isEPS=False,loadBF=False)
        
        #==============================================================================================
        # Evaluation Phase------
        # new ENV
        #env = BS(nBS=4,nUE=4,nMaxLink=2,nFile=5,nMaxCache=2,loadENV = True,SEED=0)
        env = BS(nBS=10,nUE=5,nMaxLink=3,nFile=20,nMaxCache=2,loadENV=True,SEED=envSeed)
        '''
        # shift time slot
        for i in range(15):
            env.timeVariantChannel()
        '''
        lossCount = evaluateModel(env,actMode=actMode, nItr=100,randSEED=randSEED)
        lossCountVec[randSEED] = lossCount
        
        # plot performance
        filename = 'data/'+env.TopologyCode+'/EvaluationPhase/'+'['+ str(randSEED) +']'+ env.TopologyName +'_Evaluation_'
        plotHistory(env,filename,isPlotEE=True,isPlotTP=True,isPlotPsys=True,isPlotHR=True,isEPS=False,loadBF=False)
        #plotHistory(env,filename,isPlotEE=True,isPlotTP=True,isPlotPsys=True,isPlotHR=True,isEPS=False,loadBF=True)

    good_index = [i for i, x in enumerate(lossCountVec) if x<99 ]
    #==============================================================================================
    # plot Evaluation Final for 4.4.5.2

    #==============================================================================================    
    # plot Evaluation Final for 10.5.20.2
    
    #==============================================================================================
    # multi-instance training
    '''
    nJob=2
    #with concurrent.futures.ProcessPoolExecutor(max_workers= (num_cores-2) ) as executor:
    with concurrent.futures.ProcessPoolExecutor(max_workers= nJob ) as executor:
        futures = []
        for i in range(nJob):
            future = executor.submit(trainModel,env,actMode=actMode,changeReq=False, changeChannel=True, loadActor = False,randSEED=i)
            futures.append(future)
        for future in tqdm(concurrent.futures.as_completed(futures),total=len(futures),bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            #print(future.result())
            randSEED = future.result()
            print('\n randSEED: ',randSEED,' is Completed')
    for randSEED in range(nJob):
        filename = 'data/'+env.TopologyCode+'/TrainingPhase/'+'['+ str(randSEED) +']'+ env.TopologyName +str(MAX_EPISODES*MAX_EP_STEPS)+'_Train_'
        plotHistory(filename,isPlotEE=True,isPlotTP=True,isPlotPsys=True,isPlotHR=True,isEPS=False)
    '''