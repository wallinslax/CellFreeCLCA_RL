#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 14:27:22 2019

@author: fian20819
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 17:00:16 2019

@author: fian20819
"""
import torch
from torch.autograd import Variable
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import copy
from numpy.random import randn
from random import randint
from multi_ddpg6 import weightSync, OrnsteinUhlenbeckProcess, \
                        AdaptiveParamNoiseSpec, ddpg_distance_metric, hard_update, \
                        Replay, critic, actor, DDPG, Replay_, DDPG_ ,ca_Replay,cl_Replay

from env14 import BS,plot_UE_SBS_association,UE_SBS_location_distribution,plot_UE_SBS_distribution   
import os
import csv
os.environ['CUDA_VISIBLE_DEVICES']='1'
# make variable types for automatic setting to GPU or CPU, depending on GPU availability
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
#####################  hyper parameters  ####################
LOAD_EVN = True
RESET_CHANNEL = True
REQUEST_DUPLICATE = False
          
MAX_EPISODES = 100
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
radius = 0.5 # radius of the network = 500m
lambda_bs = 3 # density of the BS
lambda_u = 10 # density of the UE
#num_bs = 3 # number of BSs
#num_u = 10 # number of UEs

h_var = 1 # channel variance
k = 1 # pathloss coefficient
alpha = 2 # pathloss exponent

d_th = 0.2*1000 # distance theshold for clustering policy candidate 200m

W = 10*15*10**3 # BW owned by BS = 10*15kHz 
M = 4 # number of subcarrier owned by BS

F = 5 # number of total files
N = 2 # capacity of BSs
beta = [0.5,1,2] # zipf parameter
n_p = len(beta) # categories of user request content popularity

rank = np.arange(F)
np.random.shuffle(rank)
rank2 = np.arange(F)
np.random.shuffle(rank2)
rank3 = np.arange(F)
np.random.shuffle(rank3)
rank_list = [rank,rank2,rank3]

P_SBS = 20 # transmit power of SBS
#P_SBS = 30 # transmit power of SBS = 30mW
P_MBS = 66 # transmit power of MBS = 66mW
P_l = 20 # data retrieval power from local cache = 20mW
P_b = 500 # data retrieval power from backhaul = 500mW
P_o_SBS = 1500 # operational power of SBS = 1500mW
P_o_MBS = 2500 # operational power of MBS =2500mW
n_var = 2*(10**-13) # thermal noise power =  -127 (dBm) 
#n_var = 1#*(10**-13) # thermal noise power
#####################################
###############################  training  ####################################
def plot(category,
        file_name,
        list1,
        list2,
        list3,
        list4,
        list5,
#        list6,
#        list7,
#        list8,
        n_realizations):
    color =np.array( ['blue', 'black', 'deepskyblue', 'gray', 'red','pink','fuchsia','darkorange'])
    plt.plot(list1, label='[1]', c = color[0])
    plt.plot(list2, label='[2]', c = color[1])
    plt.plot(list3, label='[3]', c = color[2])
    plt.plot(list4, label='[4]', c = color[3])
    plt.plot(list5, label='[5]', c = color[4])
#    plt.plot(list6, label='[6]', c = color[5])
#    plt.plot(list7, label='[7]', c = color[6])
#    plt.plot(list8, label='[8]', c = color[7])
    
    plt.legend(loc='upper left')
    plt.title(str(category)+'(average over ' +str(n_realizations)+' realizations)')
    plt.ylabel(str(category))
    plt.xlabel('training steps')
    plt.savefig(file_name) 
    plt.show()

def writecsv(file_name,
             data1, 
             data2,
             data3,
             data4,
             data5,
#             data6,
#             data7,
#             data8
             ):
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data1)
        writer.writerow(data2)
        writer.writerow(data3)
        writer.writerow(data4)
        writer.writerow(data5)
#        writer.writerow(data6)
#        writer.writerow(data7)
#        writer.writerow(data8)

def writeinfo(f,env1,env_index):
  f.write('===================env'+str(env_index)+'===================\n')
  f.write('RL_clustering_policy_num = ')
  f.write(str(RL_clustering_policy_num[env_index-1])+'\n\n')
  f.write('Cluster size of each BS (cluster_size) = ')
  f.write(str(env1.cluster_size)+'\n\n')
  f.write('Cluster member of each BS (cluster_member) = \n')
  f.write(str(env1.cluster_member)+'\n\n')
  f.write('RL_caching_policy_num = ')
  f.write(str(RL_caching_policy_num[env_index-1])+'\n\n')
  f.write('The cached files of each BS (f2b) = \n')
  f.write(str(np.where(env1.C[RL_caching_policy_num[env_index-1]]==1)[1].reshape((env1.B,N)))+'\n\n')
  f.write('Most N popular content idx in each cluster  = ')
  f.write(str(np.where(env1.C[env1.opt_caching_number.astype(int)]==1)[1].reshape((env1.B,N)))+'\n\n')
  f.write('The content request of each user (Req) = ')
  f.write(str(env1.Req)+'\n\n')
  f.write('Hit of each UE (Hit) = ')
  f.write(str(env1.Hit)+'\n\n')
  f.write('Average Hit rate of each UE (Hit_rate) = ')
  f.write(str(env1.Hit_rate)+'\n\n')
  f.write('User-subcarrier association (us) = ')
  f.write(str(env1.us)+'\n\n')
  f.write('Interference of each UE (I) = \n')
  f.write(str(env1.I)+'\n\n')
  f.write('Transmission power of associated BS of each UE (P_bs_) = ')
  f.write(str(env1.P_bs_)+'\n\n')
  f.write('Received power from associated BS of each UE (P_r) = ')
  f.write(str(env1.P_r)+'\n\n')
  f.write('SINR of each UE (SINR) = \n')
  f.write(str(env1.SINR)+'\n\n')
  f.write('Spectral efficiency of each UE (T) = \n')
  f.write(str(env1.T)+'\n\n')
  f.write('System power consumption (P_sys) = ')
  f.write(str(env1.P_sys)+'\n\n')
  f.write('Energy efficiency (E) = ')
  f.write(str(env1.E*EE_std+EE_mean)+'\n\n')
  f.write('Cluster similarity of each BS (ICS) = \n')
  f.write(str(env1.ICS)+'\n\n')
  f.write('Average cluster similarity of each BS (CS) = ')
  f.write(str(env1.CS*CS_std+CS_mean)+'\n\n')
  f.write('Cluster size constraint penalty (csc_penalty) = ')
  f.write(str(env1.csc_penalty)+'\n\n')

# save bs_coordinate,u_coordinate,B,U,P
def save_env_distribution(bs_coordinate,u_coordinate,B,U,P,up,EE_mean,EE_std,CS_mean,CS_std,pl):
    with open('./data/env_distribution.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([B])
        writer.writerow([U])
        for row in bs_coordinate:
            writer.writerow(row)
        for row in u_coordinate:
            writer.writerow(row)
        for row in P:
            writer.writerow(row)
        writer.writerow(up)
        writer.writerow([EE_mean])
        writer.writerow([EE_std])
        writer.writerow([CS_mean])
        writer.writerow([CS_std])
        for row in pl:
            writer.writerow(row)
        
def load_env_distribution():
    temp = []
    with open('./data/env_distribution.csv', 'r', newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            temp.append(np.array(row).astype(np.float))
            print(np.array(row).astype(np.float))
        B = int(temp[0][0])
        U = int(temp[1][0])
        bs_coordinate = np.array(temp[2:2+B])
        u_coordinate = np.array(temp[2+B:2+B+U])
        P = np.array(temp[2+B+U:2+B+U+U])
        up = np.array(temp[2+B+U+U]).astype(int)
        EE_mean = temp[2+B+U+U+1][0]
        EE_std = temp[2+B+U+U+2][0]
        CS_mean = temp[2+B+U+U+3][0]
        CS_std = temp[2+B+U+U+4][0]
        pl = np.array(temp[2+B+U+U+5:2+B+U+U+5+U])
        
        
    '''
    tmp = np.loadtxt("env_distribution.csv", dtype=np.str, delimiter=",")
    B = tmp[0,:].astype(np.float)
    U = tmp[1,:].astype(np.float)
    bs_coordinate = tmp[2:3+B,:].astype(np.float)
    u_coordinate = tmp[3+B:4+B+U,:].astype(np.float)
    P = tmp[4+B+U:,:].astype(np.float)
    '''
    return bs_coordinate,u_coordinate,B,U,P,up,EE_mean,EE_std,CS_mean,CS_std,pl    

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

def single_ddpg_learn(RL_s,
                      cl_ddpg,
                      ca_ddpg,
                      env,
                      cl_memory,
                      ca_memory,
                      total_reward,
                      total_ee,
                      total_cs,
                      total_csc_penalty,
                      total_hit,
                      total_itf,
                      step_counter,
                      ):
    
    cl_ddpg.noise.reset() # reset actor OU noise
    ca_ddpg.noise.reset() # reset actor OU noise
    
    cl_RL_a = cl_ddpg.action(RL_s, cl_ddpg.noise.step(), None).astype(int)# choose action
    ca_RL_a = ca_ddpg.action(RL_s, ca_ddpg.noise.step(), None).astype(int)# choose action
    
    RL_a = np.hstack((cl_RL_a,ca_RL_a))
    
    RL_clustering_policy_num,RL_caching_policy_num = env.action_vec_2_num(RL_a)
    RL_r, done,RL_EE, RL_CS, RL_csc_penalty, RL_SINR, RL_Hit_rate, RL_s_, RL_Itf = env.step(clustering_policy_num = RL_clustering_policy_num,
                                                                               caching_policy_num = RL_caching_policy_num, 
                                                                               update_Req = 1, 
                                                                               normalize = 1,
                                                                               Req_duplicate = REQUEST_DUPLICATE)
                                                                          
    cl_memory.add((RL_s,cl_RL_a,RL_r,RL_s_))
    ca_memory.add((RL_s,ca_RL_a,RL_Hit_rate,RL_s_))
    
    
    total_reward += RL_r
    total_ee += RL_EE*EE_std+EE_mean
    total_cs += RL_CS*CS_std+CS_mean
    total_csc_penalty += RL_csc_penalty
    total_hit += RL_Hit_rate
    total_itf += RL_Itf
    step_counter += 1
    
    cl_training_data = np.array(cl_memory.sample(BATCH_SIZE))
    ca_training_data = np.array(ca_memory.sample(BATCH_SIZE))
    cl_ddpg.train(cl_training_data)
    ca_ddpg.train(ca_training_data)
      
    
    return (RL_s,
            done,
            RL_clustering_policy_num,
            RL_caching_policy_num,
            cl_ddpg,
            ca_ddpg,
            env,
            cl_memory,
            ca_memory,
            total_reward,
            total_ee,
            total_cs,
            total_csc_penalty,
            total_hit,
            total_itf,
            step_counter)

def ddpg_test(RL_s,
              cl_ddpg,
              ca_ddpg,
              env,
              total_reward,
              total_ee,
              total_cs,
              total_csc_penalty,
              total_hit,
              total_itf,
              step_counter,
              ):
    
    cl_RL_a = cl_ddpg.action(RL_s, None, None).astype(int)# choose action
    ca_RL_a = ca_ddpg.action(RL_s, None, None).astype(int)# choose action
    
    RL_a = np.hstack((cl_RL_a,ca_RL_a))
    
    RL_clustering_policy_num,RL_caching_policy_num = env.action_vec_2_num(RL_a)
    RL_r, done,RL_EE, RL_CS, RL_csc_penalty, RL_SINR, RL_Hit_rate, RL_s_, RL_Itf = env.step(clustering_policy_num = RL_clustering_policy_num,
                                                                               caching_policy_num = RL_caching_policy_num, 
                                                                               update_Req = 1, 
                                                                               normalize = 1,
                                                                               Req_duplicate = REQUEST_DUPLICATE)
    total_reward += RL_r
    total_ee += RL_EE*EE_std+EE_mean
    total_cs += RL_CS*CS_std+CS_mean
    total_csc_penalty += RL_csc_penalty
    total_hit += RL_Hit_rate
    total_itf += RL_Itf
    step_counter += 1
    RL_s = RL_s_        
    return (RL_s,
            done,
            RL_clustering_policy_num,
            RL_caching_policy_num,
            env,
            total_reward,
            total_ee,
            total_cs,
            total_csc_penalty,
            total_hit,
            total_itf,
            step_counter)
    
    
if __name__ == '__main__':
    ############################################
    #
    
    reward_list = []
    EE_list = []
    CS_list = []
    csc_penalty_list = []
    Hit_list = []
    Itf_list = []
    
    reward_list2 = []
    EE_list2 = []
    CS_list2 = []
    csc_penalty_list2 = []
    Hit_list2 = []
    Itf_list2 = []
    
    reward_list3 = []
    EE_list3 = []
    CS_list3 = []
    csc_penalty_list3 = []
    Hit_list3 = []
    Itf_list3 = []
    
    reward_list4 = []
    EE_list4 = []
    CS_list4 = []
    csc_penalty_list4 = []
    Hit_list4 = []
    Itf_list4 = []
    
    reward_list5 = []
    EE_list5 = []
    CS_list5 = []
    csc_penalty_list5 = []
    Hit_list5 = []
    Itf_list5 = []
    
    reward_list6 = []
    EE_list6 = []
    CS_list6 = []
    csc_penalty_list6 = []
    Hit_list6 = []
    Itf_list6 = []
    
    reward_list7 = []
    EE_list7 = []
    CS_list7 = []
    csc_penalty_list7 = []
    Hit_list7 = []
    Itf_list7 = []
    
    reward_list8 = []
    EE_list8 = []
    CS_list8 = []
    csc_penalty_list8 = []
    Hit_list8 = []
    Itf_list8 = []

    ############################################
    if not LOAD_EVN :
        '''[1] SBS/ UE distribution'''  
        #     
        u_coordinate = UE_SBS_location_distribution(lambda0 = lambda_u) # UE
        U = len(u_coordinate)
        #
        #B = num_bs+1
        sbs_coordinate = UE_SBS_location_distribution(lambda0 = lambda_bs) # SBS
        bs_coordinate = np.concatenate((np.array([[0,0]]),sbs_coordinate),axis=0)
        B = len(bs_coordinate)
        
        '''[2] Pair-wise distance'''
        D = np.zeros((U,B))
        for u in  range(U):
            for b in range(B):
                D[u][b] = 1000*np.sqrt(sum((u_coordinate[u]-bs_coordinate[b])**2)) # km -> m
        
        '''[3] Channel gain'''
        pl = k*np.power(D, -alpha) # Path-loss
        P = channel_reset(U,B,pl)
        
        # user-popularity association
        up = np.array([randint(0, n_p-1) for _ in range(U)])
        
        #
        env = BS(bs_coordinate,u_coordinate,B,U,P,up,REQUEST_DUPLICATE,True,None,None,None,None)
        EE_mean,EE_std,CS_mean,CS_std = env.get_statistic(REQUEST_DUPLICATE)
        
        save_env_distribution(bs_coordinate,u_coordinate,B,U,P,up,EE_mean,EE_std,CS_mean,CS_std,pl)
    else :
        bs_coordinate,u_coordinate,B,U,P,up,EE_mean,EE_std,CS_mean,CS_std,pl = load_env_distribution()
#        up = np.array([0,0,0,1,1,1,2,2,2])
        '''[2] Pair-wise distance'''
        D = np.zeros((U,B))
        for u in  range(U):
            for b in range(B):
                D[u][b] = 1000*np.sqrt(sum((u_coordinate[u]-bs_coordinate[b])**2)) # km -> m
        env = BS(bs_coordinate,u_coordinate,B,U,P,up,REQUEST_DUPLICATE,False,EE_mean,EE_std,CS_mean,CS_std)   
    ############################################ 
    near_clustering_policy = np.zeros(env.U).astype(int)
    for u in  range(U):
        near_clustering_policy[u] = np.argmin(D[u])
    ############################################ 
    env1 = BS(bs_coordinate,u_coordinate,B,U,P,up,REQUEST_DUPLICATE,False,EE_mean,EE_std,CS_mean,CS_std)
    
    env2 = BS(bs_coordinate,u_coordinate,B,U,P,up,REQUEST_DUPLICATE,False,EE_mean,EE_std,CS_mean,CS_std)
    env3 = BS(bs_coordinate,u_coordinate,B,U,P,up,REQUEST_DUPLICATE,False,EE_mean,EE_std,CS_mean,CS_std)
    env4 = BS(bs_coordinate,u_coordinate,B,U,P,up,REQUEST_DUPLICATE,False,EE_mean,EE_std,CS_mean,CS_std)
    
    env5 = BS(bs_coordinate,u_coordinate,B,U,P,up,REQUEST_DUPLICATE,False,EE_mean,EE_std,CS_mean,CS_std)
    env6 = BS(bs_coordinate,u_coordinate,B,U,P,up,REQUEST_DUPLICATE,False,EE_mean,EE_std,CS_mean,CS_std)
    env7 = BS(bs_coordinate,u_coordinate,B,U,P,up,REQUEST_DUPLICATE,False,EE_mean,EE_std,CS_mean,CS_std)
    
    obs_dim = len(env.SINR)+len(env.s_)+len(env.Prof_state)
    act_dim = (env.U*env.B)+(env.B*env.F)
    
    cluster_act_dim = (env.U*env.B)
    cache_act_dim = (env.B*env.F)
    
    cl_ddpg1 = DDPG(obs_dim = obs_dim, act_dim = cluster_act_dim)
    
    cl_ddpg2 = DDPG(obs_dim = obs_dim, act_dim = cluster_act_dim)
    cl_ddpg3 = DDPG(obs_dim = obs_dim, act_dim = cluster_act_dim)
    cl_ddpg4 = DDPG(obs_dim = obs_dim, act_dim = cluster_act_dim)
    
    cl_memory1 = cl_Replay(env=env,maxlen=MEM_SIZE)
    cl_memory1.initialize(init_length= 150,Req_duplicate=REQUEST_DUPLICATE)
    cl_memory2 = cl_Replay(env=env,maxlen=MEM_SIZE) 
    cl_memory2.initialize(init_length= 150,Req_duplicate=REQUEST_DUPLICATE)
    
    ca_ddpg1 = DDPG(obs_dim = obs_dim, act_dim = cache_act_dim)
    
    ca_ddpg2 = DDPG(obs_dim = obs_dim, act_dim = cache_act_dim) 
    ca_ddpg3 = DDPG(obs_dim = obs_dim, act_dim = cache_act_dim)
    ca_ddpg4 = DDPG(obs_dim = obs_dim, act_dim = cache_act_dim)
    
    ca_memory1 = ca_Replay(env=env,maxlen=MEM_SIZE)
    ca_memory1.initialize(init_length= 150,Req_duplicate=REQUEST_DUPLICATE)
    ca_memory2 = ca_Replay(env=env,maxlen=MEM_SIZE) 
    ca_memory2.initialize(init_length= 150,Req_duplicate=REQUEST_DUPLICATE)

    
    
    env_list = [env1,env2,env3,env4,env5,env6,env7]
    cl_ddpg_list = [cl_ddpg1,cl_ddpg2,cl_ddpg3,cl_ddpg4]
    cl_memory_list = [cl_memory1,cl_memory2]
    ca_ddpg_list = [ca_ddpg1,ca_ddpg2,ca_ddpg3,ca_ddpg4]
    ca_memory_list = [ca_memory1,ca_memory2]
    
    t1 = time.time()
#    f = open('./data/info.txt','w')
#    f.write('Number of BS (B) = ')
#    f.write(str(env1.B)+'\n\n')
#    f.write('Number of UE (U) = ') 
#    f.write(str(env1.U)+'\n\n')
#    f.write('Number of total content (F) = ')
#    f.write(str(env1.F)+'\n\n')
#    f.write('User-popularity association (up) = ')
#    f.write(str(env1.up)+'\n\n')
#    f.write('Content popularity of each file (zipf_content_popularity) = \n')
#    f.write(str(env1.zipf_content_popularity)+'\n\n')
#    f.write('Most N popular content idx (top_N_idx) = ')
#    f.write(str(env1.top_N_idx)+'\n\n')
    step = 0
    idx_1 = 0
    ep = 0
    while True:
#    for ep in range(MAX_EPISODES):
        '''
        [1]single DDPG 
        [2]multi-DDPG
        [3]opt popularity-based caching & RL clustering
        [4]opt popularity-based caching & random clustering
        [5]opt popularity-based caching & near clustering
        '''
        total_reward = [0 for i in range(len(env_list))]
        step_counter = [0 for i in range(len(env_list))] # distributor agent's episode length (related to convergence rate)
        done = [0 for i in range(len(env_list))]
        total_ee = [0 for i in range(len(env_list))]
        total_cs = [0 for i in range(len(env_list))]
        total_csc_penalty = [0 for i in range(len(env_list))]
        total_hit = [0 for i in range(len(env_list))]
        total_itf = [0 for i in range(len(env_list))]
        RL_clustering_policy_num = [0 for i in range(len(env_list))]
        RL_caching_policy_num = [0 for i in range(len(env_list))]
        # get initial state
        RL_s = [env_list[i].reset() for i in range(len(env_list))]
        
        
        ####################################################################
#        # reset actor OU noise
#        for i in range(len(ddpg_list)):
#            ddpg_list[i].noise.reset() # reset actor OU noise
        ####################################################################
        step = 0
        while True:
            step += 1
            
            '''[1]single DDPG'''
            if not done[0]:
                (RL_s[0],done[0],RL_clustering_policy_num[0],RL_caching_policy_num[0],\
                cl_ddpg_list[0],ca_ddpg_list[0],env_list[0],cl_memory_list[0],ca_memory_list[0],\
                total_reward[0],total_ee[0],total_cs[0],\
                total_csc_penalty[0],total_hit[0],total_itf[0],\
                step_counter[0]) = \
                single_ddpg_learn(RL_s = RL_s[0], 
                                  cl_ddpg = cl_ddpg_list[0],
                                  ca_ddpg = ca_ddpg_list[0],
                                  env = env_list[0],
                                  cl_memory = cl_memory_list[0],
                                  ca_memory = ca_memory_list[0],
                                  total_reward = total_reward[0],
                                  total_ee = total_ee[0],
                                  total_cs = total_cs[0],
                                  total_csc_penalty = total_csc_penalty[0],
                                  total_hit = total_hit[0],
                                  total_itf = total_itf[0],
                                  step_counter = step_counter[0],
                                  )
            
            
            '''[2]multi-DDPG'''
            if not done[1]:
                (RL_s[1],done[1],RL_clustering_policy_num[1],RL_caching_policy_num[1],\
                cl_ddpg_list[1],ca_ddpg_list[1],env_list[1],cl_memory_list[1],ca_memory_list[1],\
                total_reward[1],total_ee[1],total_cs[1],\
                total_csc_penalty[1],total_hit[1],total_itf[1],\
                step_counter[1]) = \
                single_ddpg_learn(RL_s = RL_s[1],
                                  cl_ddpg = cl_ddpg_list[1],
                                  ca_ddpg = ca_ddpg_list[1],
                                  env = env_list[1],
                                  cl_memory = cl_memory_list[1],###
                                  ca_memory = ca_memory_list[1],####
                                  total_reward = total_reward[1],
                                  total_ee = total_ee[1],
                                  total_cs = total_cs[1],
                                  total_csc_penalty = total_csc_penalty[1],
                                  total_hit = total_hit[1],
                                  total_itf = total_itf[1],
                                  step_counter = step_counter[1],
                                  )
            if not done[2]:
                (RL_s[2],done[2],RL_clustering_policy_num[2],RL_caching_policy_num[2],\
                cl_ddpg_list[2],ca_ddpg_list[2],env_list[2],cl_memory_list[1],ca_memory_list[1],\
                total_reward[2],total_ee[2],total_cs[2],\
                total_csc_penalty[2],total_hit[2],total_itf[2],\
                step_counter[2]) = \
                single_ddpg_learn(RL_s = RL_s[2],
                                  cl_ddpg = cl_ddpg_list[2],
                                  ca_ddpg = ca_ddpg_list[2],
                                  env = env_list[2],
                                  cl_memory = cl_memory_list[1],###
                                  ca_memory = ca_memory_list[1],###
                                  total_reward = total_reward[2],
                                  total_ee = total_ee[2],
                                  total_cs = total_cs[2],
                                  total_csc_penalty = total_csc_penalty[2],
                                  total_hit = total_hit[2],
                                  total_itf = total_itf[2],
                                  step_counter = step_counter[2],
                                  )
            if not done[3]:
                (RL_s[3],done[3],RL_clustering_policy_num[3],RL_caching_policy_num[3],\
                cl_ddpg_list[3],ca_ddpg_list[3],env_list[3],cl_memory_list[1],ca_memory_list[1],\
                total_reward[3],total_ee[3],total_cs[3],\
                total_csc_penalty[3],total_hit[3],total_itf[3],\
                step_counter[3]) = \
                single_ddpg_learn(RL_s = RL_s[3],
                                  cl_ddpg = cl_ddpg_list[3],
                                  ca_ddpg = ca_ddpg_list[3],
                                  env = env_list[3],
                                  cl_memory = cl_memory_list[1],###
                                  ca_memory = ca_memory_list[1],###
                                  total_reward = total_reward[3],
                                  total_ee = total_ee[3],
                                  total_cs = total_cs[3],
                                  total_csc_penalty = total_csc_penalty[3],
                                  total_hit = total_hit[3],
                                  total_itf = total_itf[3],
                                  step_counter = step_counter[3],
                                  )                       
            '''[3]opt popularity-based caching & RL clustering'''
            if not done[4]:
                r1, done[4],EE1, CS1, csc_penalty1, SINR1, Hit_rate1, s1_, Itf1= env_list[4].step(clustering_policy_num = RL_clustering_policy_num[idx_1],
                                                                                                     caching_policy_num = (env_list[4].opt_caching_number).astype(int),
                                                                                                     update_Req = 1, 
                                                                                                     normalize = 1,
                                                                                                     Req_duplicate = REQUEST_DUPLICATE)
                
                total_reward[4] += r1
                total_ee[4] += EE1*EE_std+EE_mean
                total_cs[4] += CS1*CS_std+CS_mean
                total_csc_penalty[4] += csc_penalty1
                total_hit[4] += Hit_rate1
                total_itf[4] += Itf1
                step_counter[4] += 1
                RL_s[4] = s1_
            
            '''[4]opt popularity-based caching & random clustering'''
            if not done[5]:
                r, done[5],EE, CS, csc_penalty, SINR, Hit_rate, s_, Itf= env_list[5].step(clustering_policy_num = np.random.randint(0,env_list[5].B-1,size=env_list[5].U),#RL_clustering_policy_num[idx_1],
                                                                                              caching_policy_num = (env_list[5].opt_caching_number).astype(int),
                                                                                              update_Req = 1, 
                                                                                              normalize = 1,
                                                                                              Req_duplicate = REQUEST_DUPLICATE)
                total_reward[5] += r
                total_ee[5] += EE*EE_std+EE_mean
                total_cs[5] += CS*CS_std+CS_mean
                total_csc_penalty[5] += csc_penalty
                total_hit[5] += Hit_rate
                total_itf[5] += Itf
                step_counter[5] += 1
                RL_s[5] = s_
            
                    
            '''[5]opt popularity-based caching & near clustering'''
            if not done[6]:
                RL_clustering_policy_num[6] = near_clustering_policy
                RL_caching_policy_num[6] = (env_list[6].opt_caching_number).astype(int)
                r, done[6],EE, CS, csc_penalty, SINR, Hit_rate, s_, Itf= env_list[6].step(clustering_policy_num = RL_clustering_policy_num[6],
                                                                                              caching_policy_num = RL_caching_policy_num[6],
                                                                                              update_Req = 1, 
                                                                                              normalize = 1,
                                                                                              Req_duplicate = REQUEST_DUPLICATE)
                total_reward[6] += r
                total_ee[6] += EE*EE_std+EE_mean
                total_cs[6] += CS*CS_std+CS_mean
                total_csc_penalty[6] += csc_penalty
                total_hit[6] += Hit_rate
                total_itf[6] += Itf
                step_counter[6] += 1
                RL_s[6] = s_
                
            if (done[0] and 
                done[1] and 
                done[2] and 
                done[3] and 
                done[4] and 
                done[5] and
                done[6])or(step > MAX_EP_STEPS):
                break

        # select distributor agent among N agents based on total reward of each agent
        idx_1 = np.array(total_reward[1:4]).argmax()
        cl_ddpgs_1 = cl_ddpg_list[1:4]
        cl_myddpg_1 = cl_ddpgs_1[idx_1] # distributor agent 
        ca_ddpgs_1 = ca_ddpg_list[1:4]
        ca_myddpg_1 = ca_ddpgs_1[idx_1] # distributor agent 
        idx_1 += 1
        
        # print information
        print('myddpg_total_reward = '+str(total_reward[idx_1]))
        RL_clustering_policy_num_ = RL_clustering_policy_num[idx_1]
        RL_caching_policy_num_ = RL_caching_policy_num[idx_1] 
        print(RL_clustering_policy_num_)
        print(RL_caching_policy_num_)
        
        
        # trace the average reward of the episode        
        reward_list.append(total_reward[0]/step_counter[0]) # average reward of the episode
        EE_list.append((total_ee[0])/step_counter[0])
        CS_list.append((total_cs[0])/step_counter[0])
        csc_penalty_list.append(total_csc_penalty[0]/step_counter[0])
        Hit_list.append(total_hit[0]/step_counter[0])
        Itf_list.append(total_itf[0]/step_counter[0])

        reward_list2.append(total_reward[idx_1]/step_counter[idx_1]) # average reward of the episode
        EE_list2.append((total_ee[idx_1])/step_counter[idx_1])
        CS_list2.append((total_cs[idx_1])/step_counter[idx_1])
        csc_penalty_list2.append(total_csc_penalty[idx_1]/step_counter[idx_1])
        Hit_list2.append(total_hit[idx_1]/step_counter[idx_1])
        Itf_list2.append(total_itf[idx_1]/step_counter[idx_1])
        
        reward_list3.append(total_reward[4]/step_counter[4]) # average reward of the episode
        EE_list3.append((total_ee[4])/step_counter[4])
        CS_list3.append((total_cs[4])/step_counter[4])
        csc_penalty_list3.append(total_csc_penalty[4]/step_counter[4])
        Hit_list3.append(total_hit[4]/step_counter[4])
        Itf_list3.append(total_itf[4]/step_counter[4])
        
        reward_list4.append(total_reward[5]/step_counter[5]) # average reward of the episode
        EE_list4.append((total_ee[5])/step_counter[5])
        CS_list4.append((total_cs[5])/step_counter[5])
        csc_penalty_list4.append(total_csc_penalty[5]/step_counter[5])
        Hit_list4.append(total_hit[5]/step_counter[5])
        Itf_list4.append(total_itf[5]/step_counter[5])
        
        reward_list5.append(total_reward[6]/step_counter[6]) # average reward of the episode
        EE_list5.append((total_ee[6])/step_counter[6])
        CS_list5.append((total_cs[6])/step_counter[6])
        csc_penalty_list5.append(total_csc_penalty[6]/step_counter[6])
        Hit_list5.append(total_hit[6]/step_counter[6])
        Itf_list5.append(total_itf[6]/step_counter[6])
        
        # copy weights of the distributor agent to N agents
        cl_ddpg_list[1] = copy.deepcopy(cl_myddpg_1)
        cl_ddpg_list[2] = copy.deepcopy(cl_myddpg_1)
        cl_ddpg_list[3] = copy.deepcopy(cl_myddpg_1)
        
        ca_ddpg_list[1] = copy.deepcopy(ca_myddpg_1)
        ca_ddpg_list[2] = copy.deepcopy(ca_myddpg_1)
        ca_ddpg_list[3] = copy.deepcopy(ca_myddpg_1)
 
        # reset small-scale fading
        if (RESET_CHANNEL):
            P = channel_reset(U,B,pl)
            for i in range(len(env_list)):
                env_list[i].P = P
        
        # learning rate decay
        if (ep>100):
            LR_A=LR_A/10
            LR_C=LR_C/10
            for i in range(len(cl_ddpg_list)):
                cl_ddpg_list[i].LR_A = LR_A
                cl_ddpg_list[i].LR_C = LR_C
                ca_ddpg_list[i].LR_A = LR_A
                ca_ddpg_list[i].LR_C = LR_C
                
        if (ep > 30) :
           
            delta_EE = 0.0025
            delta_CS = 0.025
            delta_Hit = 0.025
            
            EE_mean_ = np.mean(EE_list[ep-30:])
            CS_mean_  = np.mean(CS_list[ep-30:])
            Hit_mean_  = np.mean(Hit_list[ep-30:])
            
            EE_not_cvg  = np.where(EE_list[ep-30:]>EE_mean_ +delta_EE)[0].size\
                        +np.where(EE_list[ep-30:]<EE_mean_ -delta_EE)[0].size
            CS_not_cvg  = np.where(CS_list[ep-30:]>CS_mean_ +delta_CS)[0].size\
                        +np.where(CS_list[ep-30:]<CS_mean_ -delta_CS)[0].size
            Hit_not_cvg  = np.where(Hit_list[ep-30:]>Hit_mean_ +delta_Hit)[0].size\
                        +np.where(Hit_list[ep-30:]<Hit_mean_ -delta_Hit)[0].size
            
            EE_mean_2 = np.mean(EE_list2[ep-30:])
            CS_mean_2  = np.mean(CS_list2[ep-30:])
            Hit_mean_2  = np.mean(Hit_list2[ep-30:])
            
            EE_not_cvg_2  = np.where(EE_list2[ep-30:]>EE_mean_2 +delta_EE)[0].size\
                        +np.where(EE_list2[ep-30:]<EE_mean_2 -delta_EE)[0].size
            CS_not_cvg_2  = np.where(CS_list2[ep-30:]>CS_mean_2 +delta_CS)[0].size\
                        +np.where(CS_list2[ep-30:]<CS_mean_2 -delta_CS)[0].size
            Hit_not_cvg_2  = np.where(Hit_list2[ep-30:]>Hit_mean_2 +delta_Hit)[0].size\
                        +np.where(Hit_list2[ep-30:]<Hit_mean_2 -delta_Hit)[0].size
            
            if ((EE_not_cvg  == 0)and
                (CS_not_cvg  == 0)and
                (Hit_not_cvg  == 0) and
                (EE_not_cvg_2  == 0)and
                (CS_not_cvg_2  == 0)and
                (Hit_not_cvg_2  == 0)
                ):
                print('mddpg training stable')
                break
                       
        if (ep>MAX_EPISODES):
            break
        ep += 1
  
    print('Running time: ', time.time() - t1)
#    f.close()
    ############################################
    # plot clustering result
    f2b=np.where(env.C[RL_caching_policy_num_]==1)[1].reshape((env.B,N))
    plot_UE_SBS_association(env.bs_coordinate[1:,0],env.bs_coordinate[1:,1],env.u_coordinate[:,0],env.u_coordinate[:,1],RL_clustering_policy_num_,env.Req,f2b,env.up,'./data/train_result[2].png','./data/train_result[2].csv')
    #
    plot('reward',
        './data/reward.png',
        reward_list,
        reward_list2,
        reward_list3,
        reward_list4,
        reward_list5,
        n_realizations)
    plot('EE',
        './data/EE.png',
        list1 = EE_list,
        list2 = EE_list2,
        list3 = EE_list3,
        list4 = EE_list4,
        list5 = EE_list5,
        n_realizations = n_realizations)
    plot('CS',
        './data/CS.png',
        CS_list,
        CS_list2,
        CS_list3,
        CS_list4,
        CS_list5,
        n_realizations)
    plot('csc_penalty',
        './data/csc_penalty.png',
        csc_penalty_list,
        csc_penalty_list2,
        csc_penalty_list3,
        csc_penalty_list4,
        csc_penalty_list5,
        n_realizations)
    plot('Hit_rate',
        './data/Hit_rate.png',
        Hit_list,
        Hit_list2,
        Hit_list3,
        Hit_list4,
        Hit_list5,
        n_realizations)
    plot('Itf',
        './data/Itf.png',
        Itf_list,
        Itf_list2,
        Itf_list3,
        Itf_list4,
        Itf_list5,
        n_realizations = n_realizations)
        
    #
    writecsv('./data/reward_data.csv',
             reward_list,
             reward_list2,
             reward_list3,
             reward_list4,
             reward_list5,
             )
    writecsv('./data/EE_data.csv',
             EE_list,
             EE_list2,
             EE_list3,
             EE_list4,
             EE_list5,
             )
    writecsv('./data/CS_data.csv',
             CS_list,
             CS_list2,
             CS_list3,
             CS_list4,
             CS_list5,
             )
    writecsv('./data/csc_penalty_data.csv',
             csc_penalty_list,
             csc_penalty_list2,
             csc_penalty_list3,
             csc_penalty_list4,
             csc_penalty_list5,
             )
    writecsv('./data/Hit_data.csv',
             Hit_list,
             Hit_list2,
             Hit_list3,
             Hit_list4,
             Hit_list5,
             )
    writecsv('./data/interference_data.csv', 
             Itf_list,
             Itf_list2,
             Itf_list3,
             Itf_list4,
             Itf_list5,
             )

    
    # clear the list
    reward_list.clear()
    EE_list.clear()
    CS_list.clear()
    csc_penalty_list.clear()
    Hit_list.clear()
    Itf_list.clear()
    
    reward_list2.clear()
    EE_list2.clear()
    CS_list2.clear()
    csc_penalty_list2.clear()
    Hit_list2.clear()
    Itf_list2.clear()
    
    reward_list3.clear()
    EE_list3.clear()
    CS_list3.clear()
    csc_penalty_list3.clear()
    Hit_list3.clear()
    Itf_list3.clear()
    
    reward_list4.clear()
    EE_list4.clear()
    CS_list4.clear()
    csc_penalty_list4.clear()
    Hit_list4.clear()
    Itf_list4.clear()
#    
    reward_list5.clear()
    EE_list5.clear()
    CS_list5.clear()
    csc_penalty_list5.clear()
    Hit_list5.clear()
    Itf_list5.clear()
    
    
    torch.save(cl_ddpg_list[idx_1].actor, './data/cl_mddpg_actor.pt')
    torch.save(ca_ddpg_list[idx_1].actor, './data/ca_mddpg_actor.pt')
    cl_ddpg_list[idx_1].actor = torch.load('./data/cl_mddpg_actor.pt')
    ca_ddpg_list[idx_1].actor = torch.load('./data/ca_mddpg_actor.pt')
    #
    torch.save(cl_ddpg_list[0].actor, './data/cl_sddpg_actor.pt')
    torch.save(ca_ddpg_list[0].actor, './data/ca_sddpg_actor.pt')
    cl_ddpg_list[0].actor = torch.load('./data/cl_sddpg_actor.pt')
    ca_ddpg_list[0].actor = torch.load('./data/ca_sddpg_actor.pt')
    ########################################### Testing ###########################################
    f = open('./data/info_test.txt','w')
    f.write('Number of BS (B) = ')
    f.write(str(env1.B)+'\n\n')
    f.write('Number of UE (U) = ') 
    f.write(str(env1.U)+'\n\n')
    f.write('Number of total content (F) = ')
    f.write(str(env1.F)+'\n\n')
    f.write('Cache size (N) = ')
    f.write(str(env1.N)+'\n\n')
    f.write('User-popularity association (up) = ')
    f.write(str(env1.up)+'\n\n')
    f.write('Content popularity of each file (zipf_content_popularity) = \n')
    f.write(str(env1.zipf_content_popularity)+'\n\n')
    
    for ep in range(100):
        '''
        [1]single DDPG 
        [2]multi-DDPG
        [3]popularity-based caching & random clustering
        [4]popularity-based caching & RL clustering
        [5]opt popularity-based caching & near clustering
        '''
        total_reward = [0 for i in range(len(env_list))]
        step_counter = [0 for i in range(len(env_list))] # distributor agent's episode length (related to convergence rate)
        done = [0 for i in range(len(env_list))]
        total_ee = [0 for i in range(len(env_list))]
        total_cs = [0 for i in range(len(env_list))]
        total_csc_penalty = [0 for i in range(len(env_list))]
        total_hit = [0 for i in range(len(env_list))]
        total_itf = [0 for i in range(len(env_list))]
        RL_clustering_policy_num = [0 for i in range(len(env_list))]
        RL_caching_policy_num = [0 for i in range(len(env_list))]
        # get initial state
        RL_s = [env_list[i].reset() for i in range(len(env_list))]
        
        
        step = 0
        
        # write info 
        f.write('===================episode['+str(ep)+']===================\n')
        while True:
            step += 1
            
            '''[1]single DDPG'''
            if not done[0]:
                (RL_s[0],done[0],RL_clustering_policy_num[0],RL_caching_policy_num[0],\
                env_list[0],\
                total_reward[0],total_ee[0],total_cs[0],\
                total_csc_penalty[0],total_hit[0],total_itf[0],\
                step_counter[0]) = \
                ddpg_test(RL_s = RL_s[0], 
                          cl_ddpg = cl_ddpg_list[0],
                          ca_ddpg = ca_ddpg_list[0],
                          env = env_list[0],
                          total_reward = total_reward[0],
                          total_ee = total_ee[0],
                          total_cs = total_cs[0],
                          total_csc_penalty = total_csc_penalty[0],
                          total_hit = total_hit[0],
                          total_itf = total_itf[0],
                          step_counter = step_counter[0],
                          )
            '''[2]multi-DDPG --> myddpg_1'''
            if not done[1]:
                (RL_s[1],done[1],RL_clustering_policy_num[1],RL_caching_policy_num[1],\
                env_list[1],\
                total_reward[1],total_ee[1],total_cs[1],\
                total_csc_penalty[1],total_hit[1],total_itf[1],\
                step_counter[1]) = \
                ddpg_test(RL_s = RL_s[1], 
                          cl_ddpg = cl_ddpg_list[idx_1],
                          ca_ddpg = ca_ddpg_list[idx_1],
                          env = env_list[1],
                          total_reward = total_reward[1],
                          total_ee = total_ee[1],
                          total_cs = total_cs[1],
                          total_csc_penalty = total_csc_penalty[1],
                          total_hit = total_hit[1],
                          total_itf = total_itf[1],
                          step_counter = step_counter[1],
                          )
#                print(RL_clustering_policy_num[1])
#                print(RL_caching_policy_num[1])
            '''[3]popularity-based caching & RL clustering'''
            if not done[2]:
                RL_clustering_policy_num[2] = RL_clustering_policy_num[1]
                RL_caching_policy_num[2] = env_list[2].get_opt_caching_number(RL_clustering_policy_num[2]).astype(int)
                r1, done[2],EE1, CS1, csc_penalty1, SINR1, Hit_rate1, s1_, Itf1= env_list[2].step(clustering_policy_num = RL_clustering_policy_num[2],
                                                                                                     caching_policy_num = RL_caching_policy_num[2],
                                                                                                     update_Req = 1, 
                                                                                                     normalize = 1,
                                                                                                     Req_duplicate = REQUEST_DUPLICATE)
                total_reward[2] += r1
                total_ee[2] += EE1*EE_std+EE_mean
                total_cs[2] += CS1*CS_std+CS_mean
                total_csc_penalty[2] += csc_penalty1
                total_hit[2] += Hit_rate1
                total_itf[2] += Itf1
                step_counter[2] += 1
                RL_s[2] = s1_
           
                
                
            '''[4]popularity-based caching & random clustering'''
            if not done[3]:
                RL_clustering_policy_num[3] = np.random.randint(0,env_list[3].B-1,size=env_list[3].U)
                RL_caching_policy_num[3] = (env_list[3].opt_caching_number).astype(int)
                r, done[3],EE, CS, csc_penalty, SINR, Hit_rate, s_, Itf= env_list[3].step(clustering_policy_num = RL_clustering_policy_num[3],
                                                              caching_policy_num = (env_list[3].pop_caching_number*np.ones(env.B)).astype(int),
                                                              update_Req = 1, 
                                                              normalize = 1,
                                                              Req_duplicate = REQUEST_DUPLICATE)
                total_reward[3] += r
                total_ee[3] += EE*EE_std+EE_mean
                total_cs[3] += CS*CS_std+CS_mean
                total_csc_penalty[3] += csc_penalty
                total_hit[3] += Hit_rate
                total_itf[3] += Itf
                step_counter[3] += 1
                RL_s[3] = s_
                
            '''[5]popularity-based caching & near clustering'''
            if not done[4]:
                RL_clustering_policy_num[4] = near_clustering_policy
                RL_caching_policy_num[4] = env_list[4].get_opt_caching_number(RL_clustering_policy_num[4]).astype(int)
                r, done[4],EE, CS, csc_penalty, SINR, Hit_rate, s_, Itf= env_list[4].step(clustering_policy_num = RL_clustering_policy_num[4],
                                                              caching_policy_num = (env_list[4].pop_caching_number*np.ones(env.B)).astype(int),
                                                              update_Req = 1, 
                                                              normalize = 1,
                                                              Req_duplicate = REQUEST_DUPLICATE)
                total_reward[4] += r
                total_ee[4] += EE*EE_std+EE_mean
                total_cs[4] += CS*CS_std+CS_mean
                total_csc_penalty[4] += csc_penalty
                total_hit[4] += Hit_rate
                total_itf[4] += Itf
                step_counter[4] += 1
                RL_s[4] = s_
            
            # write info 
            f.write('===================step['+str(step)+']===================\n')
            f.write('SDDPG_clustering_policy_num = ')
            f.write(str(RL_clustering_policy_num[0])+'\n\n')
            f.write('SDDPG_caching_policy_num = ')
            f.write(str(RL_caching_policy_num[0])+'\n\n')
            f.write('MDDPG_clustering_policy_num = ')
            f.write(str(RL_clustering_policy_num[1])+'\n\n')
            f.write('MDDPG_caching_policy_num = ')
            f.write(str(RL_caching_policy_num[1])+'\n\n')
            writeinfo(f,env_list[0],1)
            writeinfo(f,env_list[1],2)
            writeinfo(f,env_list[2],3)
            writeinfo(f,env_list[3],4)
            writeinfo(f,env_list[4],5)
            
            if (step > 0):
                break
       
         # trace the average reward of the episode        
        reward_list.append(total_reward[0]/step_counter[0]) # average reward of the episode
        EE_list.append((total_ee[0])/step_counter[0])
        CS_list.append((total_cs[0])/step_counter[0])
        csc_penalty_list.append(total_csc_penalty[0]/step_counter[0])
        Hit_list.append(total_hit[0]/step_counter[0])
        Itf_list.append(total_itf[0]/step_counter[0])

        reward_list2.append(total_reward[1]/step_counter[1]) # average reward of the episode
        EE_list2.append((total_ee[1])/step_counter[1])
        CS_list2.append((total_cs[1])/step_counter[1])
        csc_penalty_list2.append(total_csc_penalty[1]/step_counter[1])
        Hit_list2.append(total_hit[1]/step_counter[1])
        Itf_list2.append(total_itf[1]/step_counter[1])
        
        reward_list3.append(total_reward[2]/step_counter[2]) # average reward of the episode
        EE_list3.append((total_ee[2])/step_counter[2])
        CS_list3.append((total_cs[2])/step_counter[2])
        csc_penalty_list3.append(total_csc_penalty[2]/step_counter[2])
        Hit_list3.append(total_hit[2]/step_counter[2])
        Itf_list3.append(total_itf[2]/step_counter[2])
        
        reward_list4.append(total_reward[3]/step_counter[3]) # average reward of the episode
        EE_list4.append((total_ee[3])/step_counter[3])
        CS_list4.append((total_cs[3])/step_counter[3])
        csc_penalty_list4.append(total_csc_penalty[3]/step_counter[3])
        Hit_list4.append(total_hit[3]/step_counter[3])
        Itf_list4.append(total_itf[3]/step_counter[3])  
        
        reward_list5.append(total_reward[4]/step_counter[4]) # average reward of the episode
        EE_list5.append((total_ee[4])/step_counter[4])
        CS_list5.append((total_cs[4])/step_counter[4])
        csc_penalty_list5.append(total_csc_penalty[4]/step_counter[4])
        Hit_list5.append(total_hit[4]/step_counter[4])
        Itf_list5.append(total_itf[4]/step_counter[4])
        
        # reset small-scale fading
        if (RESET_CHANNEL):
            P = channel_reset(U,B,pl)
            for i in range(len(env_list)):
                env_list[i].P = P

    # plot and save the clustering result
    #
    plot_UE_SBS_association(xx_bs = env.bs_coordinate[1:,0],
                            yy_bs = env.bs_coordinate[1:,1],
                            xx_u = env.u_coordinate[:,0],
                            yy_u = env.u_coordinate[:,1],
                            L_ii = RL_clustering_policy_num[0],
                            Req = env_list[0].Req,
                            f2b = np.where(env.C[RL_caching_policy_num[0]]==1)[1].reshape((env.B,N)),
                            up = env_list[0].up,
                            filename1 = './data/test_result_[1].png',
                            filename2 = './data/test_result_[1].csv')
    #
    plot_UE_SBS_association(xx_bs = env.bs_coordinate[1:,0],
                            yy_bs = env.bs_coordinate[1:,1],
                            xx_u = env.u_coordinate[:,0],
                            yy_u = env.u_coordinate[:,1],
                            L_ii = RL_clustering_policy_num[1],
                            Req = env_list[1].Req,
                            f2b = np.where(env.C[RL_caching_policy_num[1]]==1)[1].reshape((env.B,N)),
                            up = env_list[1].up,
                            filename1 = './data/test_result_[2].png',
                            filename2 = './data/test_result_[2].csv')
    #
    plot_UE_SBS_association(xx_bs = env.bs_coordinate[1:,0],
                            yy_bs = env.bs_coordinate[1:,1],
                            xx_u = env.u_coordinate[:,0],
                            yy_u = env.u_coordinate[:,1],
                            L_ii = RL_clustering_policy_num[2],
                            Req = env_list[2].Req,
                            f2b = np.where(env.C[RL_caching_policy_num[2]]==1)[1].reshape((env.B,N)),
                            up = env_list[2].up,
                            filename1 = './data/test_result_[3].png',
                            filename2 = './data/test_result_[3].csv')
    #
    plot_UE_SBS_association(xx_bs = env.bs_coordinate[1:,0],
                            yy_bs = env.bs_coordinate[1:,1],
                            xx_u = env.u_coordinate[:,0],
                            yy_u = env.u_coordinate[:,1],
                            L_ii = RL_clustering_policy_num[3],
                            Req = env_list[3].Req,
                            f2b = np.where(env.C[RL_caching_policy_num[3]]==1)[1].reshape((env.B,N)),
                            up = env_list[3].up,
                            filename1 = './data/test_result_[4].png',
                            filename2 = './data/test_result_[4].csv')
    #
    plot_UE_SBS_association(xx_bs = env.bs_coordinate[1:,0],
                            yy_bs = env.bs_coordinate[1:,1],
                            xx_u = env.u_coordinate[:,0],
                            yy_u = env.u_coordinate[:,1],
                            L_ii = RL_clustering_policy_num[4],
                            Req = env_list[4].Req,
                            f2b = np.where(env.C[RL_caching_policy_num[4]]==1)[1].reshape((env.B,N)),
                            up = env_list[4].up,
                            filename1 = './data/test_result_[5].png',
                            filename2 = './data/test_result_[5].csv')
   #
    plot('reward',
        './data/reward_test.png',
        reward_list,
        reward_list2,
        reward_list3,
        reward_list4,
        reward_list5,
        n_realizations)
    plot('EE',
        './data/EE_test.png',
        list1 = EE_list,
        list2 = EE_list2,
        list3 = EE_list3,
        list4 = EE_list4,
        list5 = EE_list5,
        n_realizations = n_realizations)
    plot('CS',
        './data/CS_test.png',
        CS_list,
        CS_list2,
        CS_list3,
        CS_list4,
        CS_list5,
        n_realizations)
    plot('csc_penalty',
        './data/csc_penalty_test.png',
        csc_penalty_list,
        csc_penalty_list2,
        csc_penalty_list3,
        csc_penalty_list4,
        csc_penalty_list5,
        n_realizations)
    plot('Hit_rate',
        './data/Hit_rate_test.png',
        Hit_list,
        Hit_list2,
        Hit_list3,
        Hit_list4,
        Hit_list5,
        n_realizations)
    plot('Itf',
        './data/Itf_test.png',
        Itf_list,
        Itf_list2,
        Itf_list3,
        Itf_list4,
        Itf_list5,
        n_realizations = n_realizations)
       
    writecsv('./data/reward_data_test.csv',
             reward_list,
             reward_list2,
             reward_list3,
             reward_list4,
             reward_list5,
             )
    writecsv('./data/EE_data_test.csv',
             EE_list,
             EE_list2,
             EE_list3,
             EE_list4,
             EE_list5,
             )
    writecsv('./data/CS_data_test.csv',
             CS_list,
             CS_list2,
             CS_list3,
             CS_list4,
             CS_list5,
             )
    writecsv('./data/csc_penalty_data_test.csv',
             csc_penalty_list,
             csc_penalty_list2,
             csc_penalty_list3,
             csc_penalty_list4,
             csc_penalty_list5,
             )
    writecsv('./data/Hit_data_test.csv',
             Hit_list,
             Hit_list2,
             Hit_list3,
             Hit_list4,
             Hit_list5,
             )
    writecsv('./data/interference_data_test.csv', 
             Itf_list,
             Itf_list2,
             Itf_list3,
             Itf_list4,
             Itf_list5,
             )
    
    f.close()
    
    

    