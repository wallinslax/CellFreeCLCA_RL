#!/usr/bin/env python3
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Public Lib
import gym
from gym import spaces
import numpy as np
from numpy import linalg as LA
from numpy.random import randn
from random import randint
#import scipy.stats
import os,math,random,itertools,csv,pickle,inspect,torch
from itertools import combinations,permutations,product
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.pyplot import cm
import multiprocessing
from tqdm import tqdm
import concurrent.futures
from datetime import date

today = date.today()
#####################################
num_cores = multiprocessing.cpu_count()
# ENV Parameter
radius = 0.5 # radius of the network = 500m
lambda_bs = 6 # density of the BS
lambda_u = 10 # density of the UE
h_var = 1 # channel variance
k = 1 # pathloss coefficient
alpha = 2 # pathloss exponent
beta = 1 # zipf parameter

d_th = 0.2*1000 # distance theshold for clustering policy candidate 200m
'''
nBS = 4  # number of BSs
nUE = 12 # number of UEs
nMaxLink = 2 # Max allowed connected BS for a single UE
F = 5 # number of total files
N = 2 # capacity of BSs
'''
P_t = 10 # transmit power of SBS 10dbm = 10mW
P_MBS = 31.6 # transmit power of MBS 15dbm = 31.6mW
P_l = 20 # data retrieval power from local cache = 20mW
P_bh = 500 # data retrieval power from backhaul = 500mW (AP--CPU)
P_bb = 500 # data retrieval power from backbone = 500mW (CPU--Backbone)
P_o_SBS = 1500 # operational power of SBS = 1500mW
P_o_MBS = 2500 # operational power of MBS =2500mW
n_var = 7.457*(10**-10) # 300K/ 20MHz: 7.457e-13 (W) (mW)
# Johnson–Nyquist noise (thermal noise)
# https://www.everythingrf.com/rf-calculators/noise-power-calculator
# https://www.rapidtables.com/convert/power/dBm_to_Watt.html
# https://en.wikipedia.org/wiki/Johnson%E2%80%93Nyquist_noise
# 25C/ 50 kHz => -126.86714407 dBm = 1.995262315e-13(mW)
# 300K/ 20MHz  => -101 dbm = 7.94e-11 (mW)
# 300K/ 1GHz  => -84 dbm = 3.981e-9(mW)
'''
bandwidth =20MHz
kB = 1.381e-23 J/K
T0 = 300K
noise figure = 9dB

[1]wiki的公式是: noise power = bandwidth × kB × T0
noise power = 20MHz × 1.381e-23 × 300K =8.286e-14(W)

[2] 參照wiki table
300K/ 20MHz  =>noise power = -101 dbm = 7.94e-14 (W)

[3] Small Cells多了noise figure:  noise power = bandwidth × kB × T0 × noise figure
noise power = 20MHz × 1.381e-23 × 300K × 9dB =7.457e-13 (W)
'''
#####################################
# plot size

font = {'family' : 'Verdana',
        'weight' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)
markerSize = 20*4**1

#####################################
def UE_SBS_location_distribution(lambda0): #PPP
    xx0=0; yy0=0; # centre of disk
    areaTotal=np.pi*radius**2; # area of disk
    # Number~ PPP
    lambda0=lambda0; #intensity (ie mean density) of the Poisson process
    numbPoints = scipy.stats.poisson( lambda0*areaTotal ).rvs()#Poisson number of points
    # Location~ Uniform 
    points = np.random.rand(numbPoints, 2)-0.5
    return points

def plot_UE_BS_distribution_Cache(env,clustering_policy_UE,caching_policy_BS,EE,filename,isDetail=False,isEPS=False):
    #drive, path_and_file = os.path.splitdrive(filename)
    #print(filename.split('_')[-1])
    methodName = filename.split('_')[-1]
    path, filenameO = os.path.split(filename)
    if 'Training' in filename:
        phaseName = 'Training Phase'
    elif 'Evaluation' in filename:
        phaseName = 'Evaluation Phase'
    #plt.cla()
    plt.clf()
    # AP
    xx_bs = env.bs_coordinate[:,0]
    yy_bs = env.bs_coordinate[:,1]
    plt.scatter(xx_bs, yy_bs, edgecolor='k', facecolor='k',marker='^', alpha=1 ,label='AP',s=markerSize)# AP marker
    b = 0
    for x,y in zip(xx_bs, yy_bs):
        #plt.annotate("%s" % (b+1), xy=(x,y), xytext=(x, y-0.06),color='k')# AP index 4.4.5.2
        plt.annotate("%s" % (b+1), xy=(x,y), xytext=(x, y-0.07),color='k')# AP index 10.5.20.2
        if caching_policy_BS:
            #plt.annotate("%s" % str(list(caching_policy_BS[b])), xy=(x,y), xytext=(x-0.03, y+0.03),color='k')# AP cache
            plt.annotate("%s" % str([ x+1 for x in caching_policy_BS[b] ]), xy=(x,y), xytext=(x-0.03, y+0.03),color='k')#AP cache
        b = b+1
    # UE
    nUE=len(env.u_coordinate)
    color=cm.rainbow(np.linspace(0,1,nUE))
    for u in range(nUE):
        xx_u = env.u_coordinate[u,0]
        yy_u = env.u_coordinate[u,1]
        plt.scatter(xx_u, yy_u, edgecolor=color[u], facecolor='none',marker='X', alpha=0.5 ,label='UE'+str(u+1),s=markerSize)# UE marker
        #plt.annotate("%s" % u, xy=(xx_u,yy_u), xytext=(xx_u, yy_u-0.04),color=color[u])# UE index
        plt.annotate("%s" % '['+str(env.Req[u]+1)+']', xy=(xx_u,yy_u), xytext=(xx_u-0.03, yy_u+0.03),color=color[u])# UE Req
        if isDetail:
            EE=env.calEE(clustering_policy_UE,caching_policy_BS)
            # plot P_r
            plt.annotate("%s" % 'P_r='+str( "{:.2f}".format(env.P_r[u]) ), xy=(xx_u,yy_u), xytext=(xx_u-0.025, yy_u-0.03),color=color[u], fontsize=10)
            # plot I
            plt.annotate("%s" % 'I='+str( "{:.2f}".format(env.I[u]) ), xy=(xx_u,yy_u), xytext=(xx_u-0.025, yy_u-0.06),color=color[u], fontsize=10)
            # plot SINR
            plt.annotate("%s" % 'SINR='+str( "{:.2f}".format(env.SINR[u]) ), xy=(xx_u,yy_u), xytext=(xx_u-0.025, yy_u-0.09),color=color[u], fontsize=10)
            # plot TP
            plt.annotate("%s" % 'TP='+str( "{:.2f}".format(env.Throughput[u]) ), xy=(xx_u,yy_u), xytext=(xx_u-0.025, yy_u-0.12),color=color[u], fontsize=10)
            # activeBS
            activatedBS = list(set([item for sublist in clustering_policy_UE for item in sublist]))
            plt.annotate("%s" % '#actBS='+str( "{:.2f}".format(len(activatedBS)) ), xy=(xx_u,yy_u), xytext=(xx_u-0.025, yy_u-0.18),color=color[u], fontsize=10)
            # MCAP
            plt.annotate("%s" % 'MCAP='+str( "{:.2f}".format(env.missCounterAP) ), xy=(xx_u,yy_u), xytext=(xx_u-0.025, yy_u-0.21),color=color[u], fontsize=10)
            # MCCPU
            plt.annotate("%s" % 'MCCPU='+str( "{:.2f}".format(env.missCounterCPU) ), xy=(xx_u,yy_u), xytext=(xx_u-0.025, yy_u-0.24),color=color[u], fontsize=10)
            plt.title(methodName+'EE:'+str("{:.2f}".format(EE) )\
                                +';TP:'+str("{:.2f}".format(sum(env.Throughput)) )\
                                +';Psys:'+str("{:.2f}".format(env.P_sys) ) 
                    )
        # plot Clustering
        if clustering_policy_UE:
            useBS = clustering_policy_UE[u]
            for bs in useBS:
                xx_bs = env.bs_coordinate[bs,0]
                yy_bs = env.bs_coordinate[bs,1]
                plt.plot([xx_u,xx_bs],[yy_u,yy_bs],linestyle='-',color=color[u])
    
    plt.xlabel("x (km)"); plt.ylabel("y (km)")
    plt.tight_layout()
    EE = "{:.2f}".format(EE)
    #plt.axis('equal')
    plt.axis([-0.7, 0.6, -0.6, 0.6]) # 10.5.20.2
    #plt.axis([-0.2, 0.5, -0.2, 0.5]) # 4.4.5.2
    plt.legend(loc = 'lower left', fontsize=10) # 10.5.20.2
    #plt.legend(loc = 'upper right', fontsize=14) # 4.4.5.2
    if isDetail:
        plt.savefig(filename +'Detailed.png', format='png',dpi=120)
    else:
        plt.savefig(filename +'.png', format='png',dpi=120)
        if isEPS:
            plt.savefig(filename +'.eps', format='eps',dpi=120)

class BS(gym.Env):

    def get_statistic(self):
        self.EE_mean = 0
        self.EE_std = 1
        self.CS_mean = 0
        self.CS_std = 1
        print('Calculating statistic...')
        EE_sample_list = []
        CS_sample_list = []
        for i in range(10**5):
            random_clustering_policy_UE = []
            for u in range(self.U):
                bsSet = np.arange(self.B)
                np.random.shuffle(bsSet)
                random_clustering_policy_UE.append(bsSet[:self.L])

            random_caching_policy_BS = []
            for b in range(self.B):
                fileSet = np.arange(self.F)
                np.random.shuffle(fileSet)
                random_caching_policy_BS.append(fileSet[:self.N])
            EE = self.calEE(random_clustering_policy_UE, random_caching_policy_BS)
            EE_sample_list.append(EE)
        EE_mean = np.mean(EE_sample_list)
        EE_std = np.std(EE_sample_list, ddof=1)
        print('EE_mean = '+str(EE_mean))
        print('EE_std = '+str(EE_std))
        return EE_mean,EE_std

    def genUserRequest(self, userPreference):
        zipf_pmf_numerator = [] 
        for j in range(len(userPreference)):
            zipf_pmf_numerator.append((j+1)**(-beta))
        zipf_pmf = list(np.true_divide(zipf_pmf_numerator, sum(zipf_pmf_numerator)))
        pmf = np.array(zipf_pmf)
        zipf_pmf_1=np.zeros(len(userPreference))
        j=0
        for i in userPreference:
            zipf_pmf_1[j]=pmf[i]
            j=j+1

        x = random.random()*sum(zipf_pmf) # U[0,1]
        k = 0
        p = pmf[k]
        
        while x>p :
            k += 1
            p = p+pmf[k]

        index=np.where(zipf_pmf_1==pmf[k])
        #print(index)
        return (index[0])

    def resetChannel(self):
        '''[2] Pair-wise distance'''
        self.D = np.zeros((self.B,self.U))
        for b in  range(self.B):
            for u in range(self.U):
                self.D[b][u] = 1000*np.sqrt(sum((self.u_coordinate[u]-self.bs_coordinate[b])**2)) # km -> m
        D0=min(self.D.reshape(self.B*self.U))
        self.D = self.D/D0
        '''[3] Large scale fading'''
        self.pl = k*np.power(self.D, -alpha) # Path-loss

        '''[4] Small scale fading'''
        self.h = np.sqrt(h_var/2) * (randn(self.B,self.U)+1j*randn(self.B,self.U)) # h~CN(0,1); |h|~Rayleigh fading
        self.g = self.pl * self.h
        #return self.g
        
    def timeVariantChannel(self):
        noise = np.sqrt(h_var/2) * (randn(self.B,self.U)+1j*randn(self.B,self.U))
        h_next =  np.sqrt(1 - self.epsilon**2) * self.h + self.epsilon * noise
        self.h = h_next
        self.g = self.pl * self.h

    def resetReq(self):
        '''[3] Generate User request'''
        for u in range(self.U):
            self.Req[u] = self.genUserRequest(self.userPreference[u])

    def __init__(self,nBS,nUE,nMaxLink,nFile,nMaxCache,loadENV,SEED=0,obsIdx=1):
        self.SEED=SEED
        self.obsIdx = obsIdx
        self.B = nBS # number of BS
        self.U = nUE # number of UE
        self.L = nMaxLink # Max Link Capability of UE
        self.F = nFile # number of total files
        self.N = nMaxCache # Max cache size of BS
        self.TopologyName = str(self.B)+'AP_'+str(self.U)+'UE_' + str(self.F) + 'File_'+ str(self.N) +'Cache_'
        self.TopologyCode = str(self.B)+'.'+str(self.U)+'.' + str(self.F) + '.'+ str(self.N)
        self.epsilon = 0.01
        filename = 'data/'+self.TopologyCode+'/Topology/['+str(self.SEED)+']Topology_'+ self.TopologyName #+ str(today)
        if(loadENV):# load Topology
            with open(filename + '.pkl','rb') as f: 
                self.bs_coordinate, self.u_coordinate, self.pl, self.h,  self.g, self.userPreference, self.Req = pickle.load(f)
        else:
            '''[1] SBS/ UE distribution''' 
            self.u_coordinate = np.random.rand(self.U, 2)-0.5
            sbs_coordinate = np.random.rand(self.B-1, 2)-0.5
            self.bs_coordinate = np.concatenate((np.array([[0,0]]),sbs_coordinate),axis=0)
            '''[2] Generate g_mk g_bu''' 
            self.resetChannel() 
            '''[3] Generate User Preference'''
            # User Preference is a score list of for each file. Score 0 is the most favorite.
            # i.e. userPreference[0] = [3 2 0 1 4], the most favorate file of UE0 is 2th file, the second favorite file is 3th file
            self.userPreference = np.zeros((self.U,self.F),dtype=int)
            for u in range(self.U):
                seedPreference = np.arange(self.F)
                np.random.shuffle(seedPreference)
                self.userPreference[u] = seedPreference
            #print(userPreference)
            '''[4] Generate User request'''
            self.Req = [self.F]*self.U
            #self.resetReq()
            # force users always require most preferable file to ensure BM1's performance is in competitive case ######
            for u in range(self.U):
                self.Req[u] =  list(self.userPreference[u]).index(0)
            # check topology
            plot_UE_BS_distribution_Cache(self,None,None,0,filename,isEPS=False)
            # save Topology
            '''
            np.savez(filename+'_np',bs_coordinate= self.bs_coordinate, u_coordinate= self.u_coordinate, pl = self.pl, h=self.h, g=self.g, userPreference=self.userPreference, Req=self.Req)
            npzfile = np.load(filename+'_np.npz')
            print(npzfile['bs_coordinate'])
            '''
            with open(filename + '.pkl', 'wb') as f: 
                pickle.dump([self.bs_coordinate, self.u_coordinate, self.pl, self.h, self.g, self.userPreference, self.Req], f)
            #self.EE_mean,self.EE_std,self.CS_mean,self.CS_std = self.get_statistic()
        '''
        # Debug: self.userPreference
        print('self.userPreference[0]=',self.userPreference[0])
        cumulate = np.zeros(self.F)
        for i in range(100):
            req = self.genUserRequest(self.userPreference[0])
            print(req)
            cumulate[req]+=1
        print('self.userPreference[0]=',self.userPreference[0])
        print(cumulate)
        '''
        self.done = False
        '''[9] popular method to determine clustering and caching policy'''
        self.nearestClustering = np.zeros([self.U,self.B],dtype=int)
        self.optCacheTopN = np.zeros([self.B,self.N],dtype=int)
        self.estCacheTopN = np.zeros([self.B,self.N],dtype=int)
        '''[10] Content request profile of each UE'''
        self.reqStatistic = np.zeros([self.U,self.F],dtype=int)
        self.userSimilarity = np.zeros([self.U,self.U],dtype=int)

        '''[11] System power consumption'''
        self.P_sys = 0
        '''[12] Energy efficiency'''
        self.EE = 0 
        '''[15] Content request profile similarity'''
        self.ueSimilarity = np.zeros([self.U, self.U]) 
        '''[16] Intra-cluster similarity'''
        self.ICS = np.zeros(self.B) 
        '''[17] Cluster Similarity'''
        self.CS = 0
        '''[19] State'''
        self.SINR = np.zeros(self.U)
        self.clustering_state = np.zeros([self.B,self.U])
        self.caching_state = np.zeros([self.B,self.F])
        self.reqStatistic_norm = np.zeros(self.U*self.F)
        self.s_ = self.reset()
        #------------------------------------------------------------------
        
        self.dimActCL = self.B*self.U
        self.dimActCA = self.B*self.F
        self.dimAct = self.dimActCL + self.dimActCA
        self.dimObs = len(self.s_)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.dimAct,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=float("inf"), shape=(len(self.s_),), dtype=np.float32)

    def reset(self):
        if self.obsIdx==1:  # OBS1 [Paper Oberservation]
            self.s_ = np.hstack([  self.g.real.flatten(),
                                    self.g.imag.flatten(),
                                    self.clustering_state.flatten(),
                                    self.caching_state.flatten(),
                                    self.reqStatistic_norm.flatten()]) 
        elif self.obsIdx==2:  # OBS2
            self.s_ = np.hstack([  self.g.real.flatten(),
                                    self.g.imag.flatten(),
                                    self.clustering_state.flatten(),
                                    self.caching_state.flatten()]) 
        elif self.obsIdx==3:# OBS3
            self.s_ = np.hstack([#self.SINR,
                                self.g.real.flatten(),
                                self.g.imag.flatten(),    
                                self.reqStatistic_norm.flatten()]) 
        #------------------------------------------------------------------
        return self.s_

    def updateReqStatistic(self):
        '''[10] Content Request Statistic of each UE'''
        for u in range(self.U):
            self.reqStatistic[u][self.Req[u]] += 1
        '''[15] Content request profile similarity'''
        #print('LA.norm(self.reqStatistic, axis=1)=',LA.norm(self.reqStatistic, axis=1))
        self.reqStatistic_norm = self.reqStatistic/(LA.norm(self.reqStatistic, axis=1)).reshape((self.U,1))
        self.ueSimilarity = np.matmul(self.reqStatistic_norm, self.reqStatistic_norm.T) 
    
    def action2Policy(self,action):
        a_cl = action[0:self.dimActCL]
        a_ca = action[-self.dimActCA:]

        # Convert action value to policy //Clustering Part
        connectionScore = np.reshape(a_cl, (self.U,self.B) ) #[env.U x env.B]
        #connectionScore = np.around(connectionScore)
        clustering_policy_UE = []
        for u in range(self.U): 
            maxLBS = connectionScore[u].argsort()[::-1][:self.L] # limit RL connection number to L
            positiveBS = [ i for (i,v) in enumerate(connectionScore[u]) if v >= 0 ] # unlimited
            selectedBS = np.intersect1d(maxLBS,positiveBS) # <=L
            if selectedBS.size == 0:# gurantee all users are served
                selectedBS = connectionScore[u].argsort()[::-1][:1]
            clustering_policy_UE.append(selectedBS)
        
        # Convert action value to policy //Caching Part
        cacheScore = np.reshape(a_ca, (self.B,self.F) )
        caching_policy_BS = []
        for b in range(self.B):
            top_N_idx = np.sort(cacheScore[b].argsort()[-self.N:])# pick up N file with highest score, N is capacity of BSs
            caching_policy_BS.append(top_N_idx)

        return clustering_policy_UE, caching_policy_BS 

    def step(self,action):
        clustering_policy_UE, caching_policy_BS = self.action2Policy(action)
        #[EE]##############################################################################################################
        self.EE = self.calEE(clustering_policy_UE, caching_policy_BS)
        #[HR]############################################################################################################## 
        self.HR = self.calHR(clustering_policy_UE, caching_policy_BS)
        #[CS]##############################################################################################################
        self.updateReqStatistic()
        '''[16] Intra-cluster similarity
        self.ICS = np.zeros(self.B) # intra-cluster similarity of each cluster(BS)
        for b in range(self.B):
            UE_pair = [list(i) for i in list( combinations (self.clustering_policy_BS[b], 2))]
            for jj, pair in enumerate(UE_pair):
                self.ICS[b] = self.ICS[b] + self.ueSimilarity[pair[0]][pair[1]]
            if len(UE_pair):
                self.ICS[b] = self.ICS[b]/len(UE_pair)
        
        self.CS = sum(self.ICS)/self.B
        self.CS_norm = (self.CS - self.CS_mean)/self.CS_std # Z-score normalization'''
        ###############################################################################################################
        '''[19] State'''
        #convert culstering policy to binary form
        clustering_state = np.zeros([self.U,self.B])
        for u in range(self.U):
            clustering_state[u][ list(clustering_policy_UE[u]) ]=1
        self.clustering_state = clustering_state
        #convert caching policy to binary form
        caching_state = np.zeros([self.B,self.F])
        for b in range(self.B):
            caching_state[b][ list(caching_policy_BS[b]) ] = 1
        self.caching_state = caching_state
        #------------------------------------------------------------------
        '''[20] Whether episode done'''
        observation = self.reset()
        reward = self.EE
        done = self.done
        info = {"HR":self.HR}

        #return self.EE, self.HR, self.s_, self.done
        return observation, reward, done, info

    def calHR(self,clustering_policy_UE,caching_policy_BS):
        '''[13] Hit event'''
        self.Hit = np.zeros(self.U)
        for u in range(self.U):
            useBS = clustering_policy_UE[u]
            counter = 0
            for bs in useBS:
                if self.Req[u] in caching_policy_BS[bs]:
                    counter+=1
            if counter == len(useBS):
                self.Hit[u]=1
        self.HR = sum(self.Hit)/len(self.Hit)
        return self.HR
    
    def calEE(self,clustering_policy_UE,caching_policy_BS):
        '''[5-11] [5] clustering_policy_BS/[6] rho_b/[7] received power P_r/[8] Activated BS set/[8.5] Activated UE set/[9] Interference I/[10] SINR/[11] Throughput'''
        self.Throughput = self.calTP(clustering_policy_UE)

        '''[12] System power consumption'''
        self.P_sys = self.callPsys(clustering_policy_UE,caching_policy_BS)

        '''[13] Energy efficiency'''
        sumThroughput = sum(self.Throughput)
        if (self.P_sys>0):
            self.EE = sumThroughput/(self.P_sys/1000) # Bits/s*W mW->W
        else:
            self.EE = 0
        #self.EE_norm = (self.EE-self.EE_mean)/self.EE_std # Z-score normalization
        return self.EE
    
    def calTP(self,clustering_policy_UE): # calculate throughput
        # S_k = S_u = clustering_policy_UE
        # C_m = C_b = clustering_policy_BS
        '''[5] clustering_policy_BS'''
        clustering_policy_BS = []
        for b in range(self.B):
            competeUE = []
            for u in range(self.U):
                if b in clustering_policy_UE[u]:
                    competeUE.append(u) #the UE set in b-th cluster   
            clustering_policy_BS.append(competeUE)
        
        '''[6] rho_b'''
        self.rho = np.zeros(self.B)
        for b in range(self.B):
            competeUE = clustering_policy_BS[b]
            if len(competeUE) != 0:
                #print(self.g[b][competeUE])
                #print(abs(self.g[b][competeUE]))
                #print(np.power(abs(self.g[b][competeUE]),2))
                #print(sum( np.power(abs(self.g[b][competeUE]),2) ))
                self.rho[b] = P_t / sum( np.power(abs(self.g[b][competeUE]),2) )
            #print( self.rho[b] )
        
        '''[7] received power P_r'''
        self.P_r = np.zeros(self.U)
        for u in range(self.U):
            for b in clustering_policy_UE[u]: #S_u = clustering_policy_UE[u]
                #print(np.power(abs(self.g[b][u]),2))
                #print(self.g[b][u]*self.g[b][u].conjugate())
                self.P_r[u] += np.sqrt(self.rho[b]) * np.power(abs(self.g[b][u]),2)
            self.P_r[u] = np.power(self.P_r[u],2)

        '''[8] Activated BS set: S = Union S_u'''
        activatedBS = list(set([item for sublist in clustering_policy_UE for item in sublist]))
        '''[8.5] Activated UE set: C = Union C_b'''
        activatedUE = [ u for u in range(self.U) if list(clustering_policy_UE[u] ) != [] ] 
        '''[9] Interference I'''
        self.I = np.zeros(self.U)
        #***DEBUG***
        #print('self.rho=',self.rho)
        #if len(activatedUE)!=self.U:
        #    print('watch')
        #***DEBUG***
        #for u in range(self.U):
        #    other_u = list(range(self.U))
        for u in activatedUE:
            other_u = activatedUE.copy()
            other_u.remove(u)
            for uu in other_u: # set C != all UE
                sum_b = 0
                for b in activatedBS:# set S != all BS
                    #chk = self.g[b][u] * self.g[b][uu].conjugate()
                    #print(chk)
                    #***DEBUG***
                    #Ibuu=np.sqrt(self.rho[b]) * self.g[b][u] * self.g[b][uu].conjugate()
                    #print('For',u,'th UE: Ibuu from',b,'th AP to',uu,'th UE=',Ibuu)
                    #print('For',u,'th UE: Ibuu from',b,'th AP to',uu,'th UE=',np.power(abs(Ibuu),2))
                    #***DEBUG***
                    sum_b +=  np.sqrt(self.rho[b]) * self.g[b][u] * self.g[b][uu].conjugate()
                #***DEBUG***
                #print('For',u,'th UE: sum_b=',sum_b)
                #print('For',u,'th UE: sum_b from',uu,'th UE=',sum_b)
                #***DEBUG***
                self.I[u] = self.I[u] + np.power(abs(sum_b),2)
        
        '''[10] SINR/ [11]Throughput of UE'''
        self.SINR = np.zeros(self.U)
        self.Throughput = np.zeros(self.U)
        for u in range(self.U):
            self.SINR[u] = self.P_r[u]/(self.I[u] + n_var)
            self.Throughput[u] = math.log2(1+self.SINR[u]) #Bits/s

        return self.Throughput

    def callPsys(self,clustering_policy_UE,caching_policy_BS):
        '''[8] Activated BS set: S = Union S_u'''
        activatedBS = list(set([item for sublist in clustering_policy_UE for item in sublist]))
        '''[12] System power consumption'''
        missFileAP = [ [] for i in range(self.B)]
        for u in range(self.U):
            for bs in clustering_policy_UE[u]:
                if self.Req[u] not in caching_policy_BS[bs]: #Miss
                    missFileAP[bs].append(self.Req[u])
        # Derive F^miss_m for all m
        self.missCounterAP = 0
        for bs in range(self.B): 
            missFileAP[bs] = list(set(missFileAP[bs]))
            self.missCounterAP += len(missFileAP[bs])
        # Derive union F^miss_m for all m
        missFileCPU=[]
        for bs in range(self.B):
            missFileCPU.extend(missFileAP[bs])
        missFileCPU = list(set(missFileCPU))
        self.missCounterCPU = len(missFileCPU)
        self.P_sys = P_t*len(activatedBS) + P_bh * self.missCounterAP + P_bb * self.missCounterCPU  #+ self.B*P_o_SBS + P_o_MBS
        return self.P_sys
    
    def getRand_CL_Policy(self,nLink):
        CL_Policy_UE = [random.sample(range(self.B), nLink) for i in range(self.U)]
        return CL_Policy_UE

    def getFile_CL_Policy(self,nLink,CA_Policy_BS):
        g_abs = abs(self.g) # g  = [B*U]
        g_absT = g_abs.T # g_absT= [U*B]
        # kth UE determine the AP set (S_k)
        '''     
        #　caching_state to CA_Policy_BS
        CA_Policy_BS = [[] for i in range(self.B)]
        for b in range(self.B):# inherit CA policy in previous time slot
            CA_Policy_BS[b] = [fileIdx for fileIdx, flag in enumerate(self.caching_state[b]) if flag == 1]
        '''
        CL_Policy_UE = [[] for i in range(self.U)]
        nonServedUE=[]
        for u in range(self.U):
            competeBS = [b for b in range(self.B) if self.Req[u] in CA_Policy_BS[b] ]
            if len(competeBS) == 0:
                # no BS cache the desired file
                # link nearest non avtive BS and put the request file into the cache
                CL_Policy_UE[u] = [randint(0,self.B-1)]
                nonServedUE.append(u)
            elif len(competeBS) <= nLink:
                CL_Policy_UE[u] = competeBS
            else:
                noCareBS = list(set(range(self.B))-set(competeBS))
                g_absT[u][noCareBS] = 0
                #CL_Policy_UE[u] = g_absT[u].argsort()[::-1][:nLink]
                CL_Policy_UE[u] = random.sample(competeBS, nLink)
            #print('CL_Policy_UE['+str(u)+']=',CL_Policy_UE[u])
        '''
        for u in nonServedUE:
            activatedBS = list(set([item for sublist in CL_Policy_UE for item in sublist]))
            g_absT[u][activatedBS] = 0
            selectBS = g_absT[u].argsort()[::-1][0]
            #nonActiveBS = list(set(range(self.B))-set(activatedBS))
            #selectBS = random.sample(nonActiveBS, 1)[0]
            CA_Policy_BS[selectBS] = [self.Req[u]]
            CL_Policy_UE[u] = [selectBS]
        '''
        return CL_Policy_UE

    def getSNR_CL_Policy(self,nLink):
        g_abs = abs(self.g) # g  = [B*U]
        g_absT = g_abs.T # g_absT= [U*B]
        SNR_CL_Policy_UE = []  
        # kth UE determine the AP set (S_k)     
        for u in range(self.U):
            bestBS = g_absT[u].argsort()[::-1][:nLink]
            SNR_CL_Policy_UE.append(bestBS)
        return SNR_CL_Policy_UE

    def getPOP_CA_Policy_Local(self,clustering_policy_UE,cacheMode):
         # transform clustering_policy_UE to clustering_policy_BS
        clustering_policy_BS = []
        for b in range(self.B):
            competeUE = []
            for u in range(self.U):
                if b in clustering_policy_UE[u]:
                    competeUE.append(u) #the UE set in b-th cluster   
            clustering_policy_BS.append(competeUE)
        '''[] caching based on Req top_N_idx '''
        reqCacheTopN = []
        for b in range (self.B):
            fileCount=np.zeros(self.F)
            for u in clustering_policy_BS[b]:
                fileCount[ self.Req[u] ]+=1
            #print(fileCount.argsort()) 
            top_N_idx = fileCount.argsort()[-self.N:]
            reqCacheTopN.append(top_N_idx)

        '''[] caching based on userPreference top_N_idx '''
        optCacheTopN = []
        for b in range (self.B):
            #sumUserPreferenceInCluster = np.sum(self.userPreference[ clustering_policy_BS[b] ],axis=0)
            #top_N_idx = sumUserPreferenceInCluster.argsort()[0:self.N]
            sumUserPreferenceInCluster = np.sum(1/np.power(self.userPreference[ clustering_policy_BS[b] ]+1,2),axis=0)
            top_N_idx = sumUserPreferenceInCluster.argsort()[::-1][0:self.N]
            optCacheTopN.append(top_N_idx)

        '''[] caching based on reqStatistic [estimated] top_N_idx'''
        estCacheTopN = []
        for b in range (self.B):
            sumUserPreferenceInCluster = np.sum(self.reqStatistic[ clustering_policy_BS[b] ],axis=0)
            top_N_idx = sumUserPreferenceInCluster.argsort()[0:self.N]
            estCacheTopN.append(top_N_idx)
        
        if cacheMode == 'req':
            POP_CA_Policy_BS_Local = reqCacheTopN
        elif cacheMode == 'pref':
            POP_CA_Policy_BS_Local = optCacheTopN
        elif cacheMode == 'stat':
            POP_CA_Policy_BS_Local = estCacheTopN

        return POP_CA_Policy_BS_Local

    def getPOP_CA_Policy(self):
        '''[] caching based on userPreference top_N_idx '''
        sumUserPreferenceInCluster = np.sum(self.userPreference[ : ],axis=0)
        top_N_idx = sumUserPreferenceInCluster.argsort()[0:self.N]
        POP_CA_Policy_BS = [top_N_idx] * self.B
        return POP_CA_Policy_BS
    
    def getPolicy_BM1(self,nLink,cacheMode='pref'):
        SNR_CL_Policy_UE_BM1 = self.getSNR_CL_Policy(nLink)
        POP_CA_Policy_BS_BM1 = self.getPOP_CA_Policy_Local(SNR_CL_Policy_UE_BM1,cacheMode=cacheMode)
        EE_BM1 = self.calEE(SNR_CL_Policy_UE_BM1,POP_CA_Policy_BS_BM1)
        return EE_BM1, SNR_CL_Policy_UE_BM1, POP_CA_Policy_BS_BM1

    def getPolicy_BM2(self,nLink):
        # A design based solely on ${\cal S}_1, {\cal S}_2, \ldots,{\cal S}_K$ may favor the association of the $k$th user 
        # with the AP subset ${\cal S}_k$ that provides the best channel conditions, as this may increase $R_k$ and consequently $R_{\rm sum}$. 
        CL_Policy_UE_BM2 = self.getSNR_CL_Policy(nLink)
        CA_Policy_BS_BM2 = self.getPOP_CA_Policy()
        EE_BM2 = self.calEE(CL_Policy_UE_BM2,CA_Policy_BS_BM2)
        return EE_BM2, CL_Policy_UE_BM2, CA_Policy_BS_BM2
    
    def getPolicy_BM3(self,nLink):
        # a design based solely on ${\cal F}_1, {\cal F}_2, \ldots,{\cal F}_M$ may favor the association of the $k$th user 
        # with the AP subset ${\cal S}_k$ that best aligns the content caching status and user requests, 
        # as this may increase hit events and consequently decrease $P_{\rm total}$.
        CA_Policy_BS_BM3 = self.getPOP_CA_Policy()
        CL_Policy_UE_BM3 = self.getFile_CL_Policy(nLink,CA_Policy_BS_BM3)
        #CL_Policy_UE_BM3 = self.getRand_CL_Policy(nLink)
        #CA_Policy_BS_BM3 = self.getPOP_CA_Policy_Local(CL_Policy_UE_BM3,cacheMode='pref')
        EE_BM3 = self.calEE(CL_Policy_UE_BM3,CA_Policy_BS_BM3)
        return EE_BM3, CL_Policy_UE_BM3, CA_Policy_BS_BM3

    def getOptEE_BF(self,isSave=True):
        print("this is brute force for EE")
        # generate all posible clustering_policy_UE
        choiceBS = []
        for i in range(1,self.B+1):
            subChoiceBS = list(combinations (range(self.B), i))
            choiceBS += subChoiceBS

        print('choiceBS:',choiceBS)
        print('len(choiceBS)^self.U:',pow(len(choiceBS),self.U))
        universe_clustering_policy_UE = list(product(choiceBS,repeat=self.U))

        # generate all posible caching_policy_BS
        choiceFile = list(combinations (range(self.F), self.N))
        print('choiceFile:',choiceFile)
        print('len(choiceFile)^self.B:',pow(len(choiceFile),self.B))
        universe_caching_policy_BS = list(product(choiceFile,repeat=self.B))
        # Try product function
        #allComb = list(product(universe_clustering_policy_UE,universe_caching_policy_BS))
        #print('number of iteration:',len(allComb))

        # Find Best EE
        bestEE=0
        opt_clustering_policy_UE=[]
        opt_caching_policy_BS=[]

        '''
        itr = 0
        for clustering_policy_UE in tqdm(universe_clustering_policy_UE):
            for caching_policy_BS in universe_caching_policy_BS:
                EE = self.calEE(clustering_policy_UE,caching_policy_BS)
                #print("iteration:",itr,"EE=",EE)
                if EE>bestEE:
                    bestEE = EE
                    opt_clustering_policy_UE = clustering_policy_UE
                    opt_caching_policy_BS = caching_policy_BS
                itr+=1
        '''
        with concurrent.futures.ProcessPoolExecutor(max_workers= (num_cores-2) ) as executor:
            futures = []
            for caching_policy_BS in universe_caching_policy_BS:
                #subBestEE,subOpt_clustering_policy_UE,subOpt_caching_policy_BS = self.smallPeice(universe_clustering_policy_UE,caching_policy_BS)
                future = executor.submit(self.smallPeice, universe_clustering_policy_UE,caching_policy_BS) 
                futures.append(future)
            for future in tqdm(concurrent.futures.as_completed(futures),total=len(futures),bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
                #print(future.result())
                subBestEE,subOpt_clustering_policy_UE,subOpt_caching_policy_BS = future.result()
                if subBestEE>bestEE:
                    bestEE = subBestEE
                    print('new Record EE:',bestEE)
                    opt_clustering_policy_UE = subOpt_clustering_policy_UE
                    opt_caching_policy_BS = subOpt_caching_policy_BS
        filenameBF = 'data/'+self.TopologyCode+'/BF/['+str(self.SEED)+']'+self.TopologyName+'BF'
        if isSave:
            # Save the whole environment with Optimal Clustering and Optimal Caching
            with open(filenameBF+'.pkl', 'wb') as f:  
                pickle.dump([self, bestEE, opt_clustering_policy_UE, opt_caching_policy_BS], f)
            # Load the whole environment with Optimal Clustering and Optimal Caching   
            with open(filenameBF+'.pkl','rb') as f: 
                self, bestEE, opt_clustering_policy_UE, opt_caching_policy_BS = pickle.load(f)
        
        bestEE = self.calEE(opt_clustering_policy_UE,opt_caching_policy_BS)
        return bestEE, opt_clustering_policy_UE, opt_caching_policy_BS

    def smallPeice(self,universe_clustering_policy_UE,caching_policy_BS):
        subBestEE=0
        for clustering_policy_UE in universe_clustering_policy_UE:
            EE = self.calEE(clustering_policy_UE,caching_policy_BS)
            #EE, HR, RL_s_, done = self.step(clustering_policy_UE,caching_policy_BS)
            if EE>subBestEE:
                subBestEE = EE
                subOpt_clustering_policy_UE = clustering_policy_UE
                subOpt_caching_policy_BS = caching_policy_BS
        return subBestEE,subOpt_clustering_policy_UE,subOpt_caching_policy_BS

    def close(self):
        pass

if __name__ == "__main__":
    # create Channel Trajectory
    #env = BS(nBS=4,nUE=4,nMaxLink=2,nFile=5,nMaxCache=2,loadENV = True,SEED=0,obsIdx=1)
    env = BS(nBS=10,nUE=5,nMaxLink=3,nFile=20,nMaxCache=2,loadENV = True,SEED=0,obsIdx=1)
    channelTrajectory = []
    for i in range(100):
        channelTrajectory.append(env.g)
        env.timeVariantChannel()
    filename = 'data/'+env.TopologyCode+'/Topology/['+str(env.SEED)+']Topology_'+ env.TopologyName #+ str(today)
    with open(filename+'CT.pkl', 'wb') as f:
        pickle.dump(channelTrajectory, f)
    
    with open(filename+'CT.pkl','rb') as f: 
        channelTrajectoryX = pickle.load(f)
    print(len(channelTrajectoryX))
    #-------------------------------------------------------
    nMaxLink = 3
    tryCount = 100
    # BM1 Initialization
    poolEE_BM1 = [[0]*tryCount  for i in range(nMaxLink+1)]
    poolTP_BM1 = [[0]*tryCount  for i in range(nMaxLink+1)]
    poolPsys_BM1=[[0]*tryCount  for i in range(nMaxLink+1)]
    poolHR_BM1 = [[0]*tryCount  for i in range(nMaxLink+1)]
    poolMCAP_BM1=[[0]*tryCount  for i in range(nMaxLink+1)]
    poolMCCPU_BM1=[[0]*tryCount  for i in range(nMaxLink+1)]
    poolCL_BM1 = [[0]*tryCount  for i in range(nMaxLink+1)]
    poolCA_BM1 = [[0]*tryCount  for i in range(nMaxLink+1)]
    # BM2 Initialization
    poolEE_BM2 = [[0]*tryCount  for i in range(nMaxLink+1)]
    poolTP_BM2 = [[0]*tryCount  for i in range(nMaxLink+1)]
    poolPsys_BM2=[[0]*tryCount  for i in range(nMaxLink+1)]
    poolHR_BM2 = [[0]*tryCount  for i in range(nMaxLink+1)]
    poolMCAP_BM2=[[0]*tryCount  for i in range(nMaxLink+1)]
    poolMCCPU_BM2=[[0]*tryCount  for i in range(nMaxLink+1)]
    poolCL_BM2 = [[0]*tryCount  for i in range(nMaxLink+1)]
    poolCA_BM2 = [[0]*tryCount  for i in range(nMaxLink+1)]
    # BM3 Initialization
    poolEE_BM3 = [[0]*tryCount  for i in range(nMaxLink+1)]
    poolTP_BM3 = [[0]*tryCount  for i in range(nMaxLink+1)]
    poolPsys_BM3=[[0]*tryCount  for i in range(nMaxLink+1)]
    poolHR_BM3 = [[0]*tryCount  for i in range(nMaxLink+1)]
    poolMCAP_BM3=[[0]*tryCount  for i in range(nMaxLink+1)]
    poolMCCPU_BM3=[[0]*tryCount  for i in range(nMaxLink+1)]
    poolCL_BM3 = [[0]*tryCount  for i in range(nMaxLink+1)]
    poolCA_BM3 = [[0]*tryCount  for i in range(nMaxLink+1)]
    for i in range(0,tryCount):
        print('Current Random seed:',i)
        # DDPG Parameter
        SEED =  i# random seed
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        # Build ENV
        # env = BS(nBS=4,nUE=4,nMaxLink=2,nFile=5,nMaxCache=2,loadENV = True,SEED=i)
        env = BS(nBS=10,nUE=5,nMaxLink=nMaxLink,nFile=20,nMaxCache=2,loadENV = True,SEED=i)
        env.Req[1]=10
        env.Req[2]=10
        filename = 'data/'+env.TopologyCode+'/Topology/['+str(99)+']Topology_'+ env.TopologyName #+ str(today)
        with open(filename + '.pkl', 'wb') as f: 
            pickle.dump([env.bs_coordinate, env.u_coordinate, env.pl, env.h, env.g, env.userPreference, env.Req], f)
        plot_UE_BS_distribution_Cache(env,None,None,0,filename,isEPS=False)
        filename = 'data/'+env.TopologyCode+'/EVSampledPolicy_Topology/'+ env.TopologyName +'_Evaluation_'
        # iterate each L
        for l in range(1,env.L+1):
            #------------------------------------------------------------------------------------------------
            # Benchmark 1    
            EE_BM1, CL_Policy_UE_BM1, CA_Policy_BS_BM1 = env.getPolicy_BM1(cacheMode='pref',nLink=l)
            EE_BM1 = env.calEE(CL_Policy_UE_BM1,CA_Policy_BS_BM1)
            TP_BM1 = sum(env.Throughput)
            Psys_BM1 = env.P_sys/1000 # mW->W
            HR_BM1 = env.calHR(CL_Policy_UE_BM1,CA_Policy_BS_BM1)
            plot_UE_BS_distribution_Cache(env, CL_Policy_UE_BM1, CA_Policy_BS_BM1, EE_BM1,filename+'BM1'+'_L'+str(l),isDetail=False,isEPS=False)

            poolEE_BM1[l][i]    =EE_BM1
            poolTP_BM1[l][i]    =TP_BM1
            poolPsys_BM1[l][i]  =Psys_BM1
            poolHR_BM1[l][i]    =HR_BM1
            poolMCAP_BM1[l][i]  =env.missCounterAP
            poolMCCPU_BM1[l][i] =env.missCounterCPU
            poolCL_BM1[l][i]    =CL_Policy_UE_BM1
            poolCA_BM1[l][i]    =CA_Policy_BS_BM1
            print('EE_BM1'+'_L'+str(l),'=', EE_BM1)
            #------------------------------------------------------------------------------------------------
            # Benchmark 2  
            EE_BM2, CL_Policy_UE_BM2, CA_Policy_BS_BM2 = env.getPolicy_BM2(nLink=l)
            EE_BM2 = env.calEE(CL_Policy_UE_BM2,CA_Policy_BS_BM2)
            TP_BM2 = sum(env.Throughput)
            Psys_BM2 = env.P_sys/1000 # mW->W
            HR_BM2 = env.calHR(CL_Policy_UE_BM2,CA_Policy_BS_BM2)
            plot_UE_BS_distribution_Cache(env, CL_Policy_UE_BM2, CA_Policy_BS_BM2, EE_BM2,filename+'BM2'+'_L'+str(l),isDetail=False,isEPS=False)

            poolEE_BM2[l][i]    =EE_BM2
            poolTP_BM2[l][i]    =TP_BM2
            poolPsys_BM2[l][i]  =Psys_BM2
            poolHR_BM2[l][i]    =HR_BM2
            poolMCAP_BM2[l][i]  =env.missCounterAP
            poolMCCPU_BM2[l][i] =env.missCounterCPU
            poolCL_BM2[l][i]    =CL_Policy_UE_BM2
            poolCA_BM2[l][i]    =CA_Policy_BS_BM2
            print('EE_BM2'+'_L'+str(l),'=', EE_BM2)
            #------------------------------------------------------------------------------------------------
            # testing Benchmark 3
            EE_BM3, CL_Policy_UE_BM3, CA_Policy_BS_BM3 = env.getPolicy_BM3(nLink=l)
            EE_BM3 = env.calEE(CL_Policy_UE_BM3,CA_Policy_BS_BM3)
            TP_BM3 = sum(env.Throughput)
            Psys_BM3 = env.P_sys/1000 # mW->W
            HR_BM3 = env.calHR(CL_Policy_UE_BM3,CA_Policy_BS_BM3)
            plot_UE_BS_distribution_Cache(env, CL_Policy_UE_BM3, CA_Policy_BS_BM3, EE_BM3,filename+'BM3'+'_L'+str(l),isDetail=False,isEPS=False)

            poolEE_BM3[l][i]    =EE_BM3
            poolTP_BM3[l][i]    =TP_BM3
            poolPsys_BM3[l][i]  =Psys_BM3
            poolHR_BM3[l][i]    =HR_BM3
            poolMCAP_BM3[l][i]  =env.missCounterAP
            poolMCCPU_BM3[l][i] =env.missCounterCPU
            poolCL_BM3[l][i]    =CL_Policy_UE_BM3
            poolCA_BM3[l][i]    =CA_Policy_BS_BM3
            print('EE_BM3'+'_L'+str(l),'=', EE_BM3)
            
        
        #------------------------------------------------------------------------------------------------
        # Derive Policy: BF
        #EE_BF, BF_CL_Policy_UE, BF_CA_Policy_BS = env.getOptEE_BF(isSave=True)
        #------------------------------------------------------------------------------------------------
        
    # Load the whole environment with Optimal Clustering and Optimal Caching   
    filenameBF = 'data/'+env.TopologyCode+'/BF/['+str(SEED)+']'+env.TopologyName+'BF'
    with open(filenameBF+'.pkl','rb') as f: 
        envX, EE_BF, BF_CL_Policy_UE, BF_CA_Policy_BS = pickle.load(f)
    EE_BF = envX.calEE(BF_CL_Policy_UE,BF_CA_Policy_BS)
    TP_BF = sum(envX.Throughput)
    Psys_BF = envX.P_sys/1000 # mW->W
    # Plot
    #filename = 'data/'+env.TopologyCode+'/EVSampledPolicy/'+'['+ str(SEED) +']'+ env.TopologyName +'_EVSampledPolicy_'
    filename = 'data/'+env.TopologyCode+'/BF/'+'['+ str(SEED) +']'+ env.TopologyName +'_EVSampledPolicy_'
    plot_UE_BS_distribution_Cache(envX, BF_CL_Policy_UE, BF_CA_Policy_BS, EE_BF,filename+'BF',isDetail=True,isEPS=False)