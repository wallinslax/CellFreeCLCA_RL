# Public Lib
import gym
from gym import spaces
import numpy as np
from numpy import linalg as LA
from numpy.random import randn
from random import randint
import scipy.stats
import os,math,random,itertools,csv,pickle,inspect
from itertools import combinations,permutations,product
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
beta = 2 # zipf parameter

d_th = 0.2*1000 # distance theshold for clustering policy candidate 200m
'''
nBS = 4  # number of BSs
nUE = 12 # number of UEs
nMaxLink = 2 # Max allowed connected BS for a single UE
F = 5 # number of total files
N = 2 # capacity of BSs
'''
P_t = 10 # transmit power of SBS 10dbm = 10mW
P_SBS = 10 # transmit power of SBS 10dbm = 10mW
P_MBS = 31.6 # transmit power of MBS 15dbm = 31.6mW
P_l = 20 # data retrieval power from local cache = 20mW
P_bh = 500 # data retrieval power from backhaul = 500mW (AP--CPU)
P_bb = 500 # data retrieval power from backbone = 500mW (CPU--Backbone)
P_o_SBS = 1500 # operational power of SBS = 1500mW
P_o_MBS = 2500 # operational power of MBS =2500mW
n_var = 2*(10**-13) # thermal noise power =  -127 (dBm) =1.995262315e-13(mW)
#####################################
'''
SEED = 0
np.random.seed(SEED)
'''
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

def plot_UE_BS_distribution_Cache(bs_coordinate,u_coordinate,Req,clustering_policy_UE,caching_policy_BS,EE,filename):
    plt.cla()
    # AP
    xx_bs = bs_coordinate[:,0]
    yy_bs = bs_coordinate[:,1]
    plt.scatter(xx_bs, yy_bs, edgecolor='k', facecolor='k',marker='^', alpha=1 ,label='AP')
    b = 0
    for x,y in zip(xx_bs, yy_bs):
        plt.annotate("%s" % b, xy=(x,y), xytext=(x, y-0.03),color='k')#label index
        if caching_policy_BS:
            plt.annotate("%s" % 'cache:'+str(caching_policy_BS[b]), xy=(x,y), xytext=(x, y),color='k')#label cache
        b = b+1
    # UE
    '''
    xx_u = u_coordinate[:,0]
    yy_u = u_coordinate[:,1]
    plt.scatter(xx_u, yy_u, edgecolor='b', facecolor='none',marker='X', alpha=0.5 ,label='UE')
    u = 0
    for x,y in zip(xx_u,yy_u):
        plt.annotate("%s" % u, xy=(x,y), xytext=(x, y-0.03),color='b')#label index
        plt.annotate("%s" % 'Req:'+str(Req[u]), xy=(x,y), xytext=(x, y),color='red')
        u = u+1
    '''
    #cluster plot
    #color =np.array( ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'fuchsia','peachpuff','pink'])
    nUE=len(u_coordinate)
    color=cm.rainbow(np.linspace(0,1,nUE))
    for u in range(nUE):
        xx_u = u_coordinate[u,0]
        yy_u = u_coordinate[u,1]
        plt.scatter(xx_u, yy_u, edgecolor=color[u], facecolor='none',marker='X', alpha=0.5 ,label='UE'+str(u))
        plt.annotate("%s" % u, xy=(xx_u,yy_u), xytext=(xx_u, yy_u-0.03),color=color[u])#label index
        plt.annotate("%s" % 'Req:'+str(Req[u]), xy=(xx_u,yy_u), xytext=(xx_u, yy_u),color=color[u])
        if clustering_policy_UE:
            useBS = clustering_policy_UE[u]
            for bs in useBS:
                xx_bs = bs_coordinate[bs,0]
                yy_bs = bs_coordinate[bs,1]
                plt.plot([xx_u,xx_bs],[yy_u,yy_bs],linestyle='--',color=color[u])
            
    plt.xlabel("x (km)"); plt.ylabel("y (km)")
    plt.title(filename+'\nwith EE:'+str(EE))
    plt.axis('equal')
    #plt.legend(loc='upper right')
    plt.legend()
    #plt.show()
    fig = plt.gcf()
    if filename:
        #fig.savefig(filename + '.eps', format='eps',dpi=1200)
        fig.savefig(filename + '.png', format='png',dpi=1200)
    fig.show()
    fig.canvas.draw()

def genUserRequest(userPreference):
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

class BS(gym.Env):
    def get_statistic(self):
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
        
    def channel_reset(self):
        '''[2] Pair-wise distance'''
        D = np.zeros((self.U,self.B))
        for u in  range(self.U):
            for b in range(self.B):
                D[u][b] = 1000*np.sqrt(sum((self.u_coordinate[u]-self.bs_coordinate[b])**2)) # km -> m
        D0=min(D.reshape(self.U*self.B))
        D/=D0
        '''[3] Large scale fading'''
        pl = k*np.power(D, -alpha) # Path-loss

        '''[4] Small scale fading'''
        h = np.sqrt(h_var/2) * (randn(self.U,self.B)+1j*randn(self.U,self.B)) # h~CN(0,1); |h|~Rayleigh fading
        g = np.transpose(pl*h) 
        return g

        h_conj=h.conjugate() 
        h_sqt=(h*h_conj).real

    def __init__(self,nBS,nUE,nMaxLink,nFile,nMaxCache,loadENV):
        self.B = nBS # number of BS
        self.U = nUE # number of UE
        self.L = nMaxLink # Max Link Capability of UE
        self.F = nFile # number of total files
        self.N = nMaxCache # Max cache size of BS
        
        filename = 'data/Topology_'+str(self.B)+'AP_'+str(self.U)+'UE_'+ str(self.F) + 'File_'+ str(self.N) +'Cache_' #+ str(today)
        if(loadENV):# load Topology
            with open(filename + '.pkl','rb') as f: 
                self.bs_coordinate, self.u_coordinate, self.g, self.userPreference, self.Req = pickle.load(f)
        else:
            '''[1] SBS/ UE distribution''' 
            self.u_coordinate = np.random.rand(self.U, 2)-0.5
            sbs_coordinate = np.random.rand(self.B-1, 2)-0.5
            self.bs_coordinate = np.concatenate((np.array([[0,0]]),sbs_coordinate),axis=0) 
            self.g = self.channel_reset()# received power of each UE
            '''[2] Generate User Preference'''
            # User Preference is a score list of for each file. Score 0 is the most favorite.
            # i.e. userPreference[0] = [3 2 0 1 4], the most favorate file of UE0 is 2th file, the second favorite file is 3th file
            self.userPreference = np.zeros((self.U,self.F),dtype=int)
            for u in range(self.U):
                seedPreference = np.arange(self.F)
                np.random.shuffle(seedPreference)
                self.userPreference[u] = seedPreference
            #print(userPreference)
            '''[3] Generate User request''' 
            self.Req = np.zeros(self.U,dtype=int)
            for u in range(self.U):
                self.Req[u] = genUserRequest(self.userPreference[u])
            # check topology
            plot_UE_BS_distribution_Cache(self.bs_coordinate,self.u_coordinate,self.Req,None,None,0,filename)
            # save Topology
            with open(filename + '.pkl', 'wb') as f: 
                pickle.dump([self.bs_coordinate, self.u_coordinate, self.g, self.userPreference, self.Req], f)
            #self.EE_mean,self.EE_std,self.CS_mean,self.CS_std = self.get_statistic()
        '''
        # Debug: self.userPreference
        print('self.userPreference[0]=',self.userPreference[0])
        for i in range(100):
            print(genUserRequest(self.userPreference[0]))
        '''
        self.EE_mean = 0
        self.EE_std = 1
        self.CS_mean = 0
        self.CS_std = 1
        self.done = False
        
        '''[9] popular method to determine clustering and caching policy'''
        self.nearestClustering = np.zeros([self.U,self.B],dtype=int)
        self.optCacheTopN = np.zeros([self.B,self.N],dtype=int)
        self.estCacheTopN = np.zeros([self.B,self.N],dtype=int)
        '''[10] Content request profile of each UE'''
        self.reqStatistic = np.zeros([self.U,self.F],dtype=int)
        self.userSimilarity = np.zeros([self.U,self.U],dtype=int)

        '''[13] System power consumption'''
        self.P_sys = 0
        '''[14] Energy efficiency'''
        self.EE = 0 
        '''[15] Content request profile similarity'''
        self.ueSimilarity = np.zeros([self.U, self.U]) 
        '''[16] Intra-cluster similarity'''
        self.ICS = np.zeros(self.B) 
        '''[17] Cluster Similarity'''
        self.CS = 0
        '''[19] State'''
        self.SINR = np.zeros(self.U)
        self.clustering_state = np.zeros(self.B*self.U)
        self.caching_state = np.zeros(self.B*self.F)
        self.reqStatistic_norm = np.zeros(self.U*self.F)
        
        self.s_ = np.hstack([   self.SINR,
                                self.clustering_state.flatten(),
                                self.caching_state.flatten(),
                                self.reqStatistic_norm.flatten()])
        ''' 
        self.s_ = np.hstack([  self.g.real.flatten(),
                                self.g.imag.flatten(),
                                self.clustering_state.flatten(),
                                self.caching_state.flatten(),
                                self.reqStatistic_norm.flatten(),
                                self.Req.flatten()]) 
                                
        self.s_ = np.hstack([  self.clustering_state.flatten(),
                                self.caching_state.flatten(),
                                self.reqStatistic_norm.flatten(),
                                self.Req.flatten()]) 
        ''' 
        
        self.dimActCL = self.B*self.U
        self.dimActCA = self.B*self.F
        self.dimAct = self.dimActCL + self.dimActCA
        self.dimObs = len(self.s_)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.dimAct,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=float("inf"), shape=(len(self.s_),), dtype=np.float32)

    def reset(self):  
        self.SINR = np.zeros(self.U)
        self.action_space.sample()
        self.clustering_state = np.zeros(self.B*self.U)
        self.caching_state = np.zeros(self.B*self.F)
        self.reqStatistic_norm = np.zeros(self.U*self.F)
        
        self.s_ = np.hstack([   self.SINR,
                                self.clustering_state.flatten(),
                                self.caching_state.flatten(),
                                self.reqStatistic_norm.flatten()])
        '''
        self.s_ = np.hstack([  self.g.real.flatten(),
                                self.g.imag.flatten(),
                                self.clustering_state.flatten(),
                                self.caching_state.flatten(),
                                self.reqStatistic_norm.flatten(),
                                self.Req.flatten()]) 
        
        self.s_ = np.hstack([  self.clustering_state.flatten(),
                                self.caching_state.flatten(),
                                self.reqStatistic_norm.flatten(),
                                self.Req.flatten()]) 
        '''
        return self.s_
    
    def nearestClustering_TopNCache(self,cacheMode):
        
        g_abs = abs(self.g)
        #print(g_abs)
        clustering_policy_UE = []  
        # kth UE determine the AP set (S_k)     
        for u in range(self.U):
            bestBS = g_abs[u].argsort()[::-1][:self.L]
            clustering_policy_UE.append(bestBS)

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
            sumUserPreferenceInCluster = np.sum(self.userPreference[ clustering_policy_BS[b] ],axis=0)
            top_N_idx = sumUserPreferenceInCluster.argsort()[0:self.N]
            optCacheTopN.append(top_N_idx)

        '''[] caching based on reqStatistic [estimated] top_N_idx'''
        estCacheTopN = []
        for b in range (self.B):
            sumUserPreferenceInCluster = np.sum(self.reqStatistic[ clustering_policy_BS[b] ],axis=0)
            top_N_idx = sumUserPreferenceInCluster.argsort()[0:self.N]
            estCacheTopN.append(top_N_idx)
        
        if cacheMode == 'req':
            caching_policy_BS = reqCacheTopN
        elif cacheMode == 'pref':
            caching_policy_BS = optCacheTopN
        elif cacheMode == 'stat':
            caching_policy_BS = estCacheTopN

        return clustering_policy_UE,caching_policy_BS
        
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
            #print(connectionScore[u])
            #print(connectionScore[u].argsort())
            #print(connectionScore[u].argsort()[::-1][:self.L])
            bestLBS = connectionScore[u].argsort()[::-1][:self.L]
            selectedBS = [ i for (i,v) in enumerate(connectionScore[u]) if v >= 0 ]
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
        '''
        if inspect.stack()[1][3] != 'bruteForce' :
            for u in range(self.U):
                self.Req[u] = genUserRequest(self.userPreference[u])
        '''
        '''[3] Hit event'''
        self.Hit = np.zeros(self.U)
        for u in range(self.U):
            useBS = clustering_policy_UE[u]
            counter = 0
            for bs in useBS:
                if self.Req[u] in caching_policy_BS[bs]:
                    counter+=1
            if counter == len(useBS):
                self.Hit[u]=1
        self.Hit_rate = sum(self.Hit)/len(self.Hit)
        #[CS]##############################################################################################################
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
        '''[18] State'''
        #convert culstering policy to binary form
        clustering_state = np.zeros([self.U,self.B])
        for u in range(self.U):
            clustering_state[u][ list(clustering_policy_UE[u]) ]=1

        #convert caching policy to binary form
        caching_state = np.zeros([self.B,self.F])
        for b in range(self.B):
            caching_state[b][ list(caching_policy_BS[b]) ] = 1
        
        self.s_ = np.hstack([  self.SINR,
                                self.clustering_state.flatten(),
                                self.caching_state.flatten(),
                                self.reqStatistic_norm.flatten()])
        ''' 
        self.s_ = np.hstack([  self.g.real.flatten(),
                                self.g.imag.flatten(),
                                self.clustering_state.flatten(),
                                self.caching_state.flatten(),
                                self.reqStatistic_norm.flatten(),
                                self.Req.flatten()])  
        
        self.s_ = np.hstack([   self.clustering_state.flatten(),
                                self.caching_state.flatten(),
                                self.reqStatistic_norm.flatten(),
                                self.Req.flatten()])  
        ''' 
        
        '''[19] Whether episode done'''
        observation = self.s_
        reward = self.EE
        done = self.done
        info = {"HR":self.Hit_rate}

        #return self.EE, self.Hit_rate, self.s_, self.done
        return observation, reward, done, info

    def calEE(self,clustering_policy_UE,caching_policy_BS):
        '''[1] clustering_policy_BS'''
        clustering_policy_BS = []
        for b in range(self.B):
            competeUE = []
            for u in range(self.U):
                if b in clustering_policy_UE[u]:
                    competeUE.append(u) #the UE set in b-th cluster   
            clustering_policy_BS.append(competeUE)

        '''[4] rho_b'''
        self.rho = np.zeros(self.B)
        for b in range(self.B):
            competeUE = clustering_policy_BS[b]
            #print(self.g[b][competeUE])
            #print(sum( np.power(abs(self.g[b][competeUE]),2) ))
            if len(competeUE) != 0:
                self.rho[b] = P_SBS / sum( np.power(abs(self.g[b][competeUE]),2) )
            #print( self.rho[b] )
        
        '''[5] received power'''
        self.P_r = np.zeros(self.U)
        for u in range(self.U):
            for b in clustering_policy_UE[u]:
                self.P_r[u] += np.sqrt(self.rho[b]) * np.power(abs(self.g[b][u]),2)
            self.P_r[u] = np.power(self.P_r[u],2)

        '''[6] Interference'''
        self.I = np.zeros(self.U)
        for u in range(self.U):
            other_u = list(range(self.U))
            other_u.remove(u)
            #print(other_u)
            for uu in other_u:
                sum_b = 0
                for b in range(self.U):
                    #chk = self.g[b][u] * self.g[b][uu].conjugate() 
                    #print(chk)
                    sum_b +=  np.sqrt(self.rho[b]) * self.g[b][u]*self.g[b][uu].conjugate()
                self.I[u] = self.I[u] + np.power(abs(sum_b),2)
        
        '''[7] SINR/ [8]Throughput of UE'''
        self.SINR = np.zeros(self.U) 
        self.Throughput = np.zeros(self.U)
        for u in range(self.U):
            self.SINR[u] = self.P_r[u]/(self.I[u] + n_var)
            self.Throughput[u] = math.log2(1+self.SINR[u]) #Bits/s

        '''[9] System power consumption'''
        missCounterAP = 0
        missFileCPU = []
        for u in range(self.U):
            useBS = clustering_policy_UE[u]
            for bs in useBS:
                if self.Req[u] not in caching_policy_BS[bs]: #Miss
                    missCounterAP += 1
                    missFileCPU.append(self.Req[u])
        missFileCPU = list(set(missFileCPU)) 
                     
        self.P_sys = P_t*self.B + P_bh*missCounterAP + P_bb*len(missFileCPU) # + self.B*P_o_SBS + P_o_MBS 
        
        '''[10] Energy efficiency'''
        self.EE = sum(self.Throughput)/(self.P_sys/1000) # Bits/s*W mW->W
        self.EE_norm = (self.EE-self.EE_mean)/self.EE_std # Z-score normalization
        return self.EE

    def bruteForce(self):
        print("this is brute force for EE")
        # generate all posible clustering_policy_UE
        choiceBS = []
        for i in range(self.B+1):
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
            for future in tqdm(concurrent.futures.as_completed(futures),total=len(futures)):
                #print(future.result())
                subBestEE,subOpt_clustering_policy_UE,subOpt_caching_policy_BS = future.result()
                if subBestEE>bestEE:
                    bestEE = subBestEE
                    opt_clustering_policy_UE = subOpt_clustering_policy_UE
                    opt_caching_policy_BS = subOpt_caching_policy_BS
        
        

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
    # Build ENV
    env = BS(nBS=10,nUE=4,nMaxLink=1,nFile=5,nMaxCache=2,loadENV = True)
    # env = BS(nBS=40,nUE=10,nMaxLink=2,nFile=5,nMaxCache=2,loadENV = True)
    for L in range(1,11):
        env = BS(nBS=10,nUE=4,nMaxLink=L,nFile=5,nMaxCache=2,loadENV = True)
        nearnest_clustering_policy_UE, topN_caching_policy_BS = env.nearestClustering_TopNCache(cacheMode='pref')
        #nearnest_clustering_policy_UE, topN_caching_policy_BS = env.nearestClustering_TopNCache(cacheMode='req')
        nctc_EE = env.calEE(nearnest_clustering_policy_UE,topN_caching_policy_BS)
        print('L=',L,',nctc_EE=',nctc_EE)
    #------------------------------------------------------------------------------------------------
    # Derive Policy: nearestClustering_TopNCache
    env = BS(nBS=10,nUE=4,nMaxLink=4,nFile=5,nMaxCache=2,loadENV = True)
    nearnest_clustering_policy_UE, topN_caching_policy_BS = env.nearestClustering_TopNCache(cacheMode='pref')
    nctc_EE = env.calEE(nearnest_clustering_policy_UE,topN_caching_policy_BS)
    
    # Save the whole environment
    filenameNCTC = 'data/nearCL+TopNCA_'+str(env.B)+'AP_'+str(env.U)+'UE_'+ str(env.L) + 'L_' + str(env.F) + 'File_'+ str(env.N) +'Cache_' + str(today)
    with open(filenameNCTC+'.pkl', 'wb') as f:  
        pickle.dump([env.bs_coordinate, env.u_coordinate, env.g, env.userPreference, env.Req, nctc_EE, nearnest_clustering_policy_UE, topN_caching_policy_BS], f)
    # Load the whole environment
    with open(filenameNCTC+'.pkl','rb') as f: 
        bs_coordinate, u_coordinate , g, userPreference, Req, nctc_EE, nearnest_clustering_policy_UE, topN_caching_policy_BS = pickle.load(f)
    
    # Plot
    filenameNCTC = 'data/PolicyVisualized_nearCL+TopNCA_'+str(env.B)+'AP_'+str(env.U)+'UE_'+ str(env.L) + 'L_'+ str(env.F) + 'File_'+ str(env.N) +'Cache_' + str(today)
    plot_UE_BS_distribution_Cache(bs_coordinate,u_coordinate,Req,nearnest_clustering_policy_UE,topN_caching_policy_BS,nctc_EE,filenameNCTC)
    print('nctc_EE=',nctc_EE)
    print('nearnest_clustering_policy_UE=',nearnest_clustering_policy_UE)
    print('topN_caching_policy_BS=',topN_caching_policy_BS)
    #------------------------------------------------------------------------------------------------
    # Derive Policy: BF
    bestEE, opt_clustering_policy_UE, opt_caching_policy_BS = env.bruteForce()
    
    # Save the whole environment with Best Clustering and Best Caching
    filenameBF = 'data/BF_'+str(env.B)+'AP_'+str(env.U)+'UE_'+ str(env.F) + 'File_'+ str(env.N) +'Cache_' + str(today)
    with open(filenameBF+'.pkl', 'wb') as f:  
        pickle.dump([env.bs_coordinate, env.u_coordinate, env.g, env.userPreference, env.Req, bestEE, opt_clustering_policy_UE, opt_caching_policy_BS], f)
    
    # Load the whole environment with Best Clustering and Best Caching   
    filenameBF = 'data/BF_4AP_4UE_5File_2Cache_2020-11-10'
    with open(filenameBF+'.pkl','rb') as f: 
        bs_coordinate, u_coordinate , g, userPreference, Req, bestEE, opt_clustering_policy_UE, opt_caching_policy_BS = pickle.load(f)
    
    # Plot
    filenameBF = 'data/PolicyVisualized_BF_'+str(env.B)+'AP_'+str(env.U)+'UE_'+ str(env.F) + 'File_'+ str(env.N) +'Cache_' + str(today)
    plot_UE_BS_distribution_Cache(bs_coordinate,u_coordinate,Req,opt_clustering_policy_UE,opt_caching_policy_BS,bestEE,filenameBF)
    print('bestEE=',bestEE)
    print('opt_clustering_policy_UE=',opt_clustering_policy_UE)
    print('opt_caching_policy_BS=',opt_caching_policy_BS)
    #------------------------------------------------------------------------------------------------