import numpy as np
from numpy.random import randn
import scipy.stats
import matplotlib.pyplot as plt
from random import randint
import random
import itertools
from itertools import combinations,product
import math
from numpy import linalg as LA
import csv
#####################################
radius = 0.5 # radius of the network = 500m
lambda_bs = 6 # density of the BS
lambda_u = 10 # density of the UE
num_bs = 3 # number of BSs
num_u = 10 # number of UEs

h_var = 1 # channel variance
k = 1 # pathloss coefficient
alpha = 2 # pathloss exponent

d_th = 0.2*1000 # distance theshold for clustering policy candidate 200m

W = 10*15*10**3 # BW owned by BS = 10*15kHz
M = 4 # number of subcarrier owned by BS 

F = 5 # number of total files
N = 2# capacity of BSs
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

def plot_UE_SBS_distribution(xx_bs,yy_bs,xx_u,yy_u):
    # MBS
    plt.scatter(0,0,edgecolor='r', facecolor='r',marker='^', s=100 ,alpha=1 ,label='MBS')
    # SBS
    plt.scatter(xx_bs, yy_bs, edgecolor='g', facecolor='g', alpha=1 ,label='SBS')
    b = 1
    for x,y in zip(xx_bs,yy_bs):
        plt.annotate("%s" % b, xy=(x,y), xytext=(x, y),color='g')
        b = b+1
    # UE
    plt.scatter(xx_u, yy_u, edgecolor='b', facecolor='none',marker='X', alpha=0.5 ,label='UE')
    u = 0
    for x,y in zip(xx_u,yy_u):
        plt.annotate("%s" % u, xy=(x,y), xytext=(x, y),color='b')
        u = u+1
    plt.xlabel("x (km)"); plt.ylabel("y (km)")
    plt.title('Distribution of UE & BS')
    plt.axis('equal')
    plt.legend(loc='upper right')
    plt.show()
    
def plot_UE_SBS_association(xx_bs,yy_bs,xx_u,yy_u,L_ii,Req,f2b,up,filename1,filename2):
    color =np.array( ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w','blue', 'red', 'fuchsia','peachpuff','pink'])
    # MBS
    plt.scatter(0,0,color=color[0],marker='^', s=100 ,alpha=1)
    plt.annotate("%s" % f2b[0][0], xy=(0,0), xytext=(0, 0),color='fuchsia') # cached content
    plt.annotate("%s" % f2b[0][1], xy=(0,0), xytext=(0, 0+0.05),color='fuchsia')
    # SBS
    #plt.scatter(xx_bs, yy_bs, edgecolor='g', facecolor='g', alpha=1 ,label='SBS')
    b = 1
    for x,y in zip(xx_bs,yy_bs):
        plt.scatter(x, y, edgecolor=color[b],facecolor=color[b], alpha=1 )
#        plt.scatter(x, y, alpha=1 )
        #plt.annotate("%s" % b, xy=(x,y), xytext=(x, y),color=color[b])
        plt.annotate("%s" % b, xy=(x,y), xytext=(x, y-0.05),color='g')
        plt.annotate("%s" % f2b[b][0], xy=(x,y), xytext=(x, y),color='fuchsia') # cached content
        plt.annotate("%s" % f2b[b][1], xy=(x,y), xytext=(x, y+0.05),color='fuchsia')
        b = b+1
    # UE
    #plt.scatter(xx_u, yy_u, edgecolor='b', facecolor='none',marker='X', alpha=0.5 ,label='UE')
    u = 0
    top_1_idx = np.zeros(len(Req))
    top_2_idx = np.zeros(len(Req))
    for x,y in zip(xx_u,yy_u):
        plt.scatter(x, y, edgecolor=color[L_ii[u]],facecolor='none',marker='X', alpha=1 )
#        plt.scatter(x, y, marker='X', alpha=1 )
        #plt.annotate("%s" % u, xy=(x,y), xytext=(x,y),color='black') # u
        plt.annotate("%d" % u, xy=(x,y), xytext=(x, y-0.05),color='b')
        plt.annotate("%d" % Req[u], xy=(x,y), xytext=(x, y),color='black') # Req
        top_1_idx[u] = int(np.where(rank_list[up[u]]==0)[0][0])
        top_2_idx[u] = int(np.where(rank_list[up[u]]==1)[0][0])
        plt.annotate("%d" % top_1_idx[u], xy=(x,y), xytext=(x,y+0.05),color='fuchsia') # top_1_idx
        plt.annotate("%d" % top_2_idx[u], xy=(x,y), xytext=(x,y+0.1),color='fuchsia') # top_2_idx
        u = u+1
    plt.xlabel("x (km)"); plt.ylabel("y (km)")
    plt.title('Distribution of UE & BS')
    plt.axis('equal')
    plt.legend(loc='upper right')
    plt.savefig(filename1)
    plt.show()
    #
    with open(filename2, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # xx_bs,yy_bs,xx_u,yy_u,L_ii,Req,f2b,up
        writer.writerow(xx_bs)
        writer.writerow(yy_bs)
        writer.writerow(xx_u)
        writer.writerow(yy_u)
        writer.writerow(L_ii)
        writer.writerow(Req)
        writer.writerow(f2b[:,0])
        writer.writerow(f2b[:,1])
        writer.writerow(top_1_idx)
        writer.writerow(top_2_idx)
        
    
def generate_zipf_state(a,N,rank):
    # zipf pmf
    zipf_pmf_numerator = []
    for j in range(N):
        zipf_pmf_numerator.append((j+1)**(-a))
    zipf_pmf = list(np.true_divide(zipf_pmf_numerator, sum(zipf_pmf_numerator)))
    pmf = np.array(zipf_pmf)
    return pmf[rank]

def generate_user_request(zipf_pmf, residual_content_idx, n):
    # generate the user request (zipf RV)
    ur = []
    for i in range (n):
        x = random.random()*sum(zipf_pmf) # U[0,1]
        k = 0
        p = zipf_pmf[k]
        while x>p :
            x -= p
            k += 1
            p = zipf_pmf[k]
        ur.append(residual_content_idx[k])
    return (np.array(ur))


class BS(object):
    def __init__(self,bs_coordinate,u_coordinate,B,U,P,up,Req_duplicate,Get_statistic,EE_mean,EE_std,CS_mean,CS_std):
        self.U = U # number of UE
        self.B = B # number of BS
        self.F = F
        self.N = N
        self.u_coordinate,self.bs_coordinate = u_coordinate,bs_coordinate
        self.P = P
        self.up = up
        
        self.done = False
        self.csc_penalty = 0

        self.L, self.C_, self.J, self.C, self.clustering_policy_num, \
        self.us, self.cluster_size, self.cluster_member, self.I, self.zipf_content_popularity,self.top_N_idx, self.pop_caching_number, self.opt_caching_number, \
        self.Req, self.iat_of_d_content, self.Prof, self.estimate_content_popularity,self.estimate_top_N_idx,self.estimate_pop_caching_number,self.T, self.P_r,self.SINR, self.caching_policy_num, self.Hit, self.Hit_rate, self.P_bs_, \
        self.P_sys, self.E, self.S , self.ICS, self.CS,self.r, self.clustering_state,self.s_, self.Prof_state = self.initialization() 
        
        if (Get_statistic):
            self.EE_mean,self.EE_std,self.CS_mean,self.CS_std = self.get_statistic(Req_duplicate)
        else :
            self.EE_mean,self.EE_std,self.CS_mean,self.CS_std = EE_mean,EE_std,CS_mean,CS_std
        
        
    def get_statistic(self,Req_duplicate):
        print('Calculating statistic...')
        EE_sample_list = []
        CS_sample_list = []
        for i in range(10**4):
            random_clustering_policy_num = np.random.randint(0,self.B-1,size=self.U)
            random_caching_policy_num = np.random.randint(0,self.J-1,size=self.B) # suppose randomly assign caching policy to each BSs 
            r, done,EE, CS,csc_penalty, SINR, Hit_rate, s_, Itf= self.step(clustering_policy_num = random_clustering_policy_num,
                                                          caching_policy_num = random_caching_policy_num, 
                                                          update_Req = 1, 
                                                          normalize = 0,
                                                          Req_duplicate = Req_duplicate)
            EE_sample_list.append(EE)
            CS_sample_list.append(CS)
        EE_mean = np.mean(EE_sample_list)
        EE_std = np.std(EE_sample_list, ddof=1)
        CS_mean = np.mean(CS_sample_list)
        CS_std = np.std(CS_sample_list, ddof=1)
        print('EE_mean = '+str(EE_mean))
        print('EE_std = '+str(EE_std))
        print('CS_mean = '+str(CS_mean))
        print('CS_std = '+str(CS_std))
        return EE_mean,EE_std,CS_mean,CS_std
        
    def initialization(self):
        
        plot_UE_SBS_distribution(self.bs_coordinate[1:,0],self.bs_coordinate[1:,1],self.u_coordinate[:,0],self.u_coordinate[:,1])
        
        '''[4] UE-BS associated clustering policy set'''
        L = np.zeros((self.B,self.B),dtype=int) # Clustering policy space
        for b in range(self.B):
            L[b,b]= 1
        '''[5] UE-subcarrier associated channel selection policy'''
        clustering_policy_num = np.random.randint(0,self.B-1,size=self.U) # argmax(axis=1)
        L_ii = clustering_policy_num # suppose choose first policy
        cluster_member = []
        cluster_size = []
        us = np.zeros(self.U,dtype=int)#############################################if UE did not be assigned any channel???
        for b in range(self.B):
            b_cluster_UE = np.where(L_ii==b)[0]
            b_cluster_size = len(np.where(L_ii==b)[0])
            cluster_member.append(b_cluster_UE)
            cluster_size.append(b_cluster_size)
            if (b_cluster_size<=M):
                us[b_cluster_UE]=np.arange(b_cluster_size)
            else:
                t = np.zeros(b_cluster_size)
                t[:M] = np.arange(M)
                t[M:] = np.arange(len(t[M:]))
                us[b_cluster_UE] = t
        '''[8] Content caching policy set'''
        C_ = [list(i) for i in list( combinations (np.arange(F), N))] # mapping from action_num to index_of_file_to_cache
        J = len(C_) 
        C = np.zeros((J,F))
        for jj, file_to_cache in enumerate(C_):
            action = np.zeros(F)
            action[file_to_cache] = 1
            C[jj, :] = action
        
        '''[9] User request''' 
        # content popularity
        zipf_content_popularity = np.zeros((n_p,F))
        for i in range(n_p):
            zipf_content_popularity[i,:] = (generate_zipf_state(a = beta[i], N = F, rank = rank_list[i]))
        
        
        
        # pop_caching_number
        prob = []
        for i in range(n_p):
            prob.append(len(np.where(self.up==i)[0])/len(self.up))      
        p = np.sum(np.vstack((prob[0]*zipf_content_popularity[0],\
                              prob[1]*zipf_content_popularity[1],\
                              prob[2]*zipf_content_popularity[2])),axis=0)
        top_N_idx = np.sort(p.argsort()[::-1][0:N])
        pop_caching_number = C_.index(list(top_N_idx))
        
        # optimal caching number based on cluster top_N_idx
        opt_caching_number = np.ones(self.B)
        for b in range (self.B):
            p = np.sum(zipf_content_popularity[self.up[cluster_member[b]]],axis=0)
            top_N_idx = np.sort(p.argsort()[::-1][0:N])
            opt_caching_number[b] = C_.index(list(top_N_idx))
        
        # user request
        residual_content_idx = np.arange(self.F)
        Req = np.array([generate_user_request(zipf_pmf = zipf_content_popularity[self.up[u]], 
                                              residual_content_idx = residual_content_idx, 
                                              n = 1) 
                        for u in range(self.U)]).reshape(-1)
        
        # inter-arrival time of duplicate content
        iat_of_d_content = np.zeros((self.U,F)) 
        '''
        for u in range(self.U):
            for f in range(F):
                iat_of_d_content[u][f] = np.clip(np.random.geometric(p=zipf_content_popularity[up[u]][f], size=1),1,F)
        '''
        '''[10] Content request profile of each UE'''
        Prof = np.zeros((self.U,F))
        for u in range(self.U):
            Prof[u][Req[u]] = Prof[u][Req[u]] + 1
        estimate_content_popularity = np.sum(Prof,0)/np.sum(Prof)
        estimate_top_N_idx = np.sort(estimate_content_popularity.argsort()[::-1][0:N])
        estimate_pop_caching_number = C_.index(list(estimate_top_N_idx))
        '''[12] Hit event'''
        caching_policy_num = np.random.randint(0,J-1,size=self.B) # suppose randomly assign caching policy to each BSs
        
#        print(np.where(C[caching_policy_num]==1)[1].reshape((self.B,N)))
        
        Hit = np.zeros(self.U)
        for u in range(self.U):
            C_jj = C[caching_policy_num[L_ii[u]]]
            
            if(C_jj[Req[u]]==1):
                Hit[u]=1            
#            print(L_ii[u])
#            print(C_jj)
#            print(Req[u])
#            print(Hit[u])
        
        Hit_rate = sum(Hit)/len(Hit)
        ''''[7] Interference of UE at m-th frequency'''
        I = np.zeros(self.U)
        for u in range(self.U):
            m = us[u]
            use_m_UE = np.where(us==m)[0]
            use_m_BS = L_ii[use_m_UE]
            b_ = np.delete(use_m_BS, np.where(use_m_BS==L_ii[u])[0]) # no intra-cluster interference
            I[u] = np.sum(self.P[u,b_])
#            if L_ii[u]==0: # MBS
#                I[u] = 0
#            else:    
#                m = us[u]
#                use_m_UE = np.where(us==m)[0]
#                use_m_BS = L_ii[use_m_UE]
#                b_ = np.delete(use_m_BS, np.where(use_m_BS==L_ii[u])[0]) # no intra-cluster interference
#                b__ = np.delete(b_, np.where(b_==0)[0]) # no inter-tier interference
#                I[u] = np.sum(self.P[u,b__])
            '''
            b_ = np.delete(use_m_BS, np.where(use_m_BS==L_ii[u])[0][0]) # delete desired signal
            I[u] = np.sum(self.P[u,b_])
            '''
            '''
            hit_UE = np.where(Hit==1)[0]
            hit_BS = L_ii[hit_UE]
            b__ = [i for i in b_ if i in hit_BS]
            I[u] = np.sum(P[u,b__])
            '''
        '''[11] Throughput of UE'''
        T = np.zeros(self.U)
        SINR = np.zeros(self.U)
        P_r = np.zeros(self.U) # received power of each UE
        for u in range (self.U):
            P_r[u] = self.P[u][L_ii[u]]
            SINR[u] = P_r[u]/(I[u]+n_var)
            '''
            if Hit[u]:  
                SINR[u] = P[u][L_ii[u]]/(I[u]+n_var)
            else:
                SINR[u] = 0
            '''
            #T[u] = (W/M)*math.log2(1+(P[u][L_ii[u]]/(I[u][us[u]]+n_var)))
            T[u] = math.log2(1+SINR[u])
        '''[13] System power consumption'''
        P_bs_ = np.zeros(self.U)
        for u in range(self.U):
            if(L_ii[u]==0):P_bs_[u]=P_MBS
            else:P_bs_[u]=P_SBS
        P_sys = sum(P_bs_ + Hit*P_l + (1-Hit)*P_b) + len(np.where(cluster_member!=0)[0])*P_o_SBS + P_o_MBS   
        '''[14] Energy efficiency'''
        E = sum(T)/P_sys
        '''[15] Content request profile similarity'''
        Prof_normalize = Prof/(LA.norm(Prof, axis=1)).reshape((self.U,1))
        S = np.matmul(Prof_normalize, Prof_normalize.T)
        '''[16] Intra-cluster similarity'''
        ICS = np.zeros(self.B) # intra-cluster similarity of each cluster(BS)
        for b in range(self.B):
            UE_pair = [list(i) for i in list( combinations (cluster_member[0], 2))]
            for jj, pair in enumerate(UE_pair):
                ICS[b] = ICS[b] + S[pair[0]][pair[1]]
            if (len(UE_pair)):
                ICS[b] = ICS[b]/len(UE_pair)
        CS = sum(ICS)/self.B
        '''[17] Reward'''
        r = 0.8*E + 0.2*CS #+ 100*Hit_rate # 100*
#        r = sum(T)
#        r = Hit_rate 
        #r = E
        '''[18] State'''
        clustering_state = L[clustering_policy_num].flatten()
        caching_state = C[caching_policy_num].flatten()
        s_ = np.hstack((clustering_state,caching_state))
        Prof_state = Prof_normalize.flatten()
        return L, C_, J, C, clustering_policy_num, us, cluster_size, cluster_member, I, zipf_content_popularity,top_N_idx,pop_caching_number, opt_caching_number, Req, iat_of_d_content, Prof,estimate_content_popularity,estimate_top_N_idx,estimate_pop_caching_number, T, P_r, SINR, caching_policy_num, Hit, Hit_rate, P_bs_, P_sys, E, S, ICS, CS,r, clustering_state,s_, Prof_state  
    
     
    def reset(self):
        #return np.hstack((self.clustering_policy_num/self.II,self.caching_policy_num/self.J))
        return np.hstack((self.SINR,self.s_,self.Prof_state))

    def action_vec_2_num(self,a):
        #
        clustering_policy = a[:(self.U)*(self.B)].reshape((self.U,self.B))
        clustering_policy_num = clustering_policy.argmax(axis=1)
        #
        caching_policy = a[(self.U)*(self.B):].reshape((self.B,F))
        top_N_idx = [np.sort(caching_policy[i].argsort()[::-1][0:N]) for i in range(self.B)]
        caching_policy_num = np.array([self.C_.index(list(top_N_idx[i])) for i in range(self.B)])
        return clustering_policy_num,caching_policy_num
    
    def get_opt_caching_number(self,clustering_policy_num):
        # optimal caching number based on cluster top_N_idx
        self.clustering_policy_num = clustering_policy_num
        self.cluster_size.clear()
        self.cluster_member.clear() 
        L_ii = self.clustering_policy_num
        for b in range(self.B):
            b_cluster_UE = np.where(L_ii==b)[0]        
            self.cluster_member.append(b_cluster_UE)
            p = np.sum(self.zipf_content_popularity[self.up[self.cluster_member[b]]],axis=0)
            top_N_idx = np.sort(p.argsort()[::-1][0:N])
            self.opt_caching_number[b] = self.C_.index(list(top_N_idx))
        return self.opt_caching_number
        
    def step(self,clustering_policy_num,caching_policy_num,update_Req,normalize,Req_duplicate):
        self.clustering_policy_num = clustering_policy_num
        self.caching_policy_num = caching_policy_num
        self.cluster_size.clear()
        self.cluster_member.clear()
        self.done = False
        self.csc_penalty = 0
        '''[5] UE-subcarrier associated channel selection policy'''
        L_ii = self.clustering_policy_num
        for b in range(self.B):
            b_cluster_UE = np.where(L_ii==b)[0]
            b_cluster_size = len(np.where(L_ii==b)[0])           
            self.cluster_member.append(b_cluster_UE)
            self.cluster_size.append(b_cluster_size)
            if (b_cluster_size<=M):
                self.us[b_cluster_UE]=np.arange(b_cluster_size)
            else:
                t = np.zeros(b_cluster_size)
                t[:M] = np.arange(M)
                t[M:] = np.arange(len(t[M:]))
                self.us[b_cluster_UE] = t
                
        # optimal caching number based on cluster top_N_idx
        for b in range (self.B):
            p = np.sum(self.zipf_content_popularity[self.up[self.cluster_member[b]]],axis=0)
            top_N_idx = np.sort(p.argsort()[::-1][0:N])
            self.opt_caching_number[b] = self.C_.index(list(top_N_idx))
        ####################################################################
        if(update_Req):
            '''[9] User request''' 
            # user request
            if not (Req_duplicate):
                for u in range(self.U):
                    residual_content_idx = np.where(self.iat_of_d_content[u]==0)[0]
                    residual_zipf_pmf = self.zipf_content_popularity[self.up[u]][residual_content_idx]
                    if residual_content_idx.size!=0 :
                        self.Req[u] = generate_user_request(zipf_pmf = residual_zipf_pmf, 
                                                           residual_content_idx = residual_content_idx, 
                                                           n = 1)     
                    else:
                        self.Req[u] = False
                
                # update inter-arrival time
                self.iat_of_d_content[np.where(self.iat_of_d_content!=0)] -= 1 # counter minus 1
                for u in range(self.U):
                    self.iat_of_d_content[u][self.Req[u]] =  np.clip(np.random.geometric(p=self.zipf_content_popularity[self.up[u]][self.Req[u]], size=1),1,F)
            else :
#                self.Req = self.Req
                self.Req = np.array([generate_user_request(zipf_pmf = self.zipf_content_popularity[self.up[i]], 
                                                           residual_content_idx = np.arange(self.F),
                                                           n = 1) 
                            for i in range(self.U)]).reshape(-1)
            
            '''[10] Content request profile of each UE'''
            for u in range(self.U):
                self.Prof[u][self.Req[u]] = self.Prof[u][self.Req[u]] + 1
            self.estimate_content_popularity = np.sum(self.Prof,0)/np.sum(self.Prof)
            self.estimate_top_N_idx = np.sort(self.estimate_content_popularity.argsort()[::-1][0:N])
            self.estimate_pop_caching_number = self.C_.index(list(self.estimate_top_N_idx))
            '''[15] Content request profile similarity'''
            self.Prof_normalize = self.Prof/(LA.norm(self.Prof, axis=1)).reshape((self.U,1))
            self.S = np.matmul(self.Prof_normalize, self.Prof_normalize.T) 
        ####################################################################
        '''[12] Hit event'''
        self.Hit = np.zeros(self.U)
        for u in range(self.U):
            C_jj = self.C[self.caching_policy_num[L_ii[u]]]
            if(C_jj[self.Req[u]]==1):
                self.Hit[u]=1
        self.Hit_rate = sum(self.Hit)/len(self.Hit)
        '''[7] Interference of UE at m-th frequency'''
        for u in range(self.U):
#            m = self.us[u]
#            use_m_UE = np.where(self.us==m)[0]
#            use_m_BS = L_ii[use_m_UE]
#            b_ = np.delete(use_m_BS, np.where(use_m_BS==L_ii[u])[0]) # no intra-cluster interference
#            b__ = np.delete(b_, np.where(b_==0)[0]) # no inter-tier interference
#            self.I[u] = np.sum(self.P[u,b__])
            m = self.us[u]
            use_m_UE = np.where(self.us==m)[0]
            use_m_BS = L_ii[use_m_UE]
            b_ = np.delete(use_m_BS, np.where(use_m_BS==L_ii[u])[0]) # no intra-cluster interference
            self.I[u] = np.sum(self.P[u,b_])
#            if L_ii[u]==0: # MBS
#                self.I[u] = 0
#            else: 
#                m = self.us[u]
#                use_m_UE = np.where(self.us==m)[0]
#                use_m_BS = L_ii[use_m_UE]
#                b_ = np.delete(use_m_BS, np.where(use_m_BS==L_ii[u])[0]) # no intra-cluster interference
#                b__ = np.delete(b_, np.where(b_==0)[0]) # no inter-tier interference
#                self.I[u] = np.sum(self.P[u,b__])
            '''
            b_ = np.delete(use_m_BS, np.where(use_m_BS==L_ii[u])[0][0])
            self.I[u] = np.sum(self.P[u,b_])
            '''
            '''
            hit_UE = np.where(self.Hit==1)[0]
            hit_BS = L_ii[hit_UE]
            b__ = [i for i in b_ if i in hit_BS]
            self.I[u] = np.sum(self.P[u,b__]) 
            '''
        '''[11] Throughput of UE'''
        for u in range (self.U):
            self.P_r[u] = self.P[u][L_ii[u]]
            self.SINR[u] = self.P_r[u]/(self.I[u]+n_var)
            '''
            if self.Hit[u]:
                self.SINR[u] = self.P[u][L_ii[u]]/(self.I[u]+n_var)
            else:
                self.SINR[u] = 0 
            '''
            #self.T[u] = (W/M)*math.log2(1+(self.P[u][L_ii[u]]/(self.I[u][self.us[u]]+n_var)))
            self.T[u] = math.log2(1+self.SINR[u])
        '''[13] System power consumption'''
        self.P_bs_ = np.zeros(self.U)
        for u in range(self.U):
            if(L_ii[u]==0):self.P_bs_[u]=P_MBS
            else:self.P_bs_[u]=P_SBS
        self.P_sys = sum(self.P_bs_ + self.Hit*P_l + (1-self.Hit)*P_b) + len(np.where(self.cluster_member!=0)[0])*P_o_SBS + P_o_MBS   
        '''[14] Energy efficiency'''
        if(normalize):
            self.E = ((sum(self.T)/self.P_sys)-self.EE_mean)/(self.EE_std+1e-8)
        else:
            self.E = sum(self.T)/self.P_sys
        '''[16] Intra-cluster similarity'''
        self.ICS = np.zeros(self.B) # intra-cluster similarity of each cluster(BS)
        for b in range(self.B):
            UE_pair = [list(i) for i in list( combinations (self.cluster_member[0], 2))]
            for jj, pair in enumerate(UE_pair):
                self.ICS[b] = self.ICS[b] + self.S[pair[0]][pair[1]]
            if (len(UE_pair)):
                self.ICS[b] = self.ICS[b]/len(UE_pair)
        if(normalize):
            self.CS = ((sum(self.ICS)/self.B)-self.CS_mean)/(self.CS_std+1e-8)
        else:
            self.CS = sum(self.ICS)/self.B
        '''[17] Reward'''
        self.r = 0.8*self.E \
                +0.2*self.CS
#        self.r = 0.1*sum(self.T)
        self.r = self.Hit_rate 
        #self.r = self.E
        '''[18] State'''
        self.clustering_state = self.L[self.clustering_policy_num].flatten()
        caching_state = self.C[self.caching_policy_num].flatten()
        self.s_ = np.hstack((self.clustering_state,caching_state))
        self.Prof_state = self.Prof_normalize.flatten()
        '''[19] Whether episode done'''
        # cluster size constraint 
        csc_violate = np.where(np.array(self.cluster_size)>M)[0].size # empty: 0 --> no csc_violate
        if  csc_violate :
            self.csc_penalty = sum(abs(self.cluster_size-np.ones(self.B)*(M)))
            self.done = True 
            self.r = self.r-0.8*self.csc_penalty
        # throughput constraint
        
        return self.r,self.done,self.E,self.CS,self.csc_penalty,self.SINR,self.Hit_rate,np.hstack((self.SINR,self.s_,self.Prof_state)),np.sum(self.I[u])
        

if __name__ == "__main__":
    ############################################  
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
    env = BS(bs_coordinate,u_coordinate,B,U,P,up,True,True,None,None,None,None)
    s = env.reset()
#    random_clustering_policy_num = np.random.randint(0,env.B-1,size=env.U)
#    random_caching_policy_num = np.random.randint(0,env.J-1,size=env.B) # suppose randomly assign caching policy to each BSs 
#    r, done,EE, CS, csc_penalty, SINR, Hit_rate, s_, Itf= env.step(clustering_policy_num = random_clustering_policy_num,
#                                                      caching_policy_num = random_caching_policy_num, 
#                                                      update_Req = 1, 
#                                                      normalize = 1)
#    print(done)
#    print(env.cluster_size)
    '''
    #[9] User request
    # content popularity
    zipf_content_popularity = np.zeros((n_p,F))
    for i in range(n_p):
        zipf_content_popularity[i,:] = (generate_zipf_state(a = beta[i], N = F))
    
    # user-popularity association
    up = np.array([randint(0, n_p-1) for _ in range(self.U)]) 
    
    # pop_caching_number
    prob_0 = len(np.where(up==0)[0])/len(up) # Pr(beta0)
    prob_1 = len(np.where(up==1)[0])/len(up) # Pr(beta1)
    p = np.sum(np.vstack((prob_0*zipf_content_popularity[0],prob_1*zipf_content_popularity[1])),axis=0)
    top_N_idx = np.sort(p.argsort()[::-1][0:N])
    pop_caching_number = C_.index(list(top_N_idx))
    
    # user request
    Req = np.array([generate_user_request(zipf_pmf = zipf_content_popularity[up[u]], n = 1) 
                    for u in range(self.U)]).reshape(-1)
    
    # inter-arrival time of duplicate content
    iat_of_d_content = np.zeros((self.U,F)) 
    for u in range(self.U):
        for f in range(F):
            iat_of_d_content[u][f] = np.clip(np.random.geometric(p=zipf_content_popularity[up[u]][f], size=1),1,F)
    '''
    