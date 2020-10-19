'''
torch = 0.41
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import time
import matplotlib.pyplot as plt

#####################  hyper parameters  ####################

MAX_EPISODES = 100
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
TAU = 0.01


###############################  DDPG  ####################################

class ANet(nn.Module):   # ae(s)=a
    def __init__(self,s_dim,a_dim):
        super(ANet,self).__init__()
        self.h1_dim = 30
        self.fc1 = nn.Linear(s_dim,self.h1_dim)
        #self.fc1.weight.data.normal_(0,0.1) # initialization
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        #self.bn1 = nn.BatchNorm1d(self.h1_dim)
        self.ln1 = nn.LayerNorm(self.h1_dim)
        self.out = nn.Linear(self.h1_dim,a_dim)
        self.out.weight.data.normal_(0,0.1) # initialization
    def forward(self,state):
        x = self.fc1(state)
        #x = self.bn1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.out(x)
        x = F.tanh(x)
        actions_value = x*2
        return actions_value

class CNet(nn.Module):   # ae(s)=a
    def __init__(self,s_dim,a_dim):
        super(CNet,self).__init__()
        self.h1_dim = 30
        self.h2_dim = 30
        self.fcs = nn.Linear(s_dim,self.h1_dim)
        self.fcs.weight.data.normal_(0,0.1) # initialization
        self.fca = nn.Linear(a_dim,self.h1_dim)
        self.fca.weight.data.normal_(0,0.1) # initialization
        self.out = nn.Linear(self.h2_dim,1)
        self.out.weight.data.normal_(0, 0.1)  # initialization

        self.fc1 = nn.Linear(s_dim + a_dim, self.h1_dim)
        #self.fc1.weight.data.normal_(0,0.1) # initialization
        torch.nn.init.xavier_uniform_(self.fc1.weight)
    def forward(self,s,a):
        
        x = self.fcs(s)
        y = self.fca(a)
        net = F.relu(x+y)
        actions_value = self.out(net)
        return actions_value
        '''
        x = torch.cat((s,a), dim=1)
        x = self.fc1(x)
        net = F.relu(x)
        actions_value = self.out(net)
        return actions_value
        '''



class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.actor_lr = LR_A
        self.critic_lr = LR_C
        self.pointer = 0
        #self.sess = tf.Session()
        self.Actor_eval = ANet(s_dim,a_dim)
        self.Actor_target = ANet(s_dim,a_dim)
        self.Critic_eval = CNet(s_dim,a_dim)
        self.Critic_target = CNet(s_dim,a_dim)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(),lr=LR_A)
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(),lr=LR_C)
        self.loss_td = nn.MSELoss()

    def choose_action(self, s):
        s_unsqueeze = torch.unsqueeze(torch.FloatTensor(s), 0)
        #lala = self.Actor_eval(s_unsqueeze)
        #haha = self.Actor_eval(s_unsqueeze)[0]
        #baba = self.Actor_eval(s_unsqueeze)[0].detach()
        return self.Actor_eval(s_unsqueeze)[0].detach() # ae（s）

    def learn(self):

        for x in self.Actor_target.state_dict().keys():
            eval('self.Actor_target.' + x + '.data.mul_((1-TAU))')
            eval('self.Actor_target.' + x + '.data.add_(TAU*self.Actor_eval.' + x + '.data)')
        for x in self.Critic_target.state_dict().keys():
            eval('self.Critic_target.' + x + '.data.mul_((1-TAU))')
            eval('self.Critic_target.' + x + '.data.add_(TAU*self.Critic_eval.' + x + '.data)')

        # soft target replacement()
        #self.sess.run(self.soft_replace)  # 用ae、ce更新at，ct

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = torch.FloatTensor(bt[:, :self.s_dim])
        ba = torch.FloatTensor(bt[:, self.s_dim: self.s_dim + self.a_dim])
        br = torch.FloatTensor(bt[:, -self.s_dim - 1: -self.s_dim])
        bs_ = torch.FloatTensor(bt[:, -self.s_dim:])

        a = self.Actor_eval(bs)
        q = self.Critic_eval(bs,a)  # loss=-q=-ce（s,ae（s））更新ae   ae（s）=a   ae（s_）=a_
        # 如果 a是一个正确的行为的话，那么它的Q应该更贴近0
        loss_a = -torch.mean(q) 
        #print(q)
        #print(loss_a)
        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()

        a_ = self.Actor_target(bs_)  # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
        q_ = self.Critic_target(bs_,a_)  # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
        q_target = br+GAMMA*q_  # q_target = 负的
        #print(q_target)
        q_v = self.Critic_eval(bs,ba)
        #print(q_v)
        td_error = self.loss_td(q_target,q_v)
        # td_error=R + GAMMA * ct（bs_,at(bs_)）-ce(s,ba) 更新ce ,但这个ae(s)是记忆中的ba，让ce得出的Q靠近Q_target,让评价更准确
        #print(td_error)
        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()
        loss_c = td_error
        return loss_a, loss_c

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

###############################  training  ####################################
ENV_NAME = 'Pendulum-v0'
#ENV_NAME = 'LunarLander-v2'
env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

ddpg = DDPG(a_dim, s_dim, a_bound)

var = 3  # control exploration
t1 = time.time()
flag_render = False
poolLossActor=[]
poolLossCritic=[]
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if flag_render:
            env.render()

        # Add exploration noise
        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
        s_, r, done, info = env.step(a) #in Pendulum-v0 case, done is always False

        ddpg.store_transition(s, a, r / 10, s_)

        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
            loss_a, loss_c = ddpg.learn()
            poolLossActor.append(loss_a)
            poolLossCritic.append(loss_c)
            
        s = s_
        ep_reward += r

        if j == MAX_EP_STEPS-1:
            #print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            print('Episode:{} Reward:{} Explore:{}'.format(i,ep_reward,var))
            
            if ep_reward>-300:
                flag_render = True
            break
    
print('Running time: ', time.time() - t1)

# plot Brute Force V.S. RL------------------------------------------------------------------
plt.cla()
plt.plot(range(len(poolLossActor)),poolLossActor,'r-',label='Loss of actor')
plt.plot(range(len(poolLossCritic)),poolLossCritic,'c-',label='Loss of critic')

plt.title('Loss of DDPG') # title
plt.ylabel("Bits/J") # y label
plt.xlabel("Iteration") # x label
#plt.xlim([0, len(poolEE)])
plt.grid()
plt.legend()
fig = plt.gcf()
filename = 'data/DDPG_of_pytorch_'+ENV_NAME
fig.savefig(filename + '.eps', format='eps',dpi=1200)
fig.savefig(filename + '.png', format='png',dpi=1200)
fig.show()