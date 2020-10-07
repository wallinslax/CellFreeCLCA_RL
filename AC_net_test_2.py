"""
Asynchronous Advantage Actor Critic (A3C) with discrete action space, Reinforcement Learning.

The Cartpole example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.8.0
gym 0.10.5
"""

import multiprocessing
import threading
import tensorflow as tf
import numpy as np
# import gym
import os
import shutil
import matplotlib.pyplot as plt
from MIMO_env_3 import MIMO          #import MIMO environment
from mcts_A3C import MCTSPlayer
from Detector_2 import MLDetector, ZFDetector, LMMSEDetector 
from tqdm import tqdm
import copy


OUTPUT_GRAPH = False
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()

MAX_GLOBAL_EP = 600000
EVAL_NUM = 10000

GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10


GAMMA = 0.9
ENTROPY_BETA = 0.001
LR_A = 1e-4    # learning rate for network
LR_C = 1e-4    # learning rate for network
N_PLAYOUTS = 32   # playout for mcts


GLOBAL_RUNNING_R = []
W0_SER=[]
GLOBAL_EP = 0

TX = 8
RX = 8
SNR_LOW_dB = 6.0
SNR_HIGH_dB = 14.0
# FIX_H = np.sqrt(0.5)*(np.random.randn(RX, TX) + np.random.randn(RX, TX)*1j)
FIX_H = None
MODE = 'BPSK'
env = MIMO(TX, RX, mode = MODE, fix_H = FIX_H)

# N_S = env.current_state().shape[0]
N_F = env.n_features
N_A = env.n_actions


class ACNet(object):
    def __init__(self, scope, globalAC=None, init_model=None, meta = None):
        self.old_model = init_model
        self.meta = meta
        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_F], 'S')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_F], 'S')
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.a_prob + 1e-5) * tf.one_hot(self.a_his, N_A, dtype=tf.float32), axis=1, keep_dims=True)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                             axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)
                
                # with tf.name_scope('total_loss'):
                #     self.t_loss = tf.add_n([self.a_loss, self.c_loss], name = 'total_loss')

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)


            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]

                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

 

    def _build_net(self, scope):
        w_init, b_initializer = tf.random_normal_initializer(0., 0.1), tf.constant_initializer(0.01)
        with tf.variable_scope('actor'):
            l1 = tf.compat.v1.layers.dense(self.s, 2*N_F, tf.nn.relu6, kernel_initializer=w_init,              # input current state. First layer 2*n_feature neurons
                                  bias_initializer=b_initializer, name='l1')
            l2 = tf.compat.v1.layers.dense(l1, 2*N_F, tf.nn.relu6, kernel_initializer=w_init,              # input current state. First layer 2*n_feature neurons
                                  bias_initializer=b_initializer, name='l2')
            l3 = tf.compat.v1.layers.dense(l2, 2*N_F, tf.nn.relu6, kernel_initializer=w_init,              # input current state. First layer 2*n_feature neurons
                                  bias_initializer=b_initializer, name='l3')
            l4 = tf.compat.v1.layers.dense(l3, 2*N_F, tf.nn.relu6, kernel_initializer=w_init,              # input current state. First layer 2*n_feature neurons
                                  bias_initializer=b_initializer, name='l4')    
            la1 = tf.compat.v1.layers.dense(l4, N_F, tf.nn.relu, kernel_initializer=w_init,              # input current state. First layer 2*n_feature neurons
                                  bias_initializer=b_initializer, name='la1')
            la2 = tf.compat.v1.layers.dense(la1, N_F, tf.nn.relu, kernel_initializer=w_init,              # input current state. First layer 2*n_feature neurons
                                  bias_initializer=b_initializer, name='la2')
            la3 = tf.compat.v1.layers.dense(la2, N_F, tf.nn.relu, kernel_initializer=w_init,              # input current state. First layer 2*n_feature neurons
                                  bias_initializer=b_initializer, name='la3')
            la4 = tf.compat.v1.layers.dense(la3, 4*N_A, tf.nn.relu, kernel_initializer=w_init,              # input current state. First layer 2*n_feature neurons
                                  bias_initializer=b_initializer, name='la4')
            la5 = tf.compat.v1.layers.dense(la4, 2*N_A, tf.nn.relu, kernel_initializer=w_init,              # input current state. First layer 2*n_feature neurons
                                  bias_initializer=b_initializer, name='la5')
            a_prob = tf.layers.dense(la5, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')

        with tf.variable_scope('critic'):
            l5 = tf.compat.v1.layers.dense(self.s, 2*N_F, tf.nn.relu6, kernel_initializer=w_init,              # input current state. First layer 2*n_feature neurons
                                  bias_initializer=b_initializer, name='l5')
            l6 = tf.compat.v1.layers.dense(l5, 2*N_F, tf.nn.relu6, kernel_initializer=w_init,              # input current state. First layer 2*n_feature neurons
                                  bias_initializer=b_initializer, name='l6')
            l7 = tf.compat.v1.layers.dense(l6, 2*N_F, tf.nn.relu6, kernel_initializer=w_init,              # input current state. First layer 2*n_feature neurons
                                  bias_initializer=b_initializer, name='l7')
            l8 = tf.compat.v1.layers.dense(l7, 2*N_F, tf.nn.relu6, kernel_initializer=w_init,              # input current state. First layer 2*n_feature neurons
                                  bias_initializer=b_initializer, name='l8')            
            lc1 = tf.compat.v1.layers.dense(l8, N_F, tf.nn.tanh, kernel_initializer=w_init,              # input current state. First layer 2*n_feature neurons
                                  bias_initializer=b_initializer, name='lc1')
            lc2 = tf.compat.v1.layers.dense(lc1, N_F, tf.nn.tanh, kernel_initializer=w_init,              # input current state. First layer 2*n_feature neurons
                                  bias_initializer=b_initializer, name='lc2')
            lc3 = tf.compat.v1.layers.dense(lc2, N_F, tf.nn.tanh, kernel_initializer=w_init,              # input current state. First layer 2*n_feature neurons
                                  bias_initializer=b_initializer, name='lc3')
            lc4 = tf.compat.v1.layers.dense(lc3, 4*N_A, tf.nn.tanh, kernel_initializer=w_init,              # input current state. First layer 2*n_feature neurons
                                  bias_initializer=b_initializer, name='lc4')
            lc5 = tf.compat.v1.layers.dense(lc4, 2*N_A, tf.nn.tanh, kernel_initializer=w_init,              # input current state. First layer 2*n_feature neurons
                                  bias_initializer=b_initializer, name='lc5')
            v = tf.layers.dense(lc5, 1, kernel_initializer=w_init, name='v')  # state value
        
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        self.saver = tf.train.Saver()
        # self.saver = tf.train.import_meta_graph('./model/best_policy.model.meta')


        if self.old_model is not None:
            self.restore_model(self.meta)
        return a_prob, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net



    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])
        # SESS.run(self.target_replace_op)

    def choose_action(self, s):  # run by a local
        prob_weights = SESS.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action
    
    def ACout(self, s):  # run by a local
        prob_weights = SESS.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})
        actions = np.arange(len(prob_weights[0]))
        policy_values = zip(actions, prob_weights[0])
        # action = np.random.choice(range(prob_weights.shape[1]),
        #                           p=prob_weights.ravel())  # select action w.r.t the actions prob
        state_value = SESS.run(self.v, feed_dict={self.s: s[np.newaxis, :]})
        return [policy_values, state_value.ravel()]
    
    def save_model(self, model_path):
        self.saver.save(SESS, model_path)

    def restore_model(self, meta):
        self.saver = tf.train.import_meta_graph(meta)
        self.saver.restore(SESS, self.old_model)

class Worker(object):
    def __init__(self,env , name, globalAC, init_model = None, meta=None):
        self.env = env
        self.name = name
        self.AC = ACNet(name, globalAC, init_model=init_model, meta=meta)
        self.actor = MCTSPlayer(self.AC.ACout,
                                      n_playout=N_PLAYOUTS,
                                      is_selfplay=0,epsilon=0.7,c_puct=10)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            self.env.reset(SNR_LOW_dB, SNR_HIGH_dB)
            s = self.env.current_state()
            # print(s)
            ep_r = 0
            while True:
                # if self.name == 'W_0':
                #     self.env.render()
                # a = self.actor.choose_action(self.env)
                a = self.AC.choose_action(s)
                s_, r, done = self.env.step(a)
                
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)
                

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)
                    

                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                    # if self.name == 'W_0':
                        # print(self.env.SER())
                        # W0_SER.append(self.env.SER())
                        # print(GLOBAL_EP)
                    # print(
                    #     self.name,
                    #     "Ep:", GLOBAL_EP,
                    #     "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                    #       )
                    GLOBAL_EP += 1
                    break
            if (GLOBAL_EP) % 20000 == 0:
                print(GLOBAL_EP)
                
    def detect(self):
               
        self.MMSE = LMMSEDetector(self.env) 
        self.AC.pull_global()
        snr_list = [6., 10., 14.]
        SER_RL=[]
        SER_MCTS=[]
        # SER_ML=[]
        # SER_ZF=[]
        SER_MMSE=[]
        for snr in tqdm((snr_list)):
            ser_rl = 0
            ser_mcts = 0
            # ser_ml = 0
            # ser_zf = 0
            ser_mmse = 0
            for i in range(EVAL_NUM):
                self.env.reset(snr, snr)
                env_copy = copy.deepcopy(self.env)
                while True:
                    a = self.AC.choose_action(self.env.current_state())
                    a2 = self.actor.choose_action(env_copy)
                    # a = self.AC.choose_action(s)
                    _, _, done = self.env.step(a)
                    _, _, _ = env_copy.step(a2)
                    if done:
                        s1 = self.env.SER()
                        s2 = env_copy.SER()
                        break
                s3 = self.MMSE.SER()
                # s3 = self.ZF.SER()
                # s4 = self.ML.SER()
                ser_rl += s1
                ser_mcts += s2
                ser_mmse += s3
                # ser_zf += s3
                # ser_ml += s4
                
            ser_avg_1 = ser_rl/EVAL_NUM
            ser_avg_2 = ser_mcts/EVAL_NUM
            ser_avg_3 = ser_mmse/EVAL_NUM
            # ser_avg_3 = ser_zf/n_detects
            # ser_avg_4 = ser_ml/n_detects
                       
            SER_RL.append(ser_avg_1)
            SER_MCTS.append(ser_avg_2)
            SER_MMSE.append(ser_avg_3)
            # SER_ZF.append(ser_avg_3)
            # SER_ML.append(ser_avg_4)
        
        SER =zip(snr_list, SER_RL, SER_MCTS, SER_MMSE)
        

        return SER
            
if __name__ == "__main__":
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='MOMENTUM')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='MOMENTUM')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we ondly need its params
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_AC))

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)
        
    worker_threads = []
    print(N_WORKERS)
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)
    
    SER_avg = workers[0].detect()

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()
    
    # plt.plot(np.arange(len(W0_SER)), W0_SER)
    # plt.xlabel('step')
    # plt.ylabel('SER')
    # plt.show()
    # print(W0_SER)
    print("\n")
    for snr, ser1, ser2, ser3 in SER_avg:
        print("SNR:{}, Average RL SER:{}, Average MCTS SER:{}, Average MMSE SER:{}".format(snr, ser1, ser2, ser3))
        
    #%% 
    workers[0].AC.save_model('./model/current_policy.model')
    np.savetxt('H_fix.csv', FIX_H.view(float), delimiter=',')
    
    #%% 
    
    model = tf.train.latest_checkpoint('./model/4x4_fixed')
    meta = './model/4x4_fixed/current_policy.model.meta'
    FIX_H =  np.array([[(-2.194988741331324600e-02-2.254411093831358404e+00j),	 (-4.424178704010837682e-01-4.077904052909459010e-01j),	 (4.905132923281006474e-01+6.166075029984353639e-01j),	 (-3.079444322396239775e-01+4.228392732686493682e-02j)],
                       [(-3.800085292138467019e-01-1.244595376496291511e-01j),	 (-8.940292387779757988e-01-8.099539309515890739e-01j),	 (-1.398905927707667363e-01+8.932348880932667878e-01j),	 (1.697090111107474597e-01+2.188853822924322068e-02j)],
                       [(-5.767281312843866026e-01-3.752568901929281275e-01j),	 (-7.275104023013519994e-01-9.950562411708069321e-01j),	 (1.133429810472425664e-01-6.747186280714370099e-01j),	 (2.650871887205429323e-01-2.120594914531590547e+00j)],
                       [(-1.910781424035404852e-01+3.685439937561358348e-01j),	 (-2.754733999004262968e-01+1.038464114395349774e-01j),	 (-1.271979240064478045e-01+5.973386656761248137e-01j),	 (2.435192278020653356e-01-1.109363915563880837e-01j)]])

    saved_worker = Worker('W_14', GLOBAL_AC, model, meta)
    EVAL_NUM = 1000
    SER_avg = saved_worker.detect()
    for snr, ser1, ser2, ser3 in SER_avg:
        print("SNR:{}, Average RL SER:{}, Average MCTS SER:{}, Average MMSE SER:{}".format(snr, ser1, ser2, ser3))
    
    print(saved_worker.env.H_fix)
     # #%%
    # EVAL_NUM = 100000
    # SER_avg = workers[0].detect()
    # print("\n")
    # for snr, ser1, ser2, ser3 in SER_avg:
    #     print("SNR:{}, Average RL SER:{}, Average MCTS SER:{}, Average MMSE SER:{}".format(snr, ser1, ser2, ser3))