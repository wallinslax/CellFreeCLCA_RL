#!/usr/bin/env python3 
# Proprietary Design
from newENV_ghliu import BS
from normalized_env import NormalizedEnv
from evaluator import Evaluator
from ddpg import DDPG
from util import *
from tqdm import tqdm
# Public Lib
import numpy as np
import argparse
from copy import deepcopy
import torch
import gym

#gym.undo_logger_setup()

def train(num_iterations, agent, env,  evaluate, validate_steps, output, max_episode_length=None, debug=False):
    
    #max_episode_length = 0
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    poolEE=[]
    poolPLoss=[]
    poolVLoss=[]
    maxEE=0
    for step in tqdm(range(num_iterations)):
        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)

        # agent pick action ...
        if step <= args.warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(observation)
        #------------------------------------------------------------------
        cluster_act_dim = (env.U*env.B)
        cache_act_dim = (env.B*env.F)
        a_cl = action[0:cluster_act_dim]
        a_ca = action[cluster_act_dim:]

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
        cacheScore = np.reshape(a_ca,(env.B,env.F) )
        caching_policy_BS = []
        for b in range(env.B):
            top_N_idx = np.sort(cacheScore[b].argsort()[-env.N:])# pick up N file with highest score, N is capacity of BSs
            caching_policy_BS.append(top_N_idx)
        
        EE, HR, RL_s_, done  = env.step(clustering_policy_UE,caching_policy_BS)
        observation2 = deepcopy(RL_s_)
        reward = EE
        poolEE.append(EE)
        #------------------------------------------------------------------
        # env response with next_observation, reward, terminate_info
        #observation2, reward, done, info = env.step(action)
        #bservation2 = deepcopy(observation2)
        if max_episode_length and episode_steps >= max_episode_length -1:
            done = True

        # agent observe and update policy
        agent.observe(reward, observation2, done)
        if step > args.warmup :
            policy_loss,value_loss = agent.update_policy()
            poolPLoss.append(policy_loss)
            poolVLoss.append(value_loss)
        
        # [optional] evaluate
        '''
        if evaluate is not None and validate_steps > 0 and step % validate_steps == 0:
            policy = lambda x: agent.select_action(x, decay_epsilon=False)
            validate_reward = evaluate(env, policy, debug=False, visualize=False)
            if debug: prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))

        # [optional] save intermideate model
        if step % int(num_iterations/3) == 0:
            agent.save_model(output)
        '''
        # update 
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done: # end of episode
            if debug: prGreen('#{}: episode_reward:{} steps:{}'.format(episode,episode_reward,step))

            agent.memory.append(
                observation,
                agent.select_action(observation),
                0., False
            )
            epMaxEE = max(poolEE)
            maxEE = max(epMaxEE,maxEE)
            print('\n',episode,'th episode epMaxEE=', epMaxEE, ' MaxEE=',maxEE)
            print('max policy_loss=',max(poolPLoss))
            print('max value_loss=',max(poolVLoss))
            # reset
            poolEE=[]
            poolPLoss=[]
            poolVLoss=[]
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1
    print('maxEE=',maxEE)

def test(num_episodes, agent, env, evaluate, model_path, visualize=True, debug=False):

    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    for i in range(num_episodes):
        validate_reward = evaluate(env, policy, debug=debug, visualize=visualize, save=False)
        if debug: prYellow('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--env', default='Pendulum-v0', type=str, help='open-ai gym environment')
    parser.add_argument('--hidden1', default=400, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=100, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=6000000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma') 
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu') 
    parser.add_argument('--validate_episodes', default=20, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=500, type=int, help='')
    parser.add_argument('--validate_steps', default=2000, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='') 
    parser.add_argument('--train_iter', default=200000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    # parser.add_argument('--l2norm', default=0.01, type=float, help='l2 weight decay') # TODO
    # parser.add_argument('--cuda', dest='cuda', action='store_true') # TODO

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)
    if args.resume == 'default':
        args.resume = 'output/{}-run0'.format(args.env)

    env = BS(nBS=4,nUE=4,nMaxLink=2,nFile=5,nMaxCache=2,loadENV = True)

    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)


    nb_states = len(env.s_)
    cluster_act_dim = (env.U*env.B)
    cache_act_dim = (env.B*env.F)
    nb_actions = cluster_act_dim + cache_act_dim

    agent = DDPG(nb_states, nb_actions, args)
    evaluate = Evaluator(args.validate_episodes, 
        args.validate_steps, args.output, max_episode_length=args.max_episode_length)

    if args.mode == 'train':
        train(args.train_iter, agent, env, evaluate, 
            args.validate_steps, args.output, max_episode_length=args.max_episode_length, debug=args.debug)

    elif args.mode == 'test':
        test(args.validate_episodes, agent, env, evaluate, args.resume,
            visualize=True, debug=args.debug)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
