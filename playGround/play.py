import gym
import numpy as np
from newENV import BS
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
#--------------------------------------------------------------------------
#env = BS(nBS=40,nUE=10,nMaxLink=2,nFile=50,nMaxCache=10,loadENV = True)
env = gym.make('Pendulum-v0')
check_env(env, warn=True)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
model = DDPG('MlpPolicy', env, learning_rate=0.001, action_noise=action_noise, verbose=1, tensorboard_log = 'data/tblog' )
#model = TD3('MlpPolicy', env, action_noise=action_noise, verbose=1,tensorboard_log = 'tblog' )
model.learn(total_timesteps=10000, callback=[epoch_actor_losses, epoch_critic_losses], log_interval=100, tb_log_name='DDPG')
model.save("data/ddpg_CF_MMIMO")
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()




