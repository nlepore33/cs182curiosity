import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

from ppo import PPO
from icm import ICM

# Hyperparameters
learning_rate  = 0.0003 # 0.0003
gamma          = 0.9 # 0.9
lmbda          = 0.9 # 0.9
eps_clip       = 0.2  # 0.2
K_epoch        = 10
rollout_len    = 3
buffer_size    = 30
minibatch_size = 32
policy_weight = 1.0

# ICM parameters
feature_hidden_sizes = [128,128,]
feature_size = 8
inverse_hidden_sizes = [128,128,]
forward_hidden_sizes = [128,128,]
β = 0.5

def main():
    env = gym.make('MountainCarContinuous-v0')
    model = PPO(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        learning_rate=learning_rate,
        gamma = gamma,
        lmbda = lmbda,
        eps_clip = eps_clip,
        K_epoch = K_epoch,
        buffer_size = buffer_size,
        minibatch_size = minibatch_size,
        policy_weight = policy_weight,
        icm_parameters = (feature_hidden_sizes, feature_size, inverse_hidden_sizes, forward_hidden_sizes, β)
    )
    score = 0.0
    print_interval = 10
    rollout = []

    for n_epi in range(10000):
        s = env.reset()
        done = False
        while not done:
            for t in range(rollout_len):
                mu, std = model.pi(torch.from_numpy(s).float())
                dist = Normal(mu, std)
                a = dist.sample()
                log_prob = dist.log_prob(a)
                s_prime, r, done, info = env.step(a)

                rollout.append((s, a, r/10.0, s_prime, log_prob, done))
                if len(rollout) == rollout_len:
                    model.put_data(rollout)
                    rollout = []

                s = s_prime
                score += r
                if done:
                    break

            model.train_net()

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}, opt step: {}".format(n_epi, score/print_interval, model.optimization_step))
            score = 0.0

    env.close()

main()