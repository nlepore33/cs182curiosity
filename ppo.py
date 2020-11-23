import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

from icm import ICM

class PPO(nn.Module):
    def __init__(self, state_size, action_size, learning_rate, gamma, lmbda, eps_clip, K_epoch, buffer_size, minibatch_size, policy_weight, use_icm, icm_parameters):
        super(PPO, self).__init__()
        self.data = []
        
        self.fc1   = nn.Linear(state_size,128)
        self.fc_mu = nn.Linear(128,action_size)
        self.fc_std  = nn.Linear(128,action_size)
        self.fc_v = nn.Linear(128,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.optimization_step = 0

        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.K_epoch = K_epoch
        self.buffer_size = buffer_size
        self.minibatch_size = minibatch_size
        self.policy_weight = policy_weight
        self.use_icm = use_icm
        
        if use_icm:
            self.icm = ICM(state_size, action_size, icm_parameters)

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        mu = 2.0*torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_batch, a_batch, r_batch, s_prime_batch, prob_a_batch, done_batch = [], [], [], [], [], []
        data = []

        for j in range(self.buffer_size):
            for i in range(self.minibatch_size):
                rollout = self.data.pop()
                s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []

                for transition in rollout:
                    s, a, r, s_prime, prob_a, done = transition
                    
                    s_lst.append(s)
                    a_lst.append([a])
                    r_lst.append([r])
                    s_prime_lst.append(s_prime)
                    prob_a_lst.append([prob_a])
                    done_mask = 0 if done else 1
                    done_lst.append([done_mask])

                s_batch.append(s_lst)
                a_batch.append(a_lst)
                r_batch.append(r_lst)
                s_prime_batch.append(s_prime_lst)
                prob_a_batch.append(prob_a_lst)
                done_batch.append(done_lst)
                    
            mini_batch = (
                torch.tensor(s_batch, dtype=torch.float), 
                torch.tensor(a_batch, dtype=torch.float),
                torch.tensor(r_batch, dtype=torch.float), 
                torch.tensor(s_prime_batch, dtype=torch.float),
                torch.tensor(done_batch, dtype=torch.float), 
                torch.tensor(prob_a_batch, dtype=torch.float)
            )
            data.append(mini_batch)

        return data

    def calc_advantage(self, data):
        data_with_adv = []
        for mini_batch in data:
            s, a, r, s_prime, done_mask, old_log_prob = mini_batch
            with torch.no_grad():
                td_target = r + self.gamma * self.v(s_prime) * done_mask
                delta = td_target - self.v(s)
            delta = delta.numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)
            data_with_adv.append((s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage))

        return data_with_adv

        
    def train_net(self):
        if len(self.data) == self.minibatch_size * self.buffer_size:
            data = self.make_batch()
            data = self.calc_advantage(data)

            for i in range(self.K_epoch):
                for mini_batch in data:
                    s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage = mini_batch

                    mu, std = self.pi(s, softmax_dim=1)
                    dist = Normal(mu, std)
                    log_prob = dist.log_prob(a)
                    ratio = torch.exp(log_prob - old_log_prob)  # a/b == exp(log(a)-log(b))

                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage

                    policy_loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target)
                    
                    if self.use_icm:
                        a_hat, s_hat = self.icm.predict(s, a, s_prime)
                        intrinsic_loss = self.icm.loss(s, a, a_hat, s_prime, s_hat)
                        loss = self.policy_weight * policy_loss + (1-self.policy_weight) * intrinsic_loss
                    else:
                        loss = policy_loss

                    # print(policy_loss)
                    # print(intrinsic_loss)
                    # print(loss)
                    
                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimization_step += 1
                    return loss