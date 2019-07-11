from memory import Memory # episodic memory/ hippocampus
from modelD import D # Discriminator/Actor
from modelG import G # Generator/Adversarial/Autoencoceder
from modelQ import Q_fixed # Q-Net/value/reward network

import random

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """an agent from the agency of DAQL."""
    
    def __init__(self, s_size, a_size, h_size, random_seed, buffer_size, batch_size, gamma, lr_g, lr_d):
        """Initialize an Agent object.
        """
        self.gamma = gamma
        self.batch_size = batch_size
        
        # D: Discriminator/Actor Network (with Target Network)
        self.d = D(s_size, a_size, h_size, random_seed).to(device)
        self.d_target = D(s_size, a_size, h_size, random_seed).to(device)
        self.d_optimizer = optim.Adam(self.d.parameters(), lr=lr_d)

        # G: Generator/Adv-Autoencoder Network (with Target Network)
        self.g = G(s_size, a_size, h_size, random_seed).to(device)
        self.g_target = G(s_size, a_size, h_size, random_seed).to(device)
        self.g_optimizer = optim.Adam(self.g.parameters(), lr=lr_g)
        
        # Q: Q-Network (fixed/frozen)
        self.q_fixed = Q_fixed(s_size, random_seed).to(device)
        
        # ReplayBuffer/ Memory
        self.memory = Memory(buffer_size, batch_size, random_seed)
    
    # D: Discriminator/classifier as the actor such as DQN
    def act(self, s):
        """Returns an action (a) (as per current policy) for a given state (s)."""
        s = torch.from_numpy(s).float().to(device)
        self.d.eval() # validation/test/inference
        with torch.no_grad():
            a = self.d(s).cpu().data.numpy()
        self.d.train() # train
        return a # tanh(a):[-1, 1]
    
    def start_learn(self):
        if len(self.memory) >= self.batch_size:
            E = self.memory.sample() # E: expriences
            gloss, dloss, reward, reward_in = self.learn(E, self.gamma)
            #print(dloss, gloss)
            dloss = dloss.cpu().data.numpy()
            gloss = gloss.cpu().data.numpy()
            reward = reward.cpu().data.numpy()
            reward_in = reward_in.cpu().data.numpy()
            return gloss, dloss, reward, reward_in
        else: return 0, 0, 0, 0
        
    def learn(self, E, γ): # γ: gamma, E: expriences
        """Update G and D parameters using given batch of experience (e) tuples.
        Q_target = rewards + γ * d_target(S2, g_target(S2))
        where:
            g_target(S) -> A-actions and S: states
            d_target(S, A) -> Q-values

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (states, actions, rewards, next_states, dones) 
            E           (Tuple[torch.Tensor]): tuple of (   S  ,    A   , rewards,    S2      , dones) 
            γ (float): discount factor or gamma
        """
        S, A, rewards, S2, dones = E

        # ---------------------------- update G: Generator or Adversarial/Autoencoder (predictor) --------------- #
        Q = self.q_fixed(S)
        Q2 = self.q_fixed(S2)
        rewards_in = Q - (γ * Q2)
        #print(rewards_in.shape, rewards.shape, Q.shape, Q2.shape)
        
        A2 = self.d_target(S2)
        S3 = self.g_target(S2, A2)
        Q2 = self.q_fixed(S3)
        Q = rewards + (γ * Q2 * (1 - dones))
        #print(Q.shape, Q2.shape)
        
        S2_ = self.g(S, A)
        Q_ = self.q_fixed(S2_)
        #print(Q_.shape)
        
        #gloss = torch.sum((Q_ - Q)**2, dim=1).mean()
        gloss = torch.sum(torch.abs(Q_ - Q)).mean()
        
        # Minimize the loss
        self.g_optimizer.zero_grad()
        gloss.backward()
        #torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.g_optimizer.step()

        # ---------------------------- update G: Generator (action generator or actor) ---------------------------- #
        # A = self.d(S)
        # S2 = self.g(S, A)
        # Q = self.q_fixed(S2)
        
        #dloss = -Q.mean()
        # OR
        A2 = self.d(S2)
        S3 = self.g(S2, A2)
        Q2 = self.q_fixed(S3)
        
        dloss = -Q2.mean()
        
        # Minimize the loss
        self.d_optimizer.zero_grad()
        dloss.backward()
        self.d_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.d, self.d_target, γ)
        self.soft_update(self.g, self.g_target, γ)
        
        return gloss, dloss, rewards.mean(), rewards_in.mean()

    def soft_update(self, local_model, target_model, γ):
        """Soft update model parameters.
        γ: GAMMA
        θ_target = (1-γ)*θ_local + γ*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            gamma (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(((1-γ)*local_param.data) + (γ*target_param.data))