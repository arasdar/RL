from memory import Memory
from model import G, D
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.99            # discount factor
LR = 1e-3        # learning rate of the critic
BATCH_SIZE = 1024         # minibatch size/ RAM size
BUFFER_SIZE = int(1e6)  # replay buffer size

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment (env)."""
    
    def __init__(self, s_size, a_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            s_size (int): dimension of each state (s)
            a_size (int): dimension of each action (a)
            random_seed (int): random seed
        """
        self.s_size = s_size
        self.a_size = a_size
        self.seed = random.seed(random_seed)
        
        # G: Generator (actor) Network (with Target Network)
        self.g = G(s_size, a_size, random_seed).to(device)
        self.g_target = G(s_size, a_size, random_seed).to(device)
        self.g_optimizer = optim.Adam(self.g.parameters(), lr=LR)

        # D: Discriminator (critic) or Decoder (predictor) Network (with Target Network)
        self.d = D(s_size, a_size, random_seed).to(device)
        self.d_target = D(s_size, a_size, random_seed).to(device)
        self.d_optimizer = optim.Adam(self.d.parameters(), lr=LR)

        # ReplayBuffer/ Memory
        self.memory = Memory(a_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    #def step(self, state, action, reward, next_state, done/terminal):
    def step(self, s, a, r, s2, done):
        """Save experience (e) in replay memory, and use random sample from buffer to learn."""
        # Save experience (e) / reward (r)
        self.memory.add(s, a, r, s2, done)

    def act(self, s):
        """Returns an action (a) (as per current policy) for a given state (s)."""
        s = torch.from_numpy(s).float().to(device)
        self.g.eval() # validation/test/inference
        with torch.no_grad():
            a = self.g(s).cpu().data.numpy()
        self.g.train() # train
        return a # tanh(a):[-1, 1]

    def start_learn(self):
        if len(self.memory) > BATCH_SIZE:
            E = self.memory.sample() # E: expriences
            gloss, dloss = self.learn(E, GAMMA)
            #print(dloss, gloss)
            dloss = dloss.cpu().data.numpy()
            gloss = gloss.cpu().data.numpy()
            #print(dloss, gloss)
            return gloss, dloss
        else: return 0, 0
        
    def learn(self, E, γ): # γ: gamma
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

        # ---------------------------- update D: Discriminator (exacminer/evaluator) & Decoder (predictor) --------------- #
        A2 = self.g_target(S2)
        S3_ = self.d_target(S2, A2)
        #Q = rewards + (γ * Q2 * (1 - dones))
        S3_ *= (1 - dones)
        
        # Compute dloss
        S2_ = self.d(S, A)
        dist = torch.sum((S2_ - S3_)**2, dim=1)**.5 # [0, inf]
        rewards_ = torch.exp(-dist) # [0, 1]
        dloss = -(torch.log((rewards_ - rewards)**2)/2).mean() # [0, inf]
        
        # Minimize the loss
        self.d_optimizer.zero_grad()
        dloss.backward()
        #torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.d_optimizer.step()

        # ---------------------------- update G: Generator (action generator or actor) ---------------------------- #
        A2 = self.g_target(S2)
        S3_ = self.d_target(S2, A2)
        #Q = rewards + (γ * Q2 * (1 - dones))
        S3_ *= (1 - dones)
        
        # Compute gloss
        #S2_ = self.d(S, A)
        A_ = self.g(S)
        S2_ = self.d(S, A_)
        dist = torch.sum((S2_ - S3_)**2, dim=1)**.5 # [0, inf]
        rewards_ = torch.exp(-dist) # [0, 1]
        gloss = (torch.log((rewards_ - rewards)**2)/2).mean() # [0, inf]
        
        # Minimize the loss
        self.g_optimizer.zero_grad()
        gloss.backward()
        self.g_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.d, self.d_target, γ)
        self.soft_update(self.g, self.g_target, γ)
        
        return gloss, dloss

    def soft_update(self, local_model, target_model, γ):
        """Soft update model parameters.
        γ: GAMMA ~ 0.9999
        θ_target = (1-γ)*θ_local + γ*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            gamma (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(((1-γ)*local_param.data) + (γ*target_param.data))