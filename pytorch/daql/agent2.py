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
    
    #def step(self, state, action, reward, next_state, done):
    def step(self, s, a, r, s2, done):
        """Save experience (e) in replay memory, and use random sample from buffer to learn."""
        # Save experience (e)
        self.memory.add(s, a, r, s2, done)

    # actor/critic using discrimnator/QNet/Qfunction
    def act(self, s):
        """Returns an action (a) and value (q) for a given state (s)."""
        s = torch.from_numpy(s).float().to(device)
        self.d.eval() # train=false
        with torch.no_grad():
            a, q = self.d(s)
            a = torch.tanh(a) #[-1, 1]
            #print(a.shape)
            #a = a.cpu().data.numpy()#.reshape([1, -1])
            a = a.numpy()
            q = q.numpy()
            #print(a.shape)
        self.d.train() # train=true
        return a, q

    # Adv/auto encoder/decoder using generator/generative model/ predictive model/ predict the next state of env
    # Model-based or based on the env model
    def env(self, s, a):
        """Requires an action (a) and a given state (s) for predicting next state (s2_)."""
        s = torch.from_numpy(s).float().to(device)
        a = torch.from_numpy(a).float().to(device)
        self.g.eval() # train=false
        #self.g_target.eval() # test/validation/inference
        with torch.no_grad():
            s2 = self.g(s, a)
            # s2 = s2.cpu().data.numpy()
            s2 = s2.numpy()
            #print(s2.shape, q.shape)
        self.g.train() # train=true
        #self.d_target.train()
        return s2
    
    def start_learn(self):
        if len(self.memory) > BATCH_SIZE:
            E = self.memory.sample()
            gloss, dloss = self.learn(E, GAMMA)
            dloss = dloss.cpu().data.numpy()
            gloss = gloss.cpu().data.numpy()
            return gloss, dloss
        else: return 0, 0
        
    def learn(self, E, γ):
        """Update G and D parameters using given batch of experience (e) tuples.
        γ = gamma
        A2 = g_target(S2)
        Q = rewards + γ * d_target(S2, A2)
        where:
            g_target(S) -> A-actions & S-states
            d_target(S, A) -> Q-values

        Params
        ======
            E           (Tuple[torch.Tensor]): tuple of (S     , A      , rewards, S2         , dones) 
            experiences (Tuple[torch.Tensor]): tuple of (states, actions, rewards, next_states, dones)
            γ (float): discount factor/ gamma
        """
        S, A, rewards, S2, dones = E # E: expriences, e: exprience

        # ---------------------------- update D: Discriminator & Actor/Critic --------------- #
        _, Q2 = self.d_target(S2)
        Q = rewards + (γ * Q2 * (1 - dones))
        _, dQ = self.d(S)
        dloss = ((dQ - Q)**2).mean()
        
        S2_ = self.g_target(S, A)
        _, Q2_ = self.d_target(S2_)
        Q_ = rewards + (γ * Q2_ * (1 - dones))
        _, dQ2 = self.d(S2)
        dQ_ = rewards + (γ * dQ2 * (1 - dones))
        dloss += ((dQ_ - Q_)**2).mean()
        
        # Minimize the loss
        self.d_optimizer.zero_grad()
        dloss.backward()
        #torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.d_optimizer.step()
        
        # ---------------------------- update G: Generator (action generator or actor) ---------------------------- #
        gS2 = self.g(S, A)
        _, gQ2 = self.d(gS2)
        gQ = rewards + (γ * gQ2 * (1 - dones))
        gloss = -gQ.mean()
        # gloss += torch.sum((gA2 - A2)**2, dim=1).mean()        

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
        θ_target = (1-γ)*θ_local + γ*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            γ (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(((1-γ)*local_param.data) + (γ*target_param.data))