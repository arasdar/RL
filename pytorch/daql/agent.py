from memory2 import Memory
from model2 import G, D
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.99            # discount factor
LR = 1e-3        # learning rate of the critic
BATCH_SIZE = 1024         # minibatch size/ RAM size
BUFFER_SIZE = int(1e6)  # replay buffer size
# H_SIZE = 400

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
    
    #def step(self, state, action, reward, next_state, done/terminal, pred_state, pred_next_state):
    def step(self, s, a, r, s2, done, s_, s2_):
        """Save experience (e) in replay memory, and use random sample from buffer to learn."""
        # Save experience (e)
        self.memory.add(s, a, r, s2, done, s_, s2_)

    def act(self, s, s_):
        """Returns an action (a) (as per current policy) for a given state (s) and a predicted state (s_)."""
        s = torch.from_numpy(s).float().to(device)
        s_ = torch.from_numpy(s_).float().to(device)
        self.g.eval() # train=false
        with torch.no_grad():
            a = self.g(s, s_)
            #print(a.shape)
            a = a.cpu().data.numpy()#.reshape([1, -1])
            #print(a.shape)
        self.g.train() # train=true
        return a # tanh(a):[-1, 1]

    def env(self, s, a):
        """Requires an action (a) (as per current policy) and a given state (s) for predicting next state (s2_) and total future rewards (q_)."""
        s = torch.from_numpy(s).float().to(device)
        a = torch.from_numpy(a).float().to(device)
        self.d.eval() # train=false
        with torch.no_grad():
            s2, q = self.d(s, a)
            #print(s2.shape, q.shape)
            q = q.cpu().data.numpy()
            s2 = s2.cpu().data.numpy()
            #print(s2.shape, q.shape)
        self.d.train() # train=true
        return s2, q
    
    def start_learn(self):
        if len(self.memory) > BATCH_SIZE:
            E = self.memory.sample()
            self.learn(E, GAMMA)
        
    def learn(self, E, γ):
        """Update G and D parameters using given batch of experience (e) tuples.
        γ = gamma
        A2 = g_target(S2, S2_)
        Q = rewards + γ * d_target(S2, A2)
        where:
            g_target(S, S_) -> A-actions in which S: states, S_: pred_states
            d_target(S, A) -> Q-values

        Params
        ======
            E (Tuple[torch.Tensor]): tuple of (S     , A      , rewards, S2         , dones, S_, S2_) tuples 
            experiences (Tuple[torch.Tensor]): tuple of (states, actions, rewards, next_states, dones) tuples .....
            ... S_: pred_states, S2_: pred_next_states
            gamma (float): discount factor
        """
        S, A, rewards, S2, dones, S_, S2_ = E # E: expriences, e: exprience

        # ---------------------------- update D: next state and final state predictor --------------- #
        # ---------------------------- update D: Discriminator (exacminer/evaluator) & Decoder (predictor) --------------- #
        # ---------------------------- update D: Discriminator (critic) & Decoder (predictor) ---------------------------- #
        A2 = self.g_target(S2, S2_)
        _, Q2 = self.d_target(S2, A2) # S3_
        Q = rewards + (γ * Q2 * (1 - dones))
        # Compute dloss
        dS2, dQ = self.d(S, A)
        dloss = ((dQ - Q)**2).mean()
        dloss += ((dS2 - S2)**2).mean()
        #dloss = F.mse_loss(Q, Q_target)
        # Minimize the loss
        self.d_optimizer.zero_grad()
        dloss.backward()
        #torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.d_optimizer.step()

        # ---------------------------- update G: Generator (action generator or actor) ---------------------------- #
        # Compute gloss
        A = self.g(S, S_)
        gS2, gQ = self.d(S, A)
        gloss = -gQ.mean() 
        gloss += -gS2.mean()
        # Minimize the loss
        self.g_optimizer.zero_grad()
        gloss.backward()
        self.g_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.d, self.d_target, γ)
        self.soft_update(self.g, self.g_target, γ)                     

    def soft_update(self, local_model, target_model, γ):
        """Soft update model parameters.
        θ_target = (1-γ)*θ_local + γ*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            gamma (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(((1-γ)*local_param.data) + (γ*target_param.data))