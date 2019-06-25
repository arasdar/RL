import numpy as np
import random
from collections import namedtuple, deque

from model import Q
from memory import Memory

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
LR = 5e-4               # learning rate
# UPDATE_EVERY = 4        # how often to update the network

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
        self.random_seed = random.seed(random_seed)

        # Q-Network
        self.q = Q(s_size, a_size, random_seed).to(device)
        self.q_target = Q(s_size, a_size, random_seed).to(device)
        self.optimizer = optim.Adam(self.q.parameters(), lr=LR)

        # Replay memory
        self.memory = Memory(a_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, s, a, r, s2, done):
        # Save/add experience in/to replay memory/buffer
        self.memory.add(s, a, r, s2, done)
        
        # Exploration vs exploitation
        # # Learn every UPDATE_EVERY time steps.
        # self.t_step = (self.t_step + 1) % UPDATE_EVERY
        # if self.t_step == 0:
        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > BATCH_SIZE:
            E = self.memory.sample() # E: expriences, e: exprience
            self.learn(E, GAMMA)

    def act(self, s, eps=0.):
        """Returns an action (a) for a given state (s) as the current policy (a).
        
        Params
        ======
            state (array_like): current state (s)
            eps (float): epsilon, for epsilon-greedy action (a) selection
        """
        s = torch.from_numpy(s).float().unsqueeze(0).to(device)
        self.q.eval()
        with torch.no_grad():
            a_values = self.q(s) # a_values: action_values
        self.q.train()

        # # Epsilon-greedy (eps) action (a) selection
        # if random.random() > eps:
        return np.argmax(a_values.cpu().data.numpy())
        # else:
        #     return random.choice(np.arange(self.a_size))

    def learn(self, E, gamma):
        """Update value parameters using given batch of experience (e) tuples.

        Params
        ======
            exprience (Tuple[torch.Tensor]): tuple of (state, action, reward, next_state, done) 
            e (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) 
            e (Tuple[torch.Tensor]): tuple of (s, a, r, s2, done) 
            gamma (float): discount factor
        """
        S, A, rewards, S2, dones = E

        # Get max predicted Q (values) for next states (S2) from target model
        Q2 = self.q_target(S2).detach().max(1)[0].unsqueeze(1)
        print(self.q_target(S2).detach().max(1)[0].unsqueeze(1))
        print(self.q_target(S2).detach().max(1)[0])
        print(self.q_target(S2).detach().max(1))
        print(self.q_target(S2).detach())
        print(self.q_target(S2))
        
        # Compute Q target for current states (S)
        Q = rewards + (gamma * Q2 * (1 - dones))

        # Get expected Q (values) from local model
        Q_ = self.q(S).gather(1, A)
        print(self.q(S).gather(1, A))
        print(self.q(S))
        
        # Compute loss
        #loss = F.mse_loss(Q_expected, Q_targets)
        loss = ((Q_ - Q)**2).mean()
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.q, self.q_target, GAMMA)                     

    def soft_update(self, local_model, target_model, gamma):
        """Soft update model parameters.
        θ_target = (1-γ)*θ_local + γ*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(((1-gamma)*local_param.data) + (gamma*target_param.data))