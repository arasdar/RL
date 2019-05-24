import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork
from memory import Memory

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment (env)."""

    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state (s)
            action_size (int): dimension of each action (a)
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.random_seed = random.seed(random_seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, random_seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, random_seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = Memory(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    #def step(self, s, a, r, s2, done):
    def step(self, state, action, reward, next_state, done):
        # Save/add experience in/to replay memory/buffer
        self.memory.add(state, action, reward, next_state, done)
        #self.memory.add(s, a, r, s2, done)
        
        # Exploration vs exploitation
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
    #def act(self, s, eps=0.):
        """Returns actions (A) for given state (s) as per current policy (a).
        
        Params
        ======
            state (array_like): current state (s)
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy (eps) action (a) selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
            #return np.argmax(Q.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience (e) tuples.

        Params
        ======
            e (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) 
            e (Tuple[torch.Tensor]): tuple of (s, a, r, s2, done) 
            exprience (Tuple[torch.Tensor]): tuple of (state, action, reward, next_state, done) 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        #S, A, rewards, S2, dones = E

        # Get max predicted Q (values) for next states (S2) from target model
        #Q2_target = self.q_target(S2).detach().max(1)[0].unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q target for current states (S) 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        #Q_target = rewards + (gamma * Q2_target * (1 - dones))

        # Get expected Q (values) from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        #Q = self.q_local(S).gather(1, A)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        #qloss = ((Q - Q_target)**2).mean()
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # self.q_optimizer.zero_grad() # init with zero
        # qloss.backward() # backprop
        # self.q_optimizer.step() # update weight by applying the gradients

        # ------------------- update target network ------------------- #
        #self.soft_update(self.q, self.q_target, GAMMA)                     
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)