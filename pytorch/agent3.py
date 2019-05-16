import numpy as np
import random
import copy
from collections import namedtuple, deque

from model3 import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0.0000     # L2 weight decay
BATCH_SIZE = 1024         # minibatch size
BUFFER_SIZE = int(1e6)  # replay buffer size

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, S_size, A_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            S_size (int): dimension of each state
            A_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.S_size = S_size
        self.A_size = A_size
        self.seed = random.seed(random_seed)
        
        # Actor Network (w/ Target Network)
        self.actor = Actor(S_size, A_size, random_seed).to(device)
        self.actor_target = Actor(S_size, A_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic = Critic(S_size, A_size, random_seed).to(device)
        self.critic_target = Critic(S_size, A_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Replay memory
        self.memory = ReplayBuffer(A_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, S, A, r, S2, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(S, A, r, S2, done)

    def act(self, S):
        """Returns actions for given state as per current policy."""
        S = torch.from_numpy(S).float().to(device)
        self.actor.eval()
        with torch.no_grad():
            A = self.actor(S).cpu().data.numpy()
        self.actor.train()
        return A # action [-1, 1]

    def start_learn(self):
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)
        
    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_target = r + γ * critic_target(S2, actor_target(S2))
        where:
            actor_target(S) -> A
            critic_target(S, A) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (S-states, A-actions, r-rewards, S2-next_states, dones) tuples 
            gamma (float): discount factor
        """
        S, A, r, S2, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        gA2 = self.actor_target(S2)
        Q2 = self.critic_target(S2, gA2)
        Q_target = r + (gamma * Q2 * (1 - dones))
        # Compute critic loss
        Q = self.critic(S, A)
        critic_loss = ((Q - Q_target)**2).mean()
        #critic_loss = F.mse_loss(Q, Q_target)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        gA = self.actor(S)
        gQ = self.critic(S, gA)
        actor_loss = -gQ.mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, A_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, S, A, r, S2, done):
        """Add a new experience to memory."""
        e = self.experience(S, A, r, S2, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        S = torch.from_numpy(np.vstack([e.S for e in experiences if e is not None])).float().to(device)
        A = torch.from_numpy(np.vstack([e.A for e in experiences if e is not None])).float().to(device)
        r = torch.from_numpy(np.vstack([e.r for e in experiences if e is not None])).float().to(device)
        S2 = torch.from_numpy(np.vstack([e.S2 for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (S, A, r, S2, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)