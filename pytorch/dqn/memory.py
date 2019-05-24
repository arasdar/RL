import numpy as np
import random
from collections import namedtuple, deque

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    """Fixed-size buffer to store experience (e) tuples."""

    def __init__(self, action_size, buffer_size, batch_size, random_seed):
    #def __init__(self, a_size, buffer_size, batch_size, random_seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action (a)
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            random_seed (int): random seed
        """
        self.action_size = action_size
        #self.a_size = a_size
        self.buffer = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        #self.e = namedtuple("e", field_names=["s", "a", "r", "s2", "done"])
        self.random_seed = random.seed(random_seed)
    
    def add(self, state, action, reward, next_state, done):
    #def add(self, s, a, r, s2, done):
        """Add a new experience to memory buffer."""
        e = self.experience(state, action, reward, next_state, done)
        #e = self.e(s, a, r, s2, done)
        self.buffer.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from buffer."""
        experiences = random.sample(self.buffer, k=self.batch_size)
        #E = random.sample(self.buffer, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        # S = torch.from_numpy(np.vstack([e.s for e in E if e is not None])).float().to(device)
        # A = torch.from_numpy(np.vstack([e.a for e in E if e is not None])).long().to(device)
        # rewards = torch.from_numpy(np.vstack([e.r for e in E if e is not None])).float().to(device)
        # S2 = torch.from_numpy(np.vstack([e.s2 for e in E if e is not None])).float().to(device)
        # dones = torch.from_numpy(np.vstack([e.done for e in E if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)
        #return (S, A, rewards, S2, dones)

    def __len__(self):
        """Return the current size of internal buffer."""
        return len(self.buffer)