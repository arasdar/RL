import numpy as np
import random
import copy
from collections import namedtuple, deque
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, A_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.A_size = A_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["s", "a", "reward", "s2", "done"])
        self.seed = random.seed(seed)
    
    def add(self, s, a, reward, s2, done):
        """Add a new experience to memory."""
        e = self.experience(s, a, reward, s2, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        S = torch.from_numpy(np.vstack([e.s for e in experiences if e is not None])).float().to(device)
        A = torch.from_numpy(np.vstack([e.a for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        S2 = torch.from_numpy(np.vstack([e.s2 for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (S, A, rewards, S2, dones)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)