import numpy as np
import random
import copy
from collections import namedtuple, deque
import torch

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, random_seed, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        random.seed(random_seed)
        
        self.buffer = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.device = device
        
        #self.experience = namedtuple("exprience", field_names=["state", "action", "reward", "next_state", "done"])
        self.e = namedtuple("e", field_names=["s", "a", "r", "s2", "done"])
    
    def add(self, s, a, r, s2, done):
        """Add a new experience (e) to memory."""
        e_ = self.e(s, a, r, s2, done)
        self.buffer.append(e_)
    
    def sample(self):
        """Randomly sample a batch of experiences (E) from memory."""
        E = random.sample(self.buffer, k=self.batch_size)

        S = torch.from_numpy(np.vstack([e.s for e in E if e is not None])).float().to(self.device)
        A = torch.from_numpy(np.vstack([e.a for e in E if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.r for e in E if e is not None])).float().to(self.device)
        S2 = torch.from_numpy(np.vstack([e.s2 for e in E if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in E if e is not None]).astype(np.uint8)).float().to(self.device)

        return (S, A, rewards, S2, dones)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)