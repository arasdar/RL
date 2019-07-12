import numpy as np
import random
import copy
from collections import namedtuple, deque
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    """Episodic memory: Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, random_seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        random.seed(random_seed)
        
        self.buffer = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        
        #("e: exprience", field_names=["s: state", "a: action", "r: reward", "s2: next_state", "done/teminal"])
        self.e = namedtuple("e", field_names=["s", "a", "r", "s2", "done"])
    
    def add(self, s, a, r, s2, done):
        """Add a new experience (e) to memory."""
        e = self.e(s, a, r, s2, done)
        self.buffer.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences (E) from episodic memory."""
        E = random.sample(self.buffer, k=self.batch_size)

        S = torch.from_numpy(np.vstack([e.s for e in E if e is not None])).float().to(device)
        A = torch.from_numpy(np.vstack([e.a for e in E if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.r for e in E if e is not None])).float().to(device)
        S2 = torch.from_numpy(np.vstack([e.s2 for e in E if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in E if e is not None]).astype(np.uint8)).float().to(device)

        return (S, A, rewards, S2, dones)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)