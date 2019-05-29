import numpy as np
import random
import copy
from collections import namedtuple, deque
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    """Fixed-size buffer to store experience (e) tuples."""
    """Experiences (E)"""

    def __init__(self, a_size, buffer_size, batch_size, random_seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.a_size = a_size
        self.buffer = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        #self.experience = namedtuple("exprience", field_names=["state", "action", "reward", "next_state", "done/terminal"])
        #s_: predicted state", s2_: predicted next state
        self.e = namedtuple("e", field_names=["s", "a", "r", "s2", "done", "s_", "s2_"])
        self.random_seed = random.seed(random_seed)
    
    def add(self, s, a, r, s2, done, s_, s2_):
        """Add a new experience to memory."""
        e = self.e(s, a, r, s2, done, s_, s2_)
        self.buffer.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences (E) from memory."""
        E = random.sample(self.buffer, k=self.batch_size)

        S = torch.from_numpy(np.vstack([e.s for e in E if e is not None])).float().to(device)
        A = torch.from_numpy(np.vstack([e.a for e in E if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.r for e in E if e is not None])).float().to(device)
        S2 = torch.from_numpy(np.vstack([e.s2 for e in E if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in E if e is not None]).astype(np.uint8)).float().to(device)
        S_ = torch.from_numpy(np.vstack([e.s_ for e in E if e is not None])).float().to(device)
        S2_ = torch.from_numpy(np.vstack([e.s2_ for e in E if e is not None])).float().to(device)

        return (S, A, rewards, S2, dones, S_, S2_)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)