import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class D(nn.Module): # Discriminator
    """Actor (policy) Model."""

    def __init__(self, s_size, a_size, h_size, random_seed):
        """Initialize parameters and build model.
        Params
        ======
            s_size (int): Dimension of each state (s)
            a_size (int): Dimension of each action (a)
            random_seed (int): Random seed
            h_size (int): Number of nodes in hidden layer
        """
        super(D, self).__init__()
        
        torch.manual_seed(random_seed)
        
        self.fc1 = nn.Linear(s_size, h_size)
        
        self.bn1 = nn.BatchNorm1d(h_size)
        
        self.fc2 = nn.Linear(h_size, a_size)
        
        self.init_parameters()

    def init_parameters(self):
        self.fc1.weight.data.uniform_(-3e-3, 3e-3) # normal (0, 1)
        self.fc2.weight.data.uniform_(-3e-3, 3e-3) # normal (0, 1)
        
    def forward(self, S):
        """Build an actor (policy) network that maps states (S) -> actions (A)."""
        H = F.leaky_relu(self.bn1(self.fc1(S))) # H: hiddden layer/output

        return torch.tanh(self.fc2(H)) # [-1, +1]