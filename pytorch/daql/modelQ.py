import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Q_fixed(nn.Module):
    """Q-Network Model for Q-learning or reward reward function."""

    def __init__(self, s_size, random_seed):
        """Initialize parameters and build model.
        Params
        ======
            s_size (int): Dimension of each state (s)
            random_seed (int): Random seed
        """
        super(Q_fixed, self).__init__()
        
        torch.manual_seed(random_seed)
        
        self.fc = nn.Linear(s_size, 1)
        
        self.init_parameters()

    def init_parameters(self):
        self.fc.weight.data.uniform_(-3e-3, 3e-3) # normal (0, 1)
        self.fc.weight.requires_grad = False # normal (0, 1)
        self.fc.bias.requires_grad = False # normal (0, 1)
        
    def forward(self, S):
        """Build a Q-value network that maps States -> Q-value"""
        return self.fc(S)