import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class G(nn.Module):
    """Action generator or actor (Policy) Model."""

    def __init__(self, s_size, a_size, seed, h_size=400):
        """Initialize parameters and build model.
        Params
        ======
            s_size (int): Dimension of each state
            a_size (int): Dimension of each action
            seed (int): Random seed
            h_size (int): Number of nodes in first hidden layer
        """
        super(G, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(s_size, h_units)
        
        self.bn1 = nn.BatchNorm1d(h_units)
        
        self.fc2 = nn.Linear(h_units, a_size)
        
        self.init_parameters()

    def init_parameters(self):
        self.fc1.weight.data.uniform_(-3e-3, 3e-3)
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, S):
        """Build an actor (policy) network that maps states -> actions."""
        H = F.leaky_relu(self.bn1(self.fc1(S))) # H: hiddden layer/output

        return torch.tanh(self.fc2(H))


class D(nn.Module):
    """Next state predictor/Decoder & final state predictor/Discriminator (Value) Model."""

    def __init__(self, s_size, a_size, seed, h_units=400):
        """Initialize parameters and build model.
        Params
        ======
            s_size (int): Dimension of each state
            a_size (int): Dimension of each action
            seed (int): Random seed
            h_units (int): Number of nodes in first hidden layer
            h_units (int): Number of nodes in second hidden layer
        """
        super(D, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(s_size, h_units)
        self.fc2 = nn.Linear(h_units+a_size, h_units)
        
        self.bn1 = nn.BatchNorm1d(h_units)
        self.bn2 = nn.BatchNorm1d(h_units)
        
        self.fc3 = nn.Linear(h_units, 1)
        
        self.init_parameters()

    def init_parameters(self):
        self.fc1.weight.data.uniform_(-3e-3, 3e-3)
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, S, A):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        H = F.leaky_relu(self.bn1(self.fc1(S)))
        
        H = torch.cat((H, A), dim=1)
        
        H = F.leaky_relu(self.bn2(self.fc2(H)))
        
        return self.fc3(H)