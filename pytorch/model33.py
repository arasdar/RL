import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, S_size, A_size, seed, fc_units=400):
        """Initialize parameters and build model.
        Params
        ======
            S_size (int): Dimension of each state
            A_size (int): Dimension of each action
            seed (int): Random seed
            fc_units (int): Number of nodes in first hidden layer
            fc_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(S_size, fc_units)
        self.fc2 = nn.Linear(fc_units, fc_units)
        
        self.bn1 = nn.BatchNorm1d(fc_units)
        self.bn2 = nn.BatchNorm1d(fc_units)
        
        self.fc3 = nn.Linear(fc_units, A_size)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(-3e-3, 3e-3)
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, S):
        """Build an actor (policy) network that maps states -> actions."""
        H = F.leaky_relu(self.bn1(self.fc1(S))) # H: hiddden layer/output
        H = F.leaky_relu(self.bn2(self.fc2(H)))

        return torch.tanh(self.fc3(H))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, S_size, A_size, seed, fc_units=400):
        """Initialize parameters and build model.
        Params
        ======
            S_size (int): Dimension of each state
            A_size (int): Dimension of each action
            seed (int): Random seed
            fc_units (int): Number of nodes in first hidden layer
            fc_units (int): Number of nodes in second hidden layer
        """
        super(Critic, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(S_size, fc_units)
        self.fc2 = nn.Linear(fc_units+A_size, fc_units)
        
        self.bn1 = nn.BatchNorm1d(fc_units)
        self.bn2 = nn.BatchNorm1d(fc_units)
        
        self.fc3 = nn.Linear(fc_units, 1)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(-3e-3, 3e-3)
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, S, A):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        H = F.leaky_relu(self.bn1(self.fc1(S)))
        
        H = torch.cat((H, A), dim=1)
        
        H = F.leaky_relu(self.bn2(self.fc2(H)))
        
        return self.fc3(H)