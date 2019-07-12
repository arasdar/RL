import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class G(nn.Module): # Generator
    """Generative Model or Autoencoder or Advencoder."""

    def __init__(self, s_size, a_size, random_seed, h_size=400):
        """Initialize parameters and build model.
        Params
        ======
            s_size (int): Dimension of each state (s)
            a_size (int): Dimension of each action (a)
            random_seed (int): Random seed
            h_size (int): Number of nodes in hidden layer
        """
        super(G, self).__init__()
        
        torch.manual_seed(random_seed)
        
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size+a_size, h_size)
        
        self.bn1 = nn.BatchNorm1d(h_size)
        self.bn2 = nn.BatchNorm1d(h_size)
        
        self.fc3 = nn.Linear(h_size, s_size)
        
        self.init_parameters()

    def init_parameters(self):
        self.fc1.weight.data.uniform_(-3e-3, 3e-3) # normal (0, 1)
        self.fc2.weight.data.uniform_(-3e-3, 3e-3) # normal (0, 1)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3) # normal (0, 1)
        
    def forward(self, S, A):
        """Build a Generator network that maps (states, actions) pairs -> next/pred states."""
        H = F.leaky_relu(self.bn1(self.fc1(S)))
        
        HA = torch.cat((H, A), dim=1)
        
        H = F.leaky_relu(self.bn2(self.fc2(HA)))
        
        return self.fc3(H)