import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class D(nn.Module):
    """D: Discriminator/Classifier and Actor/Critic (policy/value) Model."""

    def __init__(self, s_size, a_size, random_seed, h_size=400):
        """Initialize parameters and build model.
        Params
        ======
            s_size (int): Dimension of each state (s)
            a_size (int): Dimension of each action (a)
            random_seed (int): Random seed
            h_size (int): Number of nodes in first hidden layer
        """
        super(G, self).__init__()
        
        self.random_seed = torch.manual_seed(random_seed)
        
        self.fc1 = nn.Linear(s_size, h_size)
        self.bn1 = nn.BatchNorm1d(h_size)
        
        self.fc2 = nn.Linear(h_size, h_size)
        self.bn2 = nn.BatchNorm1d(h_size)
        
        self.fc3 = nn.Linear(h_size, a_size)
        self.fc4 = nn.Linear(h_size, 1)
        
        self.init_parameters()

    def init_parameters(self):
        self.fc1.weight.data.uniform_(-3e-3, 3e-3) # normal (0, 1)
        self.fc2.weight.data.uniform_(-3e-3, 3e-3) # normal (0, 1)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3) # normal (0, 1)
        self.fc4.weight.data.uniform_(-3e-3, 3e-3) # normal (0, 1)
        
        # how to freeze the last layer Q-value
        self.fc4.weight.requires_grad = False
        self.fc4.bias.requires_grad = False
        
    def forward(self, S):
        H = F.leaky_relu(self.bn1(self.fc1(S))) # H: hiddden layer/output
        
        H = F.leaky_relu(self.bn2(self.fc2(H))) # H: hidden layer/ output
        
        A = torch.tanh(self.fc3(H)) # has to be within [-1, +1]
        Q = self.fc4(A)
        
        return A, Q


class G(nn.Module):
    """Autoencoder (next state predictor) & Generator/generative Model."""

    def __init__(self, s_size, a_size, random_seed, h_size=400):
        """Initialize parameters and build model.
        Params
        ======
            s_size (int): Dimension of each state (s)
            a_size (int): Dimension of each action (a)
            random_seed (int): Random seed
            h_size (int): Number of nodes in first hidden layer
            h_size (int): Number of nodes in second hidden layer
        """
        super(D, self).__init__()
        
        self.random_seed = torch.manual_seed(random_seed)
        
        self.fc1 = nn.Linear(s_size, h_size)
        self.bn1 = nn.BatchNorm1d(h_size)
        
        self.fc2 = nn.Linear(h_size+a_size, h_size)
        self.bn2 = nn.BatchNorm1d(h_size)
        
        self.fc3 = nn.Linear(h_size, h_size)
        self.bn3 = nn.BatchNorm1d(h_size)
        
        self.fc4 = nn.Linear(h_size, s_size)
        
        self.init_parameters()

    def init_parameters(self):
        self.fc1.weight.data.uniform_(-3e-3, 3e-3) # normal (0, 1)
        self.fc2.weight.data.uniform_(-3e-3, 3e-3) # normal (0, 1)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3) # normal (0, 1)
        self.fc4.weight.data.uniform_(-3e-3, 3e-3) # normal (0, 1)
        
    def forward(self, S, A):
        H = F.leaky_relu(self.bn1(self.fc1(S)))
        
        HA = torch.cat((H, A), dim=1)
        
        H = F.leaky_relu(self.bn2(self.fc2(HA)))
                         
        H = F.leaky_relu(self.bn3(self.fc3(H)))
        
        return self.fc4(H)