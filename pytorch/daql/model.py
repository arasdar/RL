import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class G(nn.Module):
    """Action generator or actor (policy) Model."""

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
        
        self.init_parameters()

    def init_parameters(self):
        self.fc1.weight.data.uniform_(-3e-3, 3e-3) # normal (0, 1)
        self.fc2.weight.data.uniform_(-3e-3, 3e-3) # normal (0, 1)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3) # normal (0, 1)
        
    def forward(self, S):
        """Build an actor (policy) network that maps states (S) and pred_states (S_) -> actions (A)."""
        """Build a generator network that maps state (s) and pred_state (s_) -> action (a)."""
        H = F.leaky_relu(self.bn1(self.fc1(S))) # H: hiddden layer/output
        
        H = F.leaky_relu(self.bn2(self.fc2(H))) # H: hidden layer/ output
        #H = F.leaky_relu(self.fc2(H)) # H: hidden layer/ output
        
        return torch.tanh(self.fc3(H)) # [-1, +1]


class D(nn.Module):
    """Decoder (next/final state predictor) & Discriminator (Value/evaluator/critic/examiner/final state predictor) Model."""

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
        
        self.fc4_s = nn.Linear(h_size, s_size) # Decoding/predicting next state: Decoder
        self.fc4_q = nn.Linear(h_size, 1) # Decoding/predicting final state: Discriminator
        
        self.init_parameters()

    def init_parameters(self):
        self.fc1.weight.data.uniform_(-3e-3, 3e-3) # normal (0, 1)
        self.fc2.weight.data.uniform_(-3e-3, 3e-3) # normal (0, 1)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3) # normal (0, 1)
        self.fc4_s.weight.data.uniform_(-3e-3, 3e-3) # normal (0, 1)
        self.fc4_q.weight.data.uniform_(-3e-3, 3e-3) # normal
        
        # how to freeze the output layers
        self.fc4_s.weight.requires_grad = False
        self.fc4_s.bias.requires_grad = False
        self.fc4_q.weight.requires_grad = False
        self.fc4_q.bias.requires_grad = False
        
    def forward(self, S, A):
        """Build a Descriminator/Decoder (predictor) network that maps (states, actions) pairs -> values."""
        """Build a Descriminator/Decoder (predictor) network that maps (S, A) pairs -> Q."""
        """Build a critic (value) network that maps (state, action) pairs -> value."""
        """Build a Descriminator/Decoder (predictor) network that maps (s, a) pairs -> q."""
        H = self.bn1(self.fc1(S))
        
        HA = torch.cat((H, A), dim=1)
        
        H = F.leaky_relu(self.bn2(self.fc2(HA)))
        #H = self.bn2(self.fc2(HA))
                         
        H = F.leaky_relu(self.bn3(self.fc3(H)))
        #H = F.leaky_relu(self.fc3(H))
        
        S2_ = self.fc4_s(H)
        Q_ = self.fc4_q(H)
        
        return S2_, Q_