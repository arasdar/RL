import torch
import torch.nn as nn
import torch.nn.functional as F

class Q(nn.Module):
    """Value network and actor (Policy) Model/network at the same time."""

    def __init__(self, s_size, a_size, random_seed, h_size=64):
        """Initialize parameters and build model.
        Params
        ======
            s_size (int): Dimension of each state (s)
            a_size (int): Dimension of each action (a)
            random_seed (int): Random seed
            h_size (int): Number of nodes in first hidden layer (h)
        """
        super(Q, self).__init__()
        self.random_seed = torch.manual_seed(random_seed)
        
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

        self.bn1 = nn.BatchNorm1d(h_size)

        self.init_parameters()

    def forward(self, S):
        """Build a Q-network that maps states -> actions-values."""
        """Build a Q-network that maps S -> A-Q."""
        H = F.leaky_relu(self.bn1(self.fc1(S)))
        return self.fc2(H)

    def init_parameters(self):
        self.fc1.weight.data.uniform_(-3e-3, 3e-3) # normal (0, 1)
        self.fc2.weight.data.uniform_(-3e-3, 3e-3) # normal (0, 1)

class G(nn.Module):
    """Generator network for enviornment modeling/predicting."""

    def __init__(self, s_size, a_size, random_seed, h_size=64):
        """Initialize parameters and build model.
        Params
        ======
            s_size (int): Dimension of each state (s)
            a_size (int): Dimension of each action (a)
            random_seed (int): Random seed
            h_size (int): Number of nodes in first hidden layer (h)
        """
        super(G, self).__init__()
        self.random_seed = torch.manual_seed(random_seed)
        
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size+a_size)
        self.fc3 = nn.Linear(h_size, s_size)

        self.bn1 = nn.BatchNorm1d(h_size)
        self.bn1 = nn.BatchNorm1d(h_size)

        self.init_parameters()

    def forward(self, S, A):
        """Build a G-network that maps states, actions-> next states."""
        """Build a G-network that maps S, A -> S2"""
        H = F.leaky_relu(self.bn1(self.fc1(S)))
        
        SA = torch.concat(dim=1, [S, A])
        
        H = F.leaky_relu(self.bn2(self.fc2(S)))
        
        return self.fc3(H)

    def init_parameters(self):
        self.fc1.weight.data.uniform_(-3e-3, 3e-3) # normal (0, 1)
        self.fc2.weight.data.uniform_(-3e-3, 3e-3) # normal (0, 1)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)