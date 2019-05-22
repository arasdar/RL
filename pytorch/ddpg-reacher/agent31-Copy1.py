from memory3 import ReplayBuffer
from model34 import G, D # Generator, Discriminator/Decoder
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.99            # discount factor
TAU = 1- GAMMA    # soft update rate
LR = 1e-3         # learning rate 
BATCH_SIZE = 1024         # minibatch size/ RAM size
BUFFER_SIZE = int(1e6)  # replay buffer size

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, S_size, A_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            S_size (int): dimension of each state
            A_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.S_size = S_size
        self.A_size = A_size
        self.seed = random.seed(random_seed)
        
        # Actor Network (w/ Target Network)
        self.G = G(S_size, A_size, random_seed).to(device)
        self.G_target = G(S_size, A_size, random_seed).to(device)
        self.G_optimizer = optim.Adam(self.actor.parameters(), lr=LR)

        # Critic Network (w/ Target Network)
        self.D = D(S_size, A_size, random_seed).to(device)
        self.D_target = D(S_size, A_size, random_seed).to(device)
        self.D_optimizer = optim.Adam(self.critic.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(A_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, s, a, reward, s2, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(s, a, reward, s2, done)

    def act(self, S):
        """Returns actions for given state as per current policy."""
        S = torch.from_numpy(S).float().to(device)
        self.G.eval()
        with torch.no_grad():
            A = self.G(S).cpu().data.numpy()
        self.G.train()
        return A # action [-1, 1]

    def start_learn(self):
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            GAMMA = 1 - LR # 0.999
            self.learn(experiences, GAMMA)
        
    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_target = r + γ * critic_target(S2, actor_target(S2))
        where:
            actor_target(S) -> A
            critic_target(S, A) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (S-states, A-actions, rewards, S2-next_states, dones) tuples 
            gamma (float): discount factor
        """
        S, A, rewards, S2, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        gA2 = self.G_target(S2)
        gQ2 = self.D_target(S2, gA2)
        gQ_target = rewards + (gamma * gQ2 * (1 - dones))
        # Compute critic loss
        dQ = self.D(S, A)
        dloss = ((dQ - gQ_target)**2).mean()
        #critic_loss = F.mse_loss(Q, Q_target)
        # Minimize the loss
        self.d_optimizer.zero_grad()
        dloss.backward()
        #torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.d_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        gA = self.G(S)
        gQ = self.D(S, gA)
        gloss = -gQ.mean()
        # Minimize the loss
        self.g_optimizer.zero_grad()
        gloss.backward()
        self.g_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic, self.critic_target, TAU)
        self.soft_update(self.actor, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, TAU):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_((TAU*local_param.data) + ((1-TAU)*target_param.data))