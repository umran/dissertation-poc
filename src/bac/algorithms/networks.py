import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(QNetwork, self).__init__()

        self.core = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.core(x)

class PolicyNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_min: torch.Tensor,
        action_max: torch.Tensor,
    ):
        super(PolicyNetwork, self).__init__()

        assert action_min.shape == (action_dim,)
        assert action_max.shape == (action_dim,)

        self.action_min = action_min
        self.action_max = action_max

        self.core = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
    
    def forward(self, state):
        raw = torch.tanh(self.core(state))

        action_min = self.action_min
        action_max = self.action_max

        scaled = 0.5 * (raw + 1.0) * (action_max - action_min) + action_min
        return scaled