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

class MultiHeadQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, n_heads: int):
        super(MultiHeadQNetwork, self).__init__()

        self.n_heads = n_heads

        # shared core over (s, a)
        self.core = nn.Sequential(
            nn.Linear(state_dim + action_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # independent heads: (32 -> head_hidden -> 1)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
            for _ in range(n_heads)
        ])
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        state:  [B, state_dim]
        action: [B, action_dim]
        returns: Q per head -> [B, H, 1]
        """
        x = torch.cat([state, action], dim=-1)       # [B, state+action]
        z = self.core(x)                              # [B, 32]
        outs = [head(z) for head in self.heads]       # list of [B, 1]
        return torch.stack(outs, dim=1)

class MultiHeadPolicyNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_min: torch.Tensor,
        action_max: torch.Tensor,
        n_heads: int
    ):
        super(MultiHeadPolicyNetwork, self).__init__()
        
        assert action_min.shape == (action_dim,)
        assert action_max.shape == (action_dim,)

        self.n_heads = n_heads
        self.action_dim = action_dim

        # register as buffers so they follow .to(device), save/load, etc.
        self.register_buffer("action_min", action_min.clone().detach())
        self.register_buffer("action_max", action_max.clone().detach())

        # shared core over state
        self.core = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # independent head MLPs: 32 -> 16 -> action_dim
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, action_dim)
            )
            for _ in range(n_heads)
        ])

    def _scale(self, raw: torch.Tensor) -> torch.Tensor:
        """
        raw: tanh output in [-1, 1], shape [..., action_dim]
        returns scaled to [action_min, action_max], same leading dims as raw
        """
        # Broadcast action bounds across batch and heads
        return 0.5 * (raw + 1.0) * (self.action_max - self.action_min) + self.action_min

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: [B, state_dim]
        returns per-head actions scaled to bounds: [B, H, action_dim]
        """
        z = self.core(state)  # [B, 32]
        outs = [head(z) for head in self.heads]           # list of [B, A]
        raw = torch.tanh(torch.stack(outs, dim=1))        # [B, H, A] in [-1,1]
        # scale per action dim (broadcast over B,H)
        scaled = self._scale(raw)                         # [B, H, A]
        return scaled