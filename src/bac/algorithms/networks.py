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

        self.register_buffer("action_min", action_min.clone().detach())
        self.register_buffer("action_max", action_max.clone().detach())

        self.core = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
    
    def forward(self, state):
        raw = torch.tanh(self.core(state))
        scaled = self._scale(raw) 
        
        return scaled

    def _scale(self, raw: torch.Tensor) -> torch.Tensor:
        return 0.5 * (raw + 1.0) * (self.action_max - self.action_min) + self.action_min

class MultiHeadQNetwork(nn.Module):
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        n_heads: int, 
        shared_core: bool = False
    ):
        super(MultiHeadQNetwork, self).__init__()

        self.n_heads = n_heads

        if shared_core:
            self.core = nn.Sequential(
                nn.Linear(state_dim + action_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU()
            )
        else:
            self.core = nn.Identity()

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(32 if shared_core else state_dim + action_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
            for _ in range(n_heads)
        ])
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if action.dim() == 2:
            # one action for all heads
            x = torch.cat([state, action], dim=-1)
            z = self.core(x)
            outs = [head(z) for head in self.heads]
            return torch.stack(outs, dim=1)

        elif action.dim() == 3:
            B, H, A = action.shape
            assert H == self.n_heads, f"action has {H} heads, expected {self.n_heads}"
            
            # expand state per head and run shared core on (s, a_h)
            x = torch.cat([state.unsqueeze(1).expand(-1, H, -1), action], dim=-1)
            z = self.core(x)
            outs = [self.heads[h](z[:, h, :]) for h in range(self.n_heads)]
            
            return torch.stack(outs, dim=1)

        else:
            raise ValueError(f"Unsupported action.dim()={action.dim()}; expected 2 or 3")

class MultiHeadPolicyNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_min: torch.Tensor,
        action_max: torch.Tensor,
        n_heads: int,
        shared_core: bool = False
    ):
        super(MultiHeadPolicyNetwork, self).__init__()
        
        assert action_min.shape == (action_dim,)
        assert action_max.shape == (action_dim,)

        self.n_heads = n_heads
        self.action_dim = action_dim

        self.register_buffer("action_min", action_min.clone().detach())
        self.register_buffer("action_max", action_max.clone().detach())

        
        if shared_core:
            self.core = nn.Sequential(
                nn.Linear(state_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU()
            )
        else:
            self.core = nn.Identity()

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(32 if shared_core else state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim)
            )
            for _ in range(n_heads)
        ])

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        z = self.core(state)
        outs = [head(z) for head in self.heads]
        raw = torch.tanh(torch.stack(outs, dim=1))
        scaled = self._scale(raw)

        return scaled
    
    def _scale(self, raw: torch.Tensor) -> torch.Tensor:
        return 0.5 * (raw + 1.0) * (self.action_max - self.action_min) + self.action_min