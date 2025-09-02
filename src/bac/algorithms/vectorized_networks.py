import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, n_heads: int, shared_core: bool = False):
        super().__init__()
        self.n_heads = n_heads

        if shared_core:
            self.core = nn.Sequential(
                nn.Linear(state_dim + action_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU()
            )
            core_out = 32
        else:
            self.core = nn.Identity()
            core_out = state_dim + action_dim

        self.heads = PerHeadQNetwork(n_heads, core_out)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if action.dim() == 2:
            # one action for all heads
            x = torch.cat([state, action], dim=-1)
            z = self.core(x)
            z = z.unsqueeze(1).expand(-1, self.n_heads, -1)
            q = self.heads(z)
            return q

        elif action.dim() == 3:
            B, H, A = action.shape
            assert H == self.n_heads, f"action has {H} heads, expected {self.n_heads}"
            x = torch.cat([state.unsqueeze(1).expand(-1, H, -1), action], dim=-1)
            z = self.core(x)
            q = self.heads(z)
            return q

        else:
            raise ValueError(f"Unsupported action.dim: {action.dim()}; expected 2 or 3")

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
        super().__init__()

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
            core_out = 32
        else:
            self.core = nn.Identity()
            core_out = state_dim

        self.heads = PerHeadPolicyNetwork(n_heads, core_out, 128, action_dim)

    def _scale(self, raw: torch.Tensor) -> torch.Tensor:
        return 0.5 * (raw + 1.0) * (self.action_max - self.action_min) + self.action_min

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        z = self.core(state)
        z = z.unsqueeze(1).expand(-1, self.n_heads, -1)
        raw = torch.tanh(self.heads(z))
        return self._scale(raw)

class PerHeadQNetwork(nn.Module):
    def __init__(self, n_heads: int, in_dim: int):
        super().__init__()
        self.l1 = GroupedLinear(n_heads, in_dim, 128)
        self.l2 = GroupedLinear(n_heads, 128, 128)
        self.l3 = GroupedLinear(n_heads, 128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        y = self.l3(h)
        return y

class PerHeadPolicyNetwork(nn.Module):
    def __init__(self, n_heads: int, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.l1 = GroupedLinear(n_heads, in_dim, hidden)
        self.l2 = GroupedLinear(n_heads, hidden, hidden)
        self.l3 = GroupedLinear(n_heads, hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        y = self.l3(h)
        return y

class GroupedLinear(nn.Module):
    def __init__(self, n_heads: int, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_heads, out_dim, in_dim))
        self.bias = nn.Parameter(torch.empty(n_heads, out_dim)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        for h in range(self.weight.size(0)):
            nn.init.kaiming_uniform_(self.weight[h], a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.size(-1)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.einsum('bhi,hoi->bho', x, self.weight)
        if self.bias is not None:
            y = y + self.bias
        return y
