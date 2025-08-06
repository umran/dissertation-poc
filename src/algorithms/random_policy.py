import torch
from typing import Tuple

from algorithms.policy import Policy

class RandomPolicy(Policy):
    def __init__(self, action_shape: Tuple[int, ...], action_min: torch.Tensor, action_max: torch.Tensor):
        self.action_shape = action_shape
        self.action_min = action_min
        self.action_range = action_max - action_min
    
    def action(self, state: torch.Tensor) -> torch.Tensor:
        batch_shape = state.shape[:-1]  # assumes last dim is feature dim; adjust if needed
        # Reshape for broadcasting: (batch_shape..., *action_shape)
        rand_shape = batch_shape + self.action_shape

        return self.action_min + self.action_range * torch.rand(rand_shape, dtype=torch.float32, device=self.action_min.device)