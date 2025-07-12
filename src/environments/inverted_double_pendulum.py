import torch
import gymnasium as gym
from typing import Tuple, Any

from environments.environment import Environment

class InvertedDoublePendulum(Environment):
    def __init__(self, render_mode: str = "rgb_array", device: torch.device = torch.device("cpu")):
        self.env = gym.make("InvertedDoublePendulum-v5", render_mode=render_mode)
        self.device = device

    def state_shape(self) -> Tuple[int, ...]:
        return (9, )

    def action_shape(self) -> Tuple[int, ...]:
        return (1, )
    
    def action_min(self) -> torch.Tensor:
        return torch.tensor([-1.0], dtype=torch.float32, device=self.device)

    def action_max(self) -> torch.Tensor:
        return torch.tensor([1.0], dtype=torch.float32, device=self.device)

    def reset(self) -> torch.Tensor:
        observation, _ = self.env.reset()
        observation = torch.tensor(observation, dtype=torch.float32, device=self.device)

        return observation

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        action = action.detach().cpu().numpy()
        observation, reward, terminated, truncated, info = self.env.step(action)

        observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        terminated = torch.tensor(terminated, dtype=torch.bool, device=self.device)
        truncated = torch.tensor(truncated, dtype=torch.bool, device=self.device)

        return (observation, reward, terminated, truncated, info)