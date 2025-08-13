import torch
import gymnasium as gym
from typing import Tuple, Any

from .environment import Environment

class Pendulum(Environment):
    def __init__(self, render_mode: str = "rgb_array", device: torch.device = torch.device("cpu")):
        self.env = gym.make("Pendulum-v1", render_mode=render_mode)
        self.device = device

        self.env.reset(seed=torch.randint(0, 1 << 32, (1,), dtype=torch.int64).item())

    def state_shape(self) -> Tuple[int, ...]:
        return (3, )

    def action_shape(self) -> Tuple[int, ...]:
        return (1, )
    
    def action_min(self) -> torch.Tensor:
        return torch.tensor([-2.0], dtype=torch.float32, device=self.device)

    def action_max(self) -> torch.Tensor:
        return torch.tensor([2.0], dtype=torch.float32, device=self.device)

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