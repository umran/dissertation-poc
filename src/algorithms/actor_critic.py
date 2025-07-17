import torch
from abc import ABC, abstractmethod

from algorithms.common import ReplayBuffer
from algorithms.policy import Policy

class ActorCritic(ABC):
    @abstractmethod
    def update(self, replay_buffer: ReplayBuffer, steps: int):
        pass

    @abstractmethod
    def compute_td_target(self, next_state: torch.Tensor, reward: torch.Tensor, term: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_optimal_policy(self) -> Policy:
        pass

    @abstractmethod
    def get_exploration_policy(self) -> Policy:
        pass