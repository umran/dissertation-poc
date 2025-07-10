from abc import ABC, abstractmethod

from algorithms.common import ReplayBuffer
from algorithms.policy import Policy

class ActorCritic(ABC):
    @abstractmethod
    def update(self, step: int, replay_buffer: ReplayBuffer):
        pass

    @abstractmethod
    def get_optimal_policy(self) -> Policy:
        pass

    @abstractmethod
    def get_exploration_policy(self) -> Policy:
        pass