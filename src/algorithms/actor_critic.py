from abc import ABC, abstractmethod

from algorithms.common import ReplayBuffer
from algorithms.policy import Policy
from algorithms.networks import QNetwork

class ActorCritic(ABC):
    @abstractmethod
    def update(self, replay_buffer: ReplayBuffer, steps: int, gamma: float):
        pass

    @abstractmethod
    def get_optimal_policy(self) -> Policy:
        pass

    @abstractmethod
    def get_exploration_policy(self) -> Policy:
        pass

    @abstractmethod
    def get_critic_network(self) -> QNetwork:
        pass