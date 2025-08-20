from abc import ABC, abstractmethod

from bac.algorithms.common import MaskedReplayBuffer
from bac.algorithms.policy import Policy
from bac.algorithms.networks import MultiHeadQNetwork, MultiHeadPolicyNetwork

class MultiHeadActorCritic(ABC):
    @abstractmethod
    def update(self, replay_buffer: MaskedReplayBuffer, steps: int, gamma: float):
        pass

    @abstractmethod
    def get_optimal_policy(self) -> Policy:
        pass

    @abstractmethod
    def get_exploration_policy(self) -> Policy:
        pass

    @abstractmethod
    def get_critic_network(self) -> MultiHeadQNetwork:
        pass

    @abstractmethod
    def get_actor_network(self) -> MultiHeadPolicyNetwork:
        pass

    @abstractmethod
    def get_n_heads(self) -> int:
        pass