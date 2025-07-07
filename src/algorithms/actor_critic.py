from abc import ABC, abstractmethod
from util.replay_buffer import ReplayBuffer
from policy import Policy

class ActorCritic(ABC):
    @abstractmethod
    def update(self, step: int, replay_buffer: ReplayBuffer):
        pass

    @abstractmethod
    def get_optimal_policy(self) -> Policy:
        pass

    @abstractmethod
    def get_exploratory_policy(self) -> Policy:
        pass