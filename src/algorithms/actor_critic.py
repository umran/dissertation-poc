from abc import ABC, abstractmethod
import torch

from algorithms.common import ReplayBuffer, sample_gaussian
from algorithms.policy import Policy
from algorithms.networks import QNetwork, PolicyNetwork

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

    @abstractmethod
    def get_actor_network(self) -> PolicyNetwork:
        pass

class OptimalPolicy(Policy):
    def __init__(self, policy_net: PolicyNetwork):
        self.policy_net = policy_net

    def action(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.policy_net(state)

class ExplorationPolicy(Policy):
    def __init__(self, policy_net: PolicyNetwork, noise: float, action_min: torch.Tensor, action_max: torch.Tensor):
        self.policy_net = policy_net
        self.noise = noise
        self.action_min = action_min
        self.action_max = action_max
    
    def action(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action = self.policy_net(state)
        
        noise = sample_gaussian(0.0, self.noise, action.shape, device=action.device)

        return torch.clamp(action + noise, self.action_min, self.action_max)