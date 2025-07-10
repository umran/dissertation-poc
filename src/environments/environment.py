import torch
from abc import ABC, abstractmethod
from typing import Tuple, Any

class Environment(ABC):
    @abstractmethod
    def state_shape(self) -> Tuple[int, ...]:
        """Return the shape of the environment state."""
        pass

    @abstractmethod
    def action_shape(self) -> Tuple[int, ...]:
        """Return the shape of the environment action."""
        pass

    @abstractmethod
    def reset(self) -> torch.Tensor:
        """
        Reset the environment and return the initial state tensor.
        The returned tensor must have shape equal to `self.state_shape()`.
        """
        pass

    @abstractmethod
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """
        Take an action `a` and return:
        - next_state: torch.Tensor (must have shape equal to self.state_shape())
        - reward: torch.Tensor float
        - terminated: torch.Tensor bool
        - truncated: torch.Tensor bool
        - info: Any
        """
        pass