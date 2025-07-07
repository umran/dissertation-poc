import torch
from abc import ABC, abstractmethod
from typing import Tuple, Any

class Environment(ABC):
    @abstractmethod
    def state_shape(self) -> torch.Size:
        """Return the shape of the environment state."""
        pass

    @abstractmethod
    def action_shape(self) -> torch.Size:
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
    def step(self, a: torch.Tensor) -> Tuple[torch.Tensor, float, bool, bool, Any]:
        """
        Take an action `a` and return:
        - next_state: torch.Tensor (must have shape equal to self.state_shape())
        - reward: float
        - terminated: bool
        - truncated: bool
        - info: Any
        """
        pass