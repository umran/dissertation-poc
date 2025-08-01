import torch
from abc import ABC, abstractmethod
from typing import Optional

from algorithms.common import PolicyNetwork

class Policy(ABC):
    @abstractmethod
    def action(self, state: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_policy_net(self) -> Optional[PolicyNetwork]:
        pass