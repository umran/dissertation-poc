import torch
from abc import ABC, abstractmethod

class Policy(ABC):
    @abstractmethod
    def action(self, state: torch.Tensor) -> torch.Tensor:
        pass