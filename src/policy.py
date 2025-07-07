import torch
from abc import ABC, abstractmethod
from typing import Tuple, Any

class Policy(ABC):
    @abstractmethod
    def action(s: torch.Tensor) -> torch.Tensor:
        pass