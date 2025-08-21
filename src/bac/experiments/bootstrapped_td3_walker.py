import torch

from bac.algorithms.actor_critic import MultiHeadTD3
from bac.environments import Walker2D
from bac.manifest import Manifest

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(100)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(100)

test = Manifest("./out/prod", device=DEVICE)

test.bootstrapped("td3_walker", Walker2D, MultiHeadTD3, 1_000_000)