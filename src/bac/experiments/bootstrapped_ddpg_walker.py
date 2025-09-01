import torch

from bac.algorithms.actor_critic import MultiHeadDDPG
from bac.environments import Walker2D
from bac.manifest import Manifest

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(102)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(102)

test = Manifest("./out/prod", device=DEVICE)

test.bootstrapped("ddpg_walker_102", Walker2D, MultiHeadDDPG, 1_000_000)