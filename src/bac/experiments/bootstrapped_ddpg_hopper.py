import torch

from bac.algorithms.actor_critic import MultiHeadDDPG
from bac.environments import Hopper
from bac.manifest import Manifest

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(101)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(101)

test = Manifest("./out/prod", device=DEVICE)

test.bootstrapped("ddpg_hopper_101", Hopper, MultiHeadDDPG, 1_000_000)