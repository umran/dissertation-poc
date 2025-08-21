import torch

from bac.algorithms.actor_critic import MultiHeadDDPG
from bac.environments import Walker
from bac.manifest import Manifest

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(100)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(100)

test = Manifest("./out/test", device=DEVICE)

test.bootstrapped("ddpg_walker", Walker, MultiHeadDDPG, 20_000)