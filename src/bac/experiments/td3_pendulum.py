import torch

from bac.algorithms.actor_critic import TD3
from bac.environments import Pendulum
from bac.manifest import Manifest

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(100)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(100)

prod = Manifest("./out/prod", device=DEVICE)

prod.baselines("td3_pendulum", Pendulum, TD3, 1_000_000)
prod.ablation("td3_pendulum", Pendulum, TD3, 1_000_000)