import torch

from bac.algorithms.actor_critic import TD3
from bac.environments import Pendulum
from bac.manifest import Manifest

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(101)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(101)

prod = Manifest("./out/prod", device=DEVICE)

prod.hmc_baselines("td3_pendulum_101", Pendulum, TD3, 1_000_000)
prod.hmc_ablation("td3_pendulum_101", Pendulum, TD3, 1_000_000)