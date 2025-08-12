import torch

from algorithms.td3 import TD3
from environments.pendulum import Pendulum
from experiments import Experiments

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(100)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(100)

prod = Experiments("./out/prod", device=DEVICE)

prod.ablation("td3_pendulum", Pendulum, TD3, 1_000_000)
# prod.baselines("td3_pendulum", Pendulum, TD3, 1_000_000)