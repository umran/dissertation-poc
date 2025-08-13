import torch

from bac.algorithms.actor_critic import TD3
from bac.environments.pendulum import Pendulum
from bac.manifest import Manifest

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(100)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(100)

test = Manifest("./out/test", device=DEVICE)

test.baselines("td3_pendulum", Pendulum, TD3, 20_000)
test.ablation("td3_pendulum", Pendulum, TD3, 20_000)