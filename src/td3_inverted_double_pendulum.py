import torch

from algorithms.td3 import TD3
from environments.inverted_double_pendulum import InvertedDoublePendulum
from experiments import Experiments

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(100)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(100)

prod = Experiments("./out/prod", device=DEVICE)

prod.baselines("td3_inverted_double_pendulum", InvertedDoublePendulum, TD3, 1_000_000)
prod.ablation("td3_inverted_double_pendulum", InvertedDoublePendulum, TD3, 1_000_000)