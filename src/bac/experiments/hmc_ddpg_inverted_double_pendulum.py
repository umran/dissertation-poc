import torch
import argparse

from bac.algorithms.actor_critic import DDPG
from bac.environments import InvertedDoublePendulum
from bac.manifest import Manifest

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed (integer, required)"
    )
    args = parser.parse_args()
    seed = args.seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    prod = Manifest("./out/prod", device=DEVICE)
    prod.hmc_baselines(f"ddpg_inverted_double_pendulum_{seed}", InvertedDoublePendulum, DDPG, 1_000_000)
    prod.hmc_ablation(f"ddpg_inverted_double_pendulum_{seed}", InvertedDoublePendulum, DDPG, 1_000_000)

if __name__ == "__main__":
    main()