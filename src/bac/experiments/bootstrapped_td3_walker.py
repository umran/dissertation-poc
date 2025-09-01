import torch
import argparse

from bac.algorithms.actor_critic import MultiHeadTD3
from bac.environments import Walker2D
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
    prod.bootstrapped(f"td3_walker_{seed}", Walker2D, MultiHeadTD3, 1_000_000)

if __name__ == "__main__":
    main()