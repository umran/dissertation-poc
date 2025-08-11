from typing import List, Tuple, Dict
from numpy.typing import ArrayLike
import numpy as np
import matplotlib.pyplot as plt

def plot_ablation(filename: str):
    ablation = load_from_npy(filename)

    results_100_50k = ablation["hmc_ac_100_50k_results"]
    results_1000_50k = ablation["hmc_ac_1000_50k_results"]
    results_100_5k = ablation["hmc_ac_100_5k_results"]
    results_1000_5k = ablation["hmc_ac_1000_5k_results"]

    plot_multiple_benchmarks(
        [   
            (results_100_50k, "HMC 100 50K"),
            (results_1000_50k, "HMC 1000 50K"),
            (results_100_5k, "HMC 100 5K"),
            (results_1000_5k, "HMC 1000 5K")
        ],
        colors=["red", "blue", "green", "yellow"]
    )

def plot_comparative(filename: str):
    comparative = load_from_npy(filename)

    results_random = comparative["random_results"]
    results_vanilla = comparative["vanilla_results"]
    results_hmc_ac = comparative["hmc_ac_results"]

    plot_multiple_benchmarks(
        [   
            (results_random, "Uniform Random Exploration"),
            (results_vanilla, "Optimal Exploration with Gaussian Noise"),
            (results_hmc_ac, "HMC Posterior Sampling Based Exploration"),
        ],
        colors=["red", "blue", "green"]
    )

def plot_multiple_benchmarks(
    benchmark_sets: List[Tuple[List[Dict[str, float]], str]],  # (results, label)
    colors: List[str] = None,
    title: str = "Performance Over Time"
):
    plt.figure(figsize=(10, 6))

    for i, (results, label) in enumerate(benchmark_sets):
        steps = np.array([r["step"] for r in results])
        means = np.array([r["mean"] for r in results])
        sds = np.array([r["sd"] for r in results])
        ci = 1.96 * sds

        color = colors[i] if colors is not None and i < len(colors) else None

        plt.plot(steps, means, label=label, color=color)
        plt.fill_between(steps, means - ci, means + ci, alpha=0.2, color=color, linewidth=0)

    plt.xlabel("Step")
    plt.ylabel("Episodic Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def save_to_npy(data: ArrayLike, filename: str):
    np.save(filename, data, allow_pickle=True)

def load_from_npy(filename: str) -> ArrayLike:
    return np.load(filename, allow_pickle=True).item()