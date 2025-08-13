from typing import List, Tuple, Dict, Optional
from numpy.typing import ArrayLike
import numpy as np
import matplotlib.pyplot as plt

def plot_performance(
    benchmark_sets: List[Tuple[List[Dict[str, float]], str]],
    colors: List[str] = None,
    title: str = "Performance Over Time",
    truncate_after: Optional[int] = None
):
    plt.figure(figsize=(10, 6))

    for i, (results, label) in enumerate(benchmark_sets):
        steps = np.array([r["step"] for r in results])
        means = np.array([r["mean"] for r in results])
        sds = np.array([r["sd"] for r in results])
        ci = 1.96 * sds

        if truncate_after is not None:
            mask = steps <= truncate_after
            steps = steps[mask]
            means = means[mask]
            ci = ci[mask]

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

def plot_variance(
    benchmark_sets: List[Tuple[List[Dict[str, float]], str]],
    colors: List[str] = None,
    title: str = "Variance Over Time",
    truncate_after: Optional[int] = None
):
    plt.figure(figsize=(10, 6))

    for i, (results, label) in enumerate(benchmark_sets):
        steps = np.array([r["step"] for r in results])
        sds = np.array([r["sd"] for r in results])

        if truncate_after is not None:
            mask = steps <= truncate_after
            steps = steps[mask]
            sds = sds[mask]

        color = colors[i] if colors is not None and i < len(colors) else None

        plt.plot(steps, sds, label=label, color=color)

    plt.xlabel("Step")
    plt.ylabel("Reward Variance")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def save_to_npy(data: ArrayLike, filename: str):
    np.save(filename, data, allow_pickle=True)

def load_from_npy(filename: str) -> ArrayLike:
    return np.load(filename, allow_pickle=True)