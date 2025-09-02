import os
from pathlib import Path
import torch
from typing import Type, Optional, List, Dict
from collections import defaultdict

from bac.util import save_to_npy, load_from_npy, plot_performance, plot_variance, plot_cumulative_reward
from bac.algorithms.common import new_observer, new_sample_observer
from bac.algorithms.actor_critic import ActorCritic, MultiHeadActorCritic
from bac.algorithms import HMCActorCritic, VanillaActorCritic, BootstrappedActorCritic
from bac.environments import Environment

HMC_AC_SMALL_100_5K = {
    "batch_size": 128,
    "num_warmup": 500,
    "num_samples": 1000,
    "update_every": 5000,
    "exploration_policy_lr": 1e-2,
    "exploration_policy_optimization_steps": 100,
    "exploration_policy_optimization_batch_size": 128,
    "replay_buffer_size": 1_000_000,
    "episodic_replay_buffer_size": 1_000_000,
}

HMC_AC_SMALL_100_50K = {
    "batch_size": 128,
    "num_warmup": 500,
    "num_samples": 1000,
    "update_every": 50_000,
    "exploration_policy_lr": 1e-2,
    "exploration_policy_optimization_steps": 100,
    "exploration_policy_optimization_batch_size": 128,
    "replay_buffer_size": 1_000_000,
    "episodic_replay_buffer_size": 1_000_000,
}

HMC_AC_LARGE_100_5K = {
    "batch_size": 1024,
    "num_warmup": 500,
    "num_samples": 1000,
    "update_every": 5_000,
    "exploration_policy_lr": 1e-2,
    "exploration_policy_optimization_steps": 100,
    "exploration_policy_optimization_batch_size": 128,
    "replay_buffer_size": 1_000_000,
    "episodic_replay_buffer_size": 1_000_000,
}

HMC_AC_LARGE_100_50K = {
    "batch_size": 1024,
    "num_warmup": 500,
    "num_samples": 1000,
    "update_every": 50_000,
    "exploration_policy_lr": 1e-2,
    "exploration_policy_optimization_steps": 100,
    "exploration_policy_optimization_batch_size": 128,
    "replay_buffer_size": 1_000_000,
    "episodic_replay_buffer_size": 1_000_000,
}

BS_AC_1_P100 = {
    "n_heads": 1,
    "p": 1.0,
    "replay_buffer_size": 1_000_000,
}

BS_AC_5_P50 = {
    "n_heads": 5,
    "p": 0.5,
    "replay_buffer_size": 1_000_000,
}

BS_AC_5_P80 = {
    "n_heads": 5,
    "p": 0.8,
    "replay_buffer_size": 1_000_000,
}

BS_AC_5_P95 = {
    "n_heads": 5,
    "p": 0.95,
    "replay_buffer_size": 1_000_000,
}

BS_AC_10_P50 = {
    "n_heads": 10,
    "p": 0.5,
    "replay_buffer_size": 1_000_000,
}

BS_AC_10_P80 = {
    "n_heads": 10,
    "p": 0.8,
    "replay_buffer_size": 1_000_000,
}

BS_AC_10_P95 = {
    "n_heads": 10,
    "p": 0.95,
    "replay_buffer_size": 1_000_000,
}

BS_AC_20_P50 = {
    "n_heads": 20,
    "p": 0.5,
    "replay_buffer_size": 1_000_000,
}

BS_AC_20_P80 = {
    "n_heads": 20,
    "p": 0.8,
    "replay_buffer_size": 1_000_000,
}

BS_AC_20_P95 = {
    "n_heads": 20,
    "p": 0.95,
    "replay_buffer_size": 1_000_000,
}

class Manifest:
    def __init__(self, outdir: str, device: torch.device = torch.device("cpu")):
        self.outdir = outdir
        self.device = device

    def hmc_baselines(self, prefix: str, environment_cls: Type[Environment], actor_critic_cls: Type[ActorCritic], steps: int):
        vanilla_results_url = self.make_url(prefix, "hmc_vanilla_results")
        random_results_url = self.make_url(prefix, "hmc_random_results")

        if not os.path.isfile(vanilla_results_url):
            vanilla, vanilla_observer, vanilla_results = self.prepare_actor_critic(environment_cls, actor_critic_cls)
            vanilla.train(steps=steps, observer=vanilla_observer)
            
            save_to_npy(vanilla_results, vanilla_results_url)

        if not os.path.isfile(random_results_url):
            random, random_observer, random_results = self.prepare_actor_critic(environment_cls, actor_critic_cls)
            random.train(steps=steps, start_steps=steps, observer=random_observer)
            
            save_to_npy(random_results, random_results_url)

    def hmc_ablation(self, prefix: str, environment_cls: Type[Environment], actor_critic_cls: Type[ActorCritic], steps: int):
        large_100_5k_results_url = self.make_url(prefix, "hmc_large_100_5k_results")
        small_100_5k_results_url = self.make_url(prefix, "hmc_small_100_5k_results")
        large_100_50k_results_url = self.make_url(prefix, "hmc_large_100_50k_results")
        small_100_50k_results_url = self.make_url(prefix, "hmc_small_100_50k_results")

        if not os.path.isfile(large_100_5k_results_url):  
            large_100_5k, large_100_5k_observer, large_100_5k_results, large_100_5k_sample_observer, large_100_5k_posterior_samples = self.prepare_hmc_actor_critic(environment_cls, actor_critic_cls)      
            large_100_5k.train(steps=steps, observer=large_100_5k_observer, sample_observer=large_100_5k_sample_observer, **HMC_AC_LARGE_100_5K)

            large_100_5k_posterior_samples_url = self.make_url(prefix, "hmc_large_100_5k_posterior_samples")

            save_to_npy(large_100_5k_results, large_100_5k_results_url)
            save_to_npy(large_100_5k_posterior_samples, large_100_5k_posterior_samples_url)
        
        if not os.path.isfile(small_100_5k_results_url):
            small_100_5k, small_100_5k_observer, small_100_5k_results, small_100_5k_sample_observer, small_100_5k_posterior_samples = self.prepare_hmc_actor_critic(environment_cls, actor_critic_cls)
            small_100_5k.train(steps=steps, observer=small_100_5k_observer, sample_observer=small_100_5k_sample_observer, **HMC_AC_SMALL_100_5K)
        
            small_100_5k_posterior_samples_url = self.make_url(prefix, "hmc_small_100_5k_posterior_samples")

            save_to_npy(small_100_5k_results, small_100_5k_results_url)
            save_to_npy(small_100_5k_posterior_samples, small_100_5k_posterior_samples_url)

        if not os.path.isfile(large_100_50k_results_url):
            large_100_50k, large_100_50k_observer, large_100_50k_results, large_100_50k_sample_observer, large_100_50k_posterior_samples = self.prepare_hmc_actor_critic(environment_cls, actor_critic_cls)
            large_100_50k.train(steps=steps, observer=large_100_50k_observer, sample_observer=large_100_50k_sample_observer, **HMC_AC_LARGE_100_50K)
        
            large_100_50k_posterior_samples_url = self.make_url(prefix, "hmc_large_100_50k_posterior_samples")

            save_to_npy(large_100_50k_results, large_100_50k_results_url)
            save_to_npy(large_100_50k_posterior_samples, large_100_50k_posterior_samples_url)

        if not os.path.isfile(small_100_50k_results_url):
            small_100_50k, small_100_50k_observer, small_100_50k_results, small_100_50k_sample_observer, small_100_50k_posterior_samples = self.prepare_hmc_actor_critic(environment_cls, actor_critic_cls)
            small_100_50k.train(steps=steps, observer=small_100_50k_observer, sample_observer=small_100_50k_sample_observer, **HMC_AC_SMALL_100_50K)

            small_100_50k_posterior_samples_url = self.make_url(prefix, "hmc_small_100_50k_posterior_samples")

            save_to_npy(small_100_50k_results, small_100_50k_results_url)
            save_to_npy(small_100_50k_posterior_samples, small_100_50k_posterior_samples_url)

    def bootstrapped(self, prefix: str, environment_cls: Type[Environment], actor_critic_cls: Type[MultiHeadActorCritic], steps: int):
        bs_20_p50_results_url = self.make_url(prefix, "bs_20_p50_results")
        bs_20_p80_results_url = self.make_url(prefix, "bs_20_p80_results")
        bs_20_p95_results_url = self.make_url(prefix, "bs_20_p95_results")
        bs_10_p50_results_url = self.make_url(prefix, "bs_10_p50_results")
        bs_10_p80_results_url = self.make_url(prefix, "bs_10_p80_results")
        bs_10_p95_results_url = self.make_url(prefix, "bs_10_p95_results")
        bs_5_p50_results_url = self.make_url(prefix, "bs_5_p50_results")
        bs_5_p80_results_url = self.make_url(prefix, "bs_5_p80_results")
        bs_5_p95_results_url = self.make_url(prefix, "bs_5_p95_results")
        bs_1_p100_results_url = self.make_url(prefix, "bs_1_p100_results")

        if not os.path.isfile(bs_20_p50_results_url):
            bs_20_p50, bs_20_p50_observer, bs_20_p50_results = self.prepare_bs_actor_critic(environment_cls, actor_critic_cls, n_heads=BS_AC_20_P50["n_heads"])
            bs_20_p50.train(p=BS_AC_20_P50["p"], steps=steps, observer=bs_20_p50_observer)
            
            save_to_npy(bs_20_p50_results, bs_20_p50_results_url)
        
        if not os.path.isfile(bs_20_p80_results_url):
            bs_20_p80, bs_20_p80_observer, bs_20_p80_results = self.prepare_bs_actor_critic(environment_cls, actor_critic_cls, n_heads=BS_AC_20_P80["n_heads"])
            bs_20_p80.train(p=BS_AC_20_P80["p"], steps=steps, observer=bs_20_p80_observer)
            
            save_to_npy(bs_20_p80_results, bs_20_p80_results_url)

        if not os.path.isfile(bs_20_p95_results_url):
            bs_20_p95, bs_20_p95_observer, bs_20_p95_results = self.prepare_bs_actor_critic(environment_cls, actor_critic_cls, n_heads=BS_AC_20_P95["n_heads"])
            bs_20_p95.train(p=BS_AC_20_P95["p"], steps=steps, observer=bs_20_p95_observer)
            
            save_to_npy(bs_20_p95_results, bs_20_p95_results_url)
        
        if not os.path.isfile(bs_10_p50_results_url):
            bs_10_p50, bs_10_p50_observer, bs_10_p50_results = self.prepare_bs_actor_critic(environment_cls, actor_critic_cls, n_heads=BS_AC_10_P50["n_heads"])
            bs_10_p50.train(p=BS_AC_10_P50["p"], steps=steps, observer=bs_10_p50_observer)
            
            save_to_npy(bs_10_p50_results, bs_10_p50_results_url)
        
        if not os.path.isfile(bs_10_p80_results_url):
            bs_10_p80, bs_10_p80_observer, bs_10_p80_results = self.prepare_bs_actor_critic(environment_cls, actor_critic_cls, n_heads=BS_AC_10_P80["n_heads"])
            bs_10_p80.train(p=BS_AC_10_P80["p"], steps=steps, observer=bs_10_p80_observer)
            
            save_to_npy(bs_10_p80_results, bs_10_p80_results_url)

        if not os.path.isfile(bs_10_p95_results_url):
            bs_10_p95, bs_10_p95_observer, bs_10_p95_results = self.prepare_bs_actor_critic(environment_cls, actor_critic_cls, n_heads=BS_AC_10_P95["n_heads"])
            bs_10_p95.train(p=BS_AC_10_P95["p"], steps=steps, observer=bs_10_p95_observer)
            
            save_to_npy(bs_10_p95_results, bs_10_p95_results_url)

        if not os.path.isfile(bs_5_p50_results_url):
            bs_5_p50, bs_5_p50_observer, bs_5_p50_results = self.prepare_bs_actor_critic(environment_cls, actor_critic_cls, n_heads=BS_AC_5_P50["n_heads"])
            bs_5_p50.train(p=BS_AC_5_P50["p"], steps=steps, observer=bs_5_p50_observer)
            
            save_to_npy(bs_5_p50_results, bs_5_p50_results_url)
        
        if not os.path.isfile(bs_5_p80_results_url):
            bs_5_p80, bs_5_p80_observer, bs_5_p80_results = self.prepare_bs_actor_critic(environment_cls, actor_critic_cls, n_heads=BS_AC_5_P80["n_heads"])
            bs_5_p80.train(p=BS_AC_5_P80["p"], steps=steps, observer=bs_5_p80_observer)
            
            save_to_npy(bs_5_p80_results, bs_5_p80_results_url)

        if not os.path.isfile(bs_5_p95_results_url):
            bs_5_p95, bs_5_p95_observer, bs_5_p95_results = self.prepare_bs_actor_critic(environment_cls, actor_critic_cls, n_heads=BS_AC_5_P95["n_heads"])
            bs_5_p95.train(p=BS_AC_5_P95["p"], steps=steps, observer=bs_5_p95_observer)
            
            save_to_npy(bs_5_p95_results, bs_5_p95_results_url)

        if not os.path.isfile(bs_1_p100_results_url):
            bs_1_p100, bs_1_p100_observer, bs_1_p100_results = self.prepare_bs_actor_critic(environment_cls, actor_critic_cls, n_heads=BS_AC_1_P100["n_heads"])
            bs_1_p100.train(p=BS_AC_1_P100["p"], steps=steps, observer=bs_1_p100_observer)
            
            save_to_npy(bs_1_p100_results, bs_1_p100_results_url)

    def prepare_actor_critic(self, environment_cls: Type[Environment], actor_critic_cls: Type[ActorCritic]):
        ac = VanillaActorCritic(environment_cls(device=self.device), actor_critic_cls(environment_cls(device=self.device), device=self.device), device=self.device)
        observer, results = new_observer(environment_cls(device=self.device))

        return ac, observer, results

    def prepare_hmc_actor_critic(self, environment_cls: Type[Environment], actor_critic_cls: Type[ActorCritic]):
        hmc_ac = HMCActorCritic(environment_cls(device=self.device), actor_critic_cls(environment_cls(device=self.device), device=self.device), device=self.device)
        observer, results = new_observer(environment_cls(device=self.device))
        sample_observer, posterior_samples = new_sample_observer()

        return hmc_ac, observer, results, sample_observer, posterior_samples
    
    def prepare_bs_actor_critic(self, environment_cls: Type[Environment], actor_critic_cls: Type[MultiHeadActorCritic], n_heads: int):
        ac = BootstrappedActorCritic(environment_cls(device=self.device), actor_critic_cls(environment_cls(device=self.device), n_heads=n_heads, device=self.device), device=self.device)
        observer, results = new_observer(environment_cls(device=self.device))

        return ac, observer, results
    
    def make_url(self, prefix: str, name: str):
        return f"{self.outdir}/{prefix}__{name}.npy"
    
    def plot_hmc_comparison(self, prefix: str, truncate_after: Optional[int] = None):
        random = []
        vanilla = []
        large_5k = []
        
        directory = Path(self.outdir)
        matches = list(directory.glob(f"{prefix}_*__hmc_*_results.npy"))

        for path in matches:
            data = load_from_npy(str(path))
            id = path.stem.split("__hmc_", 1)[1].removesuffix("_results")

            match id:
                case "random":
                    random.append(data)
                case "vanilla":
                    vanilla.append(data)
                case "large_100_5k":
                    large_5k.append(data)
                case _:
                    pass

        # compute means across all seeds
        random = compute_means(random)
        vanilla = compute_means(vanilla)
        large_5k = compute_means(large_5k)

        results = []
        colors = []

        results.append((random, "Uniform Random"))
        colors.append("red")

        results.append((vanilla, "Vanilla Actor Critic"))
        colors.append("blue")

        results.append((large_5k, "HMC Posterior Sampling"))
        colors.append("green")
                
        plot_performance(results, colors=colors, truncate_after=truncate_after)
        plot_cumulative_reward(results, colors=colors, truncate_after=truncate_after)
        plot_variance(results, colors=colors, truncate_after=truncate_after)
    
    def plot_hmc_ablation(self, prefix: str, truncate_after: Optional[int] = None):
        random = []
        vanilla = []
        large_5k = []
        small_5k = []
        large_50k = []
        small_50k = []

        directory = Path(self.outdir)
        matches = list(directory.glob(f"{prefix}_*__hmc_*_results.npy"))

        for path in matches:
            data = load_from_npy(str(path))
            id = path.stem.split("__hmc_", 1)[1].removesuffix("_results")

            match id:
                case "random":
                    random.append(data)
                case "vanilla":
                    vanilla.append(data)
                case "large_100_5k":
                    large_5k.append(data)
                case "small_100_5k":
                    small_5k.append(data)
                case "large_100_50k":
                    large_50k.append(data)
                case "small_100_50k":
                    small_50k.append(data)
                case _:
                    pass

        # compute means across all seeds
        random = compute_means(random)
        vanilla = compute_means(vanilla)
        large_5k = compute_means(large_5k)
        small_5k = compute_means(small_5k)
        large_50k = compute_means(large_50k)
        small_50k = compute_means(small_50k)

        results_large = []
        colors_large = []
        results_small = []
        colors_small = []
        results_50k = []
        colors_50k = []
        results_5k = []
        colors_5k = []

        # add random baseline to all plots
        results_large.append((random, "Uniform Random"))
        colors_large.append("red")
        results_small.append((random, "Uniform Random"))
        colors_small.append("red")
        results_50k.append((random, "Uniform Random"))
        colors_50k.append("red")
        results_5k.append((random, "Uniform Random"))
        colors_5k.append("red")

        # add vanilla baseline to all plots
        results_large.append((vanilla, "Vanilla Actor Critic"))
        colors_large.append("blue")
        results_small.append((vanilla, "Vanilla Actor Critic"))
        colors_small.append("blue")
        results_50k.append((vanilla, "Vanilla Actor Critic"))
        colors_50k.append("blue")
        results_5k.append((vanilla, "Vanilla Actor Critic"))
        colors_5k.append("blue")

        # add large_5k to large and 5k plots
        results_large.append((large_5k, "HMC AC Large 5K"))
        colors_large.append("green")
        results_5k.append((large_5k, "HMC AC Large 5K"))
        colors_5k.append("green")

        # add small_5k to small and 5k plots
        results_small.append((small_5k, "HMC AC Small 5K"))
        colors_small.append("orange")
        results_5k.append((small_5k, "HMC AC Small 5K"))
        colors_5k.append("orange")

        # add large_50k to large and 50k plots
        results_large.append((large_50k, "HMC AC Large 50K"))
        colors_large.append("purple")
        results_50k.append((large_50k, "HMC AC Large 50K"))
        colors_50k.append("purple")

        # add small_50k to small and 50k plots
        results_small.append((small_50k, "HMC AC Small 50K"))
        colors_small.append("brown")
        results_50k.append((small_50k, "HMC AC Small 50K"))
        colors_50k.append("brown")

        plot_performance(results_large, colors=colors_large, truncate_after=truncate_after, title="1024 Observations Per Update")
        plot_performance(results_small, colors=colors_small, truncate_after=truncate_after, title="128 Observations Per Update")
        plot_performance(results_5k, colors=colors_5k, truncate_after=truncate_after, title="Updates Every 5K Steps")
        plot_performance(results_50k, colors=colors_50k, truncate_after=truncate_after, title="Updates Every 50K Steps")
    
    def plot_bootstrapped(self, prefix: str, truncate_after: Optional[int] = None):
        h1_p100 = []
        h5_p50 = []
        h5_p80 = []
        h5_p95 = []
        h10_p50 = []
        h10_p80 = []
        h10_p95 = []
        h20_p50 = []
        h20_p80 = []
        h20_p95 = []

        directory = Path(self.outdir)
        matches = list(directory.glob(f"{prefix}_*__bs_*_results.npy"))

        for path in matches:
            data = load_from_npy(str(path))
            id = path.stem.split("__bs_", 1)[1].removesuffix("_results")

            match id:
                case "1_p100":
                    h1_p100.append(data)
                case "5_p50":
                    h5_p50.append(data)
                case "5_p80":
                    h5_p80.append(data)
                case "5_p95":
                    h5_p95.append(data)
                case "10_p50":
                    h10_p50.append(data)
                case "10_p80":
                    h10_p80.append(data)
                case "10_p95":
                    h10_p95.append(data)
                case "20_p50":
                    h20_p50.append(data)
                case "20_p80":
                    h20_p80.append(data)
                case "20_p95":
                    h20_p95.append(data)
                case _:
                    pass

        # compute means across all seeds
        h1_p100 = compute_means(h1_p100)
        h5_p50 = compute_means(h5_p50)
        h5_p80 = compute_means(h5_p80)
        h5_p95 = compute_means(h5_p95)
        h10_p50 = compute_means(h10_p50)
        h10_p80 = compute_means(h10_p80)
        h10_p95 = compute_means(h10_p95)
        h20_p50 = compute_means(h20_p50)
        h20_p80 = compute_means(h20_p80)
        h20_p95 = compute_means(h20_p95)

        results_5 = []
        colors_5 = []

        results_10 = []
        colors_10 = []

        results_20 = []
        colors_20 = []

        results_p50 = []
        colors_p50 = []

        results_p80 = []
        colors_p80 = []

        results_p95 = []
        colors_p95 = []

        # add baseline to all plots
        results_5.append((h1_p100, "Baseline"))
        colors_5.append("black")
        results_10.append((h1_p100, "Baseline"))
        colors_10.append("black")
        results_20.append((h1_p100, "Baseline"))
        colors_20.append("black")
        results_p50.append((h1_p100, "Baseline"))
        colors_p50.append("black")
        results_p80.append((h1_p100, "Baseline"))
        colors_p80.append("black")
        results_p95.append((h1_p100, "Baseline"))
        colors_p95.append("black")

        # add 5_p50 to 5 and p50 plots
        results_5.append((h5_p50, "5H P50"))
        colors_5.append("green")
        results_p50.append((h5_p50, "5H P50"))
        colors_p50.append("green")

        # add 5_p80 to 5 and p80 plots
        results_5.append((h5_p80, "5H P80"))
        colors_5.append("orange")
        results_p80.append((h5_p80, "5H P80"))
        colors_p80.append("orange")

        # add 5_p95 to 5 and p95 plots
        results_5.append((h5_p95, "5H P95"))
        colors_5.append("purple")
        results_p95.append((h5_p95, "5H P95"))
        colors_p95.append("purple")

        # add 10_p50 to 10 and p50 plots
        results_10.append((h10_p50, "10H P50"))
        colors_10.append("blue")
        results_p50.append((h10_p50, "10H P50"))
        colors_p50.append("blue")

        # add 10_p80 to 10 and p80 plots
        results_10.append((h10_p80, "10H P80"))
        colors_10.append("red")
        results_p80.append((h10_p80, "10H P80"))
        colors_p80.append("red")

        # add 10_p95 to 10 and p95 plots
        results_10.append((h10_p95, "10H P95"))
        colors_10.append("yellow")
        results_p95.append((h10_p95, "10H P95"))
        colors_p95.append("yellow")

        # add 20_p50 to 20 and p50 plots
        results_20.append((h20_p50, "20H P50"))
        colors_20.append("pink")
        results_p50.append((h20_p50, "20H P50"))
        colors_p50.append("pink")

        # add 20_p80 to 20 and p80 plots
        results_20.append((h20_p80, "20H P80"))
        colors_20.append("brown")
        results_p80.append((h20_p80, "20H P80"))
        colors_p80.append("brown")

        # add 20_p95 to 20 and p95 plots
        results_20.append((h20_p95, "20H P95"))
        colors_20.append("violet")
        results_p95.append((h20_p95, "20H P95"))
        colors_p95.append("violet")

        plot_performance(results_5, colors=colors_5, truncate_after=truncate_after, title="5 Bootstrapped Heads")
        plot_cumulative_reward(results_5, colors=colors_5, truncate_after=truncate_after, title="5 Bootstrapped Heads")

        plot_performance(results_10, colors=colors_10, truncate_after=truncate_after, title="10 Bootstrapped Heads")
        plot_cumulative_reward(results_10, colors=colors_10, truncate_after=truncate_after, title="10 Bootstrapped Heads")

        plot_performance(results_20, colors=colors_20, truncate_after=truncate_after, title="20 Bootstrapped Heads")
        plot_cumulative_reward(results_20, colors=colors_20, truncate_after=truncate_after, title="20 Bootstrapped Heads")

        plot_performance(results_p50, colors=colors_p50, truncate_after=truncate_after, title="Bernoulli Mask 0.5")
        plot_cumulative_reward(results_p50, colors=colors_p50, truncate_after=truncate_after, title="Bernoulli Mask 0.5")

        plot_performance(results_p80, colors=colors_p80, truncate_after=truncate_after, title="Bernoulli Mask 0.8")
        plot_cumulative_reward(results_p80, colors=colors_p80, truncate_after=truncate_after, title="Bernoulli Mask 0.8")

        plot_performance(results_p95, colors=colors_p95, truncate_after=truncate_after, title="Bernoulli Mask 0.95")
        plot_cumulative_reward(results_p95, colors=colors_p95, truncate_after=truncate_after, title="Bernoulli Mask 0.95")


def compute_means(data_list: List[List[Dict[str, float]]]) -> List[Dict[str, float]]:
    if not data_list:
        return []
    
    num_steps = len(data_list[0])
    result = []
    
    for step_idx in range(num_steps):
        sums = defaultdict(float)
        counts = defaultdict(int)
        for seed in data_list:
            step_data = seed[step_idx]
            for k, v in step_data.items():
                sums[k] += v
                counts[k] += 1
        means = {k: sums[k] / counts[k] for k in sums}
        result.append(means)
    
    return result