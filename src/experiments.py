import os
from pathlib import Path
import torch
from typing import Type, Optional

from util import save_to_npy, load_from_npy, plot_multiple_benchmarks, plot_multiple_variances
from algorithms.common import new_observer, new_sample_observer
from algorithms.hybrid_hmc import HybridHMC
from algorithms.actor_critic import ActorCritic
from algorithms.vanilla_actor_critic import VanillaActorCritic
from environments.environment import Environment

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

HMC_AC_SMALL_1000_5K = {
    "batch_size": 128,
    "num_warmup": 500,
    "num_samples": 1000,
    "update_every": 5000,
    "exploration_policy_lr": 1e-2,
    "exploration_policy_optimization_steps": 1000,
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

HMC_AC_SMALL_1000_50K = {
    "batch_size": 128,
    "num_warmup": 500,
    "num_samples": 1000,
    "update_every": 50_000,
    "exploration_policy_lr": 1e-2,
    "exploration_policy_optimization_steps": 1000,
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

class Experiments:
    def __init__(self, outdir: str, device: torch.device = torch.device("cpu")):
        self.outdir = outdir
        self.device = device

    def baselines(self, prefix: str, environment_cls: Type[Environment], actor_critic_cls: Type[ActorCritic], steps: int):
        vanilla_results_url = self.make_url(prefix, "vanilla_results")
        random_results_url = self.make_url(prefix, "random_results")
        
        if not os.path.isfile(vanilla_results_url):
            vanilla, vanilla_observer, vanilla_results = self.prepare_actor_critic(environment_cls, actor_critic_cls)
            vanilla.train(steps=steps, observer=vanilla_observer)
            
            save_to_npy(vanilla_results, vanilla_results_url)

        if not os.path.isfile(random_results_url):
            random, random_observer, random_results = self.prepare_actor_critic(environment_cls, actor_critic_cls)
            random.train(steps=steps, start_steps=steps, observer=random_observer)
            
            save_to_npy(random_results, random_results_url)

    def ablation(self, prefix: str, environment_cls: Type[Environment], actor_critic_cls: Type[ActorCritic], steps: int):
        large_100_5k_results_url = self.make_url(prefix, "large_100_5k_results")
        small_100_5k_results_url = self.make_url(prefix, "small_100_5k_results")
        large_100_50k_results_url = self.make_url(prefix, "large_100_50k_results")
        small_100_50k_results_url = self.make_url(prefix, "small_100_50k_results")

        if not os.path.isfile(large_100_5k_results_url):  
            large_100_5k, large_100_5k_observer, large_100_5k_results, large_100_5k_sample_observer, large_100_5k_posterior_samples = self.prepare_hmc_actor_critic(environment_cls, actor_critic_cls)      
            large_100_5k.train(steps=steps, observer=large_100_5k_observer, sample_observer=large_100_5k_sample_observer, **HMC_AC_LARGE_100_5K)

            large_100_5k_posterior_samples_url = self.make_url(prefix, "large_100_5k_posterior_samples")

            save_to_npy(large_100_5k_results, large_100_5k_results_url)
            save_to_npy(large_100_5k_posterior_samples, large_100_5k_posterior_samples_url)
        
        if not os.path.isfile(small_100_5k_results_url):
            small_100_5k, small_100_5k_observer, small_100_5k_results, small_100_5k_sample_observer, small_100_5k_posterior_samples = self.prepare_hmc_actor_critic(environment_cls, actor_critic_cls)
            small_100_5k.train(steps=steps, observer=small_100_5k_observer, sample_observer=small_100_5k_sample_observer, **HMC_AC_SMALL_100_5K)
        
            small_100_5k_posterior_samples_url = self.make_url(prefix, "small_100_5k_posterior_samples")

            save_to_npy(small_100_5k_results, small_100_5k_results_url)
            save_to_npy(small_100_5k_posterior_samples, small_100_5k_posterior_samples_url)

        if not os.path.isfile(large_100_50k_results_url):
            large_100_50k, large_100_50k_observer, large_100_50k_results, large_100_50k_sample_observer, large_100_50k_posterior_samples = self.prepare_hmc_actor_critic(environment_cls, actor_critic_cls)
            large_100_50k.train(steps=steps, observer=large_100_50k_observer, sample_observer=large_100_50k_sample_observer, **HMC_AC_LARGE_100_50K)
        
            large_100_50k_posterior_samples_url = self.make_url(prefix, "large_100_50k_posterior_samples")

            save_to_npy(large_100_50k_results, large_100_50k_results_url)
            save_to_npy(large_100_50k_posterior_samples, large_100_50k_posterior_samples_url)

        if not os.path.isfile(small_100_50k_results_url):
            small_100_50k, small_100_50k_observer, small_100_50k_results, small_100_50k_sample_observer, small_100_50k_posterior_samples = self.prepare_hmc_actor_critic(environment_cls, actor_critic_cls)
            small_100_50k.train(steps=steps, observer=small_100_50k_observer, sample_observer=small_100_50k_sample_observer, **HMC_AC_SMALL_100_50K)

            small_100_50k_posterior_samples_url = self.make_url(prefix, "small_100_50k_posterior_samples")

            save_to_npy(small_100_50k_results, small_100_50k_results_url)
            save_to_npy(small_100_50k_posterior_samples, small_100_50k_posterior_samples_url)

    def prepare_actor_critic(self, environment_cls: Type[Environment], actor_critic_cls: Type[ActorCritic]):
        ac = VanillaActorCritic(environment_cls(device=self.device), actor_critic_cls(environment_cls(device=self.device), device=self.device), device=self.device)
        observer, results = new_observer(environment_cls(device=self.device))

        return ac, observer, results

    def prepare_hmc_actor_critic(self, environment_cls: Type[Environment], actor_critic_cls: Type[ActorCritic]):
        hmc_ac = HybridHMC(environment_cls(device=self.device), actor_critic_cls(environment_cls(device=self.device), device=self.device), device=self.device)
        observer, results = new_observer(environment_cls(device=self.device))
        sample_observer, posterior_samples = new_sample_observer()

        return hmc_ac, observer, results, sample_observer, posterior_samples
    
    def make_url(self, prefix: str, name: str):
        return f"{self.outdir}/{prefix}__{name}.npy"
    
    def plot_comparison(self, prefix: str, truncate_after: Optional[int] = None):
        results = []
        colors = []

        directory = Path(self.outdir)
        matches = list(directory.glob(f"{prefix}__*_results.npy"))

        for path in matches:
            data = load_from_npy(str(path))
            id = path.stem.split("__", 1)[1].removesuffix("_results")

            match id:
                case "random":
                    results.append((data, "Uniform Random"))
                    colors.append("red")
                case "vanilla":
                    results.append((data, "Vanilla Actor Critic"))
                    colors.append("blue")
                case "large_100_5k":
                    results.append((data, "HMC Posterior Sampling"))
                    colors.append("green")
                case _:
                    pass
                
        plot_multiple_benchmarks(results, colors=colors, truncate_after=truncate_after)
        plot_multiple_variances(results, colors=colors, truncate_after=truncate_after)
    
    def plot_ablation(self, prefix: str, truncate_after: Optional[int] = None):
        results_large = []
        colors_large = []

        results_small = []
        colors_small = []

        results_50k = []
        colors_50k = []

        results_5k = []
        colors_5k = []

        directory = Path(self.outdir)
        matches = list(directory.glob(f"{prefix}__*_results.npy"))

        for path in matches:
            data = load_from_npy(str(path))
            id = path.stem.split("__", 1)[1].removesuffix("_results")

            match id:
                case "large_100_5k":
                    results_large.append((data, "HMC AC Large 5K"))
                    colors_large.append("green")

                    results_5k.append((data, "HMC AC Large 5K"))
                    colors_5k.append("green")
                case "small_100_5k":
                    results_small.append((data, "HMC AC Small 5K"))
                    colors_small.append("orange")

                    results_5k.append((data, "HMC AC Small 5K"))
                    colors_5k.append("orange")
                case "large_100_50k":
                    results_large.append((data, "HMC AC Large 50K"))
                    colors_large.append("purple")

                    results_50k.append((data, "HMC AC Large 50K"))
                    colors_50k.append("purple")
                case "small_100_50k":
                    results_small.append((data, "HMC AC Small 50K"))
                    colors_small.append("brown")

                    results_50k.append((data, "HMC AC Small 50K"))
                    colors_50k.append("brown")
                case _:
                    pass

        # plot_multiple_benchmarks(results, colors=colors, truncate_after=truncate_after)
        plot_multiple_benchmarks(results_large, colors=colors_large, truncate_after=truncate_after, title="1024 Observations Per Update")
        plot_multiple_benchmarks(results_small, colors=colors_small, truncate_after=truncate_after, title="128 Observations Per Update")
        plot_multiple_benchmarks(results_5k, colors=colors_5k, truncate_after=truncate_after, title="Updates Every 5K Steps")
        plot_multiple_benchmarks(results_50k, colors=colors_50k, truncate_after=truncate_after, title="Updates Every 50K Steps")