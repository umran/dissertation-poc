import torch
from typing import Type, Dict, Optional, Any

from util import plot_multiple_benchmarks
from environments.environment import Environment
from algorithms.actor_critic import ActorCritic
from algorithms.common import new_observer, new_sample_observer
from algorithms.vanilla_actor_critic import VanillaActorCritic
from algorithms.hybrid_hmc import HybridHMC

def bench_hmc(
    environment_cls: Type[Environment],
    actor_critic_cls: Type[ActorCritic],
    actor_critic_args: Optional[Dict[str, Any]] = None,
    hybrid_hmc_args: Optional[Dict[str, Any]] = None,
    steps: int = 200_000,
    device: torch.device = torch.device("cpu")
):
    actor_critic_args = actor_critic_args or {}
    hybrid_hmc_args = hybrid_hmc_args or {}

    random_ac = VanillaActorCritic(environment_cls(device=device), actor_critic_cls(environment_cls(device=device), device=device))
    random_ac_observer, random_ac_results = new_observer(environment_cls(device=device))
    random_ac.train(steps=steps, start_steps=steps, observer=random_ac_observer, **actor_critic_args)

    vanilla_ac = VanillaActorCritic(environment_cls(device=device), actor_critic_cls(environment_cls(device=device), device=device))
    vanilla_ac_observer, vanilla_ac_results = new_observer(environment_cls(device=device))
    vanilla_ac.train(steps=steps, observer=vanilla_ac_observer, **actor_critic_args)

    hybrid_hmc = HybridHMC(environment_cls(device=device), actor_critic_cls(environment_cls(device=device), device=device), device=device)
    hybrid_hmc_observer, hybrid_hmc_results = new_observer(environment_cls(device=device))
    sample_observer, posterior_samples = new_sample_observer()
    hybrid_hmc.train(steps=steps, observer=hybrid_hmc_observer, sample_observer=sample_observer, **hybrid_hmc_args)

    plot_multiple_benchmarks(
        [
            (random_ac_results, "Uniform Random Exploration"),
            (vanilla_ac_results, "Vanilla Actor Critic"),
            (hybrid_hmc_results, "HMC Actor Critic")
        ],
        colors=["red", "blue", "green"]
    )

    return [
        random_ac.actor_critic.get_optimal_policy(),
        vanilla_ac.actor_critic.get_optimal_policy(),
        hybrid_hmc.actor_critic.get_optimal_policy()
    ]