import sys
import torch
from typing import Type

from environments.environment import Environment
from algorithms.actor_critic import ActorCritic
from algorithms.common import new_observer, plot_multiple_benchmarks
from algorithms.vanilla_actor_critic import VanillaActorCritic
from algorithms.hybrid_hmc import HybridHMC

def bench_hmc(environment_cls: Type[Environment], actor_critic_cls: Type[ActorCritic], device: torch.device = torch.device("cpu")):
    random_ac = VanillaActorCritic(environment_cls(device=device), actor_critic_cls(environment_cls(device=device), device=device))
    random_ac_observer, random_ac_results = new_observer(environment_cls(device=device))
    random_ac.train(start_steps=sys.maxsize, observer=random_ac_observer)

    vanilla_ac = VanillaActorCritic(environment_cls(device=device), actor_critic_cls(environment_cls(device=device), device=device))
    vanilla_ac_observer, vanilla_ac_results = new_observer(environment_cls(device=device))
    vanilla_ac.train(observer=vanilla_ac_observer)

    hybrid_hmc = HybridHMC(environment_cls(device=device), actor_critic_cls(environment_cls(device=device), device=device), device=device)
    hybrid_hmc_observer, hybrid_hmc_results = new_observer(environment_cls(device=device))
    hybrid_hmc.train(observer=hybrid_hmc_observer)

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