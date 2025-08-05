from algorithms.common import new_observer, plot_multiple_benchmarks
from algorithms.vanilla_actor_critic import VanillaActorCritic
from algorithms.td3 import TD3
from algorithms.hybrid_hmc import HybridHMC
from environments.pendulum import Pendulum

def bench_pendulum_td3():
    random_ddpg = VanillaActorCritic(Pendulum(), TD3(Pendulum()))
    random_ddpg_observer, random_ddpg_results = new_observer(Pendulum())
    random_ddpg.train(start_steps=200_000, observer=random_ddpg_observer)

    vanilla_ddpg = VanillaActorCritic(Pendulum(), TD3(Pendulum()))
    vanilla_ddpg_observer, vanilla_ddpg_results = new_observer(Pendulum())
    vanilla_ddpg.train(observer=vanilla_ddpg_observer)

    hybrid_hmc = HybridHMC(Pendulum(), TD3(Pendulum()))
    hybrid_hmc_observer, hybrid_hmc_results = new_observer(Pendulum())
    hybrid_hmc.train(observer=hybrid_hmc_observer)

    plot_multiple_benchmarks(
        [
            (random_ddpg_results, "TD3 Random Exploration"),
            (vanilla_ddpg_results, "Vanilla TD3"),
            (hybrid_hmc_results, "Hybrid HMC")
        ],
        colors=["red", "blue", "green"]
    )

    return [
        random_ddpg.actor_critic.get_optimal_policy(),
        vanilla_ddpg.actor_critic.get_optimal_policy(),
        hybrid_hmc.actor_critic.get_optimal_policy()
    ]