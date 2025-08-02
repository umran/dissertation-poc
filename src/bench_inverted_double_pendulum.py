from algorithms.common import new_observer, plot_multiple_benchmarks
from algorithms.vanilla_actor_critic import VanillaActorCritic
from algorithms.ddpg import DDPG
from algorithms.hybrid_hmc import HybridHMC
from environments.inverted_double_pendulum import InvertedDoublePendulum

def bench_inverted_double_pendulum():
    random_ddpg = VanillaActorCritic(InvertedDoublePendulum(), DDPG(InvertedDoublePendulum()))
    random_ddpg_observer, random_ddpg_results = new_observer(InvertedDoublePendulum())
    random_ddpg.train(start_steps=200_000, observer=random_ddpg_observer)

    vanilla_ddpg = VanillaActorCritic(InvertedDoublePendulum(), DDPG(InvertedDoublePendulum()))
    vanilla_ddpg_observer, vanilla_ddpg_results = new_observer(InvertedDoublePendulum())
    vanilla_ddpg.train(observer=vanilla_ddpg_observer)

    hybrid_hmc = HybridHMC(InvertedDoublePendulum(), DDPG(InvertedDoublePendulum()))
    hybrid_hmc_observer, hybrid_hmc_results = new_observer(InvertedDoublePendulum())
    hybrid_hmc.train(update_every=10_000, observer=hybrid_hmc_observer)

    plot_multiple_benchmarks(
        [
            (random_ddpg_results, "DDPG Random Exploration"),
            (vanilla_ddpg_results, "Vanilla DDPG"),
            (hybrid_hmc_results, "Hybrid HMC")
        ],
        colors=["red", "blue", "green"]
    )

    return [
        random_ddpg.actor_critic.get_optimal_policy(),
        vanilla_ddpg.actor_critic.get_optimal_policy(),
        hybrid_hmc.actor_critic.get_optimal_policy()
    ]