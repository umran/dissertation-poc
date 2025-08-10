import torch
from typing import Type

from util import save_dict_to_npy
from algorithms.common import new_observer, new_sample_observer
from algorithms.hybrid_hmc import HybridHMC
from algorithms.actor_critic import ActorCritic
from algorithms.vanilla_actor_critic import VanillaActorCritic
from algorithms.ddpg import DDPG
from algorithms.td3 import TD3
from environments.environment import Environment
from environments.pendulum import Pendulum
from environments.inverted_pendulum import InvertedPendulum
from environments.inverted_double_pendulum import InvertedDoublePendulum

OUTDIR = "./out"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")

# HMC_AC Configurations
HMC_AC_100_5K = {
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

HMC_AC_1000_5K = {
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

HMC_AC_100_50K = {
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

HMC_AC_1000_50K = {
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

def run_all(steps: int):
    ddpg_pendulum(steps)
    ddpg_inverted_pendulum(steps)
    ddpg_inverted_double_pendulum(steps)

    td3_pendulum(steps)
    td3_inverted_pendulum(steps)
    td3_inverted_double_pendulum(steps)

def run_test():
    ddpg_pendulum(20_000)
    td3_pendulum(20_000)

def ddpg_pendulum(steps: int):
    comparative = comparative_study(Pendulum, DDPG, steps)
    save_dict_to_npy(comparative, f"{OUTDIR}/ddpg_pendulum_comparative.npy")
    
    ablation = ablation_study(Pendulum, DDPG, steps)
    save_dict_to_npy(ablation, f"{OUTDIR}/ddpg_pendulum_ablation.npy")

def ddpg_inverted_pendulum(steps: int):
    comparative = comparative_study(InvertedPendulum, DDPG, steps)
    save_dict_to_npy(comparative, f"{OUTDIR}/ddpg_inverted_pendulum_comparative.npy")
    
    ablation = ablation_study(InvertedPendulum, DDPG, steps)
    save_dict_to_npy(ablation, f"{OUTDIR}/ddpg_inverted_pendulum_ablation.npy")

def ddpg_inverted_double_pendulum(steps: int):
    comparative = comparative_study(InvertedDoublePendulum, DDPG, steps)
    save_dict_to_npy(comparative, f"{OUTDIR}/ddpg_inverted_double_pendulum_comparative.npy")
    
    ablation = ablation_study(InvertedDoublePendulum, DDPG, steps)
    save_dict_to_npy(ablation, f"{OUTDIR}/ddpg_inverted_double_pendulum_ablation.npy")

def td3_pendulum(steps: int):
    comparative = comparative_study(Pendulum, TD3, steps)
    save_dict_to_npy(comparative, f"{OUTDIR}/td3_pendulum_comparative.npy")
    
    ablation = ablation_study(Pendulum, TD3, steps)
    save_dict_to_npy(ablation, f"{OUTDIR}/td3_pendulum_ablation.npy")

def td3_inverted_pendulum(steps: int):
    comparative = comparative_study(InvertedPendulum, TD3, steps)
    save_dict_to_npy(comparative, f"{OUTDIR}/td3_inverted_pendulum_comparative.npy")
    
    ablation = ablation_study(InvertedPendulum, TD3, steps)
    save_dict_to_npy(ablation, f"{OUTDIR}/td3_inverted_pendulum_ablation.npy")

def td3_inverted_double_pendulum(steps: int):
    comparative = comparative_study(InvertedDoublePendulum, TD3, steps)
    save_dict_to_npy(comparative, f"{OUTDIR}/td3_inverted_double_pendulum_comparative.npy")
    
    ablation = ablation_study(InvertedDoublePendulum, TD3, steps)
    save_dict_to_npy(ablation, f"{OUTDIR}/td3_inverted_double_pendulum_ablation.npy")

def comparative_study(environment_cls: Type[Environment], actor_critic_cls: Type[ActorCritic], steps: int):
    random, random_observer, random_results = prepare_actor_critic(environment_cls, actor_critic_cls)
    vanilla, vanilla_observer, vanilla_results = prepare_actor_critic(environment_cls, actor_critic_cls)

    hmc_ac, hmc_ac_observer, hmc_ac_results, sample_observer, posterior_samples = prepare_hmc_actor_critic(environment_cls, actor_critic_cls)
    
    random.train(steps=steps, start_steps=steps, observer=random_observer)
    vanilla.train(steps=steps, observer=vanilla_observer)
    hmc_ac.train(steps=steps, observer=hmc_ac_observer, sample_observer=sample_observer, **HMC_AC_1000_5K)

    return {
        "random_results": random_results,
        "vanilla_results": vanilla_results,
        "hmc_ac_results": hmc_ac_results,
        "posterior_samples": posterior_samples,
    }

def ablation_study(environment_cls: Type[Environment], actor_critic_cls: Type[ActorCritic], steps: int):
    hmc_ac_100_50k, hmc_ac_100_50k_observer, hmc_ac_100_50k_results, sample_observer_100_50k, posterior_samples_100_50k = prepare_hmc_actor_critic(environment_cls, actor_critic_cls)
    hmc_ac_1000_50k, hmc_ac_1000_50k_observer, hmc_ac_1000_50k_results, sample_observer_1000_50k, posterior_samples_1000_50k = prepare_hmc_actor_critic(environment_cls, actor_critic_cls)
    hmc_ac_100_5k, hmc_ac_100_5k_observer, hmc_ac_100_5k_results, sample_observer_100_5k, posterior_samples_100_5k = prepare_hmc_actor_critic(environment_cls, actor_critic_cls)
    hmc_ac_1000_5k, hmc_ac_1000_5k_observer, hmc_ac_1000_5k_results, sample_observer_1000_5k, posterior_samples_1000_5k = prepare_hmc_actor_critic(environment_cls, actor_critic_cls)

    hmc_ac_100_50k.train(steps=steps, observer=hmc_ac_100_50k_observer, sample_observer=sample_observer_100_50k, **HMC_AC_100_50K)
    hmc_ac_1000_50k.train(steps=steps, observer=hmc_ac_1000_50k_observer, sample_observer=sample_observer_1000_50k, **HMC_AC_1000_50K)
    hmc_ac_100_5k.train(steps=steps, observer=hmc_ac_100_5k_observer, sample_observer=sample_observer_100_5k, **HMC_AC_100_5K)
    hmc_ac_1000_5k.train(steps=steps, observer=hmc_ac_1000_5k_observer, sample_observer=sample_observer_1000_5k, **HMC_AC_1000_5K)

    return {
        "hmc_ac_100_50k_results": hmc_ac_100_50k_results,
        "hmc_ac_1000_50k_results": hmc_ac_1000_50k_results,
        "hmc_ac_100_5k_results": hmc_ac_100_5k_results,
        "hmc_ac_1000_5k_results": hmc_ac_1000_5k_results,
        "posterior_samples_100_50k": posterior_samples_100_50k,
        "posterior_samples_1000_50k": posterior_samples_1000_50k,
        "posterior_samples_100_5k": posterior_samples_100_5k,
        "posterior_samples_1000_5k": posterior_samples_1000_5k,
    }

def prepare_actor_critic(environment_cls: Type[Environment], actor_critic_cls: Type[ActorCritic], device: torch.device = DEVICE):
    ac = VanillaActorCritic(environment_cls(device=device), actor_critic_cls(environment_cls(device=device), device=device), device=device)
    observer, results = new_observer(environment_cls(device=device))

    return ac, observer, results

def prepare_hmc_actor_critic(environment_cls: Type[Environment], actor_critic_cls: Type[ActorCritic], device: torch.device = DEVICE):
    hmc_ac = HybridHMC(environment_cls(device=device), actor_critic_cls(environment_cls(device=device), device=device), device=device)
    observer, results = new_observer(environment_cls(device=device))
    sample_observer, posterior_samples = new_sample_observer()

    return hmc_ac, observer, results, sample_observer, posterior_samples