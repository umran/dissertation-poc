import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict, Callable

from algorithms.policy import Policy
from environments.environment import Environment

class ReplayBuffer:
    def __init__(self, 
        capacity: int,
        state_shape: Tuple[int, ...], 
        action_shape: Tuple[int, ...], 
        state_dtype: torch.dtype = torch.float32, 
        action_dtype: torch.dtype = torch.float32, 
        device: torch.device = torch.device("cpu")
    ):
        self.capacity = capacity
        self.device = device
        self.state_dtype = state_dtype
        self.action_dtype = action_dtype
        self.ptr = 0
        self.size = 0

        self.states = torch.zeros((capacity, *state_shape), dtype=state_dtype, device=device)
        self.actions = torch.zeros((capacity, *action_shape), dtype=action_dtype, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, *state_shape), dtype=state_dtype, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.bool, device=device)

    def add(self, state, action, reward, next_state, done):
        self.add_batch(
            torch.tensor(state, dtype=self.state_dtype, device=self.device).unsqueeze(0),
            torch.tensor(action, dtype=self.action_dtype, device=self.device).unsqueeze(0),
            torch.tensor(reward, dtype=torch.float32, device=self.device).view(1, 1),
            torch.tensor(next_state, dtype=self.state_dtype, device=self.device).unsqueeze(0),
            torch.tensor(done, dtype=torch.bool, device=self.device).view(1, 1)
        )

    def add_batch(self, states, actions, rewards, next_states, dones):
        batch_size = states.shape[0]
        assert states.shape == (batch_size, *self.states.shape[1:])
        assert actions.shape == (batch_size, *self.actions.shape[1:])
        assert rewards.shape == (batch_size, 1)
        assert next_states.shape == (batch_size, *self.next_states.shape[1:])
        assert dones.shape == (batch_size, 1)

        end = self.ptr + batch_size
        if end <= self.capacity:
            self.states[self.ptr:end] = states
            self.actions[self.ptr:end] = actions
            self.rewards[self.ptr:end] = rewards
            self.next_states[self.ptr:end] = next_states
            self.dones[self.ptr:end] = dones
        else:
            first = self.capacity - self.ptr
            second = batch_size - first
            self.states[self.ptr:] = states[:first]
            self.actions[self.ptr:] = actions[:first]
            self.rewards[self.ptr:] = rewards[:first]
            self.next_states[self.ptr:] = next_states[:first]
            self.dones[self.ptr:] = dones[:first]

            self.states[:second] = states[first:]
            self.actions[:second] = actions[first:]
            self.rewards[:second] = rewards[first:]
            self.next_states[:second] = next_states[first:]
            self.dones[:second] = dones[first:]

        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx]
        )

    def __len__(self):
        return self.size


class EpisodicReplayBuffer:
    def __init__(
        self, 
        capacity: int, 
        state_shape: Tuple[int, ...], 
        action_shape: Tuple[int, ...], 
        state_dtype: torch.dtype = torch.float32, 
        action_dtype: torch.dtype = torch.float32, 
        device: torch.device = torch.device("cpu")
    ):
        self.capacity = capacity
        self.device = device
        self.state_dtype = state_dtype
        self.action_dtype = action_dtype
        self.ptr = 0
        self.size = 0

        self.states = torch.zeros((capacity, *state_shape), dtype=state_dtype, device=device)
        self.actions = torch.zeros((capacity, *action_shape), dtype=action_dtype, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.mc_returns = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, *state_shape), dtype=state_dtype, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.bool, device=device)

    def add(self, state, action, reward, mc_return, next_state, done):
        self.add_batch(
            torch.tensor(state, dtype=self.state_dtype, device=self.device).unsqueeze(0),
            torch.tensor(action, dtype=self.action_dtype, device=self.device).unsqueeze(0),
            torch.tensor(reward, dtype=torch.float32, device=self.device).view(1, 1),
            torch.tensor(mc_return, dtype=torch.float32, device=self.device).view(1, 1),
            torch.tensor(next_state, dtype=self.state_dtype, device=self.device).unsqueeze(0),
            torch.tensor(done, dtype=torch.bool, device=self.device).view(1, 1)
        )

    def add_batch(self, states, actions, rewards, mc_returns, next_states, dones):
        batch_size = states.shape[0]
        assert states.shape == (batch_size, *self.states.shape[1:])
        assert actions.shape == (batch_size, *self.actions.shape[1:])
        assert rewards.shape == (batch_size, 1)
        assert mc_returns.shape == (batch_size, 1)
        assert next_states.shape == (batch_size, *self.next_states.shape[1:])
        assert dones.shape == (batch_size, 1)

        end = self.ptr + batch_size
        if end <= self.capacity:
            self.states[self.ptr:end] = states
            self.actions[self.ptr:end] = actions
            self.rewards[self.ptr:end] = rewards
            self.mc_returns[self.ptr:end] = mc_returns
            self.next_states[self.ptr:end] = next_states
            self.dones[self.ptr:end] = dones
        else:
            first = self.capacity - self.ptr
            second = batch_size - first
            self.states[self.ptr:] = states[:first]
            self.actions[self.ptr:] = actions[:first]
            self.rewards[self.ptr:] = rewards[:first]
            self.mc_returns[self.ptr:] = mc_returns[:first]
            self.next_states[self.ptr:] = next_states[:first]
            self.dones[self.ptr:] = dones[:first]

            self.states[:second] = states[first:]
            self.actions[:second] = actions[first:]
            self.rewards[:second] = rewards[first:]
            self.mc_returns[:second] = mc_returns[first:]
            self.next_states[:second] = next_states[first:]
            self.dones[:second] = dones[first:]

        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.mc_returns[idx],
            self.next_states[idx],
            self.dones[idx]
        )

    def __len__(self):
        return self.size

ObserverType = Callable[
    [int, Policy],
    None
]

def new_observer(bench_env: Environment, bench_every = 10_000, bench_episodes = 10):
    bench_results = []

    def observer(
        step: int,
        policy: Policy
    ):
        if step % bench_every == 0:
            results = run_bench(bench_env, policy, bench_episodes)
            results["step"] = step
            bench_results.append(results)

    return observer, bench_results

def run_bench(env: Environment, policy: Policy, num_episodes: int) -> Dict[str, float]:
    episode_rewards = np.zeros(num_episodes)
    
    for i in range(num_episodes):
        state = env.reset()

        while True:
            action = policy.action(state)
            state, reward, terminated, truncated, _ = env.step(action)
        
            episode_rewards[i] += reward

            if terminated or truncated:
                break

    return {
        "mean": episode_rewards.mean(),
        "sd": episode_rewards.std(),
        "min": episode_rewards.min(),
        "max": episode_rewards.max()
    }

def plot_benchmark(results: List[Dict[str, float]], *, label: str = "Policy", color: str = "blue"):
    steps = np.array([r["step"] for r in results])
    means = np.array([r["mean"] for r in results])
    sds = np.array([r["sd"] for r in results])
    ci = 1.96 * sds  # 95% confidence interval

    plt.figure(figsize=(8, 5))
    plt.plot(steps, means, label=label, color=color)
    plt.fill_between(steps, means - ci, means + ci, alpha=0.3, color=color, label="95% CI")

    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.title("Performance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_multiple_benchmarks(
    benchmark_sets: List[Tuple[List[Dict[str, float]], str]],  # (results, label)
    colors: List[str] = None
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
    plt.ylabel("Cumulative Reward")
    plt.title("Performance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def sample_gaussian(
    mean: float,
    std: float,
    size: Tuple[int, ...],
    clip: Optional[Tuple[float, float]]=None,
    device: torch.device = torch.device("cpu")
):
    sample = torch.normal(mean=mean, std=std, size=size, device=device)

    if clip is None:
        return sample
    
    return torch.clamp(sample, min=clip[0], max=clip[1])

# copies params from a source to a target network
def copy_params(target_net: nn.Module, source_net: nn.Module):
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(source_param.data)

# performs an update of the target network parameters via Polyak averaging
# where target_params = p * target_params + (1 - p) * source_params
def polyak_update(target_net: nn.Module, source_net: nn.Module, p: float):
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(p * target_param.data + (1 - p) * source_param.data)