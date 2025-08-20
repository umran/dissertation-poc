import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Callable, Any

from bac.algorithms.policy import Policy
from bac.algorithms.networks import QNetwork
from bac.environments import Environment

class MaskedReplayBuffer:
    def __init__(self, 
        capacity: int,
        n_heads: int,
        p: float,
        state_shape: Tuple[int, ...], 
        action_shape: Tuple[int, ...], 
        state_dtype: torch.dtype = torch.float32, 
        action_dtype: torch.dtype = torch.float32, 
        device: torch.device = torch.device("cpu")
    ):
        self.capacity = capacity
        self.n_heads = n_heads
        self.p = p
        self.device = device
        self.state_dtype = state_dtype
        self.action_dtype = action_dtype
        self.ptr = 0
        self.size = 0

        self.states = torch.zeros((capacity, *state_shape), dtype=state_dtype, device=device)
        self.actions = torch.zeros((capacity, *action_shape), dtype=action_dtype, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, *state_shape), dtype=state_dtype, device=device)
        self.terms = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.masks = torch.zeros((capacity, n_heads), dtype=torch.float32, device=device)

    def add(self, state, action, reward, next_state, term):
        self.add_batch(
            state.unsqueeze(0),
            action.unsqueeze(0),
            reward.view(1, 1),
            next_state.unsqueeze(0),
            term.view(1, 1)
        )

    def add_batch(self, states, actions, rewards, next_states, terms):
        batch_size = states.shape[0]
        assert states.shape == (batch_size, *self.states.shape[1:])
        assert actions.shape == (batch_size, *self.actions.shape[1:])
        assert rewards.shape == (batch_size, 1)
        assert next_states.shape == (batch_size, *self.next_states.shape[1:])
        assert terms.shape == (batch_size, 1)

        if self.n_heads == 1:
            masks = torch.ones((batch_size, 1), dtype=torch.float32, device=self.device)
        else:
            masks = torch.bernoulli(
                torch.full((batch_size, self.n_heads), self.p, dtype=torch.float32, device=self.device)
            )

        end = self.ptr + batch_size
        if end <= self.capacity:
            self.states[self.ptr:end] = states
            self.actions[self.ptr:end] = actions
            self.rewards[self.ptr:end] = rewards
            self.next_states[self.ptr:end] = next_states
            self.terms[self.ptr:end] = terms
            self.masks[self.ptr:end] = masks
        else:
            first = self.capacity - self.ptr
            second = batch_size - first
            self.states[self.ptr:] = states[:first]
            self.actions[self.ptr:] = actions[:first]
            self.rewards[self.ptr:] = rewards[:first]
            self.next_states[self.ptr:] = next_states[:first]
            self.terms[self.ptr:] = terms[:first]
            self.masks[self.ptr:] = masks[:first]

            self.states[:second] = states[first:]
            self.actions[:second] = actions[first:]
            self.rewards[:second] = rewards[first:]
            self.next_states[:second] = next_states[first:]
            self.terms[:second] = terms[first:]
            self.masks[:second] = masks[first:]

        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.terms[idx],
            self.masks[idx]
        )

    def __len__(self):
        return self.size

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
        self.terms = torch.zeros((capacity, 1), dtype=torch.float32, device=device)

    def add(self, state, action, reward, next_state, term):
        self.add_batch(
            state.unsqueeze(0),
            action.unsqueeze(0),
            reward.view(1, 1),
            next_state.unsqueeze(0),
            term.view(1, 1)
        )

    def add_batch(self, states, actions, rewards, next_states, terms):
        batch_size = states.shape[0]
        assert states.shape == (batch_size, *self.states.shape[1:])
        assert actions.shape == (batch_size, *self.actions.shape[1:])
        assert rewards.shape == (batch_size, 1)
        assert next_states.shape == (batch_size, *self.next_states.shape[1:])
        assert terms.shape == (batch_size, 1)

        end = self.ptr + batch_size
        if end <= self.capacity:
            self.states[self.ptr:end] = states
            self.actions[self.ptr:end] = actions
            self.rewards[self.ptr:end] = rewards
            self.next_states[self.ptr:end] = next_states
            self.terms[self.ptr:end] = terms
        else:
            first = self.capacity - self.ptr
            second = batch_size - first
            self.states[self.ptr:] = states[:first]
            self.actions[self.ptr:] = actions[:first]
            self.rewards[self.ptr:] = rewards[:first]
            self.next_states[self.ptr:] = next_states[:first]
            self.terms[self.ptr:] = terms[:first]

            self.states[:second] = states[first:]
            self.actions[:second] = actions[first:]
            self.rewards[:second] = rewards[first:]
            self.next_states[:second] = next_states[first:]
            self.terms[:second] = terms[first:]

        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.terms[idx]
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
        self.terms = torch.zeros((capacity, 1), dtype=torch.float32, device=device)

    def add(self, state, action, reward, mc_return, next_state, term):
        self.add_batch(
            state.unsqueeze(0),
            action.unsqueeze(0),
            reward.view(1, 1),
            mc_return.view(1, 1),
            next_state.unsqueeze(0),
            term.view(1, 1),
        )

    def add_batch(self, states, actions, rewards, mc_returns, next_states, terms):
        batch_size = states.shape[0]
        assert states.shape == (batch_size, *self.states.shape[1:])
        assert actions.shape == (batch_size, *self.actions.shape[1:])
        assert rewards.shape == (batch_size, 1)
        assert mc_returns.shape == (batch_size, 1)
        assert next_states.shape == (batch_size, *self.next_states.shape[1:])
        assert terms.shape == (batch_size, 1)

        end = self.ptr + batch_size
        if end <= self.capacity:
            self.states[self.ptr:end] = states
            self.actions[self.ptr:end] = actions
            self.rewards[self.ptr:end] = rewards
            self.mc_returns[self.ptr:end] = mc_returns
            self.next_states[self.ptr:end] = next_states
            self.terms[self.ptr:end] = terms
        else:
            first = self.capacity - self.ptr
            second = batch_size - first
            self.states[self.ptr:] = states[:first]
            self.actions[self.ptr:] = actions[:first]
            self.rewards[self.ptr:] = rewards[:first]
            self.mc_returns[self.ptr:] = mc_returns[:first]
            self.next_states[self.ptr:] = next_states[:first]
            self.terms[self.ptr:] = terms[:first]

            self.states[:second] = states[first:]
            self.actions[:second] = actions[first:]
            self.rewards[:second] = rewards[first:]
            self.mc_returns[:second] = mc_returns[first:]
            self.next_states[:second] = next_states[first:]
            self.terms[:second] = terms[first:]

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
            self.terms[idx]
        )
    
    def new_per_sampler(self, q_net: QNetwork):
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        # compute disagreement scores
        delta = compute_disagreement(q_net, self.states[:self.size], self.actions[:self.size], self.mc_returns[:self.size])

        # sort transitions by error in descending order
        sorted_indices = torch.argsort(delta, descending=True)

        def sample_per(batch_size: int):
            # divide into k segments of equal probability mass
            segment_size = self.size // batch_size
            sample_indices = []

            for i in range(batch_size):
                start = i * segment_size
                end = (i + 1) * segment_size if i < batch_size - 1 else self.size
                if end > start:
                    segment = sorted_indices[start:end]
                    rand_idx = torch.randint(len(segment), (1,)).item()
                    sample_indices.append(segment[rand_idx])

            idx = torch.tensor(sample_indices, dtype=torch.long, device=self.device)

            return (
                self.states[idx],
                self.actions[idx],
                self.rewards[idx],
                self.mc_returns[idx],
                self.next_states[idx],
                self.terms[idx]
            )

        return sample_per

    def __len__(self):
        return self.size

ObserverType = Callable[
    [int, Policy],
    None
]

SampleObserverType = Callable[
    [int, Dict[str, Any]],
    None
]

def new_observer(bench_env: Environment, bench_every = 1_000, bench_episodes = 10):
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

def new_sample_observer():
    posterior_samples = []

    def observer(
        step: int,
        samples: Dict[str, Any]
    ):
        posterior_samples.append({
            "step": step,
            "samples": samples
        })
    
    return observer, posterior_samples



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

def compute_disagreement(q_net, states: torch.Tensor, actions: torch.Tensor, mc_returns: torch.Tensor, batch_size: int = 8192):
    N = states.shape[0]
    deltas = []

    # flatten returns
    mc_returns = mc_returns.view(-1)

    with torch.no_grad():
        for i in range(0, N, batch_size):
            s_batch = states[i:i+batch_size]
            a_batch = actions[i:i+batch_size]
            g_batch = mc_returns[i:i+batch_size]

            q_batch = q_net(s_batch, a_batch)

            if q_batch.ndim == 2 and q_batch.shape[1] == 1:
                q_batch = q_batch[:, 0]
            elif q_batch.ndim == 1:
                pass
            else:
                raise ValueError(f"Unexpected Q output shape: {q_batch.shape}")

            assert q_batch.shape == g_batch.shape, \
                f"Shape mismatch: g_batch {g_batch.shape}, q_batch {q_batch.shape}"

            delta_batch = torch.abs(g_batch - q_batch)
            deltas.append(delta_batch)

    return torch.cat(deltas, dim=0)