import torch
import torch.optim as optim
import torch.nn as nn
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from typing import Tuple, List, Optional

from algorithms.common import ReplayBuffer, EpisodicReplayBuffer, ObserverType, copy_params
from algorithms.networks import PolicyNetwork, QNetwork
from algorithms.policy import Policy
from algorithms.random_policy import RandomPolicy
from algorithms.actor_critic import ActorCritic
from environments.environment import Environment

class HybridHMC:
    def __init__(
        self,
        env: Environment,
        actor_critic: ActorCritic,
        rng_key = jax.random.key(0),
        device: torch.device = torch.device("cpu"),
    ):
        self.env = env
        self.actor_critic = actor_critic
        self.rng_key = rng_key
        self.device = device

        self.random_policy = RandomPolicy(
            env.action_shape(),
            env.action_min(),
            env.action_max()
        )

        self.q_weight_posterior = None
        self.exploration_policy_net = None

    def train(
        self,
        steps=200_000,
        update_after=10_000,
        update_every=5_000,
        update_actor_critic_every=50,
        gamma=0.99,
        observer: Optional[ObserverType] = None,
    ):
        replay_buffer = ReplayBuffer(
            1_000_000,
            self.env.state_shape(),
            self.env.action_shape(),
            device=self.device
        )

        episodic_replay_buffer = EpisodicReplayBuffer(
            1_000_000,
            self.env.state_shape(),
            self.env.action_shape(),
            device=self.device
        )

        policy = self.sample_policy(episodic_replay_buffer)
        state = self.env.reset()
        episode_steps: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

        for step in range(steps):
            # coarse progress tracking
            if step % 5_000 == 0:
                print(step)

            action = policy.action(state)
            next_state, reward, term, trunc, _ = self.env.step(action)
            done = term or trunc

            # append state, action, reward, done, next_state to episode_steps and replay buffer
            episode_steps.append((state, action, reward, next_state, term.to(torch.float32)))
            replay_buffer.add(state, action, reward, next_state, term.to(torch.float32))

            if done:
                # we've reached the end of an episode
                # calculate discounted monte carlo returns per step within the episode and add to episodic_replay_buffer
                mc_return = 0
                for (state, action, reward, next_state, term) in reversed(episode_steps):
                    mc_return = reward + gamma * mc_return
                    episodic_replay_buffer.add(state, action, reward, mc_return, next_state, term)

                # reset episode_steps
                episode_steps = []
                # resample policy
                policy = self.sample_policy(episodic_replay_buffer)
                # get new initial state
                state = self.env.reset()
            else:
                state = next_state

            # call the actor critic update method
            if step % update_actor_critic_every == 0:
                self.actor_critic.update(replay_buffer, update_actor_critic_every, gamma)

            # update posterior
            if step >= update_after and (step - update_after) % update_every == 0:
                self.update_posterior(episodic_replay_buffer)
               
            
            if observer is not None:
                observer(step, self.actor_critic.get_optimal_policy())

    def update_posterior(self, episodic_replay_buffer: EpisodicReplayBuffer):
        q_net = self.actor_critic.get_critic_network()
        state, action, _, mc_return, _, _ = episodic_replay_buffer.sample_per(q_net, 1000)

        state = jnp.array(state.cpu().numpy())
        action = jnp.array(action.cpu().numpy())
        target = jnp.array(mc_return.cpu().numpy())

        prior_params = None
        if self.q_weight_posterior is not None:
            prior_params = extract_qnetwork_params(q_net)

        kernel = NUTS(q_model)
        mcmc = MCMC(
            kernel,
            num_warmup=500,
            num_samples=1000,
            num_chains=1
        )
        
        mcmc.run(self.next_rng_key(), state=state, action=action, y=target, prior_params=prior_params)
        mcmc.print_summary()
        
        self.q_weight_posterior = mcmc.get_samples()
        
        # instantiate a new policy
        self.exploration_policy_net = PolicyNetwork(
            self.env.state_shape()[0],
            self.env.action_shape()[0],
            self.env.action_min(),
            self.env.action_max()
        ).to(self.device)

        copy_params(self.exploration_policy_net, self.actor_critic.get_actor_network())
        

    def sample_policy(self, episodic_replay_buffer: EpisodicReplayBuffer) -> Policy:
        # this method should return the policy corresponding to a Q function
        # randomly sampled from the posterior estimated by HMC if a posterior
        # has been estimated, otherwise returns the random policy
        if self.q_weight_posterior is None:
            return self.random_policy

        # sample a set of q network parameters from mcmc samples
        idx = jax.random.randint(self.next_rng_key(), shape=(), minval=0, maxval=self.q_weight_posterior['layer1_w'].shape[0])
        q_params = {
            k: jax.device_get(v[idx])  # Convert from JAX DeviceArray
            for k, v in self.q_weight_posterior.items()
        }

        # construct a q network from sampled parameters
        q_net = SampledQNetwork(q_params).to(self.device)

        # instantiate optimizer for policy parameters
        optimizer = optim.Adam(self.exploration_policy_net.parameters(), lr=1e-2)

        state, _, _, _, _, _ = episodic_replay_buffer.sample_per(q_net, 1000)
        for _ in range(100):
            action = self.exploration_policy_net(state)
            value = q_net(state, action)
            loss = -value.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return SampledPolicy(self.exploration_policy_net)

    def next_rng_key(self):
        self.rng_key, subkey = jax.random.split(self.rng_key)
        
        return subkey

class SampledQNetwork(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = nn.ParameterDict({
            k: nn.Parameter(torch.tensor(v, dtype=torch.float32), requires_grad=False)
            for k, v in weights.items()
        })

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)

        z1 = torch.relu(torch.matmul(x, self.weights["layer1_w"]) + self.weights["layer1_b"])
        out = torch.matmul(z1, self.weights["output_w"]) + self.weights["output_b"]
       
        return out

class SampledPolicy(Policy):
    def __init__(self, policy_net: nn.Module):
        self.policy_net = policy_net
    
    def action(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.policy_net(state)

def q_model(state, action, y=None, prior_params=None, h1_dim=64):
    x = jnp.concatenate([state, action], axis=-1)
    n, input_dim = x.shape

    z1 = relu(linear("layer1", x, input_dim, h1_dim,
                         prior=prior_params["layer1"] if prior_params else None))

    out = linear("output", z1, h1_dim, 1,
                     prior=prior_params["output"] if prior_params else None)
    
    assert out.shape == (n, 1)

    if y is not None:
        assert y.shape == (n, 1)
        sigma = numpyro.sample("sigma", dist.Gamma(1.0, 1.0))
        with numpyro.plate("data", n):
            numpyro.sample("y", dist.Normal(out.squeeze(-1), sigma), obs=y.squeeze(-1))

def relu(x):
    return jnp.maximum(0, x)

def linear(name, x, in_dim, out_dim, prior=None):
    weight_scale = numpyro.sample(f"{name}_weight_scale", dist.Gamma(1.0, 1.0).expand([in_dim, out_dim]))
    bias_scale = numpyro.sample(f"{name}_bias_scale", dist.Gamma(1.0, 1.0).expand([out_dim]))

    if prior:
        weight_mu = jnp.asarray(prior["w"])
        bias_mu = jnp.asarray(prior["b"])
        
        # Check prior shapes
        assert weight_mu.shape == (in_dim, out_dim), (
            f"Expected prior['w'] to have shape {(in_dim, out_dim)}, got {weight_mu.shape}"
        )
        assert bias_mu.shape == (out_dim,), (
            f"Expected prior['b'] to have shape ({out_dim},), got {bias_mu.shape}"
        )
    else:
        weight_mu = jnp.zeros((in_dim, out_dim))
        bias_mu = jnp.zeros((out_dim,))

    w = numpyro.sample(f"{name}_w", dist.Normal(loc=weight_mu, scale=1. / jnp.sqrt(weight_scale)))
    b = numpyro.sample(f"{name}_b", dist.Normal(loc=bias_mu, scale=1. / jnp.sqrt(bias_scale)))

    return jnp.dot(x, w) + b

def extract_qnetwork_params(q_model: QNetwork):
    param_dict = {
        name: param.clone().detach().cpu().numpy()
        for name, param in q_model.named_parameters()
    }

    return {
        "layer1": {
            "w": param_dict['core.0.weight'].T,  # Transpose to [in, out] for JAX
            "b": param_dict['core.0.bias'],
        },
        "output": {
            "w": param_dict['core.2.weight'].T,
            "b": param_dict['core.2.bias'],
        },
    }
