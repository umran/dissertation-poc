import torch
import torch.optim as optim
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpy.typing import ArrayLike
from numpyro.infer import MCMC, NUTS
from typing import Callable

def relu(x):
    return jnp.maximum(0, x)

def ard_linear(name, x, in_dim, out_dim):
    # ARD: log-precision for each weight and bias
    weight_scale = numpyro.sample(f"{name}_weight_scale", dist.Gamma(1.0, 1.0).expand([in_dim, out_dim]))
    bias_scale = numpyro.sample(f"{name}_bias_scale", dist.Gamma(1.0, 1.0).expand([out_dim]))

    w = numpyro.sample(f"{name}_w", dist.Normal(0., 1. / jnp.sqrt(weight_scale)))
    b = numpyro.sample(f"{name}_b", dist.Normal(0., 1. / jnp.sqrt(bias_scale)))

    return jnp.dot(x, w) + b

def q_model(state, action, hidden_sizes=[32, 32], y=None):
    x = jnp.concatenate([state, action], axis=-1)
    n, input_dim = x.shape
    h1_dim, h2_dim = hidden_sizes

    # First layer
    z1 = relu(ard_linear("layer1", x, input_dim, h1_dim))

    # Second layer
    z2 = relu(ard_linear("layer2", z1, h1_dim, h2_dim))

    # Output layer
    out = ard_linear("output", z2, h2_dim, 1)
    assert out.shape == (n, 1)

    if y is not None:
        assert y.shape == (n, 1)
        sigma_obs = numpyro.sample("sigma_obs", dist.Exponential(1.0))
        with numpyro.plate("data", n):
            numpyro.sample("y", dist.Normal(out.squeeze(-1), sigma_obs), obs=y.squeeze(-1))

def run_mcmc(state, action, target, num_samples=1000, num_warmup=500, rng_key=jax.random.PRNGKey(0)):
    kernel = NUTS(q_model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=1)
    
    mcmc.run(rng_key, state=state, action=action, y=target)
    mcmc.print_summary()
    
    samples = mcmc.get_samples()
    return samples


# to be refined
type QFunction = Callable[[ArrayLike, ArrayLike], jnp.ndarray]

def sample_q_function(samples, rng_key) -> QFunction:
    idx = jax.random.randint(rng_key, shape=(), minval=0, maxval=samples['layer1_w'].shape[0])
    q_params = {k: v[idx] for k, v in samples.items()}
    
    def q_fn(s: ArrayLike, a: ArrayLike) -> jnp.ndarray:
        x = jnp.concatenate([s, a], axis=-1)
        z1 = relu(jnp.dot(x, q_params['layer1_w']) + q_params['layer1_b'])
        z2 = relu(jnp.dot(z1, q_params['layer2_w']) + q_params['layer2_b'])
        out = jnp.dot(z2, q_params['output_w']) + q_params['output_b']
        return out
    
    return q_fn

def optimize_policy(q_fn: QFunction, policy_cls: torch.Module, state_batch: torch.Tensor, num_steps=10, lr=1e-2, device: torch.device = torch.device("cpu")):
    policy = policy_cls().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    for _ in range(num_steps):
        optimizer.zero_grad()
        actions = policy(state_batch)

        # Convert to numpy → JAX → evaluate Q
        s_np = state_batch.detach().cpu().numpy()
        a_np = actions.detach().cpu().numpy()
        q_values = q_fn(s_np, a_np)
        q_values = jnp.array(q_values).squeeze()  # JAX DeviceArray → NumPy → scalar

        # Convert to torch tensor and take negative (gradient ascent)
        q_tensor = torch.tensor(q_values, dtype=torch.float32, device=device)
        loss = -q_tensor.mean()  # gradient ascent

        loss.backward()
        optimizer.step()

    return policy  # Can be used for action selection during next episode