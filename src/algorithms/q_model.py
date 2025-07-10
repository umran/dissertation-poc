import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

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