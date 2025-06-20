# Bayesian Exploration in Actor Critic Frameworks

## Bootstrapped Actor Critic as An Extension of Bootstrapped DQN (Osband et al)

## Hamiltonian Monte Carlo Based Posterior Estimation for Exploratory Action Selection in Off Policy Actor Critic Algorithms

### Framework

#### Existing Actor Critic Networks

Use existing Actor Critic machinery to indirectly estimate the policy network via estimation of the Q Network

#### Parallel HMC Based Q Network Estimation

A process that runs in parallel, on an interval (every N steps for a sufficiently large N).
Sample a subset of the Replay buffer of size d, at uniform random

-   use a MCMC sampler against the sampled subset to estimate a set of weights for the Q Network.
-   possibly use a model that incorporates both the Q Network and its corresponding Policy Network in order to simultaneously estimate a posterior over both networks
-   barring the above or alternatively, use the estimated posterior over the Q Network to sample a finite set of corresponding Policy Networks

#### Exploratory Action Selection

If a posterior over the policy network is estimated directly via MCMC, sample a policy network from that posterior.
Else, if a subset of policy networks are sampled from the estimated posterior over Q Networks, sample a policy network at random. Then use the sampled policy network for action selection.

Some considerations:

-   the posterior estimates will be lagging or may not be available at all until the first N steps
-   action selection using posterior estimates may be too poor for the first few steps (something to be determined, and will probably vary with the environment and problem to be solved)
-   in such cases, perhaps allow initial learning to occur at a reasonable rate by falling back to action selection via additive Gaussian or OU noise, i.e, noise added directly to actions selected from the policy being learned.
