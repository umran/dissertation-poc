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

### Hamiltonian Monte Carlo

#### Theoretical Model

The exploration strategy used here involves maintaining an approximation of the following probability distribution, the posterior distribution over the weights of a Q function centered around some prior weights theta hat, given observed monte carlo returns associated with state action pairs visited in past episodes:

P(theta | D, theta_hat) ~ P(D | theta) P(theta | theta_hat, alpha) P(alpha)

where:
P(D | theta) ~ Gaussian(f(x, theta), sigma), P(sigma) ~ HalfCauchy(1)
P(theta | theta_hat, alpha) ~ Gaussian(theta_hat, alpha), P(alpha) ~ Gamma(1, 1)

Note that the above model places an ARD-like prior on theta, where instead of centering around zero, we center around the prior parameter theta_hat. The intuition is that this expresses a preference for keeping most weights of the Q Network unchanged, while varying only the weights most likely to explain the data sampled during the current approximation.

Notably, the data D (x, y) is a set of K transitions sampled from a replay buffer with the probability of a transition being drawn proportional to |f(D.x, theta_hat) - D.y| / Sum_over_transitions(|f(D.x, theta_hat) - D.y|). Also worth noting, is that transitions present in the replay buffer are generated as a result of acting greedily with respect to a Q function sampled from the approximated Q posterior for the entire duration of an episode, and so on for each traininig episode. An episode consists of a collection of transition tuples of the following form: (state, action, reward, next_state, done). The Monte Carlo return associated with (state, action) pairs in the episode is calculated as follows:

Sum_over_i_to_t(gamma^i \* reward_t), where t is the duration of the episode (or number of transitions) and i is the order of occurrence the transition within the episode, indexed from 0.

The approximated Q posterior remains unchanged until a reapproximation occurs every K training steps, where during each approximation, theta_hat is assumed to be closer to the weights of the Q function under the optimal policy than it was during the previous approximation, on average (This is to do with the involvement of the actor-critic component, which carries out policy improvement according to the standard mechanics of actor-critic frameworks, covered elsewhere in this paper). Thus, the idea is that with each approximation, we induce a distribution over plausible Q functions around what can be thought of as the best guess we have so far. The hypothesis is that this approximates a distribution over plausible Q functions, which may include Q functions that remain unexplored and yet may be closer to the Q function under the optimal policy than any that may have been explored before.

#### Implementation
