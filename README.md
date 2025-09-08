<!-- about 500 words -->

# Introduction

We propose an interface for Bayesian exploration in existing Actor-Critic algorithms that accommodates the use of a compatible posterior-sampling method for action-selection during training, which we call Bayesian Actor Critic (BAC). Additionally, we propose specifications for and concrete implementations of two different algorithms that satisfy the BAC interface:

-   Hamiltonian Monte Carlo Actor-Critic (HMC-AC)
-   Bootstrapped Actor-Critic (BS-AC)

We also conduct and report the results of experiments evaluating both HMC-AC and BS-AC, in benchmark OpenAI Gym environments for continuous control in the sections that follow.

<!-- about 3000 words -->

# Literature and Technology Survey

# Bayesian Actor Critic

We define Bayesian Actor Critic as a method of exploration in reinforcement learning in continuous action spaces that utilizes an Actor-Critic training loop as the primary learning mechanism, and a method of posterior sampling over plausible Q functions to obtain behaviour policies during exploration. The exploration strategy involves sampling a plausible Q function at the beginning of each episode, performing policy iteration to obtain a greedy behaviour policy with respect to the sampled Q-function, and following it for the duration of the episode. Throughout training, the Actor-Critic component periodically updates its approximation of the optimal Q function and policy using transitions observed during training.

<!-- about 2000 words -->

# Hamiltonian Monte Carlo Actor Critic (HMC-AC)

HMC-AC uses a Bayesian model to approximate a distribution over plausible Q-functions, using the Actor-Critic approximation of the optimal Q-function (Q\*) as a prior. Since the Actor-Critic approximation is non-stationary throughout training, and ideally moves closer to the true optimal Q-function, we design HMC Actor Critic so that the posterior over plausible Q-functions is periodically reapproximated, so as to maintain diversity around the most current approximation of Q\*. Thus, the idea is that with each approximation, we induce a distribution over plausible Q functions around what can be thought of as the best guess about Q\* we have so far, where the posterior ideally includes Q functions that remain unexplored and yet may be closer to Q\* than any that may have been explored in past episodes.

## Theoretical Model

The exploration strategy used in HMC-AC involves maintaining an approximation of the Bayesian posterior over the parameters of plausible Q functions given approximated parameters $\hat{\theta}$ of Q\* (obtained via Actor-Critic updates, explained in section ?) and observations of discounted returns associated with state-action pairs from trajectories resulting from previously sampled behaviour policies. We formally define our observations as follows:

$$
\mathcal D \;=\; \big\{\, (s_i,\ a_i,\ G_i) \,\big\}_{i=1}^{N}
$$

For each episode $e$ with length $T_e$ and discount factor $\gamma\in[0,1)$, the return from
time $t$ is

$$
G^{(e)}_t \;=\; \sum_{k=0}^{T_e-t-1} \gamma^{k}\, r^{(e)}_{t+k+1}.
$$

Since our environments involve continuous states and actions, we record one sample for **every** occurrence of a state–action pair within an episode, as opposed to _single-visit_ Monte Carlo, where only the first occurrence is recorded. Sutton and Barto show that in the limit, both approaches lead to an unbiased estimate of the state-action value given the behaviour policy.

We may begin to express this posterior with a Gaussian likelihood centered at $Q(s_i, a_i; \theta)$ with known noise $\sigma > 0$, and a Gaussian prior on $\theta$ centered at $\hat{\theta}$ with fixed width $\alpha > 0$.

```math
p(\theta \mid D, \hat{\theta}, \sigma, \alpha) \;\propto\;
\prod_{i=1}^{K} \mathcal N\!\big(y_i \mid Q(s_i, a_i; \theta), \sigma^2\big)
\times
\prod_{j=1}^{d} \mathcal N\!\big(\theta_j \mid \hat{\theta}_j, \alpha^2\big)
```

In our setting, the “observations” are Monte Carlo returns generated under past behaviour policies, rather than samples from an optimal policy. Accordingly, the Gaussian likelihood should be interpreted not as a noise model around a fixed ground truth, but as a compatibility score between a candidate Q-function and the empirical mixture of returns obtained during exploration under various behaviour policies. The noise variance $\sigma^2$ captures both stochasticity in the environment as well as the additional variability introduced by mixing across multiple policies. The resulting posterior therefore represents a distribution over Q-functions that are plausible given the trajectory of past policies, rather than one centered on a single optimal solution.

We note that so far, given a linear $Q$ and fixed $\sigma$ and $\alpha$, the above posterior has a closed form solution. To make our model more robust to estimation of observation noise, we introduce an uninformative, log-uniform prior on sigma, no longer fixing it, and instead reformulating our posterior as the joint posterior over the parameters and noise variance $\sigma^2$:

```math
p(\theta, \sigma \mid D, \hat{\theta}, \alpha) \;\propto\;
\prod_{i=1}^{K} \mathcal N\!\big(y_i \mid Q(x_i; \theta), \sigma^2\big)
\times \prod_{j=1}^{d} \mathcal N\!\big(\theta_j \mid \hat{\theta}_j, \alpha^2\big)
\times p(\sigma)
```

```math
\sigma \sim \mathrm{LogUniform}\!\left(10^{-4},\,10^{4}\right)
```

No longer conditioning on $\sigma$ and the introduction of the relatively broad prior on $\sigma$ results in a posterior that does not have a closed form solution.

```math
\begin{aligned} p(\theta,\boldsymbol{\alpha},\sigma \mid \mathcal D,\hat{\theta}) \ &\propto\ \underbrace{\prod_{i=1}^K \mathcal N\!\big(y_i \mid f(x_i;\theta),\,\sigma^2\big)}_{\text{data likelihood}} \;\times\; \underbrace{\prod_{j=1}^d \mathcal N\!\big(\theta_j \mid \hat{\theta}_j,\, \alpha_j^2\big)}_{\text{centered ARD prior}} \\[6pt] &\hspace{3.5cm}\times\; \underbrace{\prod_{j=1}^d p(\alpha_j)}_{\alpha_j^{-2}\sim \mathrm{Gamma}(0, 0)} \;\times\; \underbrace{p(\sigma)}_{\sigma\sim \text{Gamma}(0, 0)} \end{aligned}
```

Note that the above model places an ARD-like prior on theta, where instead of centering around zero, we center around the prior parameter theta_hat. The intuition is that this expresses a preference for keeping most weights of the Q Network unchanged, while varying only the weights most likely to explain the data sampled during the current approximation.

Notably, the data D (x, y) is a set of K transitions sampled from a replay buffer with the probability of a transition being drawn proportional to |f(D.x, theta_hat) - D.y| / Sum_over_transitions(|f(D.x, theta_hat) - D.y|). Also worth noting, is that transitions present in the replay buffer are generated as a result of acting greedily with respect to a Q function sampled from the approximated Q posterior for the entire duration of an episode, and so on for each traininig episode. An episode consists of a collection of transition tuples of the following form: (state, action, reward, next_state, done). The Monte Carlo return associated with the ith (state, action) pair in an episode is calculated as follows:

Sum_over_k_to_T_minus_one_minus_i(gamma^k \* reward_i_plus_k), where i is the ith transition of the episode and T is the total number of transitions in the episode.

## Implementation

<!-- about 1000 words -->

# Bootstrapped Actor Critic

## Theoretical Model

## Implementation

<!-- about 1000 words -->

# Methodology

<!-- about 500 words -->

# Experiments

<!-- about 1000 words -->

# Results and Discussion

<!-- about 1000 words -->

# Conclusion and Future Work

# Appendices

## Appendix A - On Policy Actor Critic Algorithms

## A.1 DDPG

## A.2 TD3

## Appendix B - Behaviour Policy Optimization in HMC

## B.1 OneCycleLR

## Appendix C - Alternatively Using Proper Gamma(1, 1) Priors
