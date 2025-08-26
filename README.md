<!-- about 500 words -->

# Introduction

<!-- about 3000 words -->

# Literature and Technology Survey

<!-- about 2000 words -->

# HMC Actor Critic

At the heart of HMC Actor Critic is a Bayesian model used to continuously approximate a distribution over plausible Q-functions, centered around the Actor-Critic approximation of the optimal Q-function. Since the Actor-Critic approximation is non-stationary throughout training, and ideally moves closer to the true optimal Q-function, we design HMC Actor Critic so that the posterior over plausible Q-functions also shifts, so as to maintain diversity around the best approximation. Thus, the idea is that with each approximation, we induce a distribution over plausible Q functions around what can be thought of as the best guess we have so far, where the posterior ideally includes Q functions that remain unexplored and yet may be closer to the Q function under the optimal policy than any that may have been explored before.

## Theoretical Model

The exploration strategy used here involves maintaining an approximation of the Bayesian posterior over the parameters of plausible Q functions given approximated parameters $\hat{\theta}$ and Monte Carlo returns associated with state-action pairs observed during exploration $D = \{ (s_i, a_i, y_i) \}_{i=1}^{K}$.

We begin to express this posterior with a Gaussian likelihood centered at $Q(s_i, a_i; \theta)$ with known noise $\sigma > 0$, and a Gaussian prior on $\theta$ centered at $\hat{\theta}$ with fixed width $\alpha > 0$.

$$
p(\theta \mid D, \hat{\theta}, \sigma, \alpha) \;\propto\;
\prod_{i=1}^{K} \mathcal{N}\!\big(y_i \mid Q(s_i, a_i; \theta), \sigma^2\big)
\times
\prod_{j=1}^{d} \mathcal{N}\!\big(\theta_j \mid \hat{\theta}_j, \alpha^2\big)
$$

In our setting, the “observations” are Monte Carlo returns generated under a sequence of exploratory policies, rather than samples from an optimal policy. Accordingly, the Gaussian likelihood should be interpreted not as a noise model around a fixed ground truth, but as a compatibility score between a candidate Q-function and the empirical mixture of returns obtained during exploration. The noise parameter $\sigma$ captures both stochasticity in the environment as well as the additional variability introduced by mixing across multiple policies. The resulting posterior therefore represents a distribution over Q-functions that are plausible given the trajectory of past policies, rather than one centered on a single optimal solution.

We note that so far, given a linear $Q$ and fixed $\sigma$ and $\alpha$, the above posterior has a closed form solution. Making our model more robust to estimation of observation noise, we introduce an uninformative prior on sigma, which yields the following posterior:

$$
p(\theta, \sigma \mid D, \hat{\theta}, \alpha) \;\propto\;
\Bigg[ \prod_{i=1}^{K} \mathcal{N}\!\big(y_i \mid Q(x_i; \theta), \sigma^2\big) \Bigg]
\times
\Bigg[ \prod_{j=1}^{d} \mathcal{N}\!\big(\theta_j \mid \hat{\theta}_j, \alpha^2\big) \Bigg]
\times p(\sigma)
$$

$$
\sigma \sim \text{Gamma}(0, 0)
$$

No longer conditioning on $\sigma$ and the introduction of the relatively broad prior on $\sigma$ results in a posterior that does not have a closed form solution even given a linear $Q$ function.

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
