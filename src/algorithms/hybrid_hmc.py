import torch
import jax.numpy as jnp
# from tqdm import tqdm
from typing import Tuple, List

from algorithms.common import ReplayBuffer, EpisodicReplayBuffer
from algorithms.policy import Policy
from algorithms.actor_critic import ActorCritic
from algorithms.q_model import run_mcmc
from environments.environment import Environment

class HybridHMC:
    def __init__(self, env: Environment, actor_critic: ActorCritic, device: torch.device = torch.device("cpu")):
        self.env = env
        self.actor_critic = actor_critic
        self.device = device
        
        self.replay_buffer = ReplayBuffer(
            1_000_000,
            env.state_shape(),
            env.action_shape(),
            device=device
        )

        self.episodic_replay_buffer = EpisodicReplayBuffer(
            1_000_000,
            env.state_shape(),
            env.action_shape(),
            device=device
        )
    
    def update_posteior(self):
        state, action, _, mc_return, _, _ = self.episodic_replay_buffer.sample(128)

        state = jnp.array(state.numpy())
        action = jnp.array(action.numpy())
        target = jnp.array(mc_return.numpy())

        samples = run_mcmc(state, action, target)
        

    def sample_policy(self) -> Policy:
        # this method should return the policy corresponding to a Q function
        # randomly sampled from the posterior estimated by HMC

        # the following is just a placeholder for now
        return self.actor_critic.get_exploration_policy()

    def train(self, steps=1_000_000, start_steps=10_000, update_after=100_000, update_every=100_000, gamma=0.99):    
        policy = self.sample_policy()
        state = self.env.reset()
        episode_steps: List[Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, bool]] = []

        for step in range(steps):
            # coarse progress tracking
            if step % 10_000 == 0:
                print(step) 
            
            if step < start_steps:
                # need to refactor this for the generic use case where the range
                # can be arbitrary
                action = 2 * torch.rand(self.env.action_shape(), device=self.device) - 1
            else:
                action = policy.action(state)
            
            next_state, reward, term, trunc, _ = self.env.step(action)
            done = term or trunc

            # append state, action, reward, done, next_state to episode_steps and replay buffer
            episode_steps.append((state, action, reward, next_state, done))
            self.replay_buffer.add(state, action, reward, next_state, done)

            if done:
                # we've reached the end of an episode
                # calculate discounted monte carlo returns per step within the episode and add to episodic_replay_buffer
                mc_return = 0
                for (state, action, reward, next_state, done) in reversed(episode_steps):
                    mc_return = reward + gamma * mc_return
                    self.episodic_replay_buffer.add(state, action, reward, mc_return, next_state, done)

                # reset episode_steps
                episode_steps = []
                # resample policy
                policy = self.sample_policy()
                # get new initial state
                state = self.env.reset() 
            else:
                state = next_state
                

            # call the actor critic update method
            self.actor_critic.update(step, self.replay_buffer)

            # update posterior
            if step >= update_after and step % update_every == 0:
                print("updating posterior")
                self.update_posteior()
    