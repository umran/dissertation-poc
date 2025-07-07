import torch
from typing import Tuple, List
from util.replay_buffer import ReplayBuffer
from util.episodic_replay_buffer import EpisodicReplayBuffer
from environment import Environment
from policy import Policy
from actor_critic import ActorCritic

class HybridHMC:
    def __init__(self, env: Environment, actor_critic: ActorCritic):
        self.env = env
        self.actor_critic = actor_critic
        
        self.replay_buffer = ReplayBuffer(
            100_000,
            env.state_shape(),
            env.action_shape()
        )

        self.episodic_replay_buffer = EpisodicReplayBuffer(
            100_000,
            env.state_shape(),
            env.action_shape()
        )
    
    def sample_policy(self) -> Policy:
        # this method should return the policy corresponding to a Q function
        # randomly sampled from the posterior estimated by HMC

        # the following is just a placeholder for now
        self.actor_critic.get_exploratory_policy()

    def train(self, gamma=0.99, steps=100_000):    
        policy = self.sample_policy()
        state = self.env.reset()
        episode_steps: List[Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, bool]] = []


        for step in range(steps):
            action = policy.action(state)
            next_state, reward, term, trunc = self.env.step(action)
            done = term or trunc

            # append state, action, reward, done, next_state to episode_steps and replay buffer
            episode_steps.append((state, action, reward, next_state, done))
            self.replay_buffer.add(state, action, reward, next_state, done)

            if not done:
                state = next_state
                continue

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

            # call the actor critic update method
            self.actor_critic.update(step, self.replay_buffer)
    