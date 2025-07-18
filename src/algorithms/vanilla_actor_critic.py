import torch
from typing import Optional

from algorithms.common import ReplayBuffer, ObserverType
from algorithms.random_policy import RandomPolicy
from algorithms.actor_critic import ActorCritic
from environments.environment import Environment

class VanillaActorCritic:
    def __init__(
        self,
        env: Environment,
        actor_critic: ActorCritic,
        device: torch.device = torch.device("cpu"),
    ):
        self.env = env
        self.actor_critic = actor_critic
        self.device = device

        self.random_policy = RandomPolicy(
            env.action_shape(),
            env.action_min(),
            env.action_max()
        )

    def train(
        self,
        steps=200_000,
        start_steps=10_000,
        update_after=10_000,
        update_every=50,
        gamma=0.99,
        observer: Optional[ObserverType] = None,
    ):
        replay_buffer = ReplayBuffer(
            1_000_000,
            self.env.state_shape(),
            self.env.action_shape(),
            device=self.device
        )

        policy = self.actor_critic.get_exploration_policy()
        state = self.env.reset()

        for step in range(steps):
            # coarse progress tracking
            if step % 10_000 == 0:
                print(step)

            if step < start_steps:
                action = self.random_policy.action(state)
            else:
                action = policy.action(state)
            
            next_state, reward, term, trunc, _ = self.env.step(action)
            done = term or trunc

            # append state, action, reward, done, next_state to replay buffer
            replay_buffer.add(state, action, reward, next_state, term)

            if done:
                # we've reached the end of an episode
                # get new initial state
                state = self.env.reset()
            else:
                state = next_state

            # call the actor critic update method
            if step >= update_after and step % update_every == 0:
                self.actor_critic.update(replay_buffer, update_every, gamma)
            
            if observer is not None:
                observer(step, self.actor_critic.get_optimal_policy())