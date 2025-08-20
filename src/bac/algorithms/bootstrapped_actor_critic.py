import torch
from tqdm import tqdm
from typing import Optional

from bac.algorithms.common import MaskedReplayBuffer, ObserverType
from bac.algorithms.actor_critic import MultiHeadActorCritic
from bac.algorithms.policy import RandomPolicy
from bac.environments import Environment

class BootstrappedActorCritic:
    def __init__(
        self,
        env: Environment,
        actor_critic: MultiHeadActorCritic,
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
        p=1.0,
        steps=200_000,
        start_steps=10_000,
        update_after=10_000,
        update_every=50,
        gamma=0.99,
        observer: Optional[ObserverType] = None,
    ):
        replay_buffer = MaskedReplayBuffer(
            1_000_000,
            self.actor_critic.get_n_heads(),
            p,
            self.env.state_shape(),
            self.env.action_shape(),
            device=self.device
        )

        # sample initial exploration policy
        policy = self.actor_critic.get_exploration_policy()
        # sample initial state
        state = self.env.reset()

        for step in tqdm(range(steps)):
            if step < start_steps:
                action = self.random_policy.action(state)
            else:
                action = policy.action(state)
            
            next_state, reward, term, trunc, _ = self.env.step(action)
            done = term or trunc

            # append state, action, reward, done, next_state to replay buffer
            replay_buffer.add(state, action, reward, next_state, term.to(torch.float32))

            if done:
                # we've reached the end of an episode
                # sample new exploration policy
                policy = self.actor_critic.get_exploration_policy()
                # get new initial state
                state = self.env.reset()
            else:
                state = next_state

            # call the actor critic update method
            if step >= update_after and step % update_every == 0:
                self.actor_critic.update(replay_buffer, update_every, gamma)
            
            if observer is not None:
                observer(step, self.actor_critic.get_optimal_policy())