import torch
import torch.nn as nn
import torch.optim as optim
from typing import Type

from algorithms.actor_critic import ActorCritic
from algorithms.policy import Policy
from algorithms.common import ReplayBuffer, copy_params, polyak_update, sample_gaussian

class DDPG(ActorCritic):
    def __init__(
        self,
        q_cls: Type[nn.Module],
        policy_cls: Type[nn.Module],
        batch_size: int = 128,
        update_after: int = 10_000,
        update_every: int = 50,
        gamma: float = 0.99,
        polyak: float = 0.995,
        q_lr: float = 1e-4,
        policy_lr: float = 1e-4,
        exploration_noise: float = 0.2,
        device: torch.device = torch.device("cpu")
    ):
        self.q = q_cls().to(device)
        self.q_target = q_cls().to(device)
        self.policy = policy_cls().to(device)
        self.policy_target = policy_cls().to(device)

        # initially set parameters of the target networks 
        # to those from the actual networks
        copy_params(self.q_target, self.q)
        copy_params(self.policy_target, self.policy)


        self.batch_size = batch_size
        self.update_after = update_after
        self.update_every = update_every
        self.gamma = gamma
        self.polyak = polyak

        # initialize optimizers for the q and policy networks
        self.q_optimizer = optim.Adam(self.q.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)

        # define the loss function
        self.loss = nn.MSELoss()

        # define the optimal policy
        self.optimal_policy = OptimalPolicy(self.policy)
        self.exploration_policy = ExplorationPolicy(self.policy, exploration_noise)

        # define the exploration policy
    
    def update(self, step: int, replay_buffer: ReplayBuffer):
        if step < self.update_after or (step - self.update_after) % self.update_every != 0:
            return

        for _ in range(self.update_every):
            state, action, reward, next_state, done = replay_buffer.sample(self.batch_size)

            with torch.no_grad():
                target = reward + self.gamma * (1 - done.to(torch.float32)) * self.q_target(next_state, self.policy_target(next_state))
            
            # do a gradient descent update of the
            # q network to minimize the MSBE loss
            predicted = self.q(state, action)
            loss = self.loss(predicted, target)

            self.q_optimizer.zero_grad()
            loss.backward()
            self.q_optimizer.step()

            # do a gradient ascent update of the policy
            # network to maximize the average state-action value
            policy_loss = -1 * self.q(state, self.policy(state)).mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # shift target networks forward
            polyak_update(self.q_target, self.q, self.polyak)
            polyak_update(self.policy_target, self.policy, self.polyak)

    def get_optimal_policy(self) -> Policy:
        return self.optimal_policy

    def get_exploration_policy(self) -> Policy:
        return self.exploration_policy

class OptimalPolicy(Policy):
    def __init__(self, policy_net: nn.Module):
        self.policy_net = policy_net

    def action(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.policy_net(state)

class ExplorationPolicy(Policy):
    def __init__(self, policy_net: nn.Module, noise: float):
        self.policy_net = policy_net
        self.noise = noise
    
    def action(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action = self.policy_net(state)
        
        noise = sample_gaussian(0.0, self.noise, action.shape, device=action.device)

        return action + noise