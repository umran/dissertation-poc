import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.actor_critic import ActorCritic
from algorithms.policy import Policy
from algorithms.common import ReplayBuffer, copy_params, polyak_update, sample_gaussian
from algorithms.networks import QNetwork, PolicyNetwork
from environments.environment import Environment

class TD3(ActorCritic):
    def __init__(
        self,
        env: Environment,
        batch_size: int = 128,
        polyak: float = 0.995,
        q_lr: float = 1e-4,
        policy_lr: float = 1e-4,
        exploration_noise: float = 0.2,
        target_noise: float = 0.1,
        target_noise_clip: float = 0.5,
        policy_delay: int = 2,
        device: torch.device = torch.device("cpu")
    ):
        state_shape = env.state_shape()
        action_shape = env.action_shape()
        action_min = env.action_min()
        action_max = env.action_max()

        self.action_min = action_min
        self.action_max = action_max
        self.target_noise = target_noise
        self.target_noise_clip = target_noise_clip
        self.policy_delay = policy_delay

        self.q1_net = QNetwork(state_shape[0], action_shape[0]).to(device)
        self.q1_net_target = QNetwork(state_shape[0], action_shape[0]).to(device)

        self.q2_net = QNetwork(state_shape[0], action_shape[0]).to(device)
        self.q2_net_target = QNetwork(state_shape[0], action_shape[0]).to(device)
        
        self.policy_net = PolicyNetwork(state_shape[0], action_shape[0], action_min, action_max).to(device)
        self.policy_net_target = PolicyNetwork(state_shape[0], action_shape[0], action_min, action_max).to(device)

        # initially set parameters of the target networks 
        # to those from the actual networks
        copy_params(self.q1_net_target, self.q1_net)
        copy_params(self.q2_net_target, self.q2_net)
        copy_params(self.policy_net_target, self.policy_net)


        self.batch_size = batch_size
        self.polyak = polyak

        # initialize optimizers for the q and policy networks
        self.q1_optimizer = optim.Adam(self.q1_net.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.q2_net.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        # define the loss function
        self.loss = nn.MSELoss()

        # define the optimal policy
        self.optimal_policy = OptimalPolicy(self.policy_net)
        
        # define the exploration policy
        self.exploration_policy = ExplorationPolicy(self.policy_net, exploration_noise, action_min, action_max)
    
    def update(self, replay_buffer: ReplayBuffer, steps: int, gamma: float):
        if len(replay_buffer) == 0:
            return

        for i in range(steps):
            state, action, reward, next_state, done = replay_buffer.sample(self.batch_size)

            with torch.no_grad():
                policy_action = self.policy_net_target(next_state)
                target_action = torch.clamp(
                    policy_action + sample_gaussian(0, self.target_noise, policy_action.shape, (-self.target_noise_clip, self.target_noise_clip), device=policy_action.device),
                    self.action_min,
                    self.action_max
                )

                target = reward + gamma * (1 - done.to(torch.float32)) * torch.min(self.q1_net_target(next_state, target_action), self.q2_net_target(next_state, target_action))
            
            # do a gradient descent update of the
            # q networks to minimize the MSBE loss
            q1_loss = self.loss(self.q1_net(state, action), target)
            self.q1_optimizer.zero_grad()
            q1_loss.backward()
            self.q1_optimizer.step()

            q2_loss = self.loss(self.q2_net(state, action), target)
            self.q2_optimizer.zero_grad()
            q2_loss.backward()
            self.q2_optimizer.step()

            if i % self.policy_delay == 0:
                # do a gradient ascent update of the policy
                # network to maximize the average state-action value
                policy_loss = -1 * self.q1_net(state, self.policy_net(state)).mean()
                
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                # shift target networks forward
                polyak_update(self.q1_net_target, self.q1_net, self.polyak)
                polyak_update(self.q2_net_target, self.q2_net, self.polyak)
                polyak_update(self.policy_net_target, self.policy_net, self.polyak)

    def get_optimal_policy(self) -> Policy:
        return self.optimal_policy

    def get_exploration_policy(self) -> Policy:
        return self.exploration_policy
    
    def get_critic_network(self) -> QNetwork:
        return self.q1_net
    
    def get_actor_network(self) -> PolicyNetwork:
        return self.policy_net

class OptimalPolicy(Policy):
    def __init__(self, policy_net: nn.Module):
        self.policy_net = policy_net

    def action(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.policy_net(state)

class ExplorationPolicy(Policy):
    def __init__(self, policy_net: nn.Module, noise: float, action_min: torch.Tensor, action_max: torch.Tensor):
        self.policy_net = policy_net
        self.noise = noise
        self.action_min = action_min
        self.action_max = action_max
    
    def action(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action = self.policy_net(state)
        
        noise = sample_gaussian(0.0, self.noise, action.shape, device=action.device)

        return torch.clamp(action + noise, self.action_min, self.action_max)
