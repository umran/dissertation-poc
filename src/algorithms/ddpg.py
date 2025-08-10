import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.actor_critic import ActorCritic, OptimalPolicy, ExplorationPolicy
from algorithms.policy import Policy
from algorithms.common import ReplayBuffer, copy_params, polyak_update
from algorithms.networks import QNetwork, PolicyNetwork
from environments.environment import Environment

class DDPG(ActorCritic):
    def __init__(
        self,
        env: Environment,
        batch_size: int = 128,
        polyak: float = 0.995,
        q_lr: float = 1e-4,
        policy_lr: float = 1e-4,
        exploration_noise: float = 0.2,
        device: torch.device = torch.device("cpu")
    ):
        state_shape = env.state_shape()
        action_shape = env.action_shape()
        action_min = env.action_min()
        action_max = env.action_max()

        self.q_net = QNetwork(state_shape[0], action_shape[0]).to(device)
        self.q_net_target = QNetwork(state_shape[0], action_shape[0]).to(device)
        self.policy_net = PolicyNetwork(state_shape[0], action_shape[0], action_min, action_max).to(device)
        self.policy_net_target = PolicyNetwork(state_shape[0], action_shape[0], action_min, action_max).to(device)

        # initially set parameters of the target networks 
        # to those from the actual networks
        copy_params(self.q_net_target, self.q_net)
        copy_params(self.policy_net_target, self.policy_net)

        self.batch_size = batch_size
        self.polyak = polyak

        # initialize optimizers for the q and policy networks
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        # define the optimal policy
        self.optimal_policy = OptimalPolicy(self.policy_net)
        
        # define the exploration policy
        self.exploration_policy = ExplorationPolicy(self.policy_net, exploration_noise, action_min, action_max)
    
    def update(self, replay_buffer: ReplayBuffer, steps: int, gamma: float):
        if len(replay_buffer) == 0:
            return

        for _ in range(steps):
            state, action, reward, next_state, done = replay_buffer.sample(self.batch_size)

            with torch.no_grad():
                target = reward + gamma * (1 - done) * self.q_net_target(next_state, self.policy_net_target(next_state))
            
            # do a gradient descent update of the
            # q network to minimize the MSBE loss
            q_loss = nn.MSELoss()(self.q_net(state, action), target)
            self.q_optimizer.zero_grad()
            q_loss.backward()
            self.q_optimizer.step()

            # do a gradient ascent update of the policy
            # network to maximize the average state-action value
            policy_loss = -1 * self.q_net(state, self.policy_net(state)).mean() 
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # shift target networks forward
            polyak_update(self.q_net_target, self.q_net, self.polyak)
            polyak_update(self.policy_net_target, self.policy_net, self.polyak)

    def get_optimal_policy(self) -> Policy:
        return self.optimal_policy

    def get_exploration_policy(self) -> Policy:
        return self.exploration_policy
    
    def get_critic_network(self) -> QNetwork:
        return self.q_net
    
    def get_actor_network(self) -> PolicyNetwork:
        return self.policy_net