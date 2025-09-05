import torch
import torch.optim as optim

from bac.algorithms.policy import Policy
from bac.algorithms.common import MaskedReplayBuffer, copy_params, polyak_update, masked_mean
from bac.algorithms.vectorized_networks import MultiHeadQNetwork, MultiHeadPolicyNetwork
from bac.environments import Environment
from .multi_head_actor_critic import MultiHeadActorCritic, OptimalPolicy, SampledPolicy, NoisyPolicy

class MultiHeadDDPG(MultiHeadActorCritic):
    def __init__(
        self,
        env: Environment,
        n_heads: int = 1,
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

        self.n_heads = n_heads
        self.q_net = MultiHeadQNetwork(state_shape[0], action_shape[0], n_heads).to(device)
        self.q_net_target = MultiHeadQNetwork(state_shape[0], action_shape[0], n_heads).to(device)
        self.policy_net = MultiHeadPolicyNetwork(state_shape[0], action_shape[0], action_min, action_max, n_heads).to(device)
        self.policy_net_target = MultiHeadPolicyNetwork(state_shape[0], action_shape[0], action_min, action_max, n_heads).to(device)

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
        self.optimal_policy = OptimalPolicy(self.policy_net, self.q_net)

        # set a noisy exploration policy for when n_heads = 1
        self.exploration_policy = NoisyPolicy(self.policy_net, exploration_noise, action_min, action_max)
    
    def update(self, replay_buffer: MaskedReplayBuffer, steps: int, gamma: float):
        if len(replay_buffer) == 0:
            return

        for _ in range(steps):
            state, action, reward, next_state, done, mask = replay_buffer.sample(self.batch_size)

            if mask.sum() == 0:
                continue

            with torch.no_grad():
                a_next = self.policy_net_target(next_state)
                q_next = self.q_net_target(next_state, a_next).squeeze(-1)
                r = reward.squeeze(-1)
                d = done.squeeze(-1)
                target = r[:, None] + gamma * (1.0 - d[:, None]) * q_next

            a_b = action.unsqueeze(1).expand(-1, self.n_heads, -1)
            
            q_pred = self.q_net(state, a_b).squeeze(-1)
            se = (q_pred - target).pow(2)
            q_loss = masked_mean(se, mask)
            self.q_optimizer.zero_grad()
            q_loss.backward()
            self.q_optimizer.step()

            a_pi = self.policy_net(state)
            q_pi = self.q_net(state, a_pi).squeeze(-1)
            policy_loss = -masked_mean(q_pi, mask)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # shift target networks forward
            polyak_update(self.q_net_target, self.q_net, self.polyak)
            polyak_update(self.policy_net_target, self.policy_net, self.polyak)

    def get_optimal_policy(self) -> Policy:
        return self.optimal_policy

    def get_exploration_policy(self) -> Policy:
        if self.n_heads == 1:
            return self.exploration_policy
        
        head_idx = torch.randint(low=0, high=self.n_heads, size=(1,)).item()
        return SampledPolicy(self.policy_net, head_idx)
    
    def get_critic_network(self) -> MultiHeadQNetwork:
        return self.q_net
    
    def get_actor_network(self) -> MultiHeadPolicyNetwork:
        return self.policy_net
    
    def get_n_heads(self):
        return self.n_heads