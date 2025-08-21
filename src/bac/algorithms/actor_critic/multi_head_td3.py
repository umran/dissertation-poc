import torch
import torch.optim as optim

from bac.algorithms.policy import Policy
from bac.algorithms.common import MaskedReplayBuffer, copy_params, polyak_update, sample_gaussian, masked_mean
from bac.algorithms.networks import MultiHeadQNetwork, MultiHeadPolicyNetwork
from bac.environments import Environment
from .multi_head_actor_critic import MultiHeadActorCritic, OptimalPolicy, SampledPolicy, NoisyPolicy

class MultiHeadTD3(MultiHeadActorCritic):
    def __init__(
        self,
        env: Environment,
        n_heads: int = 1,
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

        self.n_heads = n_heads
        self.action_min = action_min
        self.action_max = action_max
        self.target_noise = target_noise
        self.target_noise_clip = target_noise_clip
        self.policy_delay = policy_delay

        self.q1_net = MultiHeadQNetwork(state_shape[0], action_shape[0], n_heads).to(device)
        self.q1_net_target = MultiHeadQNetwork(state_shape[0], action_shape[0], n_heads).to(device)

        self.q2_net = MultiHeadQNetwork(state_shape[0], action_shape[0], n_heads).to(device)
        self.q2_net_target = MultiHeadQNetwork(state_shape[0], action_shape[0], n_heads).to(device)
        
        self.policy_net = MultiHeadPolicyNetwork(state_shape[0], action_shape[0], action_min, action_max, n_heads).to(device)
        self.policy_net_target = MultiHeadPolicyNetwork(state_shape[0], action_shape[0], action_min, action_max, n_heads).to(device)

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

        # define the optimal policy
        self.optimal_policy = OptimalPolicy(self.policy_net)
        
        # set a noisy exploration policy for when n_heads = 1
        self.exploration_policy = NoisyPolicy(self.policy_net, exploration_noise, action_min, action_max)
    
    def update(self, replay_buffer: MaskedReplayBuffer, steps: int, gamma: float):
        if len(replay_buffer) == 0:
            return

        for i in range(steps):
            state, action, reward, next_state, done, mask = replay_buffer.sample(self.batch_size)

            if mask.sum() == 0:
                continue

            with torch.no_grad():
                policy_action = self.policy_net_target(next_state)
                a_next = torch.clamp(
                    policy_action + sample_gaussian(0, self.target_noise, policy_action.shape, (-self.target_noise_clip, self.target_noise_clip), device=policy_action.device),
                    self.action_min,
                    self.action_max
                )
                q1_next = self.q1_net_target(next_state, a_next).squeeze(-1)
                q2_next = self.q2_net_target(next_state, a_next).squeeze(-1)
                r = reward.squeeze(-1)
                d = done.squeeze(-1)

                target = r[:, None] + gamma * (1 - d[:, None]) * torch.min(q1_next, q2_next)
            
            a_b = action.unsqueeze(1).expand(-1, self.n_heads, -1)
            
            # do a gradient descent update of the
            # q networks to minimize the MSBE loss
            q1_pred = self.q1_net(state, a_b).squeeze(-1)
            se_1 = (q1_pred - target).pow(2)
            q1_loss = masked_mean(se_1, mask)
            self.q1_optimizer.zero_grad()
            q1_loss.backward()
            self.q1_optimizer.step()


            q2_pred = self.q2_net(state, a_b).squeeze(-1)
            se_2 = (q2_pred - target).pow(2)
            q2_loss = masked_mean(se_2, mask)
            self.q2_optimizer.zero_grad()
            q2_loss.backward()
            self.q2_optimizer.step()

            if i % self.policy_delay == 0:
                # do a gradient ascent update of the policy
                # network to maximize the average state-action value
                a_pi = self.policy_net(state)
                q_pi = self.q1_net(state, a_pi).squeeze(-1)
                policy_loss = -masked_mean(q_pi, mask)
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
        if self.n_heads == 1:
            return self.exploration_policy
        
        head_idx = torch.randint(low=0, high=self.n_heads, size=(1,)).item()
        return SampledPolicy(self.policy_net, head_idx)
    
    def get_critic_network(self) -> MultiHeadQNetwork:
        return self.q1_net
    
    def get_actor_network(self) -> MultiHeadPolicyNetwork:
        return self.policy_net

    def get_n_heads(self):
        return self.n_heads