import torch
import torch.nn as nn
import torch.optim as optim

from bac.algorithms.policy import Policy
from bac.algorithms.common import MaskedReplayBuffer, copy_params, polyak_update
from bac.algorithms.networks import MultiHeadQNetwork, MultiHeadPolicyNetwork, QNetwork, PolicyNetwork
from bac.environments import Environment
from .actor_critic import ActorCritic, OptimalPolicy, ExplorationPolicy

class BootstrappedDDPG(ActorCritic):
    def __init__(
        self,
        env: Environment,
        batch_size: int = 128,
        polyak: float = 0.995,
        q_lr: float = 1e-4,
        policy_lr: float = 1e-4,
        exploration_noise: float = 0.2,
        n_heads: int = 1,
        device: torch.device = torch.device("cpu")
    ):
        state_shape = env.state_shape()
        action_shape = env.action_shape()
        action_min = env.action_min()
        action_max = env.action_max()

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
        self.optimal_policy = OptimalPolicy(self.policy_net)
        
        # define the exploration policy
        self.exploration_policy = ExplorationPolicy(self.policy_net, exploration_noise, action_min, action_max)
    
    def update(self, replay_buffer: MaskedReplayBuffer, steps: int, gamma: float):
        if len(replay_buffer) == 0:
            return

        for _ in range(steps):
            state, action, reward, next_state, done, mask = replay_buffer.sample(self.batch_size)

            with torch.no_grad():
                # Target actions per head: [B, H, A]
                a_next = self.policy_net_target(next_state)                           # [B,H,A]
                # Target Q per head at next state-action: [B, H]
                q_next = self.q_net_target(next_state, a_next).squeeze(-1)           # [B,H]
                r = reward.squeeze(-1)                                               # [B]
                d = done.squeeze(-1)                                                 # [B]
                target = r[:, None] + gamma * (1.0 - d[:, None]) * q_next            # [B,H]

            # Pred Q per head at (s, a_behavior): expand a across heads
            a_b = action.unsqueeze(1).expand(-1, self.n_heads, -1)                   # [B,H,A]
            q_pred = self.q_net(state, a_b).squeeze(-1)                              # [B,H]

            # ----- Critic update (masked MSE) -----
            se = (q_pred - target).pow(2)                                            # [B,H]
            q_loss = masked_mean(se, mask)                                          # scalar

            if mask.sum() == 0:
                # No active (b,h) in this batch → skip both updates
                continue

            self.q_optimizer.zero_grad(set_to_none=True)
            q_loss.backward()
            self.q_optimizer.step()

            a_pi = self.policy_net(state)                                            # [B,H,A]
            q_pi = self.q_net(state, a_pi).squeeze(-1)                               # [B,H]
            policy_loss = -masked_mean(q_pi, mask)                                   # scalar

            self.policy_optimizer.zero_grad(set_to_none=True)
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
        pass
    
    def get_actor_network(self) -> PolicyNetwork:
        pass

def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask if x.dtype.is_floating_point else mask.float()
    denom = m.sum().clamp_min(1.0)
    return (x * m).sum() / denom