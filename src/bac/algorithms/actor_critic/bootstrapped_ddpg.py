import torch
import torch.nn as nn
import torch.optim as optim

from bac.algorithms.policy import Policy
from bac.algorithms.common import MaskedReplayBuffer, copy_params, polyak_update
from bac.algorithms.networks import MultiHeadQNetwork, MultiHeadPolicyNetwork, QNetwork, PolicyNetwork
from bac.environments import Environment
from .actor_critic import ActorCritic, ExplorationPolicy

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
        self.optimal_policy = MeanPolicy(self.policy_net)
    
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
        head_idx = torch.randint(low=0, high=self.n_heads, size=(1,)).item()
        return ExplorationPolicy(self.policy_net, head_idx)
    
    def get_critic_network(self) -> QNetwork:
        pass
    
    def get_actor_network(self) -> PolicyNetwork:
        pass

class MeanPolicy(Policy):
    def __init__(self, policy_net):
        self.policy_net = policy_net

    def action(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 1:
            state = state.unsqueeze(0)  # [1, S]

        # [B, H, A]
        all_actions = self.policy_net(state)
        # average over heads → [B, A]
        mean_action = all_actions.mean(dim=1)

        if mean_action.shape[0] == 1:
            return mean_action.squeeze(0)  # [A]
        
        return mean_action

class ExplorationPolicy(Policy):
    def __init__(self, policy_net, head_idx: int):
        self.policy_net = policy_net
        self.head_idx = head_idx

    def action(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: [state_dim] or [B, state_dim]
        returns: [action_dim] or [B, action_dim] from the fixed head
        """
        with torch.no_grad():
            squeeze_back = False
            if state.ndim == 1:
                state = state.unsqueeze(0)
                squeeze_back = True

            all_actions = self.policy_net(state)      # [B, H, A]
            a = all_actions[:, self.head_idx, :]     # [B, A]

            return a.squeeze(0) if squeeze_back else a

def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask if x.dtype.is_floating_point else mask.float()
    denom = m.sum().clamp_min(1.0)
    return (x * m).sum() / denom