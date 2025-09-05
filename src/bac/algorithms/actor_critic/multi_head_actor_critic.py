import torch
from abc import ABC, abstractmethod

from bac.algorithms.common import MaskedReplayBuffer, sample_gaussian
from bac.algorithms.policy import Policy
from bac.algorithms.vectorized_networks import MultiHeadQNetwork, MultiHeadPolicyNetwork

class MultiHeadActorCritic(ABC):
    @abstractmethod
    def update(self, replay_buffer: MaskedReplayBuffer, steps: int, gamma: float):
        pass

    @abstractmethod
    def get_optimal_policy(self) -> Policy:
        pass

    @abstractmethod
    def get_exploration_policy(self) -> Policy:
        pass

    @abstractmethod
    def get_critic_network(self) -> MultiHeadQNetwork:
        pass

    @abstractmethod
    def get_actor_network(self) -> MultiHeadPolicyNetwork:
        pass

    @abstractmethod
    def get_n_heads(self) -> int:
        pass

# class OptimalPolicy(Policy):
#     def __init__(self, policy_net: MultiHeadPolicyNetwork):
#         self.policy_net = policy_net

#     def action(self, state: torch.Tensor) -> torch.Tensor:
#         if state.ndim == 1:
#             state = state.unsqueeze(0)

#         all_actions = self.policy_net(state)
#         mean_action = all_actions.mean(dim=1)

#         if mean_action.shape[0] == 1:
#             return mean_action.squeeze(0)
        
#         return mean_action

class OptimalPolicy(Policy):
    def __init__(self, policy_net: MultiHeadPolicyNetwork, q_net: MultiHeadQNetwork):
        self.policy_net = policy_net
        self.q_net = q_net

    @torch.no_grad()
    def action(self, state: torch.Tensor) -> torch.Tensor:
        single = False
        if state.ndim == 1:
            state = state.unsqueeze(0)
            single = True

        A_heads = self.policy_net(state)
        B, H_act, A_dim = A_heads.shape

        # evaluate each candidate under all critic heads
        S_rep  = state.repeat_interleave(H_act, dim=0)
        A_flat = A_heads.reshape(B * H_act, A_dim)
        Q_flat = self.q_net(S_rep, A_flat).squeeze(-1)
        scores = Q_flat.mean(dim=1).view(B, H_act)

        best_idx    = scores.argmax(dim=1)
        best_action = A_heads[torch.arange(B, device=A_heads.device), best_idx]
        return best_action.squeeze(0) if single else best_action


class SampledPolicy(Policy):
    def __init__(self, policy_net: MultiHeadPolicyNetwork, head_idx: int):
        self.policy_net = policy_net
        self.head_idx = head_idx

    def action(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            squeeze_back = False
            if state.ndim == 1:
                state = state.unsqueeze(0)
                squeeze_back = True

            all_actions = self.policy_net(state)
            a = all_actions[:, self.head_idx, :]

            return a.squeeze(0) if squeeze_back else a

class NoisyPolicy(Policy):
    def __init__(
        self,
        policy_net: MultiHeadPolicyNetwork,
        noise: float,
        action_min: torch.Tensor,
        action_max: torch.Tensor,
    ):
        self.policy_net = policy_net
        self.noise = noise
        self.action_min = action_min
        self.action_max = action_max

    def action(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            squeeze_back = False
            if state.ndim == 1:
                state = state.unsqueeze(0)
                squeeze_back = True

            all_actions = self.policy_net(state)
            a = all_actions[:, 0, :]

            noise = sample_gaussian(0.0, self.noise, a.shape, device=a.device)
            a = a + noise
            
            a = torch.clamp(a, self.action_min, self.action_max)

            return a.squeeze(0) if squeeze_back else a