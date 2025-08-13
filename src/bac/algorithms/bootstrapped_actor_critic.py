import torch
from tqdm import tqdm
from typing import List, Optional

from bac.environments import Environment
from bac.algorithms.common import ReplayBuffer, ObserverType
from bac.algorithms.actor_critic import ActorCritic
from bac.algorithms.policy import Policy, RandomPolicy

class BootstrappedActorCritic:
    def __init__(self, env: Environment, ensemble: List[ActorCritic], device: torch.device = torch.device("cpu")):
        self.env = env
        self.ensemble = ensemble
        self.mean_policy = MeanPolicy(ensemble, device)
        self.random_policy = RandomPolicy(env.action_shape(), env.action_min(), env.action_max())
        self.device = device

    def train(
        self,
        p_mask: float,
        steps: int = 200_000,
        start_steps: int = 10_000,
        update_every: int = 50,
        gamma: float = 0.99,
        observer: Optional[ObserverType] = None
    ):
        replay_buffers = [ReplayBuffer(1_000_000, self.env.state_shape(), self.env.action_shape()) for _ in range(len(self.ensemble))]
        policy = self.sample_policy()
        state = self.env.reset()

        for step in tqdm(range(steps)):
            if step < start_steps:
                action = self.random_policy.action(state)
            else:
                action = policy.action(state)
                
            
            next_state, reward, term, trunc, _ = self.env.step(action)
            done = term or trunc

            mask = torch.bernoulli(torch.full((len(self.ensemble), 1), p_mask))
            for i in range(len(self.ensemble)):
                if mask[i].item() == 1:
                    replay_buffers[i].add(state, action, reward, next_state, term)

            if done:
                # we've reached the end of an episode
                # resample policy
                policy = self.sample_policy()
                # get new initial state
                state = self.env.reset()
            else:
                state = next_state

            if step % update_every == 0:
                for i, actor in enumerate(self.ensemble):
                    actor.update(replay_buffers[i], update_every, gamma)
            
            if observer is not None:
                observer(step, self.mean_policy)
        
            

    def sample_policy(self) -> Policy:
        idx = torch.randint(0, len(self.ensemble), (1,)).item()
        actor = self.ensemble[idx]

        return actor.get_optimal_policy()

    def get_mean_policy(self) -> Policy:
        return self.mean_policy

class MeanPolicy(Policy):
    def __init__(self, ensemble: List[ActorCritic], device: torch.device):
        self.ensemble = ensemble
        self.device = device
    
    def action(self, state) -> torch.Tensor:
        action_list = [actor.get_optimal_policy().action(state) for actor in self.ensemble]
        action_tensor = torch.stack(action_list, dim=0).to(self.device)
        return action_tensor.mean(dim=0)