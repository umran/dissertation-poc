from actor_critic import ActorCritic
from util.replay_buffer import ReplayBuffer
from policy import Policy

class DDPG(ActorCritic):
    def __init__(self):
        pass
    
    def update(self, step: int, replay_buffer: ReplayBuffer):
        pass

    def get_optimal_policy(self) -> Policy:
        pass

    def get_exploratory_policy(self) -> Policy:
        pass