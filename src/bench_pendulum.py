import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.common import new_observer, plot_multiple_benchmarks
from algorithms.vanilla_actor_critic import VanillaActorCritic
from algorithms.ddpg import DDPG
from algorithms.hybrid_hmc import HybridHMC
from environments.pendulum import Pendulum

# STATE_DIM = 3
# ACTION_DIM = 1

# class QNetwork(nn.Module):
#     def __init__(self, hidden_sizes=[32, 32]):
#         super(QNetwork, self).__init__()
#         input_dim = STATE_DIM + ACTION_DIM
#         self.fc1 = nn.Linear(input_dim, hidden_sizes[0])
#         self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
#         self.out = nn.Linear(hidden_sizes[1], 1)

#     def forward(self, state, action):
#         x = torch.cat([state, action], dim=-1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
        
#         return self.out(x)

# class PolicyNetwork(nn.Module):
#     def __init__(self, hidden_sizes=[32, 32]):
#         super(PolicyNetwork, self).__init__()
#         self.fc1 = nn.Linear(STATE_DIM, hidden_sizes[0])
#         self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
#         self.out = nn.Linear(hidden_sizes[1], ACTION_DIM)

#     def forward(self, state):
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         action = 2 * torch.tanh(self.out(x))
        
#         return action

def bench_pendulum():
    random_ddpg = VanillaActorCritic(Pendulum(), DDPG(Pendulum()))
    random_ddpg_observer, random_ddpg_results = new_observer(Pendulum())
    random_ddpg.train(start_steps=200_000, observer=random_ddpg_observer)

    vanilla_ddpg = VanillaActorCritic(Pendulum(), DDPG(Pendulum()))
    vanilla_ddpg_observer, vanilla_ddpg_results = new_observer(Pendulum())
    vanilla_ddpg.train(observer=vanilla_ddpg_observer)

    hybrid_hmc = HybridHMC(Pendulum(), DDPG(Pendulum()))
    hybrid_hmc_observer, hybrid_hmc_results = new_observer(Pendulum())
    hybrid_hmc.train(observer=hybrid_hmc_observer)

    plot_multiple_benchmarks(
        [
            (random_ddpg_results, "DDPG Random Exploration"),
            (vanilla_ddpg_results, "Vanilla DDPG"),
            (hybrid_hmc_results, "Hybrid HMC")
        ],
        colors=["red", "blue", "green"]
    )

    return [
        random_ddpg.actor_critic.get_optimal_policy(),
        vanilla_ddpg.actor_critic.get_optimal_policy(),
        hybrid_hmc.actor_critic.get_optimal_policy()
    ]