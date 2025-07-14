import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.common import new_observer, plot_benchmark
from algorithms.ddpg import DDPG
from algorithms.hybrid_hmc import HybridHMC
from environments.hopper import Hopper

STATE_DIM = 11
ACTION_DIM = 3

class QNetwork(nn.Module):
    def __init__(self, hidden_sizes=[32, 32]):
        super(QNetwork, self).__init__()
        input_dim = STATE_DIM + ACTION_DIM
        self.fc1 = nn.Linear(input_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.out = nn.Linear(hidden_sizes[1], 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return self.out(x)

class PolicyNetwork(nn.Module):
    def __init__(self, hidden_sizes=[32, 32]):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(STATE_DIM, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.out = nn.Linear(hidden_sizes[1], ACTION_DIM)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.out(x))
        
        return action

def demo():
    observer, bench_results = new_observer(Hopper())
    ddpg = DDPG(QNetwork, PolicyNetwork)
    hybrid_hmc = HybridHMC(Hopper(), ddpg, PolicyNetwork, observer=observer)

    hybrid_hmc.train()
    plot_benchmark(bench_results)

    optimized_policy = ddpg.get_optimal_policy()

    demo_env = Hopper(render_mode="human")
    state = demo_env.reset()

    while True:
        action = optimized_policy.action(state)

        print(action.shape, " ", action)

        next_state, _, terminated, truncated, _ = demo_env.step(action)
        if terminated or truncated:
            state = demo_env.reset()
        else:
            state = next_state

demo()