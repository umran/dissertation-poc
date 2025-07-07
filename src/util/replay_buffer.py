import torch

class ReplayBuffer:
    def __init__(self, 
        capacity: int,
        state_shape: torch.Size, 
        action_shape: torch.Size, 
        state_dtype: torch.dtype = torch.float32, 
        action_dtype: torch.dtype = torch.float32, 
        device: torch.device = 'cpu'
    ):
        self.capacity = capacity
        self.device = device
        self.state_dtype = state_dtype
        self.action_dtype = action_dtype
        self.ptr = 0
        self.size = 0

        self.states = torch.zeros((capacity, *state_shape), dtype=state_dtype, device=device)
        self.actions = torch.zeros((capacity, *action_shape), dtype=action_dtype, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, *state_shape), dtype=state_dtype, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.bool, device=device)

    def add(self, state, action, reward, next_state, done):
        self.add_batch(
            torch.tensor([state], dtype=self.state_dtype, device=self.device),
            torch.tensor([action], dtype=self.action_dtype, device=self.device),
            torch.tensor([[reward]], dtype=torch.float32, device=self.device),
            torch.tensor([next_state], dtype=self.state_dtype, device=self.device),
            torch.tensor([[done]], dtype=torch.bool, device=self.device)
        )

    def add_batch(self, states, actions, rewards, next_states, dones):
        batch_size = states.shape[0]
        assert states.shape == (batch_size, *self.states.shape[1:])
        assert actions.shape == (batch_size, *self.actions.shape[1:])
        assert rewards.shape == (batch_size, 1)
        assert next_states.shape == (batch_size, *self.next_states.shape[1:])
        assert dones.shape == (batch_size, 1)

        end = self.ptr + batch_size
        if end <= self.capacity:
            self.states[self.ptr:end] = states
            self.actions[self.ptr:end] = actions
            self.rewards[self.ptr:end] = rewards
            self.next_states[self.ptr:end] = next_states
            self.dones[self.ptr:end] = dones
        else:
            first = self.capacity - self.ptr
            second = batch_size - first
            self.states[self.ptr:] = states[:first]
            self.actions[self.ptr:] = actions[:first]
            self.rewards[self.ptr:] = rewards[:first]
            self.next_states[self.ptr:] = next_states[:first]
            self.dones[self.ptr:] = dones[:first]

            self.states[:second] = states[first:]
            self.actions[:second] = actions[first:]
            self.rewards[:second] = rewards[first:]
            self.next_states[:second] = next_states[first:]
            self.dones[:second] = dones[first:]

        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx]
        )

    def __len__(self):
        return self.size
