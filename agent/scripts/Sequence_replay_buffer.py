import random
import numpy as np
import torch
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class SequenceReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []        # list of episodes; each episode = list of Transition
        self.current_episode = []
        self.size = 0

    def __len__(self):
        return self.size

    # start episodes
    def start_episode(self):
        if len(self.current_episode) > 0:
            self.finish_episode()
        self.current_episode = []

    # safe episode in memory, control overflow
    def finish_episode(self):
        if not self.current_episode:
            return

        # append episode, evict the oldest episodes if exceeding capacity (by transitions)
        ep_len = len(self.current_episode)
        self.buffer.append(self.current_episode)
        self.size += ep_len

        # evict
        while self.size > self.capacity:
            removed = self.buffer.pop(0)
            self.size -= removed
        self.current_episode = []

    def push(self, state, action, reward, next_state, done):
        # push Transition into current episode
        self.current_episode.append(Transition(state, action, reward, next_state, done))
        if done:
            self.finish_episode()

    def sample(self,  batch_size, seq_len):
        """
        sample batch_size sequences of length seq_len.
        Returns tensors: states (B, S, obs_dim), actions (B, S), rewards (B, S),
        next_states (B, S, obs_dim), dones (B, S)
        """

        # collect eligible episodes (length >= seq_leq)
        elig_eps = [ep for ep in self.buffer if len(ep) >= seq_len]
        if len(elig_eps) == 0:
            return None

        batch = []
        for i in range(batch_size):
            ep = random.choice(elig_eps)
            start = random.randint(0, len(ep) - seq_len)
            seq = ep[start:start + seq_len]
            batch.append(seq)

        # convert to arrays
        # assume state as numpy array or torch; handle both
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for seq in batch:
            s = np.stack([t.state for t in seq])
            a = np.array([t.action for t in seq], dtype=np.int64)
            r = np.array([t.reward for t in seq], dtype=np.float32)
            ns = np.stack([t.next_state for t in seq])
            d = np.array([t.done for t in seq], dtype=np.float32)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)

        # convert to tensors
        states = torch.tensor(np.stack(states, axis=0), dtype=torch.float32)        # (B, S, obs)
        actions = torch.tensor(np.stack(actions, axis=0), dtype=torch.long)         # (B, S)
        rewards = torch.tensor(np.stack(rewards, axis=0), dtype=torch.float32)      # (B, S)
        next_states = torch.tensor(np.stack(next_states, axis=0), dtype=torch.float32)  # (B, S, obs)
        dones = torch.tensor(np.stack(dones, axis=0), dtype=torch.float32)          # (B, S)

        return states, actions, rewards, next_states, dones
