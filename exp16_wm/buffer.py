"""
Replay buffers for Exp-16.

ReplayBuffer          : vanilla circular buffer (numpy), used by VanillaSAC.
TrajectoryReplayBuffer: extends with h_real storage, used by WMSAC.
"""

import numpy as np


class ReplayBuffer:
    """
    Circular numpy replay buffer.

    Stores: (obs, action, reward, next_obs, done)
    All fields are float32.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        capacity: int = 1_000_000,
    ):
        self.capacity   = capacity
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.ptr        = 0
        self.size       = 0

        self.obs      = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.action   = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward   = np.zeros((capacity, 1),          dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.done     = np.zeros((capacity, 1),          dtype=np.float32)
        self.timeout  = np.zeros((capacity, 1),          dtype=np.float32)

    def add(
        self,
        obs:      np.ndarray,
        action:   np.ndarray,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
        timeout:  bool = False,
    ):
        i = self.ptr
        self.obs[i]      = obs
        self.action[i]   = action
        self.reward[i]   = reward
        self.next_obs[i] = next_obs
        self.done[i]     = float(done)
        self.timeout[i]  = float(timeout)

        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(
        self,
        obs:      np.ndarray,   # (n, obs_dim)
        action:   np.ndarray,   # (n, action_dim)
        reward:   np.ndarray,   # (n,)
        next_obs: np.ndarray,   # (n, obs_dim)
        done:     np.ndarray,   # (n,)  bool
        timeout:  np.ndarray,   # (n,)  bool
    ):
        n    = len(obs)
        idxs = np.arange(self.ptr, self.ptr + n) % self.capacity
        self.obs[idxs]      = obs
        self.action[idxs]   = action
        self.reward[idxs]   = reward.reshape(-1, 1)
        self.next_obs[idxs] = next_obs
        self.done[idxs]     = done.reshape(-1, 1).astype(np.float32)
        self.timeout[idxs]  = timeout.reshape(-1, 1).astype(np.float32)
        self.ptr  = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int) -> dict:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs":      self.obs[idx],
            "action":   self.action[idx],
            "reward":   self.reward[idx],
            "next_obs": self.next_obs[idx],
            # SB3-style: truncated episodes are not treated as terminal
            "done":     self.done[idx] * (1 - self.timeout[idx]),
        }

    def __len__(self):
        return self.size


class TrajectoryReplayBuffer(ReplayBuffer):
    """
    Extends ReplayBuffer with h_real storage for WM supervision.

    h_real  : (B, max_events, event_dim)  — real physics trajectory
    traj_len: (B, 1)                       — number of valid events (<= max_events)
    """

    def __init__(
        self,
        obs_dim:    int,
        action_dim: int,
        max_events: int,
        event_dim:  int,
        capacity:   int = 1_000_000,
    ):
        super().__init__(obs_dim, action_dim, capacity)
        self.max_events = max_events
        self.event_dim  = event_dim

        self.h_real   = np.zeros((capacity, max_events, event_dim), dtype=np.float32)
        self.traj_len = np.zeros((capacity, 1),                     dtype=np.int32)

    def add(
        self,
        obs:      np.ndarray,
        action:   np.ndarray,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
        timeout:  bool = False,
        h_real:   np.ndarray = None,
        traj_len: int = 0,
    ):
        i = self.ptr
        # pad h_real to (max_events, event_dim) if needed
        padded = np.zeros((self.max_events, self.event_dim), dtype=np.float32)
        if h_real is not None:
            n = min(traj_len, self.max_events)
            padded[:n] = h_real[:n]

        self.h_real[i]   = padded
        self.traj_len[i] = traj_len

        super().add(obs, action, reward, next_obs, done, timeout=timeout)

    def sample(self, batch_size: int) -> dict:
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = {
            "obs":      self.obs[idx],
            "action":   self.action[idx],
            "reward":   self.reward[idx],
            "next_obs": self.next_obs[idx],
            "done":     self.done[idx] * (1 - self.timeout[idx]),
            "h_real":   self.h_real[idx],
            "traj_len": self.traj_len[idx],
        }
        return batch
