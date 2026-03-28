"""
SAC agents for Exp-16.

VanillaSAC : standard TanhGaussian SAC (twin Q).
WMSAC      : inherits VanillaSAC, overrides only update_critic to add WM loss.
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F

from .networks import Actor, Critic, WorldModelCritic


# ──────────────────────────────────────────────
# VanillaSAC
# ──────────────────────────────────────────────

class VanillaSAC:
    """
    Soft Actor-Critic with TanhGaussian actor and twin Q-critic.

    Matches SB3 SAC defaults:
      lr=3e-4, tau=0.005, gamma=0.99, target_entropy=-action_dim
      automatic entropy tuning (log_alpha, learned)
    """

    def __init__(
        self,
        obs_dim:    int,
        action_dim: int,
        act_low:    np.ndarray,
        act_high:   np.ndarray,
        hidden:     list[int] = [256, 256],
        lr:         float     = 3e-4,
        tau:        float     = 0.005,
        gamma:      float     = 0.99,
        device:     str       = "cpu",
    ):
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.tau        = tau
        self.gamma      = gamma
        self.device     = torch.device(device)

        # ── networks ──────────────────────────────
        self.actor  = Actor(obs_dim, action_dim, act_low, act_high, hidden).to(self.device)
        self.critic = Critic(obs_dim, action_dim, hidden).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        # ── entropy coefficient ────────────────────
        self.target_entropy = -float(action_dim)
        self.log_alpha      = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha          = self.log_alpha.exp().item()

        # ── optimizers ────────────────────────────
        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.alpha_opt  = torch.optim.Adam([self.log_alpha],         lr=lr)

    # ── internal helpers ──────────────────────────

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        # as_tensor avoids copy when x is already float32 on CPU
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def _unpack(self, batch: dict):
        return (
            self._to_tensor(batch["obs"]),
            self._to_tensor(batch["action"]),
            self._to_tensor(batch["reward"]),
            self._to_tensor(batch["next_obs"]),
            self._to_tensor(batch["done"]),
        )

    # ── soft update ───────────────────────────────

    @torch.no_grad()
    def soft_update(self):
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.mul_(1 - self.tau).add_(p.data, alpha=self.tau)

    # ── update steps ─────────────────────────────

    def update_critic(self, obs, action, reward, next_obs, done) -> dict:
        with torch.no_grad():
            next_action, next_log_pi = self.actor(next_obs)
            q1_t, q2_t = self.critic_target(next_obs, next_action)
            q_target = torch.min(q1_t, q2_t) - self.alpha * next_log_pi
            y = reward + self.gamma * (1 - done) * q_target

        q1, q2 = self.critic(obs, action)
        critic_loss = 0.5 * (F.mse_loss(q1, y) + F.mse_loss(q2, y))

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        return {"critic_loss": critic_loss.item()}

    def update_actor(self, obs) -> tuple[dict, torch.Tensor]:
        action, log_pi = self.actor(obs)
        q_val = self.critic.q_min(obs, action)
        actor_loss = (self.alpha * log_pi - q_val).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        return {"actor_loss": actor_loss.item(), "log_pi": log_pi.mean().item()}, log_pi

    def update_alpha(self, log_pi: torch.Tensor) -> dict:
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy)).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self.alpha = self.log_alpha.exp().item()
        return {"alpha_loss": alpha_loss.item(), "alpha": self.alpha}

    def update(self, batch: dict) -> dict:
        """One full gradient step. Returns dict of scalar metrics."""
        # convert batch to tensors once — reused across all three updates
        obs, action, reward, next_obs, done = self._unpack(batch)

        c_info            = self.update_critic(obs, action, reward, next_obs, done)
        a_info, log_pi    = self.update_actor(obs)
        al_info           = self.update_alpha(log_pi.detach())  # reuse log_pi, no extra forward pass

        self.soft_update()

        return {**c_info, **a_info, **al_info}

    # ── inference ────────────────────────────────

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = self._to_tensor(obs[None])           # (1, obs_dim)
        return self.actor.act(obs_t, deterministic)[0]   # (action_dim,)

    def act_batch(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Vectorised action collection — single forward pass for all envs."""
        obs_t = self._to_tensor(obs)                 # (n_envs, obs_dim)
        return self.actor.act(obs_t, deterministic)  # (n_envs, action_dim)

    def evaluate(self, env, n_episodes: int = 10):
        """
        Roll out n_episodes, return (mean_reward, std_reward, pocket_rate).
        Assumes 1-step episodes with info['pocketed'].
        """
        rewards, pocketed = [], []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            ep_pocketed = False
            while not done:
                action = self.act(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward   += reward
                ep_pocketed  = info.get("pocketed", False)
            rewards.append(ep_reward)
            pocketed.append(float(ep_pocketed))

        return (
            float(np.mean(rewards)),
            float(np.std(rewards)),
            float(np.mean(pocketed)),
        )

    # ── save / load ───────────────────────────────

    def save(self, path: str):
        torch.save({
            "actor":        self.actor.state_dict(),
            "critic":       self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha":    self.log_alpha.detach().cpu(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))
        self.alpha = self.log_alpha.exp().item()


# ──────────────────────────────────────────────
# WMSAC  (World Model SAC)
# ──────────────────────────────────────────────

class WMSAC(VanillaSAC):
    """
    World Model SAC.

    Changes from VanillaSAC:
      - critic replaced by WorldModelCritic (M1+q1, M2+q2)
      - update_critic adds MSE(ĥ, h_real) world-model loss
      - buffer must supply "h_real" key (use TrajectoryReplayBuffer)

    All other methods (update_actor, update_alpha, soft_update, act, evaluate)
    are inherited unchanged.
    """

    def __init__(
        self,
        obs_dim:    int,
        action_dim: int,
        act_low:    np.ndarray,
        act_high:   np.ndarray,
        max_events: int,
        event_dim:  int,
        actor_hidden:  list[int] = [256, 256],
        wm_hidden:     list[int] = [512, 512, 512],
        q_hidden:      list[int] = [256, 256],
        lr:            float     = 3e-4,
        tau:           float     = 0.005,
        gamma:         float     = 0.99,
        wm_coef:       float     = 1.0,    # weight of WM loss relative to Bellman
        device:        str       = "cpu",
    ):
        # call parent __init__ but we'll replace the critic immediately after
        super().__init__(
            obs_dim, action_dim, act_low, act_high,
            hidden=actor_hidden, lr=lr, tau=tau, gamma=gamma, device=device,
        )

        self.wm_coef    = wm_coef
        self.max_events = max_events
        self.event_dim  = event_dim

        # ── replace critic with WorldModelCritic ──
        self.critic = WorldModelCritic(
            obs_dim, action_dim, max_events, event_dim, wm_hidden, q_hidden,
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        # reset critic optimizer to point at new params
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

    # ── override update_critic only ───────────────

    def update_critic(self, obs, action, reward, next_obs, done, h_real=None) -> dict:
        # h_real passed in as tensor (pre-converted in update)
        with torch.no_grad():
            next_action, next_log_pi = self.actor(next_obs)
            q1_t, q2_t, _, _ = self.critic_target(next_obs, next_action)
            q_target = torch.min(q1_t, q2_t) - self.alpha * next_log_pi
            y = reward + self.gamma * (1 - done) * q_target

        q1, q2, h1_hat, h2_hat = self.critic(obs, action)

        bellman_loss = 0.5 * (F.mse_loss(q1, y) + F.mse_loss(q2, y))
        wm_loss      = 0.5 * (F.mse_loss(h1_hat, h_real) + F.mse_loss(h2_hat, h_real))
        critic_loss  = bellman_loss + self.wm_coef * wm_loss

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        return {
            "critic_loss":  critic_loss.item(),
            "bellman_loss": bellman_loss.item(),
            "wm_loss":      wm_loss.item(),
        }

    def update(self, batch: dict) -> dict:
        obs, action, reward, next_obs, done = self._unpack(batch)
        h_real  = self._to_tensor(batch["h_real"])

        c_info          = self.update_critic(obs, action, reward, next_obs, done, h_real=h_real)
        a_info, log_pi  = self.update_actor(obs)
        al_info         = self.update_alpha(log_pi.detach())

        self.soft_update()

        return {**c_info, **a_info, **al_info}

    # ── save / load ───────────────────────────────

    def save(self, path: str):
        torch.save({
            "actor":         self.actor.state_dict(),
            "critic":        self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha":     self.log_alpha.detach().cpu(),
        }, path)
