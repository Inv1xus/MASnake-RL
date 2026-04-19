"""
ppo_snake_core.py (Base PPO — POMDP A/B Test Control)
=====================================================
The feedforward baseline.
- Clean Math: Grid division and scalar division bugs removed.
- POMDP: Scalars blinded, 7x7 Fog of War enforced.
"""

import copy
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.amp import GradScaler, autocast
from collections import deque
from dataclasses import dataclass, asdict, fields
from typing import Optional

from snake_env import MultiAgentSnakeEnv
from async_vec_env import C, META_DIM, get_cached_vec_env, _REWARD_KEYS

@torch.jit.script
def _gae_jit(rb_v, rb_r, rb_d, last_v, gamma: float, gae_lambda: float):
    """Compute GAE advantages/returns for batched PPO rollouts."""
    nxt_v = torch.cat([rb_v[1:], last_v.unsqueeze(0)], dim=0)
    not_done = 1.0 - rb_d
    deltas = rb_r + gamma * nxt_v * not_done - rb_v
    coeffs = gamma * gae_lambda * not_done
    advantages = torch.zeros_like(rb_v)
    last_gae = torch.zeros_like(last_v)
    for t in range(rb_v.shape[0] - 1, -1, -1):
        last_gae = deltas[t] + coeffs[t] * last_gae
        advantages[t] = last_gae
    return advantages, advantages + rb_v

def layer_init(layer, std=np.sqrt(2)):
    """Apply orthogonal init used across policy/value heads."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, 0.0)
    return layer

class ActorCritic(nn.Module):
    """Feedforward actor-critic used as the POMDP control baseline."""
    def __init__(self, grid_h=18, grid_w=24, in_channels=8, scalar_dim=10, hidden_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 64, 3, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 128, 3, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(128, 128, 3, stride=2, padding=1)), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat_dim = self.cnn(torch.zeros(1, in_channels, grid_h, grid_w)).shape[1]

        self.scalar_net = nn.Sequential(layer_init(nn.Linear(scalar_dim, 64)), nn.ReLU())
        self.trunk = nn.Sequential(
            nn.LayerNorm(flat_dim + 64),
            layer_init(nn.Linear(flat_dim + 64, hidden_dim)), nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(hidden_dim, 4), std=0.01)
        self.critic = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def forward(self, grid, scalars):
        x = torch.cat([self.cnn(grid), self.scalar_net(scalars)], dim=-1)
        h = self.trunk(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def get_action_and_value(self, grid, scalars, action=None):
        logits, value = self.forward(grid, scalars)
        dist = Categorical(logits=logits)
        if action is None: action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

@dataclass
class PPOConfig:
    grid_width:               int   = 24
    grid_height:              int   = 18
    in_channels:              int   = 8
    scalar_dim:               int   = 10
    total_timesteps:          int   = 150_000_000
    max_steps_per_episode:    int   = 1_000
    rollout_steps:            int   = 1_024
    num_envs:                 int   = 64
    num_workers:              Optional[int] = 8
    seed:                     int   = 42

    lr:                       float = 8e-5
    lr_min:                   float = 1e-5
    gamma:                    float = 0.995
    gae_lambda:               float = 0.95
    clip_eps:                 float = 0.2
    value_loss_coef:          float = 0.5
    entropy_coef:             float = 0.15
    entropy_coef_final:       float = 0.01
    max_grad_norm:            float = 0.5
    update_epochs:            int   = 4
    minibatch_size:           int   = 4096

    opponent_pool_size:       int   = 20
    opponent_pool_update_freq:int   = 15
    opponent_pool_frac:       float = 0.5
    opponent_epsilon:         float = 0.0

    survival_reward:          float = 0.01
    food_reward:              float = 5.0
    death_penalty:            float = -5.0
    win_reward:               float = 3.0
    distance_shaping:         float = 0.15

    use_compile:              bool  = True
    use_amp:                  bool  = True
    env_backend:              str   = "async"
    torch_env_compile:        bool  = False
    compile_thread_cap:       int   = 4
    reward_norm_warmup_steps: int   = 128
    speed_mode:               bool  = True

    @classmethod
    def from_dict(cls, data: dict):
        valid_keys = {f.name for f in fields(cls)}
        config_data = data.get("hyperparameters", data)
        return cls(**{k: v for k, v in config_data.items() if k in valid_keys})

class StatefulSnakeTrainer:
    """Trainer implementing rollout collection and PPO optimization."""
    def __init__(self, config: PPOConfig, device: torch.device):
        self.config, self.device = config, device
        H, W, N, T, D = config.grid_height, config.grid_width, config.num_envs, config.rollout_steps, config.scalar_dim

        self._use_amp = config.use_amp and device.type == "cuda"
        self._amp_dtype = torch.bfloat16 if self._use_amp and torch.cuda.is_bf16_supported() else torch.float16
        self._scaler = GradScaler("cuda") if self._use_amp and self._amp_dtype == torch.float16 else None

        cfg_dict  = asdict(config)
        cfg_dict["env_device"] = device
        reward_kw = {k: cfg_dict[k] for k in _REWARD_KEYS if k in cfg_dict}
        self.vec_env = get_cached_vec_env(reward_kwargs = reward_kw, **cfg_dict)
        self._is_torch_env = getattr(self.vec_env, "is_torch_env", False)

        self._raw_network = ActorCritic(H, W).to(device)
        self.network = torch.compile(self._raw_network) if config.use_compile else self._raw_network
        self._raw_opponent = ActorCritic(H, W).to(device)
        self.opponent_network = torch.compile(self._raw_opponent) if config.use_compile else self._raw_opponent

        p_actor = [p for n, p in self._raw_network.named_parameters() if "critic" not in n]
        self.optimizer = optim.Adam([{"params": p_actor, "lr": config.lr}, {"params": self._raw_network.critic.parameters(), "lr": config.lr * 2.0}], eps=1e-5)
        self.opponent_pool = deque([copy.deepcopy(self._raw_network.state_dict())], maxlen=config.opponent_pool_size)

        self._pin_grids = torch.zeros(N, 2, C, H, W, dtype=torch.float32).pin_memory()
        self._pin_meta  = torch.zeros(N, 2, META_DIM).pin_memory()
        self._pin_rews  = torch.zeros(N, 2).pin_memory()
        self._pin_dones = torch.zeros(N, 3, dtype=torch.uint8).pin_memory()
        self._pin_acts  = torch.zeros(N, 2, dtype=torch.int32).pin_memory()

        kw = dict(device=device)
        self.rb_grids = torch.zeros(T, N, 2, C, H, W, **kw)
        self.rb_scalars = torch.zeros(T, N, 2, D, **kw)
        self.rb_actions = torch.zeros(T, N, 2, dtype=torch.long, **kw)
        self.rb_log_probs = torch.zeros(T, N, 2, **kw)
        self.rb_values = torch.zeros(T, N, 2, **kw)
        self.rb_rewards = torch.zeros(T, N, 2, **kw)
        self.rb_dones = torch.zeros(T, N, 2, **kw)

        self.ep_rews, self.ep_lens = deque(maxlen=100), deque(maxlen=100)
        self._ep_accum, self._ep_steps = torch.zeros(N, 2, **kw), torch.zeros(N, dtype=torch.int32, **kw)
        self.total_steps, self.update_count, self._ent_coef = 0, 0, config.entropy_coef

    def enable_tracking(self, progress_cb=None):
        self._metrics = {"steps":[], "reward":[], "pi_loss":[], "v_loss":[], "entropy":[], "ep_len":[], "sps":[], "ent_coef":[]}
        self._metrics_t0, self._metrics_step0, self._progress_cb = time.time(), self.total_steps, progress_cb

    def _shm_to_gpu(self):
        if self._is_torch_env:
            grids, meta = self.vec_env.torch_grids, self.vec_env.torch_meta
        else:
            np.copyto(self._pin_grids.numpy(), self.vec_env.np_grids)
            np.copyto(self._pin_meta.numpy(),  self.vec_env.np_meta)
            # CLEAN MATH: Division bug removed
            grids = self._pin_grids.to(self.device, non_blocking=True)
            meta  = self._pin_meta.to(self.device,  non_blocking=True)

        N, A, C, H, W = grids.shape

        # ── 1. BLINDED SCALARS ──
        d_oh   = torch.zeros(N, 2, 4, device=self.device)
        speed  = torch.zeros(N, 2, 1, device=self.device)
        credit = torch.zeros(N, 2, 1, device=self.device)
        
        hx = meta[:, :, 3]
        hy = meta[:, :, 4]
        walls = torch.stack([hx / 23.0, (23.0 - hx) / 23.0, hy / 17.0, (17.0 - hy) / 17.0], dim=-1)
        scalars = torch.cat([d_oh, speed, credit, walls], dim=-1)

        # ── 2. FOG OF WAR (Spatial Masking) ──
        y_grid, x_grid = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device), indexing='ij')
        hx_exp, hy_exp = hx.view(N, 2, 1, 1), hy.view(N, 2, 1, 1)
        dist_x = torch.abs(x_grid.view(1, 1, H, W) - hx_exp)
        dist_y = torch.abs(y_grid.view(1, 1, H, W) - hy_exp)
        
        vision_mask = (dist_x <= 3) & (dist_y <= 3)
        grids = grids * vision_mask.unsqueeze(2).float()

        return grids.contiguous(), scalars.contiguous()

    def train_chunk(self, target_steps: int) -> float:
        N, T = self.config.num_envs, self.config.rollout_steps
        while self.total_steps < target_steps:
            self.network.eval()
            self._raw_opponent.load_state_dict(random.choice(list(self.opponent_pool)))
            l_mask = torch.rand(N, 2, device=self.device) > 0.5 

            for t in range(T):
                gs, ss = self._shm_to_gpu()
                with torch.no_grad(), autocast("cuda", enabled=self._use_amp, dtype=self._amp_dtype):
                    a, lp, _, v = self.network.get_action_and_value(gs.view(-1,C,18,24), ss.view(-1,10))
                    o_a, _, _, _ = self.opponent_network.get_action_and_value(gs.view(-1,C,18,24), ss.view(-1,10))
                
                acts = torch.where(l_mask.view(-1), a, o_a).view(N, 2)
                self.rb_grids[t], self.rb_scalars[t], self.rb_actions[t] = gs, ss, acts
                self.rb_log_probs[t], self.rb_values[t] = lp.view(N, 2), v.view(N, 2)

                if self._is_torch_env:
                    self.vec_env.send_actions(acts)
                    self.vec_env.recv_results()
                    rews, dones = self.vec_env.torch_rews.clone(), self.vec_env.torch_dones.float()
                else:
                    self._pin_acts.copy_(acts.cpu().to(torch.int32))
                    self.vec_env.send_actions(self._pin_acts.numpy())
                    self.vec_env.recv_results()
                    np.copyto(self._pin_rews.numpy(),  self.vec_env.np_rews)
                    np.copyto(self._pin_dones.numpy(), self.vec_env.np_dones)
                    rews  = self._pin_rews.to(self.device,  non_blocking=True)
                    dones = self._pin_dones.to(self.device, non_blocking=True).float()

                self.rb_rewards[t], self.rb_dones[t] = rews, dones[:, :2]
                self.total_steps += int(l_mask.sum())
                self._ep_accum += rews; self._ep_steps += 1
                
                if dones[:, 2].any():
                    idx = dones[:, 2].bool()
                    self.ep_rews.extend(self._ep_accum[idx, 0].tolist())
                    self.ep_lens.extend(self._ep_steps[idx].tolist())
                    self._ep_accum[idx], self._ep_steps[idx] = 0, 0

            with torch.no_grad():
                ng, ns = self._shm_to_gpu()
                _, _, _, lv = self.network.get_action_and_value(ng.view(-1,C,18,24), ns.view(-1,10))
                lv = lv.view(N, 2).detach()

            adv, ret = _gae_jit(self.rb_values.detach(), self.rb_rewards, self.rb_dones, lv, self.config.gamma, self.config.gae_lambda)
            progress = min(self.total_steps / self.config.total_timesteps, 1.0)
            self._ent_coef = self.config.entropy_coef * (1.0 - progress) + self.config.entropy_coef_final * progress
            
            pi_l, v_l, ent_l = self._ppo_update(
                self.rb_grids.permute(1,2,0,3,4,5)[l_mask].reshape(-1,C,18,24).detach(),
                self.rb_scalars.permute(1,2,0,3)[l_mask].reshape(-1,10).detach(),
                self.rb_actions.permute(1,2,0)[l_mask].reshape(-1).detach(),
                self.rb_log_probs.permute(1,2,0)[l_mask].reshape(-1).detach(),
                adv.permute(1,2,0)[l_mask].reshape(-1).detach(),
                ret.permute(1,2,0)[l_mask].reshape(-1).detach()
            )
            
            self.update_count += 1
            if self.update_count % self.config.opponent_pool_update_freq == 0:
                self.opponent_pool.append(copy.deepcopy(self._raw_network.state_dict()))

            if hasattr(self, "_metrics"):
                dt = max(time.time() - self._metrics_t0, 1e-3)
                self._metrics["steps"].append(self.total_steps); self._metrics["reward"].append(np.mean(self.ep_rews) if self.ep_rews else 0.0)
                self._metrics["pi_loss"].append(pi_l); self._metrics["v_loss"].append(v_l); self._metrics["entropy"].append(ent_l)
                self._metrics["ep_len"].append(np.mean(self.ep_lens) if self.ep_lens else 0.0)
                self._metrics["sps"].append((self.total_steps - self._metrics_step0)/dt); self._metrics["ent_coef"].append(self._ent_coef)
                if self._progress_cb: self._progress_cb(self)

        return float(np.mean(self.ep_rews)) if self.ep_rews else 0.0

    def _ppo_update(self, gs, ss, acts, lp_o, adv, ret):
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.network.train(); pi_s, v_s, ent_s, n_mb = 0, 0, 0, 0
        idx = torch.randperm(len(gs), device=self.device)
        
        for _ in range(self.config.update_epochs):
            for s in range(0, len(idx), self.config.minibatch_size):
                mb = idx[s : s + self.config.minibatch_size]
                with autocast("cuda", enabled=self._use_amp, dtype=self._amp_dtype):
                    _, lp, ent, v = self.network.get_action_and_value(gs[mb], ss[mb], acts[mb])
                    ratio = torch.exp(lp.float() - lp_o[mb])
                    pi_loss = -torch.min(ratio*adv[mb], ratio.clamp(1-self.config.clip_eps, 1+self.config.clip_eps)*adv[mb]).mean()
                    v_loss = 0.5 * ((v.float() - ret[mb])**2).mean()
                    loss = pi_loss + self.config.value_loss_coef * v_loss - self._ent_coef * ent.mean()
                
                self.optimizer.zero_grad(set_to_none=True)
                if self._scaler:
                    self._scaler.scale(loss).backward()
                    self._scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self._raw_network.parameters(), self.config.max_grad_norm)
                    self._scaler.step(self.optimizer); self._scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self._raw_network.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                pi_s += pi_loss.item(); v_s += v_loss.item(); ent_s += ent.mean().item(); n_mb += 1
        return pi_s/n_mb, v_s/n_mb, ent_s/n_mb

    def save_state(self, path): torch.save({"network": self._raw_network.state_dict(), "optimizer": self.optimizer.state_dict(), "config_dict": asdict(self.config), "total_steps": self.total_steps, "opponent_pool": self.opponent_pool}, path)
    def load_state(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._raw_network.load_state_dict(ckpt["network"]); self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt["total_steps"]; self.opponent_pool = ckpt["opponent_pool"]
    def close(self):
        if hasattr(self, "vec_env") and self.vec_env is not None: 
            self.vec_env.close()