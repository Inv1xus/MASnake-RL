"""
ppo_snake_epiplexity.py  (True Epiplexity — Coupled IDM)
========================================================
Recurrent PPO stabilised by sequence BPTT, augmented with a coupled Inverse 
Dynamics Model (IDM) to enforce causal structural representations.

Fixes Implemented:
1. IDM Restored: A 2-step IDM predicts `action_t` from `(h_t, h_{t+1})`.
2. Gradient Coupling: Features are NO LONGER detached. The IDM cross-entropy 
   loss flows backward through the GRU and the CNN. This forces the vision 
   system to extract action-relevant spatial features rather than generic 
   auto-encoding features, breaking the GRU optimization bottleneck.
"""

import copy
import random
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.amp import GradScaler, autocast
from collections import deque
from dataclasses import dataclass, asdict, fields
from typing import Optional

from async_vec_env import C, META_DIM, get_cached_vec_env, _REWARD_KEYS
from ppo_snake_core import layer_init, _gae_jit, PPOConfig


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EpiplexityConfig(PPOConfig):
    gru_hidden:        int   = 512
    bptt_seq_len:      int   = 16
    feat_dim:          int   = 256   

    # IDM Coefficient for auxiliary task
    idm_coef:          float = 0.1

    entropy_floor:     float = 0.10
    entropy_boost:     float = 0.02

    @classmethod
    def from_dict(cls, data: dict):
        valid = {f.name for f in fields(cls)}
        src   = data.get("hyperparameters", data)
        return cls(**{k: v for k, v in src.items() if k in valid})


def _init_gru(cell: nn.GRUCell) -> None:
    """Initialize GRU weights with stable orthogonal defaults."""
    nn.init.orthogonal_(cell.weight_ih)
    nn.init.orthogonal_(cell.weight_hh)
    nn.init.zeros_(cell.bias_ih)
    nn.init.zeros_(cell.bias_hh)


# ─────────────────────────────────────────────────────────────────────────────
# Network
# ─────────────────────────────────────────────────────────────────────────────

class RecurrentActorCritic(nn.Module):
    """Recurrent actor-critic with auxiliary heads for Epiplexity training."""
    def __init__(
        self,
        grid_h:         int = 18,
        grid_w:         int = 24,
        in_channels:    int = 8,
        scalar_dim:     int = 10,
        hidden_dim:     int = 256,
        gru_hidden:     int = 512,
        feat_dim:       int = 256,
    ):
        super().__init__()
        self.gru_hidden = gru_hidden
        self.feat_dim = feat_dim

        # ── Shared CNN & Scalar Extractors ──
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 64, 3, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 128, 3, padding=1)),          nn.ReLU(),
            layer_init(nn.Conv2d(128, 128, 3, stride=2, padding=1)), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            cnn_raw = self.cnn(torch.zeros(1, in_channels, grid_h, grid_w)).shape[1]

        self.scalar_net = nn.Sequential(layer_init(nn.Linear(scalar_dim, 64)), nn.ReLU())
        
        # ── Bottleneck (Preserves Spatial Structure) ──
        self.pre_norm = nn.LayerNorm(cnn_raw + 64)
        self.compress = nn.Sequential(
            layer_init(nn.Linear(cnn_raw + 64, feat_dim)), nn.ReLU()
        )

        # ── GRU Memory ──
        self.gru = nn.GRUCell(self.feat_dim, gru_hidden)
        _init_gru(self.gru)

        # ── PPO Heads (Skip Connection) ──
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(self.feat_dim + gru_hidden, hidden_dim)), nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(),
        )
        self.actor  = layer_init(nn.Linear(hidden_dim, 4), std=0.01)
        self.critic = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

        # ── IDM Head (Epiplexity Auxiliary) ──
        self.idm_head = nn.Sequential(
            layer_init(nn.Linear(gru_hidden * 2, 256)), nn.ReLU(),
            layer_init(nn.Linear(256, 4), std=0.01)
        )

    def extract_features(self, grid: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.cnn(grid), self.scalar_net(scalars)], dim=-1)
        x = self.pre_norm(x)
        return self.compress(x)

    def gru_step(self, feat: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.gru(feat, h.to(feat.dtype))

    def heads_from_feat_and_h(self, feat: torch.Tensor, h: torch.Tensor):
        t = self.trunk(torch.cat([feat, h], dim=-1))
        return self.actor(t), self.critic(t).squeeze(-1)

    def get_action_and_value(self, grid, scalars, h, action=None):
        feat  = self.extract_features(grid, scalars)
        h_new = self.gru_step(feat, h)
        logits, value = self.heads_from_feat_and_h(feat, h_new)
        dist = Categorical(logits=logits)
        if action is None: action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value, h_new

    def idm_logits(self, h_t: torch.Tensor, h_t1: torch.Tensor) -> torch.Tensor:
        """Predicts the action taken between h_t and h_t1."""
        return self.idm_head(torch.cat([h_t, h_t1], dim=-1))

    def get_initial_state(self, batch_size: int, device) -> torch.Tensor:
        return torch.zeros(batch_size, self.gru_hidden, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class EpiplexityTrainer:
    """Collect recurrent rollouts and optimize PPO + auxiliary losses."""
    def __init__(self, config: EpiplexityConfig, device: torch.device):
        self.config = config
        self.device = device
        H, W, N, T, D, GH = config.grid_height, config.grid_width, config.num_envs, config.rollout_steps, config.scalar_dim, config.gru_hidden
        S = T // config.bptt_seq_len

        self._use_amp = config.use_amp and device.type == "cuda"
        self._amp_dtype = torch.bfloat16 if self._use_amp and torch.cuda.is_bf16_supported() else torch.float16
        self._scaler = GradScaler("cuda") if self._use_amp and self._amp_dtype == torch.float16 else None

        cfg_dict = asdict(config)
        cfg_dict["env_device"] = device
        reward_kw = {k: cfg_dict[k] for k in _REWARD_KEYS if k in cfg_dict}
        self.vec_env = get_cached_vec_env(reward_kwargs=reward_kw, **cfg_dict)
        self._is_torch_env = getattr(self.vec_env, "is_torch_env", False)

        net_kw = dict(grid_h=H, grid_w=W, in_channels=C, scalar_dim=D, gru_hidden=GH, feat_dim=config.feat_dim)
        self._raw_network  = RecurrentActorCritic(**net_kw).to(device)
        self._raw_opponent = RecurrentActorCritic(**net_kw).to(device)
        self.network = torch.compile(self._raw_network) if config.use_compile else self._raw_network
        self.opponent_network = torch.compile(self._raw_opponent) if config.use_compile else self._raw_opponent

        # Optimizer covers base model + IDM head
        p_critic = list(self._raw_network.critic.parameters())
        p_rest   = [p for n, p in self._raw_network.named_parameters() if not n.startswith("critic")]
        self.optimizer = optim.Adam([{"params": p_rest, "lr": config.lr}, {"params": p_critic, "lr": config.lr * 2.0}], eps=1e-5)
        
        self.opponent_pool = deque([copy.deepcopy(self._raw_network.state_dict())], maxlen=config.opponent_pool_size)
        self.gru_h, self.opp_gru_h = torch.zeros(N, 2, GH, device=device), torch.zeros(N, 2, GH, device=device)

        self._pin_grids = torch.zeros(N, 2, C, H, W, dtype=torch.float32).pin_memory()
        self._pin_meta  = torch.zeros(N, 2, META_DIM).pin_memory()
        self._pin_rews  = torch.zeros(N, 2).pin_memory()
        self._pin_dones = torch.zeros(N, 3, dtype=torch.uint8).pin_memory()
        self._pin_acts  = torch.zeros(N, 2, dtype=torch.int32).pin_memory()

        kw = dict(device=device)
        self.rb_grids, self.rb_scalars, self.rb_actions = torch.zeros(T, N, 2, C, H, W, **kw), torch.zeros(T, N, 2, D, **kw), torch.zeros(T, N, 2, dtype=torch.long, **kw)
        self.rb_log_probs, self.rb_values = torch.zeros(T, N, 2, **kw), torch.zeros(T, N, 2, **kw)
        self.rb_rewards, self.rb_dones, self.rb_env_dones = torch.zeros(T, N, 2, **kw), torch.zeros(T, N, 2, **kw), torch.zeros(T, N, dtype=torch.bool, **kw)
        self.rb_h_seq_starts = torch.zeros(S + 1, N, 2, GH, **kw)

        self.ep_rews, self.ep_lens = deque(maxlen=100), deque(maxlen=100)
        self._ep_accum, self._ep_steps = torch.zeros(N, 2, **kw), torch.zeros(N, dtype=torch.int32, **kw)

        self.total_steps, self.update_count = 0, 0
        self._ent_coef = config.entropy_coef
        self._ent_ema = math.log(4)
        self._dynamic_ent_boost = 0.0

    def enable_tracking(self, progress_cb=None):
        self._metrics = {"steps": [], "reward": [], "pi_loss": [], "v_loss": [], "entropy": [], "idm_loss": [], "ep_len": [], "sps": [], "ent_coef": []}
        self._metrics_t0, self._metrics_step0, self._progress_cb = time.time(), self.total_steps, progress_cb

    def train_chunk(self, target_steps: int) -> float:
        N, T, H, W, D, GH, L = self.config.num_envs, self.config.rollout_steps, self.config.grid_height, self.config.grid_width, self.config.scalar_dim, self.config.gru_hidden, self.config.bptt_seq_len

        while self.total_steps < target_steps:
            self.network.eval()
            self._raw_opponent.load_state_dict(random.choice(list(self.opponent_pool)))
            l_mask = torch.rand(N, 2, device=self.device) < self.config.opponent_pool_frac
            self.rb_h_seq_starts[0].copy_(self.gru_h)

            for t in range(T):
                gs, ss = self._shm_to_gpu()
                with torch.no_grad(), autocast("cuda", enabled=self._use_amp, dtype=self._amp_dtype):
                    a, lp, _, v, h_new_flat = self.network.get_action_and_value(gs.view(-1, C, H, W), ss.view(-1, D), self.gru_h.view(-1, GH))
                    a, lp, v, h_new = a.view(N, 2), lp.view(N, 2), v.view(N, 2), h_new_flat.view(N, 2, GH)
                    o_a, _, _, _, oh_new_flat = self.opponent_network.get_action_and_value(gs.view(-1, C, H, W), ss.view(-1, D), self.opp_gru_h.view(-1, GH))
                    o_a, oh_new = o_a.view(N, 2), oh_new_flat.view(N, 2, GH)

                acts = torch.where(l_mask, a, o_a)
                self.rb_grids[t], self.rb_scalars[t], self.rb_actions[t], self.rb_log_probs[t], self.rb_values[t] = gs, ss, acts, lp, v

                if self._is_torch_env:
                    self.vec_env.send_actions(acts); self.vec_env.recv_results()
                    rews, dones = self.vec_env.torch_rews.clone(), self.vec_env.torch_dones.float()
                else:
                    self._pin_acts.copy_(acts.cpu().to(torch.int32)); self.vec_env.send_actions(self._pin_acts.numpy()); self.vec_env.recv_results()
                    np.copyto(self._pin_rews.numpy(), self.vec_env.np_rews); np.copyto(self._pin_dones.numpy(), self.vec_env.np_dones)
                    rews, dones = self._pin_rews.to(self.device, non_blocking=True), self._pin_dones.to(self.device, non_blocking=True).float()

                self.rb_rewards[t], self.rb_dones[t] = rews, dones[:, :2]
                env_done = dones[:, 2].bool()
                self.rb_env_dones[t] = env_done

                if env_done.any():
                    h_new = h_new.clone(); h_new[env_done] = 0.0
                    oh_new = oh_new.clone(); oh_new[env_done] = 0.0

                self.gru_h, self.opp_gru_h = h_new.float(), oh_new.float()
                self.total_steps += int(l_mask.sum())
                self._ep_accum += rews; self._ep_steps += 1

                if env_done.any():
                    self.ep_rews.extend(self._ep_accum[env_done, 0].tolist()); self.ep_lens.extend(self._ep_steps[env_done].tolist())
                    self._ep_accum[env_done], self._ep_steps[env_done] = 0.0, 0

                if (t + 1) < T and (t + 1) % L == 0:
                    self.rb_h_seq_starts[(t + 1) // L].copy_(self.gru_h)

            with torch.no_grad():
                ng, ns = self._shm_to_gpu()
                _, _, _, lv, _ = self.network.get_action_and_value(ng.view(-1, C, H, W), ns.view(-1, D), self.gru_h.view(-1, GH))
                lv = lv.view(N, 2).detach()

            adv, ret = _gae_jit(self.rb_values.detach(), self.rb_rewards, self.rb_dones, lv, self.config.gamma, self.config.gae_lambda)

            progress = min(self.total_steps / self.config.total_timesteps, 1.0)
            base_ent = self.config.entropy_coef * (1.0 - progress) + self.config.entropy_coef_final * progress
            if self._ent_ema < self.config.entropy_floor:
                self._dynamic_ent_boost += self.config.entropy_boost
            else:
                self._dynamic_ent_boost *= 0.9
            self._ent_coef = base_ent + self._dynamic_ent_boost

            # ── 1. Sequence BPTT PPO Update ──
            pi_l, v_l, ent_l = self._ppo_update(adv, ret, l_mask)
            self._ent_ema = 0.95 * self._ent_ema + 0.05 * ent_l

            # ── 2. Epiplexity IDM Update ──
            idm_l = 0.0
            if self.config.idm_coef > 0.0:
                idm_l = self._idm_update()

            self.update_count += 1
            if self.update_count % self.config.opponent_pool_update_freq == 0:
                self.opponent_pool.append(copy.deepcopy(self._raw_network.state_dict()))

            if hasattr(self, "_metrics"):
                dt = max(time.time() - self._metrics_t0, 1e-3); m = self._metrics
                m["steps"].append(self.total_steps); m["reward"].append(float(np.mean(self.ep_rews)) if self.ep_rews else 0.0)
                m["pi_loss"].append(pi_l); m["v_loss"].append(v_l); m["entropy"].append(ent_l); m["idm_loss"].append(idm_l)
                m["ep_len"].append(float(np.mean(self.ep_lens)) if self.ep_lens else 0.0)
                m["sps"].append((self.total_steps - self._metrics_step0) / dt); m["ent_coef"].append(self._ent_coef)
                if self._progress_cb: self._progress_cb(self)

        return float(np.mean(self.ep_rews)) if self.ep_rews else 0.0

    def _ppo_update(self, adv, ret, l_mask):
        T, N, L, S, GH = self.config.rollout_steps, self.config.num_envs, self.config.bptt_seq_len, self.config.rollout_steps // self.config.bptt_seq_len, self.config.gru_hidden
        l_n, l_a = l_mask.nonzero(as_tuple=True)
        K = l_n.shape[0]

        def _lrn(buf): return buf[:, l_n, l_a]
        def _to_seqs(t):
            rest = t.shape[2:]
            return t.view(S, L, K, *rest).permute(0, 2, 1, *range(3, 3 + len(rest))).reshape(S * K, L, *rest)

        adv_K = _lrn(adv)
        adv_K = (adv_K - adv_K.mean()) / (adv_K.std() + 1e-8)

        gs_seq, ss_seq, acts_seq = _to_seqs(_lrn(self.rb_grids)), _to_seqs(_lrn(self.rb_scalars)), _to_seqs(_lrn(self.rb_actions))
        lp_old_seq, v_old_seq, adv_seq, ret_seq = _to_seqs(_lrn(self.rb_log_probs)), _to_seqs(_lrn(self.rb_values)), _to_seqs(adv_K), _to_seqs(_lrn(ret))
        done_seq = _to_seqs(self.rb_env_dones[:, l_n].unsqueeze(-1)).squeeze(-1).bool()
        h_seq_starts = self.rb_h_seq_starts[:S, l_n, l_a].reshape(S * K, GH).detach()

        n_seqs, mb_seqs = S * K, max(1, self.config.minibatch_size // L)
        self.network.train()
        pi_sum, v_sum, ent_sum, n_mb = 0.0, 0.0, 0.0, 0

        for _ in range(self.config.update_epochs):
            perm = torch.randperm(n_seqs, device=self.device)
            for start in range(0, n_seqs, mb_seqs):
                mb = perm[start : start + mb_seqs]
                if len(mb) == 0: continue
                pi_l, v_l, ent_l = self._bptt_step(gs_seq[mb], ss_seq[mb], acts_seq[mb], lp_old_seq[mb], v_old_seq[mb], adv_seq[mb], ret_seq[mb], h_seq_starts[mb], done_seq[mb])
                pi_sum += pi_l; v_sum += v_l; ent_sum += ent_l; n_mb += 1

        return pi_sum / max(n_mb, 1), v_sum / max(n_mb, 1), ent_sum / max(n_mb, 1)

    def _bptt_step(self, b_gs, b_ss, b_acts, b_lp_old, b_v_old, b_adv, b_ret, h0, b_done):
        K, L = b_gs.shape[:2]
        
        with autocast("cuda", enabled=self._use_amp, dtype=self._amp_dtype):
            # ── 1. VECTORIZED CNN EXTRACTION ──
            # Flatten Batch and Time dimensions to saturate the GPU in one pass
            flat_gs = b_gs.reshape(K * L, *b_gs.shape[2:])
            flat_ss = b_ss.reshape(K * L, *b_ss.shape[2:])
            
            flat_feats = self._raw_network.extract_features(flat_gs, flat_ss)
            feats = flat_feats.view(K, L, -1) # Reshape back to [Batch, SeqLen, FeatDim]

            h = h0.to(self._amp_dtype) if self._use_amp else h0
            pi_loss_t = torch.zeros(K, L, device=self.device)
            v_loss_t  = torch.zeros(K, L, device=self.device)
            ent_t     = torch.zeros(K, L, device=self.device)

            # ── 2. SEQUENTIAL GRU UNROLLING ──
            for t in range(L):
                h = self._raw_network.gru_step(feats[:, t], h)
                logits, value = self._raw_network.heads_from_feat_and_h(feats[:, t], h)
                
                dist = Categorical(logits=logits)
                lp = dist.log_prob(b_acts[:, t])
                ent_t[:, t] = dist.entropy()
                
                ratio = torch.exp(lp.float() - b_lp_old[:, t])
                pi_loss_t[:, t] = -torch.min(
                    ratio * b_adv[:, t], 
                    ratio.clamp(1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps) * b_adv[:, t]
                )
                
                v_f = value.float()
                v_clipped = b_v_old[:, t] + (v_f - b_v_old[:, t]).clamp(-self.config.clip_eps, self.config.clip_eps)
                v_loss_t[:, t] = 0.5 * torch.max((v_f - b_ret[:, t]) ** 2, (v_clipped - b_ret[:, t]) ** 2)
                
                if t < L - 1 and b_done[:, t].any():
                    h = h * (~b_done[:, t]).float().unsqueeze(-1).to(h.dtype)

            pi_loss, v_loss, ent_ = pi_loss_t.mean(), v_loss_t.mean(), ent_t.mean()
            loss = pi_loss + self.config.value_loss_coef * v_loss - self._ent_coef * ent_

        self.optimizer.zero_grad(set_to_none=True)
        if self._scaler:
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self._raw_network.parameters(), self.config.max_grad_norm)
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self._raw_network.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

        return pi_loss.item(), v_loss.item(), ent_.item()

    def _idm_update(self):
        T, N = self.config.rollout_steps, self.config.num_envs
        IDM_BATCH = min(512, T * N // 4)

        valid_t = torch.where(~self.rb_env_dones)[0]
        valid_t = valid_t[(valid_t > 0) & (valid_t < T - 1)]
        if len(valid_t) == 0: 
            return 0.0

        si = torch.randint(0, len(valid_t), (IDM_BATCH,), device=self.device)
        t_idx = valid_t[si]
        n_idx = torch.randint(0, N, (IDM_BATCH,), device=self.device)
        a_idx = torch.randint(0, 2, (IDM_BATCH,), device=self.device)

        g_t   = self.rb_grids[t_idx, n_idx, a_idx]
        g_t1  = self.rb_grids[t_idx + 1, n_idx, a_idx]
        s_t   = self.rb_scalars[t_idx, n_idx, a_idx]
        s_t1  = self.rb_scalars[t_idx + 1, n_idx, a_idx]
        act_t = self.rb_actions[t_idx, n_idx, a_idx]

        h_prev = torch.zeros(IDM_BATCH, self.config.gru_hidden, device=self.device)

        self.network.train()
        with autocast("cuda", enabled=self._use_amp, dtype=self._amp_dtype):
            # ── 1. VECTORIZED CNN EXTRACTION ──
            g_both = torch.cat([g_t, g_t1], dim=0)
            s_both = torch.cat([s_t, s_t1], dim=0)
            
            feat_both = self._raw_network.extract_features(g_both, s_both)
            feat_t, feat_t1 = feat_both.chunk(2, dim=0)

            # ── 2. GRU STEP ──
            h_t  = self._raw_network.gru_step(feat_t, h_prev)
            h_t1 = self._raw_network.gru_step(feat_t1, h_t)

            logits = self._raw_network.idm_logits(h_t, h_t1)
            idm_loss = F.cross_entropy(logits.float(), act_t)
            loss = self.config.idm_coef * idm_loss

        self.optimizer.zero_grad(set_to_none=True)
        if self._scaler:
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self._raw_network.parameters(), self.config.max_grad_norm)
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self._raw_network.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

        return idm_loss.item()

    def _shm_to_gpu(self):
        if self._is_torch_env:
            grids, meta = self.vec_env.torch_grids, self.vec_env.torch_meta
        else:
            np.copyto(self._pin_grids.numpy(), self.vec_env.np_grids)
            np.copyto(self._pin_meta.numpy(),  self.vec_env.np_meta)
            grids, meta = self._pin_grids.to(self.device, non_blocking=True).div_(255.0), self._pin_meta.to(self.device,  non_blocking=True)

        dirs = meta[:, :, 0].long().unsqueeze(-1)
        d_oh = torch.zeros(meta.shape[0], 2, 4, device=self.device).scatter_(2, dirs, 1.0)
        walls = torch.stack([meta[:, :, 3] / 23, (23 - meta[:, :, 3]) / 23, meta[:, :, 4] / 17, (17 - meta[:, :, 4]) / 17], dim=-1)
        
        # Removed the / 50.0 normalization on speed_credit (index 2:3)
        scalars = torch.cat([d_oh, meta[:, :, 1:2], meta[:, :, 2:3], walls], dim=-1)
        return grids.contiguous(), scalars.contiguous()

    def save_state(self, path: str): torch.save({"network": self._raw_network.state_dict(), "optimizer": self.optimizer.state_dict(), "config_dict": asdict(self.config), "total_steps": self.total_steps, "opponent_pool": self.opponent_pool, "gru_h": self.gru_h, "ent_ema": self._ent_ema, "dynamic_ent_boost": self._dynamic_ent_boost}, path)
    def load_state(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._raw_network.load_state_dict(ckpt["network"]); self.optimizer.load_state_dict(ckpt["optimizer"]); self.total_steps, self.opponent_pool = ckpt["total_steps"], ckpt["opponent_pool"]
        if "gru_h" in ckpt: self.gru_h.copy_(ckpt["gru_h"])
        if "ent_ema" in ckpt: self._ent_ema = ckpt["ent_ema"]
        if "dynamic_ent_boost" in ckpt: self._dynamic_ent_boost = ckpt["dynamic_ent_boost"]
    def close(self):
        if hasattr(self, "vec_env") and self.vec_env is not None: self.vec_env.close()