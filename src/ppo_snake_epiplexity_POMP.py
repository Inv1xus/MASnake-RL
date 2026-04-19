"""
Recurrent PPO with IDM and opponent tracking for partially observable Snake.

Scalar inputs are blinded and a 7x7 fog of war masks the spatial observation.
An auxiliary head trains the GRU to predict the opponent's coordinates so the
recurrent state stays informative even when the opponent is out of view.
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


@dataclass
class EpiplexityConfig(PPOConfig):
    gru_hidden: int = 512
    bptt_seq_len: int = 16
    feat_dim: int = 256
    idm_coef: float = 0.1
    entropy_floor: float = 0.10
    entropy_boost: float = 0.02
    aux_loss_coef: float = 0.5

    @classmethod
    def from_dict(cls, data: dict):
        """Builds an EpiplexityConfig from a dict, ignoring unknown keys."""
        valid = {f.name for f in fields(cls)}
        src = data.get("hyperparameters", data)
        return cls(**{k: v for k, v in src.items() if k in valid})


def _init_gru(cell: nn.GRUCell) -> None:
    """Initializes GRU weights with orthogonal values and zero bias."""
    nn.init.orthogonal_(cell.weight_ih)
    nn.init.orthogonal_(cell.weight_hh)
    nn.init.zeros_(cell.bias_ih)
    nn.init.zeros_(cell.bias_hh)


class RecurrentActorCritic(nn.Module):
    """Recurrent policy and value network with IDM and opponent tracking auxiliary heads."""

    def __init__(
        self,
        grid_h: int = 18,
        grid_w: int = 24,
        in_channels: int = 8,
        scalar_dim: int = 10,
        hidden_dim: int = 256,
        gru_hidden: int = 512,
        feat_dim: int = 256,
    ):
        """Builds the CNN, scalar encoder, GRU, PPO heads, tracking head, and IDM head."""
        super().__init__()
        self.gru_hidden = gru_hidden
        self.feat_dim = feat_dim

        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 64, 3, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 128, 3, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(128, 128, 3, stride=2, padding=1)), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            cnn_raw = self.cnn(
                torch.zeros(
                    1,
                    in_channels,
                    grid_h,
                    grid_w)).shape[1]

        self.scalar_net = nn.Sequential(
            layer_init(
                nn.Linear(
                    scalar_dim,
                    64)),
            nn.ReLU())
        self.pre_norm = nn.LayerNorm(cnn_raw + 64)
        self.compress = nn.Sequential(
            layer_init(
                nn.Linear(
                    cnn_raw + 64,
                    feat_dim)),
            nn.ReLU())

        self.gru = nn.GRUCell(self.feat_dim, gru_hidden)
        _init_gru(self.gru)

        self.trunk = nn.Sequential(
            layer_init(
                nn.Linear(
                    self.feat_dim +
                    gru_hidden,
                    hidden_dim)),
            nn.ReLU(),
            layer_init(
                nn.Linear(
                    hidden_dim,
                    hidden_dim)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(hidden_dim, 4), std=0.01)
        self.critic = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

        # Auxiliary head predicts the opponent's absolute (X, Y) grid
        # coordinates
        self.aux_head = layer_init(nn.Linear(gru_hidden, 2), std=0.01)

        self.idm_head = nn.Sequential(
            layer_init(nn.Linear(gru_hidden * 2, 256)), nn.ReLU(),
            layer_init(nn.Linear(256, 4), std=0.01)
        )

    def extract_features(
            self,
            grid: torch.Tensor,
            scalars: torch.Tensor) -> torch.Tensor:
        """Passes the grid and scalars through encoders and returns a compressed feature vector."""
        x = torch.cat([self.cnn(grid), self.scalar_net(scalars)], dim=-1)
        x = self.pre_norm(x)
        return self.compress(x)

    def gru_step(self, feat: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Runs one GRU cell step and returns the updated hidden state."""
        return self.gru(feat, h.to(feat.dtype))

    def heads_from_feat_and_h(self, feat: torch.Tensor, h: torch.Tensor):
        """Returns action logits, value estimate, and the auxiliary tracking prediction."""
        t = self.trunk(torch.cat([feat, h], dim=-1))
        return self.actor(t), self.critic(t).squeeze(-1), self.aux_head(h)

    def get_action_and_value(self, grid, scalars, h, action=None):
        """Extracts features, steps the GRU, then samples or evaluates an action."""
        feat = self.extract_features(grid, scalars)
        h_new = self.gru_step(feat, h)
        logits, value, _ = self.heads_from_feat_and_h(feat, h_new)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value, h_new

    def idm_logits(
            self,
            h_t: torch.Tensor,
            h_t1: torch.Tensor) -> torch.Tensor:
        """Predicts the action taken between hidden states h_t and h_t1."""
        return self.idm_head(torch.cat([h_t, h_t1], dim=-1))

    def get_initial_state(self, batch_size: int, device) -> torch.Tensor:
        """Returns a zeroed initial GRU hidden state for the given batch size."""
        return torch.zeros(batch_size, self.gru_hidden, device=device)


class EpiplexityTrainer:
    """Trainer for recurrent PPO with IDM and auxiliary opponent tracking objectives."""

    def __init__(self, config: EpiplexityConfig, device: torch.device):
        """Sets up the recurrent network, optimizer, environment, and all rollout buffers."""
        self.config = config
        self.device = device
        H, W, N, T, D, GH = config.grid_height, config.grid_width, config.num_envs, config.rollout_steps, config.scalar_dim, config.gru_hidden
        S = T // config.bptt_seq_len

        self._use_amp = config.use_amp and device.type == "cuda"
        self._amp_dtype = torch.bfloat16 if self._use_amp and torch.cuda.is_bf16_supported() else torch.float16
        self._scaler = GradScaler(
            "cuda") if self._use_amp and self._amp_dtype == torch.float16 else None

        cfg_dict = asdict(config)
        cfg_dict["env_device"] = device
        reward_kw = {k: cfg_dict[k] for k in _REWARD_KEYS if k in cfg_dict}
        self.vec_env = get_cached_vec_env(reward_kwargs=reward_kw, **cfg_dict)
        self._is_torch_env = getattr(self.vec_env, "is_torch_env", False)

        net_kw = dict(
            grid_h=H,
            grid_w=W,
            in_channels=C,
            scalar_dim=D,
            gru_hidden=GH,
            feat_dim=config.feat_dim)
        self._raw_network = RecurrentActorCritic(**net_kw).to(device)
        self._raw_opponent = RecurrentActorCritic(**net_kw).to(device)
        self.network = torch.compile(
            self._raw_network) if config.use_compile else self._raw_network
        self.opponent_network = torch.compile(
            self._raw_opponent) if config.use_compile else self._raw_opponent

        p_critic = list(self._raw_network.critic.parameters())
        p_rest = [p for n, p in self._raw_network.named_parameters()
                  if not n.startswith("critic")]
        # The critic gets twice the learning rate because it has a harder job than the policy.
        # The policy just needs to rank actions; the critic needs to fit a
        # precise value function.
        self.optimizer = optim.Adam([{"params": p_rest, "lr": config.lr}, {
                                    "params": p_critic, "lr": config.lr * 2.0}], eps=1e-5)

        self.opponent_pool = deque([copy.deepcopy(
            self._raw_network.state_dict())], maxlen=config.opponent_pool_size)
        self.gru_h, self.opp_gru_h = torch.zeros(
            N, 2, GH, device=device), torch.zeros(
            N, 2, GH, device=device)

        self._pin_grids = torch.zeros(
            N, 2, C, H, W, dtype=torch.float32).pin_memory()
        self._pin_meta = torch.zeros(N, 2, META_DIM).pin_memory()
        self._pin_rews = torch.zeros(N, 2).pin_memory()
        self._pin_dones = torch.zeros(N, 3, dtype=torch.uint8).pin_memory()
        self._pin_acts = torch.zeros(N, 2, dtype=torch.int32).pin_memory()

        kw = dict(device=device)
        self.rb_grids, self.rb_scalars, self.rb_actions = torch.zeros(
            T, N, 2, C, H, W, **kw), torch.zeros(T, N, 2, D, **kw), torch.zeros(T, N, 2, dtype=torch.long, **kw)
        self.rb_log_probs, self.rb_values = torch.zeros(
            T, N, 2, **kw), torch.zeros(T, N, 2, **kw)
        self.rb_rewards, self.rb_dones, self.rb_env_dones = torch.zeros(
            T, N, 2, **kw), torch.zeros(T, N, 2, **kw), torch.zeros(T, N, dtype=torch.bool, **kw)
        # We snapshot the GRU state at the start of every BPTT window during the rollout.
        # Without this, recomputing gradients during the update would use stale hidden states
        # from the previous iteration rather than the ones that actually
        # generated the data.
        self.rb_h_seq_starts = torch.zeros(S + 1, N, 2, GH, **kw)

        # The opponent coordinates recorded here come from the actual game state, not the network.
        # They serve as ground truth targets for the auxiliary tracking loss so the GRU is forced
        # to remember where the opponent is even when it cannot see them
        # through the fog.
        self.rb_opp_coords = torch.zeros(T, N, 2, 2, **kw)

        self.ep_rews, self.ep_lens = deque(maxlen=100), deque(maxlen=100)
        self._ep_accum, self._ep_steps = torch.zeros(
            N, 2, **kw), torch.zeros(N, dtype=torch.int32, **kw)

        self.total_steps, self.update_count = 0, 0
        self._ent_coef = config.entropy_coef
        self._ent_ema = math.log(4)
        self._dynamic_ent_boost = 0.0

    def enable_tracking(self, progress_cb=None):
        """Enables metric collection and optionally registers a progress callback."""
        self._metrics = {
            "steps": [],
            "reward": [],
            "pi_loss": [],
            "v_loss": [],
            "entropy": [],
            "idm_loss": [],
            "aux_loss": [],
            "ep_len": [],
            "sps": [],
            "ent_coef": []}
        self._metrics_t0, self._metrics_step0, self._progress_cb = time.time(
        ), self.total_steps, progress_cb

    def _shm_to_gpu(self):
        """Copies observations to GPU and applies blinded scalars, fog of war, and extracts opponent coordinates."""
        if self._is_torch_env:
            grids, meta = self.vec_env.torch_grids, self.vec_env.torch_meta
        else:
            np.copyto(self._pin_grids.numpy(), self.vec_env.np_grids)
            np.copyto(self._pin_meta.numpy(), self.vec_env.np_meta)
            grids = self._pin_grids.to(self.device, non_blocking=True)
            meta = self._pin_meta.to(self.device, non_blocking=True)

        N, A, C, H, W = grids.shape

        # Direction, speed, and credit are all zeroed to enforce POMDP: the agent cannot
        # know its own speed stats directly and must infer them from what it
        # observes.
        d_oh = torch.zeros(N, 2, 4, device=self.device)
        speed = torch.zeros(N, 2, 1, device=self.device)
        credit = torch.zeros(N, 2, 1, device=self.device)

        hx = meta[:, :, 3]
        hy = meta[:, :, 4]
        # Wall distances are the one scalar signal we keep. They give the agent a sense of
        # position on the board without handing it anything about speed or
        # internal state.
        walls = torch.stack([hx / 23.0, (23.0 - hx) / 23.0,
                            hy / 17.0, (17.0 - hy) / 17.0], dim=-1)
        scalars = torch.cat([d_oh, speed, credit, walls], dim=-1)

        # Fog of war: anything more than 3 cells away in either axis is zeroed out,
        # giving each snake exactly a 7x7 window centered on its head.
        y_grid, x_grid = torch.meshgrid(
            torch.arange(
                H, device=self.device), torch.arange(
                W, device=self.device), indexing='ij')
        hx_exp, hy_exp = hx.view(N, 2, 1, 1), hy.view(N, 2, 1, 1)
        dist_x = torch.abs(x_grid.view(1, 1, H, W) - hx_exp)
        dist_y = torch.abs(y_grid.view(1, 1, H, W) - hy_exp)

        vision_mask = (dist_x <= 3) & (dist_y <= 3)
        grids = grids * vision_mask.unsqueeze(2).float()

        # Because there are exactly 2 agents, flipping the agent axis maps each entry
        # to its opponent's metadata without any extra indexing.
        opp_hx = hx.flip(dims=[1]) / 23.0
        opp_hy = hy.flip(dims=[1]) / 17.0
        opp_coords = torch.stack([opp_hx, opp_hy], dim=-1)

        return grids.contiguous(), scalars.contiguous(), opp_coords.contiguous()

    def train_chunk(self, target_steps: int) -> float:
        """Collects recurrent rollouts and updates the network until total_steps reaches target_steps."""
        N, T, H, W, D, GH, L = self.config.num_envs, self.config.rollout_steps, self.config.grid_height, self.config.grid_width, self.config.scalar_dim, self.config.gru_hidden, self.config.bptt_seq_len

        while self.total_steps < target_steps:
            self.network.eval()
            self._raw_opponent.load_state_dict(
                random.choice(list(self.opponent_pool)))
            # l_mask marks which (env, agent) slots play as the learner this rollout.
            # The rest play from a frozen snapshot in the opponent pool, giving the learner
            # a diverse set of opponents to improve against rather than just
            # itself.
            l_mask = torch.rand(
                N, 2, device=self.device) < self.config.opponent_pool_frac
            self.rb_h_seq_starts[0].copy_(self.gru_h)

            for t in range(T):
                gs, ss, opp_c = self._shm_to_gpu()
                with torch.no_grad(), autocast("cuda", enabled=self._use_amp, dtype=self._amp_dtype):
                    a, lp, _, v, h_new_flat = self.network.get_action_and_value(
                        gs.view(-1, C, H, W), ss.view(-1, D), self.gru_h.view(-1, GH))
                    a, lp, v, h_new = a.view(
                        N, 2), lp.view(
                        N, 2), v.view(
                        N, 2), h_new_flat.view(
                        N, 2, GH)
                    o_a, _, _, _, oh_new_flat = self.opponent_network.get_action_and_value(
                        gs.view(-1, C, H, W), ss.view(-1, D), self.opp_gru_h.view(-1, GH))
                    o_a, oh_new = o_a.view(N, 2), oh_new_flat.view(N, 2, GH)

                acts = torch.where(l_mask, a, o_a)

                self.rb_grids[t], self.rb_scalars[t], self.rb_actions[t], self.rb_log_probs[
                    t], self.rb_values[t], self.rb_opp_coords[t] = gs, ss, acts, lp, v, opp_c

                if self._is_torch_env:
                    self.vec_env.send_actions(acts)
                    self.vec_env.recv_results()
                    rews, dones = self.vec_env.torch_rews.clone(), self.vec_env.torch_dones.float()
                else:
                    self._pin_acts.copy_(acts.cpu().to(torch.int32))
                    self.vec_env.send_actions(self._pin_acts.numpy())
                    self.vec_env.recv_results()
                    np.copyto(self._pin_rews.numpy(), self.vec_env.np_rews)
                    np.copyto(self._pin_dones.numpy(), self.vec_env.np_dones)
                    rews, dones = self._pin_rews.to(
                        self.device, non_blocking=True), self._pin_dones.to(
                        self.device, non_blocking=True).float()

                self.rb_rewards[t], self.rb_dones[t] = rews, dones[:, :2]
                env_done = dones[:, 2].bool()
                self.rb_env_dones[t] = env_done

                # GRU state is zeroed for any env that just finished an episode.
                # Carrying memory across episode boundaries would corrupt the next episode's
                # hidden state with context that no longer applies.
                if env_done.any():
                    h_new = h_new.clone()
                    h_new[env_done] = 0.0
                    oh_new = oh_new.clone()
                    oh_new[env_done] = 0.0

                self.gru_h, self.opp_gru_h = h_new.float(), oh_new.float()
                # Only count steps where the learner was actually acting, not
                # the opponent.
                self.total_steps += int(l_mask.sum())
                self._ep_accum += rews
                self._ep_steps += 1

                if env_done.any():
                    self.ep_rews.extend(self._ep_accum[env_done, 0].tolist())
                    self.ep_lens.extend(self._ep_steps[env_done].tolist())
                    self._ep_accum[env_done], self._ep_steps[env_done] = 0.0, 0

                # Save the GRU state at the boundary between BPTT windows so we can
                # feed the correct starting state back in during the update
                # pass.
                if (t + 1) < T and (t + 1) % L == 0:
                    self.rb_h_seq_starts[(t + 1) // L].copy_(self.gru_h)

            with torch.no_grad():
                ng, ns, _ = self._shm_to_gpu()
                _, _, _, lv, _ = self.network.get_action_and_value(
                    ng.view(-1, C, H, W), ns.view(-1, D), self.gru_h.view(-1, GH))
                lv = lv.view(N, 2).detach()

            adv, ret = _gae_jit(self.rb_values.detach(
            ), self.rb_rewards, self.rb_dones, lv, self.config.gamma, self.config.gae_lambda)

            progress = min(self.total_steps / self.config.total_timesteps, 1.0)
            base_ent = self.config.entropy_coef * \
                (1.0 - progress) + self.config.entropy_coef_final * progress
            # Dynamic entropy boost acts like a thermostat: it grows when entropy drops too low
            # and decays toward zero when things are healthy. This prevents
            # premature collapse.
            if self._ent_ema < self.config.entropy_floor:
                self._dynamic_ent_boost += self.config.entropy_boost
            else:
                self._dynamic_ent_boost *= 0.9
            self._ent_coef = base_ent + self._dynamic_ent_boost

            pi_l, v_l, ent_l, aux_l = self._ppo_update(adv, ret, l_mask)
            self._ent_ema = 0.95 * self._ent_ema + 0.05 * ent_l

            idm_l = 0.0
            if self.config.idm_coef > 0.0:
                idm_l = self._idm_update()

            self.update_count += 1
            if self.update_count % self.config.opponent_pool_update_freq == 0:
                self.opponent_pool.append(
                    copy.deepcopy(
                        self._raw_network.state_dict()))

            if hasattr(self, "_metrics"):
                dt = max(time.time() - self._metrics_t0, 1e-3)
                m = self._metrics
                m["steps"].append(self.total_steps)
                m["reward"].append(float(np.mean(self.ep_rews))
                                   if self.ep_rews else 0.0)
                m["pi_loss"].append(pi_l)
                m["v_loss"].append(v_l)
                m["entropy"].append(ent_l)
                m["idm_loss"].append(idm_l)
                m["aux_loss"].append(aux_l)
                m["ep_len"].append(float(np.mean(self.ep_lens))
                                   if self.ep_lens else 0.0)
                m["sps"].append((self.total_steps - self._metrics_step0) / dt)
                m["ent_coef"].append(self._ent_coef)
                if self._progress_cb:
                    self._progress_cb(self)

        return float(np.mean(self.ep_rews)) if self.ep_rews else 0.0

    def _ppo_update(self, adv, ret, l_mask):
        """Reorganizes rollout data into BPTT sequences and runs the PPO update with the tracking loss."""
        T, N, L, S, GH = self.config.rollout_steps, self.config.num_envs, self.config.bptt_seq_len, self.config.rollout_steps // self.config.bptt_seq_len, self.config.gru_hidden
        l_n, l_a = l_mask.nonzero(as_tuple=True)
        K = l_n.shape[0]

        # _lrn slices only the (env, agent) pairs where the learner was acting.
        # _to_seqs reshapes from (segments, seq_len, sequences, ...) to (sequences, seq_len, ...)
        # so each row fed to BPTT is one contiguous L-step window, not a random
        # mix of timesteps.
        def _lrn(buf): return buf[:, l_n, l_a]

        def _to_seqs(t):
            rest = t.shape[2:]
            return t.view(
                S,
                L,
                K,
                *
                rest).permute(
                0,
                2,
                1,
                *
                range(
                    3,
                    3 +
                    len(rest))).reshape(
                S *
                K,
                L,
                *
                rest)

        adv_K = _lrn(adv)
        adv_K = (adv_K - adv_K.mean()) / (adv_K.std() + 1e-8)

        gs_seq, ss_seq, acts_seq = _to_seqs(
            _lrn(
                self.rb_grids)), _to_seqs(
            _lrn(
                self.rb_scalars)), _to_seqs(
                    _lrn(
                        self.rb_actions))
        lp_old_seq, v_old_seq, adv_seq, ret_seq = _to_seqs(
            _lrn(
                self.rb_log_probs)), _to_seqs(
            _lrn(
                self.rb_values)), _to_seqs(adv_K), _to_seqs(
                    _lrn(ret))
        opp_c_seq = _to_seqs(_lrn(self.rb_opp_coords))
        done_seq = _to_seqs(
            self.rb_env_dones[:, l_n].unsqueeze(-1)).squeeze(-1).bool()
        h_seq_starts = self.rb_h_seq_starts[:S, l_n, l_a].reshape(
            S * K, GH).detach()

        n_seqs, mb_seqs = S * K, max(1, self.config.minibatch_size // L)
        self.network.train()
        pi_sum, v_sum, ent_sum, aux_sum, n_mb = 0.0, 0.0, 0.0, 0.0, 0

        for _ in range(self.config.update_epochs):
            perm = torch.randperm(n_seqs, device=self.device)
            for start in range(0, n_seqs, mb_seqs):
                mb = perm[start: start + mb_seqs]
                if len(mb) == 0:
                    continue
                pi_l, v_l, ent_l, aux_l = self._bptt_step(
                    gs_seq[mb], ss_seq[mb], acts_seq[mb], lp_old_seq[mb], v_old_seq[mb], adv_seq[mb], ret_seq[mb], opp_c_seq[mb], h_seq_starts[mb], done_seq[mb])
                pi_sum += pi_l
                v_sum += v_l
                ent_sum += ent_l
                aux_sum += aux_l
                n_mb += 1

        return pi_sum / max(n_mb, 1), v_sum / max(n_mb,
                                                  1), ent_sum / max(n_mb, 1), aux_sum / max(n_mb, 1)

    def _bptt_step(
            self,
            b_gs,
            b_ss,
            b_acts,
            b_lp_old,
            b_v_old,
            b_adv,
            b_ret,
            b_opp_c,
            h0,
            b_done):
        """Runs one BPTT minibatch with PPO loss plus the MSE tracking auxiliary loss."""
        K, L = b_gs.shape[:2]

        with autocast("cuda", enabled=self._use_amp, dtype=self._amp_dtype):
            # All K*L frames are processed by the CNN in one big batch because the CNN
            # has no temporal dependencies. The GRU must be unrolled step by step below
            # because each hidden state depends on the previous one.
            flat_gs = b_gs.reshape(K * L, *b_gs.shape[2:])
            flat_ss = b_ss.reshape(K * L, *b_ss.shape[2:])

            flat_feats = self._raw_network.extract_features(flat_gs, flat_ss)
            feats = flat_feats.view(K, L, -1)

            h = h0.to(self._amp_dtype) if self._use_amp else h0
            pi_loss_t = torch.zeros(K, L, device=self.device)
            v_loss_t = torch.zeros(K, L, device=self.device)
            ent_t = torch.zeros(K, L, device=self.device)
            aux_loss_t = torch.zeros(K, L, device=self.device)

            for t in range(L):
                h = self._raw_network.gru_step(feats[:, t], h)
                logits, value, aux_pred = self._raw_network.heads_from_feat_and_h(
                    feats[:, t], h)

                dist = Categorical(logits=logits)
                lp = dist.log_prob(b_acts[:, t])
                ent_t[:, t] = dist.entropy()

                ratio = torch.exp(lp.float() - b_lp_old[:, t])
                pi_loss_t[:,
                          t] = -torch.min(ratio * b_adv[:,
                                                        t],
                                          ratio.clamp(1.0 - self.config.clip_eps,
                                                      1.0 + self.config.clip_eps) * b_adv[:,
                                                                                          t])

                v_f = value.float()
                v_clipped = b_v_old[:,
                                    t] + (v_f - b_v_old[:,
                                                        t]).clamp(-self.config.clip_eps,
                                                                  self.config.clip_eps)
                v_loss_t[:, t] = 0.5 * \
                    torch.max((v_f - b_ret[:, t]) ** 2, (v_clipped - b_ret[:, t]) ** 2)

                aux_loss_t[:, t] = F.mse_loss(aux_pred, b_opp_c[:, t].to(
                    aux_pred.dtype), reduction='none').mean(dim=-1)

                # Zero out hidden states mid-sequence when an episode ends so the next step
                # starts fresh. We skip this on the last step because there's
                # no next step.
                if t < L - 1 and b_done[:, t].any():
                    h = h * (~b_done[:, t]).float().unsqueeze(-1).to(h.dtype)

            pi_loss, v_loss, ent_, aux_loss = pi_loss_t.mean(
            ), v_loss_t.mean(), ent_t.mean(), aux_loss_t.mean()
            loss = pi_loss + self.config.value_loss_coef * v_loss - \
                self._ent_coef * ent_ + self.config.aux_loss_coef * aux_loss

        self.optimizer.zero_grad(set_to_none=True)
        if self._scaler:
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self._raw_network.parameters(),
                self.config.max_grad_norm)
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(
                self._raw_network.parameters(),
                self.config.max_grad_norm)
            self.optimizer.step()

        return pi_loss.item(), v_loss.item(), ent_.item(), aux_loss.item()

    def _idm_update(self):
        """Samples random transitions and trains the IDM to predict the action between consecutive states."""
        T, N = self.config.rollout_steps, self.config.num_envs
        IDM_BATCH = min(512, T * N // 4)

        # We only use timesteps where the episode was still running at t and t+1.
        # A done step at t means the next frame is a fresh reset, so (s_t, a_t, s_t+1)
        # would span two unrelated episodes which would give the IDM a bad
        # training signal.
        valid_t = torch.where(~self.rb_env_dones)[0]
        valid_t = valid_t[(valid_t > 0) & (valid_t < T - 1)]
        if len(valid_t) == 0:
            return 0.0

        si = torch.randint(0, len(valid_t), (IDM_BATCH,), device=self.device)
        t_idx = valid_t[si]
        n_idx = torch.randint(0, N, (IDM_BATCH,), device=self.device)
        a_idx = torch.randint(0, 2, (IDM_BATCH,), device=self.device)

        g_t = self.rb_grids[t_idx, n_idx, a_idx]
        g_t1 = self.rb_grids[t_idx + 1, n_idx, a_idx]
        s_t = self.rb_scalars[t_idx, n_idx, a_idx]
        s_t1 = self.rb_scalars[t_idx + 1, n_idx, a_idx]
        act_t = self.rb_actions[t_idx, n_idx, a_idx]

        h_prev = torch.zeros(
            IDM_BATCH,
            self.config.gru_hidden,
            device=self.device)

        self.network.train()
        with autocast("cuda", enabled=self._use_amp, dtype=self._amp_dtype):
            g_both = torch.cat([g_t, g_t1], dim=0)
            s_both = torch.cat([s_t, s_t1], dim=0)

            feat_both = self._raw_network.extract_features(g_both, s_both)
            feat_t, feat_t1 = feat_both.chunk(2, dim=0)

            h_t = self._raw_network.gru_step(feat_t, h_prev)
            h_t1 = self._raw_network.gru_step(feat_t1, h_t)

            logits = self._raw_network.idm_logits(h_t, h_t1)
            idm_loss = F.cross_entropy(logits.float(), act_t)
            loss = self.config.idm_coef * idm_loss

        self.optimizer.zero_grad(set_to_none=True)
        if self._scaler:
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self._raw_network.parameters(),
                self.config.max_grad_norm)
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(
                self._raw_network.parameters(),
                self.config.max_grad_norm)
            self.optimizer.step()

        return idm_loss.item()

    def save_state(self, path: str):
        """Saves the network, optimizer, config, training progress, and GRU state to a file."""
        torch.save({"network": self._raw_network.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "config_dict": asdict(self.config),
                    "total_steps": self.total_steps,
                    "opponent_pool": self.opponent_pool,
                    "gru_h": self.gru_h,
                    "ent_ema": self._ent_ema,
                    "dynamic_ent_boost": self._dynamic_ent_boost},
                   path)

    def load_state(self, path: str):
        """Loads a checkpoint and restores network weights, optimizer, training counters, and GRU state."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._raw_network.load_state_dict(ckpt["network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps, self.opponent_pool = ckpt["total_steps"], ckpt["opponent_pool"]
        if "gru_h" in ckpt:
            self.gru_h.copy_(ckpt["gru_h"])
        if "ent_ema" in ckpt:
            self._ent_ema = ckpt["ent_ema"]
        if "dynamic_ent_boost" in ckpt:
            self._dynamic_ent_boost = ckpt["dynamic_ent_boost"]

    def close(self):
        """Closes the vector environment if it exists."""
        if hasattr(self, "vec_env") and self.vec_env is not None:
            self.vec_env.close()
