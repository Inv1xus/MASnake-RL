"""
async_vec_env.py  (v4 — cached workers)
========================================
v3 recap: shared memory eliminated ~1.7 GB/rollout of pipe IPC.

v4 addition: global worker cache.
    Previously workers were spawned and killed on every DEHB trial
    (50 trials × 11 workers = 550 process spawns, ~2-5 s overhead each).
    Workers hold no trial-specific state — reward shaping values are the
    only env-level params in the DEHB search space, and those are plain
    instance attributes that can be overwritten in-place.

    New protocol command:
        "update_rewards"  data: dict of reward kwargs
            → workers overwrite env attributes for all local envs
            → send b"\\x00"

    Trial lifecycle with caching:
        first trial  → spawn workers (one-time cost)
        each trial   → send "update_rewards" + "reset_all"  (~2 ms total)
        final close  → send "close" + join + unlink shm

Cache key: (num_envs, num_workers, grid_h, grid_w)
    Reward params are NOT in the key — they are updated via update_rewards().
    Structural params (grid size, worker count) require a full respawn if
    they change, but DEHB never samples those.

Shared memory layout
────────────────────
  grids : uint8   (num_envs, 2, C, H, W)   quantised obs grids (÷255 on GPU)
  meta  : float32 (num_envs, 2, META_DIM)  [dir, speed, speed_credit, hx, hy, alive]
  rews  : float32 (num_envs, 2)            per-agent rewards
  dones : uint8   (num_envs, 3)            [done_0, done_1, done_all]
"""

import multiprocessing as mp
from multiprocessing import shared_memory
import sys

# Fix for Colab/Jupyter: spawn context calls getattr(__main__, '__spec__', None)
# which raises AttributeError in notebook environments where __spec__ doesn't
# exist as an attribute at all (vs being None).  Setting it explicitly fixes this.
import types
_main = sys.modules.get('__main__')
if _main is not None and not hasattr(_main, '__spec__'):
    _main.__spec__ = None
from typing import Dict, Optional

import numpy as np
import torch

from snake_env import BASE_SPEED, MAX_SPEED, SPEED_GROWTH, MultiAgentSnakeEnv

# ── Constants ──────────────────────────────────────────────────────────────
C        = MultiAgentSnakeEnv.N_CHANNELS   # 8
META_DIM = 6   # direction_idx | speed | speed_credit | head_x | head_y | alive

# Reward attributes that can be hot-updated between trials
_REWARD_KEYS = ("survival_reward", "food_reward", "death_penalty",
                "win_reward", "distance_shaping", "speed_mode")

# ── Global vec-env cache ──────────────────────────────────────────────────
# Key: (num_envs, num_workers, grid_h, grid_w)
# Value: AsyncVecEnv instance (workers still running, shm still allocated)
_VECENV_CACHE: Dict = {}


def get_cached_vec_env(
    reward_kwargs: dict,
    **kwargs,
) -> "AsyncVecEnv":
    """
    Return a ready-to-use AsyncVecEnv, reusing cached workers when possible.

    On cache miss  → spawn workers (first trial only, one-time cost).
    On cache hit   → send update_rewards + reset_all  (~2 ms).

    All arguments come from asdict(PPOConfig) — no explicit positional params
    needed, which avoids the duplicate-keyword-argument error when callers
    spread the full config dict.

    Parameters
    ----------
    reward_kwargs : dict
        Keys from _REWARD_KEYS — updated on workers before reset.
    **kwargs
        Full PPOConfig dict.  num_envs, num_workers, grid_height, grid_width,
        and seed are extracted here; the rest forwarded to AsyncVecEnv.
    """
    # We key the cache only by structural params; reward values are hot-swapped.
    num_envs    = kwargs.get("num_envs",    256)
    num_workers = kwargs.get("num_workers", None)
    grid_h      = kwargs.get("grid_height", 18)
    grid_w      = kwargs.get("grid_width",  24)
    env_device  = torch.device(kwargs.get("env_device", "cpu"))
    env_backend = str(kwargs.get("env_backend", "torch" if env_device.type == "cuda" else "async")).lower().strip()
    torch_env_compile = bool(kwargs.get("torch_env_compile", True))

    if env_backend == "torch":
        key = ("torch", str(env_device), num_envs, grid_h, grid_w, torch_env_compile)
        if key in _VECENV_CACHE:
            vec_env = _VECENV_CACHE[key]
            vec_env.update_rewards(reward_kwargs)
            vec_env.reset()
            print(f"[TorchSnakeVecEnv] Reusing cached env on {env_device}", flush=True)
            return vec_env

        vec_env = TorchSnakeVecEnv(device=env_device, **kwargs)
        vec_env.update_rewards(reward_kwargs)
        vec_env.reset()
        _VECENV_CACHE[key] = vec_env
        return vec_env

    key = ("async", num_envs, num_workers, grid_h, grid_w)

    if key in _VECENV_CACHE:
        vec_env = _VECENV_CACHE[key]
        vec_env.update_rewards(reward_kwargs)
        vec_env.reset()
        print("[AsyncVecEnv] Reusing cached workers — no respawn", flush=True)
        return vec_env

    # Cache miss — first time this configuration is seen
    vec_env = AsyncVecEnv(**kwargs)
    vec_env.update_rewards(reward_kwargs)
    vec_env.reset()
    _VECENV_CACHE[key] = vec_env
    return vec_env


def close_all_cached() -> None:
    """Gracefully shut down all cached workers.  Call at program exit."""
    for vec_env in _VECENV_CACHE.values():
        try:
            vec_env.close()
        except Exception:
            pass
    _VECENV_CACHE.clear()


class TorchSnakeVecEnv:
    """
    Batched torch-backed Snake env.

    Keeps env state, observations, rewards, and dones on the target device so
    PPO rollouts do not round-trip through CPU workers or shared memory.
    """

    _ENV_KEYS = frozenset([
        "grid_width", "grid_height", "num_food", "max_steps",
        "survival_reward", "food_reward", "death_penalty",
        "win_reward", "distance_shaping", "speed_mode",
    ])

    is_torch_env = True

    def __init__(
        self,
        num_envs: int,
        device,
        **kwargs,
    ):
        """Initialize batched torch state buffers and per-env simulation tensors."""
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.W = int(kwargs.get("grid_width", 24))
        self.H = int(kwargs.get("grid_height", 18))
        self.num_food = int(kwargs.get("num_food", 2))
        self.max_steps = int(kwargs.get("max_steps", 1_000))
        self.survival_reward = float(kwargs.get("survival_reward", 0.01))
        self.food_reward = float(kwargs.get("food_reward", 1.0))
        self.death_penalty = float(kwargs.get("death_penalty", -1.0))
        self.win_reward = float(kwargs.get("win_reward", 0.5))
        self.distance_shaping = float(kwargs.get("distance_shaping", 0.02))
        self.speed_mode = bool(kwargs.get("speed_mode", True))

        self.max_cells = self.W * self.H
        self._env_ids = torch.arange(self.num_envs, device=self.device)
        self._body_ord = torch.arange(self.max_cells, device=self.device).view(1, -1)
        self._dir_dx = torch.tensor([0, 0, -1, 1], dtype=torch.long, device=self.device)
        self._dir_dy = torch.tensor([-1, 1, 0, 0], dtype=torch.long, device=self.device)
        self._opp_dir = torch.tensor([1, 0, 3, 2], dtype=torch.long, device=self.device)
        self._speed_growth = torch.tensor(SPEED_GROWTH, dtype=torch.float32, device=self.device)
        self._big_dist = self.W + self.H + 1
        self._compile_requested = bool(kwargs.get("torch_env_compile", True)) and self.device.type == "cuda"
        self._use_compiled_step = False

        seed = int(kwargs.get("seed", 0))
        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(seed)

        state_shape = (self.num_envs, 2, self.max_cells)
        pair_shape = (self.num_envs, 2)
        food_shape = (self.num_envs, self.num_food)

        self.body_x = torch.zeros(state_shape, dtype=torch.long, device=self.device)
        self.body_y = torch.zeros(state_shape, dtype=torch.long, device=self.device)
        self.lengths = torch.zeros(pair_shape, dtype=torch.long, device=self.device)
        self.directions = torch.zeros(pair_shape, dtype=torch.long, device=self.device)
        self.alive = torch.zeros(pair_shape, dtype=torch.bool, device=self.device)
        self.scores = torch.zeros(pair_shape, dtype=torch.long, device=self.device)
        self.food_eaten = torch.zeros(pair_shape, dtype=torch.long, device=self.device)
        self.speed_credits = torch.zeros(pair_shape, dtype=torch.float32, device=self.device)

        self.food_x = torch.zeros(food_shape, dtype=torch.long, device=self.device)
        self.food_y = torch.zeros(food_shape, dtype=torch.long, device=self.device)
        self.food_active = torch.zeros(food_shape, dtype=torch.bool, device=self.device)

        self.step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.winner = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)

        self.torch_grids = torch.zeros(
            (self.num_envs, 2, C, self.H, self.W),
            dtype=torch.float32,
            device=self.device,
        )
        self.torch_meta = torch.zeros(
            (self.num_envs, 2, META_DIM),
            dtype=torch.float32,
            device=self.device,
        )
        self.torch_rews = torch.zeros(pair_shape, dtype=torch.float32, device=self.device)
        self.torch_dones = torch.zeros((self.num_envs, 3), dtype=torch.uint8, device=self.device)
        self.np_grids = None
        self.np_meta = None
        self.np_rews = None
        self.np_dones = None

        q1 = max(2, self.W // 4)
        q3 = min(self.W - 3, 3 * self.W // 4)
        cy = self.H // 2
        self._reset_body_x = torch.zeros((1, 2, self.max_cells), dtype=torch.long, device=self.device)
        self._reset_body_y = torch.zeros((1, 2, self.max_cells), dtype=torch.long, device=self.device)
        self._reset_body_x[0, 0, :3] = torch.tensor([q1, q1 + 1, q1 + 2], dtype=torch.long, device=self.device)
        self._reset_body_x[0, 1, :3] = torch.tensor([q3, q3 - 1, q3 - 2], dtype=torch.long, device=self.device)
        self._reset_body_y[0, :, :3] = cy
        self._reset_lengths = torch.tensor([[3, 3]], dtype=torch.long, device=self.device)
        self._reset_dirs = torch.tensor([[2, 3]], dtype=torch.long, device=self.device)
        self._reset_alive = torch.ones((1, 2), dtype=torch.bool, device=self.device)
        self._reset_pair_long = torch.zeros((1, 2), dtype=torch.long, device=self.device)
        self._reset_pair_float = torch.zeros((1, 2), dtype=torch.float32, device=self.device)
        self._reset_food_active = torch.zeros((1, self.num_food), dtype=torch.bool, device=self.device)
        self._reset_winner = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)

        self._compiled_send_actions = self._send_actions_impl
        self._compile_error = None

    def enable_compile(self) -> None:
        if not self._compile_requested or self._use_compiled_step:
            return

        print("[TorchSnakeVecEnv] Compiling send_actions hot path...", flush=True)
        compile_attempts = [
            dict(fullgraph=False, dynamic=False, options={"triton.cudagraphs": False}),
            dict(mode="max-autotune-no-cudagraphs", fullgraph=False, dynamic=False),
        ]
        last_exc = None
        for compile_kwargs in compile_attempts:
            try:
                self._compiled_send_actions = torch.compile(self._send_actions_impl, **compile_kwargs)
                self._use_compiled_step = True
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                continue
        if last_exc is not None:
            self._compile_requested = False
            self._use_compiled_step = False
            self._compiled_send_actions = self._send_actions_impl
            self._compile_error = last_exc
            print(f"[TorchSnakeVecEnv] No safe no-cudagraph compile mode available, using eager step path: {last_exc}", flush=True)

    def reset(self) -> None:
        mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._reset_mask(mask)
        self.torch_rews.zero_()
        self.torch_dones.zero_()
        self._rebuild_observations_impl()

    def update_rewards(self, reward_kwargs: dict) -> None:
        filtered = {k: v for k, v in reward_kwargs.items() if k in _REWARD_KEYS}
        for attr, val in filtered.items():
            setattr(self, attr, val)

    def reset_single(self, global_i: int) -> None:
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[int(global_i)] = True
        self._reset_mask(mask)
        self.torch_rews[int(global_i)] = 0.0
        self.torch_dones[int(global_i)] = 0
        self._rebuild_observations_impl()

    def close(self) -> None:
        return

    def recv_results(self) -> None:
        return

    def send_actions(self, actions) -> None:
        actions_t = torch.as_tensor(actions, device=self.device, dtype=torch.long)
        actions_t = actions_t.view(self.num_envs, 2)
        try:
            if self._use_compiled_step and hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            self._compiled_send_actions(actions_t)
        except Exception as exc:
            if self._use_compiled_step:
                self._use_compiled_step = False
                self._compiled_send_actions = self._send_actions_impl
                self._compile_error = exc
                self._materialize_persistent_state()
                print(f"[TorchSnakeVecEnv] torch.compile fallback to eager step path: {exc}", flush=True)
            self._send_actions_impl(actions_t)

    def _materialize_persistent_state(self) -> None:
        """
        Break any references to compiler-owned graph outputs before falling back
        to eager execution.
        """
        self.body_x = self.body_x.clone()
        self.body_y = self.body_y.clone()
        self.lengths = self.lengths.clone()
        self.directions = self.directions.clone()
        self.alive = self.alive.clone()
        self.scores = self.scores.clone()
        self.food_eaten = self.food_eaten.clone()
        self.speed_credits = self.speed_credits.clone()
        self.food_x = self.food_x.clone()
        self.food_y = self.food_y.clone()
        self.food_active = self.food_active.clone()
        self.step_count = self.step_count.clone()
        self.done = self.done.clone()
        self.winner = self.winner.clone()
        self.torch_grids = self.torch_grids.clone()
        self.torch_meta = self.torch_meta.clone()
        self.torch_rews = self.torch_rews.clone()
        self.torch_dones = self.torch_dones.clone()

    def _send_actions_impl(self, actions_t: torch.Tensor) -> None:
        self.torch_rews.zero_()
        self.torch_dones.zero_()

        # Get integer tensor of how many blocks each snake moves this tick
        moves = self._plan_tick_moves()
        max_moves = int(moves.max().item())

        # Execute the physical steps internally before querying the network again
        for sub in range(max_moves):
            moving_mask = (moves > sub) & self.alive
            if not moving_mask.any():
                break
            self._physical_substep(actions_t, moving_mask)

        terminal_envs = self.done.clone()
        self.torch_dones[:, :2] = (terminal_envs.unsqueeze(1) | (~self.alive)).to(torch.uint8)
        self.torch_dones[:, 2] = terminal_envs.to(torch.uint8)

        self._reset_mask(terminal_envs)
        self._rebuild_observations_impl()

    def _plan_tick_moves(self) -> torch.Tensor:
        if not self.speed_mode:
            return self.alive.to(torch.int32)

        speed_mult = self._speed_multiplier()
        
        self.speed_credits = torch.where(
            self.alive,
            self.speed_credits + speed_mult,
            self.speed_credits,
        )

        # Extract integer moves (e.g., 2.5 credits = 2 physical moves)
        moves = torch.floor(self.speed_credits).to(torch.int32)
        
        # Keep the fractional remainder (2.5 - 2.0 = 0.5 left over)
        self.speed_credits = torch.where(
            self.alive,
            self.speed_credits - moves.float(),
            self.speed_credits,
        )
        
        return torch.where(self.alive, moves, torch.zeros_like(moves))
        
    def _physical_substep(self, actions: torch.Tensor, moving: torch.Tensor) -> None:
        env_active = moving.any(dim=1)
        self.step_count += env_active.long()

        sub_rews = env_active.unsqueeze(1).expand(-1, 2).float() * self.survival_reward

        proposed = self.directions
        valid_actions = moving & (actions >= 0) & (actions <= 3)
        proposed = torch.where(valid_actions, actions, proposed)
        reversals = moving & (self.lengths > 1) & (proposed == self._opp_dir[self.directions])
        proposed = torch.where(reversals, self.directions, proposed)
        self.directions = torch.where(moving, proposed, self.directions)

        old_x = self.body_x[:, :, 0].clone()
        old_y = self.body_y[:, :, 0].clone()
        next_x = old_x + self._dir_dx[self.directions]
        next_y = old_y + self._dir_dy[self.directions]

        food_match = (
            (next_x.unsqueeze(-1) == self.food_x.unsqueeze(1))
            & (next_y.unsqueeze(-1) == self.food_y.unsqueeze(1))
            & self.food_active.unsqueeze(1)
        )
        will_eat = moving & food_match.any(dim=-1)

        occupied = torch.zeros((self.num_envs, self.max_cells), dtype=torch.float32, device=self.device)
        body_idx = self.body_y * self.W + self.body_x
        for aid in (0, 1):
            trim_tail = (moving[:, aid] & (~will_eat[:, aid]) & self.alive[:, aid]).long()
            eff_len = (self.lengths[:, aid] - trim_tail).clamp(min=0)
            valid = self._body_ord < eff_len.unsqueeze(-1)
            valid &= self.alive[:, aid].unsqueeze(-1)
            occupied.scatter_add_(1, body_idx[:, aid], valid.float())

        in_bounds = (
            (next_x >= 0) & (next_x < self.W)
            & (next_y >= 0) & (next_y < self.H)
        )
        next_idx = (next_y.clamp(0, self.H - 1) * self.W + next_x.clamp(0, self.W - 1)).long()
        occ_hits = occupied.gather(1, next_idx)

        dead = torch.zeros_like(self.alive)
        wall_dead = moving & (~in_bounds)
        dead |= wall_dead
        sub_rews += wall_dead.float() * self.death_penalty

        body_dead = moving & (~dead) & in_bounds & (occ_hits > 0)
        dead |= body_dead
        sub_rews += body_dead.float() * self.death_penalty

        head_to_head = (
            moving[:, 0] & moving[:, 1]
            & (~dead[:, 0]) & (~dead[:, 1])
            & (next_x[:, 0] == next_x[:, 1])
            & (next_y[:, 0] == next_y[:, 1])
        )
        dead[:, 0] |= head_to_head
        dead[:, 1] |= head_to_head
        sub_rews[:, 0] += head_to_head.float() * self.death_penalty
        sub_rews[:, 1] += head_to_head.float() * self.death_penalty

        self.alive &= ~dead
        moved_alive = moving & self.alive

        fx = self.food_x.unsqueeze(1)
        fy = self.food_y.unsqueeze(1)
        active_food = self.food_active.unsqueeze(1)
        old_d = (old_x.unsqueeze(-1) - fx).abs() + (old_y.unsqueeze(-1) - fy).abs()
        new_d = (next_x.unsqueeze(-1) - fx).abs() + (next_y.unsqueeze(-1) - fy).abs()
        old_d = torch.where(active_food, old_d, self._big_dist).amin(dim=-1)
        new_d = torch.where(active_food, new_d, self._big_dist).amin(dim=-1)
        shaping_mask = moved_alive & self.food_active.any(dim=1, keepdim=True)
        sub_rews += self.distance_shaping * (old_d - new_d).float() * shaping_mask.float()

        shifted_x = torch.cat((next_x.unsqueeze(-1), self.body_x[:, :, :-1]), dim=-1)
        shifted_y = torch.cat((next_y.unsqueeze(-1), self.body_y[:, :, :-1]), dim=-1)
        self.body_x = torch.where(moved_alive.unsqueeze(-1), shifted_x, self.body_x)
        self.body_y = torch.where(moved_alive.unsqueeze(-1), shifted_y, self.body_y)

        ate_food = moved_alive & will_eat
        ate_food_long = ate_food.long()
        self.lengths += ate_food_long
        self.scores += ate_food_long
        self.food_eaten += ate_food_long
        sub_rews += ate_food.float() * self.food_reward

        consumed = (food_match & ate_food.unsqueeze(-1)).any(dim=1)
        self.food_active &= ~consumed
        self._refill_food(consumed.any(dim=1))

        was_done = self.done.clone()
        alive_counts = self.alive.long().sum(dim=1)
        single_alive = (alive_counts == 1) & (~was_done)
        no_alive = (alive_counts == 0) & (~was_done)
        winners = self.alive.long().argmax(dim=1)
        sub_rews[:, 0] += (single_alive & (winners == 0)).float() * self.win_reward
        sub_rews[:, 1] += (single_alive & (winners == 1)).float() * self.win_reward
        self.winner = torch.where(single_alive, winners, self.winner)
        self.winner = torch.where(no_alive, self._reset_winner, self.winner)

        step_limit = (self.step_count >= self.max_steps) & (~was_done)
        self.winner = torch.where(step_limit, self._score_winner(), self.winner)

        self.done |= single_alive | no_alive | step_limit
        self.torch_rews += sub_rews

    def _speed_multiplier(self) -> torch.Tensor:
        speed = BASE_SPEED * torch.pow(self._speed_growth, self.food_eaten.float())
        speed = torch.clamp(speed, max=MAX_SPEED)
        return speed / BASE_SPEED

    def _score_winner(self) -> torch.Tensor:
        winners = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        better0 = self.scores[:, 0] > self.scores[:, 1]
        better1 = self.scores[:, 1] > self.scores[:, 0]
        winners = torch.where(better0, torch.zeros_like(winners), winners)
        winners = torch.where(better1, torch.ones_like(winners), winners)
        return winners

    def _reset_mask(self, env_mask: torch.Tensor) -> None:
        if not env_mask.any():
            return

        mask3 = env_mask.view(-1, 1, 1)
        mask2 = env_mask.view(-1, 1)
        food_mask = env_mask.view(-1, 1)
        self.body_x = torch.where(mask3, self._reset_body_x.expand(self.num_envs, -1, -1), self.body_x)
        self.body_y = torch.where(mask3, self._reset_body_y.expand(self.num_envs, -1, -1), self.body_y)
        self.lengths = torch.where(mask2, self._reset_lengths.expand(self.num_envs, -1), self.lengths)
        self.directions = torch.where(mask2, self._reset_dirs.expand(self.num_envs, -1), self.directions)
        self.alive = torch.where(mask2, self._reset_alive.expand(self.num_envs, -1), self.alive)
        self.scores = torch.where(mask2, self._reset_pair_long.expand(self.num_envs, -1), self.scores)
        self.food_eaten = torch.where(mask2, self._reset_pair_long.expand(self.num_envs, -1), self.food_eaten)
        self.speed_credits = torch.where(mask2, self._reset_pair_float.expand(self.num_envs, -1), self.speed_credits)
        self.food_active = torch.where(food_mask, self._reset_food_active.expand(self.num_envs, -1), self.food_active)
        self.step_count = torch.where(env_mask, torch.zeros_like(self.step_count), self.step_count)
        self.done = torch.where(env_mask, torch.zeros_like(self.done), self.done)
        self.winner = torch.where(env_mask, self._reset_winner, self.winner)
        self._refill_food(env_mask)

    def _refill_food(self, env_mask: torch.Tensor) -> None:
        if not env_mask.any():
            return

        for _ in range(self.num_food):
            need = env_mask & (self.food_active.long().sum(dim=1) < self.num_food)
            if not need.any():
                break
            self._spawn_one_food(need)

    def _spawn_one_food(self, env_mask: torch.Tensor) -> None:
        if not env_mask.any():
            return

        occ = torch.zeros((self.num_envs, self.max_cells), dtype=torch.float32, device=self.device)
        body_idx = self.body_y * self.W + self.body_x
        for aid in (0, 1):
            valid = self._body_ord < self.lengths[:, aid].unsqueeze(-1)
            valid &= self.alive[:, aid].unsqueeze(-1)
            occ.scatter_add_(1, body_idx[:, aid], valid.float())

        food_idx = self.food_y * self.W + self.food_x
        occ.scatter_add_(1, food_idx, self.food_active.float())

        free = occ == 0
        has_free = free.any(dim=1)
        spawn_mask = env_mask & has_free
        sample_weights = free.float()
        sample_weights[:, 0] += (~spawn_mask).float()
        samples = torch.multinomial(sample_weights, 1, generator=self._rng).squeeze(1)
        empty_slots = (~self.food_active).long().argmax(dim=1)
        slot_mask = torch.zeros_like(self.food_active)
        slot_mask.scatter_(1, empty_slots.unsqueeze(1), spawn_mask.unsqueeze(1))

        new_food_x = self.food_x.clone()
        new_food_y = self.food_y.clone()
        new_food_x.scatter_(1, empty_slots.unsqueeze(1), samples.remainder(self.W).unsqueeze(1))
        new_food_y.scatter_(1, empty_slots.unsqueeze(1), torch.div(samples, self.W, rounding_mode="floor").unsqueeze(1))
        self.food_x = torch.where(slot_mask, new_food_x, self.food_x)
        self.food_y = torch.where(slot_mask, new_food_y, self.food_y)
        self.food_active |= slot_mask

    def _rebuild_observations_impl(self) -> None:
        self.torch_grids.zero_()

        food_idx = self.food_y * self.W + self.food_x
        food_vals = self.food_active.float()
        self.torch_grids[:, 0, 4].view(self.num_envs, -1).scatter_add_(1, food_idx, food_vals)
        self.torch_grids[:, 1, 4].view(self.num_envs, -1).scatter_add_(1, food_idx, food_vals)

        body_idx = self.body_y * self.W + self.body_x

        for viewer in (0, 1):
            for src in (0, 1):
                alive_src = self.alive[:, src]

                is_self = int(src == viewer)
                head_ch = 0 if is_self else 2
                body_ch = 1 if is_self else 3
                order_ch = 5 if is_self else 6

                self.torch_grids[:, viewer, head_ch].view(self.num_envs, -1).scatter_add_(
                    1,
                    body_idx[:, src, :1],
                    alive_src.float().unsqueeze(-1),
                )

                valid = (self._body_ord < self.lengths[:, src].unsqueeze(-1)) & alive_src.unsqueeze(-1)
                tail_valid = valid & (self._body_ord > 0)
                self.torch_grids[:, viewer, body_ch].view(self.num_envs, -1).scatter_add_(
                    1,
                    body_idx[:, src],
                    tail_valid.float(),
                )

                denom = self.lengths[:, src].clamp(min=1).unsqueeze(-1)
                order_vals = (
                    (self.lengths[:, src].unsqueeze(-1) - self._body_ord).float()
                    / denom.float()
                ) * valid.float()
                self.torch_grids[:, viewer, order_ch].view(self.num_envs, -1).scatter_add_(
                    1,
                    body_idx[:, src],
                    order_vals,
                )

        speed = BASE_SPEED * torch.pow(self._speed_growth, self.food_eaten.float())
        speed = torch.clamp(speed, max=MAX_SPEED) / 100.0

        self.torch_meta[:, :, 0] = self.directions.float()
        self.torch_meta[:, :, 1] = speed
        self.torch_meta[:, :, 2] = self.speed_credits
        self.torch_meta[:, :, 3] = self.body_x[:, :, 0].float()
        self.torch_meta[:, :, 4] = self.body_y[:, :, 0].float()
        self.torch_meta[:, :, 5] = self.alive.float()


# ─────────────────────────────────────────────────────────────────────────────
# Worker subprocess
# ─────────────────────────────────────────────────────────────────────────────

def _worker_fn(
    pipe,
    env_kwargs_list: list,
    shm_names: dict,
    total_envs: int,
    env_start: int,
    H: int,
    W: int,
) -> None:
    """
    Worker entry point.

    Protocol (pipe messages):
        recv ("reset_all",      None)               → write obs, send b'\\x00'
        recv ("step",           np.int32 (K,2))     → step K envs, write, send b'\\x00'
        recv ("reset_single",   int local_idx)      → reset one env, send b'\\x00'
        recv ("update_rewards", dict)               → overwrite env attrs, send b'\\x00'
        recv ("close",          None)               → cleanup, send b'\\x00', exit
    """
    shms = {k: shared_memory.SharedMemory(name=v) for k, v in shm_names.items()}

    num_local = len(env_kwargs_list)
    sl = slice(env_start, env_start + num_local)

    all_grids = np.ndarray((total_envs, 2, C, H, W), np.float32, buffer=shms["grids"].buf)
    all_meta  = np.ndarray((total_envs, 2, META_DIM), np.float32, buffer=shms["meta"].buf)
    all_rews  = np.ndarray((total_envs, 2),            np.float32, buffer=shms["rews"].buf)
    all_dones = np.ndarray((total_envs, 3),            np.uint8,   buffer=shms["dones"].buf)

    w_grids = all_grids[sl]
    w_meta  = all_meta[sl]
    w_rews  = all_rews[sl]
    w_dones = all_dones[sl]

    envs = [MultiAgentSnakeEnv(**kw) for kw in env_kwargs_list]


    # ── Helpers ───────────────────────────────────────────────────────

    def _write_obs(li: int, raw_obs: dict) -> None:
        for aid in (0, 1):
            obs = raw_obs[aid]
            w_grids[li, aid] = obs["grid"]
            w_meta[li, aid, 0]  = obs["direction"]
            w_meta[li, aid, 1]  = obs["speed"]
            w_meta[li, aid, 2]  = obs["speed_credit"]
            w_meta[li, aid, 3]  = obs["head"][0]
            w_meta[li, aid, 4]  = obs["head"][1]
            w_meta[li, aid, 5]  = float(obs["alive"])

    def _write_step(li: int, rews: dict, dones: dict) -> None:
        w_rews[li, 0]  = rews.get(0, 0.0)
        w_rews[li, 1]  = rews.get(1, 0.0)
        w_dones[li, 0] = int(dones.get(0,         True))
        w_dones[li, 1] = int(dones.get(1,         True))
        w_dones[li, 2] = int(dones.get("__all__", False))

    # ── Main loop ─────────────────────────────────────────────────────

    try:
        while True:
            cmd, data = pipe.recv()

            if cmd == "reset_all":
                for li, env in enumerate(envs):
                    _write_obs(li, env.reset())
                w_rews[:]  = 0.0
                w_dones[:] = 0
                pipe.send(b"\x00")

            elif cmd == "step":
                for li, env in enumerate(envs):
                    actions = {0: int(data[li, 0]), 1: int(data[li, 1])}
                    obs, rews, dones, _ = env.step(actions)
                    _write_step(li, rews, dones)
                    if dones.get("__all__", False):
                        obs = env.reset()
                    _write_obs(li, obs)
                pipe.send(b"\x00")

            elif cmd == "update_rewards":
                # data: dict of reward attribute name → new value
                for env in envs:
                    for attr, val in data.items():
                        setattr(env, attr, val)
                pipe.send(b"\x00")

            elif cmd == "reset_single":
                li = data
                _write_obs(li, envs[li].reset())
                w_rews[li, :]  = 0.0
                w_dones[li, :] = 0
                pipe.send(b"\x00")

            elif cmd == "close":
                for env in envs:
                    try:
                        env.close_render()
                    except Exception:
                        pass
                for shm in shms.values():
                    shm.close()
                pipe.send(b"\x00")
                break

    except (EOFError, KeyboardInterrupt):
        for shm in shms.values():
            try:
                shm.close()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# AsyncVecEnv
# ─────────────────────────────────────────────────────────────────────────────

class AsyncVecEnv:
    """
    Shared-memory async vectorised environment.

    Normally obtained via get_cached_vec_env() rather than instantiated
    directly — the cache avoids repeated worker spawning across DEHB trials.

    Public shared-memory arrays (safe to read after reset() or recv_results(),
    until the next send_actions()):
        np_grids  : (num_envs, 2, C, H, W)  float32  observation grids
        np_meta   : (num_envs, 2, META_DIM)  float32  [dir,speed,speed_credit,hx,hy,alive]
        np_rews   : (num_envs, 2)            float32  per-agent rewards
        np_dones  : (num_envs, 3)            uint8    [done_0, done_1, done_all]
    """

    _ENV_KEYS = frozenset([
        "grid_width", "grid_height", "num_food", "max_steps",
        "survival_reward", "food_reward", "death_penalty",
        "win_reward", "distance_shaping", "speed_mode",
    ])

    def __init__(
        self,
        num_envs:    int,
        num_workers: Optional[int] = None,
        **kwargs,
    ):
        self.num_envs = num_envs
        H = kwargs.get("grid_height", 18)
        W = kwargs.get("grid_width",  24)
        self.H, self.W = H, W

        n_cpus = mp.cpu_count()
        # Divide by 2 to account for hyperthreading — Colab and cloud VMs
        # report vCPU count (e.g. 32) which is 2× physical cores.  Spawning
        # 31 Python worker processes on 16 physical cores causes massive
        # context-switching overhead and was the primary cause of the 10×
        # SPS drop when num_envs was increased from 64 to 256.
        n_physical = max(1, n_cpus // 2)
        self.num_workers = num_workers or min(num_envs, n_physical)

        base, extra = divmod(num_envs, self.num_workers)
        self._worker_sizes = [
            base + (1 if i < extra else 0) for i in range(self.num_workers)
        ]

        # ── Allocate shared memory ────────────────────────────────────
        n = num_envs

        def _alloc(shape, dtype):
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            shm = shared_memory.SharedMemory(create=True, size=max(nbytes, 1))
            arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            arr[:] = 0
            return shm, arr

        self._shms: dict = {}
        self._shms["grids"], self.np_grids = _alloc((n, 2, C, H, W), np.float32)
        self._shms["meta"],  self.np_meta  = _alloc((n, 2, META_DIM), np.float32)
        self._shms["rews"],  self.np_rews  = _alloc((n, 2),            np.float32)
        self._shms["dones"], self.np_dones = _alloc((n, 3),            np.uint8)

        shm_names = {k: v.name for k, v in self._shms.items()}

        # ── Spawn workers ─────────────────────────────────────────────
        base_kw   = {k: v for k, v in kwargs.items() if k in self._ENV_KEYS}
        base_seed = kwargs.get("seed", 0)
        ctx = mp.get_context("spawn")

        self._pipes: list = []
        self._procs: list = []
        global_start = 0

        for w, size in enumerate(self._worker_sizes):
            parent, child = ctx.Pipe(duplex=True)
            env_kwargs_list = [
                {**base_kw, "seed": base_seed + global_start + li}
                for li in range(size)
            ]
            proc = ctx.Process(
                target=_worker_fn,
                args=(child, env_kwargs_list, shm_names,
                      num_envs, global_start, H, W),
                daemon=True,
            )
            proc.start()
            child.close()
            self._pipes.append(parent)
            self._procs.append(proc)
            global_start += size

        print(
            f"[AsyncVecEnv] {num_envs} envs | {self.num_workers} workers "
            f"{self._worker_sizes} | shared memory: "
            f"{(self._shms['grids'].size + self._shms['meta'].size) / 1e6:.1f} MB",
            flush=True,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all environments. Shared memory populated on return."""
        for pipe in self._pipes:
            pipe.send(("reset_all", None))
        for pipe in self._pipes:
            pipe.recv()

    def update_rewards(self, reward_kwargs: dict) -> None:
        """
        Overwrite reward shaping attributes on all worker envs.
        Called at the start of each DEHB trial before reset().

        Parameters
        ----------
        reward_kwargs : dict
            Any subset of: survival_reward, food_reward, death_penalty,
            win_reward, distance_shaping, speed_mode.
        """
        # Filter to only valid reward keys
        filtered = {k: v for k, v in reward_kwargs.items() if k in _REWARD_KEYS}
        if not filtered:
            return
        for pipe in self._pipes:
            pipe.send(("update_rewards", filtered))
        for pipe in self._pipes:
            pipe.recv()

    def reset_single(self, global_i: int) -> None:
        """Reset one environment (blocking). Workers auto-reset on episode end."""
        w, li = self._env_location(global_i)
        self._pipes[w].send(("reset_single", li))
        self._pipes[w].recv()

    def close(self) -> None:
        """Shut down workers and release shared memory."""
        for pipe in self._pipes:
            try:
                pipe.send(("close", None))
                pipe.recv()
            except Exception:
                pass
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
        for shm in self._shms.values():
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass

    # ── Async interface ───────────────────────────────────────────────────────

    def send_actions(self, actions_np: np.ndarray) -> None:
        """
        Non-blocking. Dispatch step commands to all workers simultaneously.
        actions_np : np.int32 (num_envs, 2) — col 0 = agent-0, col 1 = agent-1.
        """
        offset = 0
        for pipe, size in zip(self._pipes, self._worker_sizes):
            pipe.send(("step", actions_np[offset : offset + size]))
            offset += size

    def recv_results(self) -> None:
        """
        Blocking. Returns when all workers have finished writing to shared
        memory. Safe to read np_grids/meta/rews/dones until next send_actions().
        """
        for pipe in self._pipes:
            pipe.recv()

    # ── Helper ────────────────────────────────────────────────────────────────

    def _env_location(self, global_i: int):
        offset = 0
        for w, size in enumerate(self._worker_sizes):
            if global_i < offset + size:
                return w, global_i - offset
            offset += size
        raise IndexError(f"global_i={global_i} out of range (num_envs={self.num_envs})")