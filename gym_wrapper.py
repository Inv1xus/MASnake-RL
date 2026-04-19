"""
gym_wrapper.py
==============
Gymnasium-compatible wrapper around MultiAgentSnakeEnv.

Design goals
────────────
  • Expose agent-0 as the single learning agent via the standard
    gymnasium.Env API (reset / step / render / close).
  • Agent-1 is controlled by an `opponent_fn(raw_obs_dict) -> int`
    that can be hot-swapped at any time with `set_opponent()`.
  • The observation returned is the *raw* dict produced by
    MultiAgentSnakeEnv — identical to what obs_to_tensors() already
    consumes — so the existing PPO training code needs zero changes.
  • gymnasium.spaces are declared for framework compliance and to
    make future wrappers (e.g. RecordEpisodeStatistics,
    TransformObservation) work out of the box.

Epiplexity hook
───────────────
  When the auxiliary self-prediction module is added, its inputs
  (past obs, past actions, past rewards) live in `self._history`.
  _history is a collections.deque already maintained here; just set
  `history_len > 0` at construction time to activate it.
  The step() method appends every transition before returning, so
  the prediction module can read it directly from the env object.

Usage (standalone)
──────────────────
  env = SnakeGymEnv(seed=0)
  obs, info = env.reset()
  obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

import random
import collections
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from snake_env import MultiAgentSnakeEnv, Action


# ─────────────────────────────────────────────────────────────────────────────
# SnakeGymEnv
# ─────────────────────────────────────────────────────────────────────────────

class SnakeGymEnv(gym.Env):
    """
    Single-agent Gymnasium wrapper for MultiAgentSnakeEnv.

    Parameters
    ----------
    opponent_fn : callable(raw_obs_dict) -> int | None
        Policy for agent-1.  None → uniform random.
    history_len : int
        Number of past (obs, action, reward) tuples to keep in
        self._history.  0 disables history (default, zero overhead).
    **env_kwargs
        Forwarded verbatim to MultiAgentSnakeEnv.
    """

    metadata = {"render_modes": ["human"]}

    # Keys that MultiAgentSnakeEnv's __init__ accepts
    _ENV_KEYS = frozenset([
        "grid_width", "grid_height", "num_food", "max_steps", "seed",
        "survival_reward", "food_reward", "death_penalty",
        "win_reward", "distance_shaping", "speed_mode",
    ])

    def __init__(
        self,
        opponent_fn: Optional[Callable] = None,
        history_len: int = 0,
        **env_kwargs,
    ):
        super().__init__()

        # ── Underlying environment ────────────────────────────────────
        clean_kw = {k: v for k, v in env_kwargs.items() if k in self._ENV_KEYS}
        self._env    = MultiAgentSnakeEnv(**clean_kw)
        self._W: int = clean_kw.get("grid_width",  24)
        self._H: int = clean_kw.get("grid_height", 18)

        # ── Opponent ──────────────────────────────────────────────────
        self._opponent_fn: Optional[Callable] = opponent_fn

        # ── History buffer (for epiplexity) ──────────────────────────
        self._history_len: int = history_len
        self._history: Deque[dict] = collections.deque(maxlen=history_len or 1)

        # ── Gymnasium spaces ──────────────────────────────────────────
        C = MultiAgentSnakeEnv.N_CHANNELS
        self.observation_space = spaces.Dict({
            # Raw grid — same shape as MultiAgentSnakeEnv produces
            "grid":       spaces.Box(0.0, 1.0, (C, self._H, self._W), np.float32),
            # Discrete direction index  (0=UP 1=DOWN 2=LEFT 3=RIGHT)
            "direction":  spaces.Discrete(4),
            # Normalised speed scalar  [0, 1]
            "speed":      spaces.Box(0.0, 1.0, (1,), np.float32),
            # Fractional speed-credit phase  [0, 1)
            "speed_credit": spaces.Box(0.0, 1.0, (1,), np.float32),
            # Raw food-eaten score (unbounded above)
            "score":      spaces.Box(0.0, np.inf, (1,), np.float32),
            # Pixel head coordinates  (x, y)
            "head":       spaces.Box(
                              np.array([0, 0], dtype=np.int32),
                              np.array([self._W - 1, self._H - 1], dtype=np.int32),
                          ),
        })
        self.action_space = spaces.Discrete(4)

        # ── Internal state ────────────────────────────────────────────
        self._last_raw_obs: Optional[Dict] = None  # {0: obs, 1: obs}

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[dict, dict]:
        if seed is not None:
            self._env.rng = random.Random(seed)

        raw_obs = self._env.reset()
        self._last_raw_obs = raw_obs
        self._history.clear()

        return raw_obs[0], {}

    def step(self, action: int) -> Tuple[dict, float, bool, bool, dict]:
        """
        Step the environment.

        Parameters
        ----------
        action : int
            Agent-0's action (0=UP 1=DOWN 2=LEFT 3=RIGHT).

        Returns
        -------
        obs          : raw obs dict for agent-0 (same format as MultiAgentSnakeEnv)
        reward       : float reward for agent-0
        terminated   : bool — episode ended naturally (death / win / timeout)
        truncated    : bool — always False (timeout handled inside the env)
        info         : dict with optional keys: death, ate_food, opponent_reward
        """
        assert self._last_raw_obs is not None, "Call reset() before step()."

        # Opponent selects action
        opp_raw = self._last_raw_obs[1]
        if self._opponent_fn is not None:
            opp_action = int(self._opponent_fn(opp_raw))
        else:
            opp_action = random.randint(0, 3)

        raw_obs, rewards, dones, info = self._env.step(
            {0: int(action), 1: opp_action}
        )
        self._last_raw_obs = raw_obs

        obs        = raw_obs[0]
        reward     = float(rewards.get(0, 0.0))
        terminated = bool(dones.get("__all__", False))
        truncated  = False

        agent_info = dict(info.get(0, {}))
        agent_info["opponent_reward"] = float(rewards.get(1, 0.0))

        # ── Epiplexity history ────────────────────────────────────────
        if self._history_len > 0:
            self._history.append({
                "obs":    obs,
                "action": int(action),
                "reward": reward,
                "done":   terminated,
            })

        return obs, reward, terminated, truncated, agent_info

    def render(self) -> None:
        self._env.render()

    def close(self) -> None:
        self._env.close_render()

    # ── Convenience helpers ───────────────────────────────────────────────────

    def set_opponent(self, fn: Optional[Callable]) -> None:
        """Hot-swap opponent without recreating the env.  Pass None for random."""
        self._opponent_fn = fn

    @property
    def history(self) -> List[dict]:
        """Ordered list of past transitions (oldest first).  Empty if history_len=0."""
        return list(self._history)
