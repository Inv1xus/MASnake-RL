"""
Gymnasium wrapper around MultiAgentSnakeEnv.

Exposes agent 0 as the single learning agent. Agent 1 is controlled by
an opponent function that can be swapped at any time via set_opponent().
"""

import random
import collections
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from snake_env import MultiAgentSnakeEnv, Action


class SnakeGymEnv(gym.Env):
    """
    Single-agent Gymnasium wrapper for MultiAgentSnakeEnv.

    Parameters
    ----------
    opponent_fn : callable or None
        Policy for agent 1. Pass None for uniform random.
    history_len : int
        How many past (obs, action, reward) tuples to keep in self._history.
        Zero disables history with no overhead.
    **env_kwargs
        Forwarded to MultiAgentSnakeEnv.
    """

    metadata = {"render_modes": ["human"]}

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
        """Sets up the wrapped environment, opponent, history buffer, and gym spaces."""
        super().__init__()

        clean_kw = {k: v for k, v in env_kwargs.items() if k in self._ENV_KEYS}
        self._env = MultiAgentSnakeEnv(**clean_kw)
        self._W: int = clean_kw.get("grid_width", 24)
        self._H: int = clean_kw.get("grid_height", 18)

        self._opponent_fn: Optional[Callable] = opponent_fn

        self._history_len: int = history_len
        self._history: Deque[dict] = collections.deque(maxlen=history_len or 1)

        C = MultiAgentSnakeEnv.N_CHANNELS
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(0.0, 1.0, (C, self._H, self._W), np.float32),
            "direction": spaces.Discrete(4),
            "speed": spaces.Box(0.0, 1.0, (1,), np.float32),
            "speed_credit": spaces.Box(0.0, 1.0, (1,), np.float32),
            "score": spaces.Box(0.0, np.inf, (1,), np.float32),
            "head": spaces.Box(
                np.array([0, 0], dtype=np.int32),
                np.array([self._W - 1, self._H - 1], dtype=np.int32),
            ),
        })
        self.action_space = spaces.Discrete(4)

        self._last_raw_obs: Optional[Dict] = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[dict, dict]:
        """Resets the environment and returns the learning agent's first observation."""
        if seed is not None:
            self._env.rng = random.Random(seed)

        raw_obs = self._env.reset()
        self._last_raw_obs = raw_obs
        self._history.clear()

        return raw_obs[0], {}

    def step(self, action: int) -> Tuple[dict, float, bool, bool, dict]:
        """Steps the environment with agent 0's action and returns the gym tuple."""
        assert self._last_raw_obs is not None, "Call reset() before step()."

        opp_raw = self._last_raw_obs[1]
        if self._opponent_fn is not None:
            opp_action = int(self._opponent_fn(opp_raw))
        else:
            opp_action = random.randint(0, 3)

        raw_obs, rewards, dones, info = self._env.step(
            {0: int(action), 1: opp_action}
        )
        self._last_raw_obs = raw_obs

        obs = raw_obs[0]
        reward = float(rewards.get(0, 0.0))
        terminated = bool(dones.get("__all__", False))
        truncated = False

        agent_info = dict(info.get(0, {}))
        agent_info["opponent_reward"] = float(rewards.get(1, 0.0))

        if self._history_len > 0:
            self._history.append({
                "obs": obs,
                "action": int(action),
                "reward": reward,
                "done": terminated,
            })

        return obs, reward, terminated, truncated, agent_info

    def render(self) -> None:
        """Forwards the render call to the wrapped multi-agent environment."""
        self._env.render()

    def close(self) -> None:
        """Closes any render resources held by the wrapped environment."""
        self._env.close_render()

    def set_opponent(self, fn: Optional[Callable]) -> None:
        """Swaps the opponent policy without recreating the environment. Pass None for random."""
        self._opponent_fn = fn

    @property
    def history(self) -> List[dict]:
        """Returns an ordered list of past transitions, oldest first. Empty if history_len is zero."""
        return list(self._history)
