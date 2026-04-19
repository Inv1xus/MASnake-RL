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

    Exposes agent 0 as the learner. Agent 1 is controlled by an opponent
    function that can be swapped at any time without recreating the environment.

    Args:
        opponent_fn (callable or None): policy for agent 1. Pass None for random.
        history_len (int): number of past transitions to keep; 0 disables history.
        **env_kwargs: passed through to MultiAgentSnakeEnv.

    Example:
        from gym_wrapper import SnakeGymEnv
        env = SnakeGymEnv(grid_width=24, grid_height=18)
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
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
        """
        Sets up the wrapped environment, opponent, history buffer, and gym spaces.

        Args:
            opponent_fn (callable or None): called with the opponent observation each step
                to produce an action. None uses random actions.
            history_len (int): max number of transitions stored in self._history.
            **env_kwargs: keyword arguments forwarded to MultiAgentSnakeEnv.
        """
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
        """
        Resets the environment and returns agent 0's first observation.

        Args:
            seed (int or None): optional seed passed to the inner environment.
            options (dict or None): unused, kept for gym API compatibility.

        Returns:
            tuple: (obs, info) where obs is a dict matching observation_space
            and info is an empty dict.

        Example:
            env = SnakeGymEnv()
            obs, info = env.reset(seed=42)
        """
        if seed is not None:
            self._env.rng = random.Random(seed)

        raw_obs = self._env.reset()
        self._last_raw_obs = raw_obs
        self._history.clear()

        return raw_obs[0], {}

    def step(self, action: int) -> Tuple[dict, float, bool, bool, dict]:
        """
        Steps the environment with agent 0's action.

        Args:
            action (int): integer in [0, 3] for UP, DOWN, LEFT, RIGHT.

        Returns:
            tuple: (obs, reward, terminated, truncated, info)
                obs (dict): new observation for agent 0.
                reward (float): reward earned this step.
                terminated (bool): True when the episode is over.
                truncated (bool): always False (handled inside the env).
                info (dict): extra data including "opponent_reward".

        Example:
            env = SnakeGymEnv()
            env.reset()
            obs, reward, terminated, truncated, info = env.step(0)
        """
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
        """
        Renders the current game state via the inner environment's pygame window.

        No return value. Requires pygame to be installed.
        """
        self._env.render()

    def close(self) -> None:
        """
        Closes any render resources held by the wrapped environment.

        Safe to call even if render was never used. No return value.
        """
        self._env.close_render()

    def set_opponent(self, fn: Optional[Callable]) -> None:
        """
        Swaps the opponent policy without recreating the environment.

        Args:
            fn (callable or None): new policy for agent 1. Pass None for random actions.

        Example:
            env = SnakeGymEnv()
            env.set_opponent(lambda obs: 0)  # always go UP
        """
        self._opponent_fn = fn

    @property
    def history(self) -> List[dict]:
        """
        Returns past transitions as an ordered list, oldest first.

        Each entry is a dict with keys: obs, action, reward, done.
        Empty if history_len was set to zero at construction.

        Returns:
            list of dict: recorded (obs, action, reward, done) tuples.
        """
        return list(self._history)
