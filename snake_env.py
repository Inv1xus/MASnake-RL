"""
snake_env.py
============
Multi-agent Snake environment for reinforcement learning.
"""

import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np

BASE_SPEED = 10.0
MAX_SPEED = 20.0
SPEED_GROWTH = 1.05


# ──────────────────────────────────────────────────────────────
# Actions
# ──────────────────────────────────────────────────────────────

class Action(IntEnum):
    """Discrete movement actions supported by both agents."""
    UP    = 0
    DOWN  = 1
    LEFT  = 2
    RIGHT = 3


_DIR_VEC = {
    Action.UP:    ( 0, -1),
    Action.DOWN:  ( 0,  1),
    Action.LEFT:  (-1,  0),
    Action.RIGHT: ( 1,  0),
}

_OPPOSITE = {
    Action.UP:    Action.DOWN,
    Action.DOWN:  Action.UP,
    Action.LEFT:  Action.RIGHT,
    Action.RIGHT: Action.LEFT,
}


# ──────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────

@dataclass
class SnakeAgent:
    """Mutable per-agent state container used inside each environment step."""
    agent_id:   int
    body:       List[Tuple[int, int]]
    direction:  Action
    alive:      bool = True
    score:      int  = 0
    food_eaten: int  = 0
    speed_credits: float = 0.0
    _speed_mult:   float = 1.0  # Caches the exponentiation math!

    @property
    def head(self) -> Tuple[int, int]:
        return self.body[0]

    def current_speed(self) -> float:
        return BASE_SPEED * self._speed_mult

    def speed_multiplier(self) -> float:
        return self._speed_mult

    def update_speed(self):
        """Called ONLY when eating food to avoid math overhead on every frame."""
        new_speed = min(MAX_SPEED, BASE_SPEED * (SPEED_GROWTH ** self.food_eaten))
        self._speed_mult = new_speed / BASE_SPEED


# ──────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────

class MultiAgentSnakeEnv:
    """Two-player Snake environment with reward shaping and optional speed mode."""

    N_CHANNELS = 8

    def __init__(
        self,
        grid_width:       int   = 24,
        grid_height:      int   = 18,
        num_food:         int   = 2,
        max_steps:        int   = 1_000,
        seed:             Optional[int] = None,
        survival_reward:  float = 0.01,
        food_reward:      float = 5.0,
        death_penalty:    float = -5.0,
        win_reward:       float = 3.0,
        distance_shaping: float = 0.15,
        speed_mode:       bool  = True,
    ):
        self.W  = grid_width
        self.H  = grid_height
        self.num_food         = num_food
        self.max_steps        = max_steps
        self.survival_reward  = survival_reward
        self.food_reward      = food_reward
        self.death_penalty    = death_penalty
        self.win_reward       = win_reward
        self.distance_shaping = distance_shaping
        self.speed_mode       = speed_mode

        self.rng = random.Random(seed)

        self._obs_buf = np.zeros((self.N_CHANNELS, self.H, self.W), dtype=np.float32)

        self.agents:         Dict[int, SnakeAgent] = {}
        self.food_positions: List[Tuple[int, int]] = []
        self.step_count:     int                   = 0
        self.done:           bool                  = False
        self.winner:         Optional[int]         = None

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def reset(self) -> Dict[int, dict]:
        """Reset board state and return observations for both agents."""
        cy = self.H // 2
        q1 = max(2, self.W // 4)
        q3 = min(self.W - 3, 3 * self.W // 4)

        self.agents = {
            0: SnakeAgent(
                agent_id  = 0,
                body      = [(q1, cy), (q1 + 1, cy), (q1 + 2, cy)],
                direction = Action.LEFT,
                speed_credits = 0.0,
                _speed_mult   = 1.0,
            ),
            1: SnakeAgent(
                agent_id  = 1,
                body      = [(q3, cy), (q3 - 1, cy), (q3 - 2, cy)],
                direction = Action.RIGHT,
                speed_credits = 0.0,
                _speed_mult   = 1.0,
            ),
        }

        self.food_positions = []
        self.step_count     = 0
        self.done           = False
        self.winner         = None
        self._obs_buf[:]    = 0.0

        self._refill_food()
        return self._observations()

    def step(
        self, actions: Dict[int, int]
    ) -> Tuple[Dict[int, dict], Dict[int, float], Dict[int, bool], Dict[int, dict]]:
        """Advance one tick using actions for alive agents and return Gym-style outputs."""
        if self.done:
            raise RuntimeError("Episode done — call reset() first.")

        moves = self._plan_tick_moves()
        m0 = moves.get(0, 0)
        m1 = moves.get(1, 0)
        max_moves = m0 if m0 > m1 else m1
        
        accumulated_rewards = {0: 0.0, 1: 0.0}
        last_dones = {0: False, 1: False, "__all__": False}
        last_info = {}

        # FAST PATH: 95% of frames use exactly 1 move. We skip loop overhead.
        if max_moves == 1:
            moving_agents = {aid for aid, m in moves.items() if m > 0}
            sub_actions = {aid: int(actions.get(aid, int(self.agents[aid].direction))) for aid in moving_agents}
            _, rews, dones, info = self._physical_step(sub_actions, moving_agents=moving_agents, build_obs=False)
            
            accumulated_rewards[0] = rews.get(0, 0.0)
            accumulated_rewards[1] = rews.get(1, 0.0)
            last_dones.update(dones)
            last_info.update(info)
            
        else:
            # SLOW PATH: Executed only during a multi-block momentum jump
            for sub in range(max_moves):
                moving_agents = {aid for aid, m in moves.items() if m > sub}
                if not moving_agents:
                    continue

                sub_actions = {aid: int(actions.get(aid, int(self.agents[aid].direction))) for aid in moving_agents}
                
                _, rews, dones, info = self._physical_step(sub_actions, moving_agents=moving_agents, build_obs=False)
                
                for aid, r in rews.items():
                    accumulated_rewards[aid] += r
                last_dones.update(dones)
                last_info.update(info)
                
                if last_dones.get("__all__", False):
                    break
        
        self.step_count += 1
        if self.step_count >= self.max_steps and not self.done:
            self.done   = True
            self.winner = self._score_winner()
            last_dones = {aid: (self.done or not a.alive) for aid, a in self.agents.items()}
            last_dones["__all__"] = True

        obs = self._observations()
        return obs, accumulated_rewards, last_dones, last_info

    def _plan_tick_moves(self) -> Dict[int, int]:
        moves = {}
        for aid, agent in self.agents.items():
            if not agent.alive:
                continue
            if not self.speed_mode:
                moves[aid] = 1
                continue

            agent.speed_credits += agent._speed_mult
            m = int(agent.speed_credits)
            agent.speed_credits -= m
            
            if m > 0:
                moves[aid] = m
                
        return moves

    def _physical_step(
        self,
        actions: Dict[int, int],
        moving_agents: Optional[set] = None,
        build_obs: bool = True,
    ) -> Tuple[Dict[int, dict], Dict[int, float], Dict[int, bool], Dict[int, dict]]:
        
        rewards: Dict[int, float] = {i: self.survival_reward for i in self.agents}
        info:    Dict[int, dict]  = {i: {}                   for i in self.agents}
        
        if moving_agents is None:
            moving_agents = {aid for aid, agent in self.agents.items() if agent.alive}

        # ── 1. Resolve actions ─────────
        for aid, agent in self.agents.items():
            if not agent.alive or aid not in moving_agents:
                continue
            try:
                act = Action(actions.get(aid, int(agent.direction)))
            except ValueError:
                act = agent.direction
            if len(agent.body) > 1 and act == _OPPOSITE[agent.direction]:
                act = agent.direction
            agent.direction = act

        # ── 2. Proposed next heads ─────────
        next_heads: Dict[int, Tuple[int, int]] = {}
        for aid in moving_agents:
            agent = self.agents[aid]
            if not agent.alive:
                continue
            dx, dy = _DIR_VEC[agent.direction]
            hx, hy = agent.head
            next_heads[aid] = (hx + dx, hy + dy)

        will_eat = {
            aid: (head in self.food_positions)
            for aid, head in next_heads.items()
        }

        # ── 3. Occupancy after tails vacate ─────────
        occupied: set = set()
        for aid, agent in self.agents.items():
            if not agent.alive:
                continue
            if aid in moving_agents:
                cells = agent.body if will_eat.get(aid, False) else agent.body[:-1]
            else:
                cells = agent.body
            occupied.update(cells)

        # ── 4. Detect deaths ─────────
        dead: set = set()

        for aid, head in next_heads.items():
            if not (0 <= head[0] < self.W and 0 <= head[1] < self.H):
                dead.add(aid)
                rewards[aid] += self.death_penalty
                info[aid]["death"] = "wall"
                continue
            if head in occupied:
                dead.add(aid)
                rewards[aid] += self.death_penalty
                info[aid]["death"] = "body"

        head_count: Dict[Tuple[int, int], int] = {}
        for aid, head in next_heads.items():
            if aid not in dead:
                head_count[head] = head_count.get(head, 0) + 1
        for aid, head in next_heads.items():
            if aid not in dead and head_count.get(head, 0) > 1:
                dead.add(aid)
                rewards[aid] += self.death_penalty
                info[aid]["death"] = "head_to_head"

        # ── 5. Apply moves + shaping + food ─────────
        eaten_food: set = set()

        for aid, agent in self.agents.items():
            if not agent.alive:
                continue
            if aid in dead:
                agent.alive = False
                continue
            if aid not in moving_agents:
                continue

            old_head = agent.head
            new_head = next_heads[aid]
            agent.body.insert(0, new_head)

            if self.food_positions and self.distance_shaping > 0:
                new_d = min(abs(new_head[0] - fx) + abs(new_head[1] - fy) for fx, fy in self.food_positions)
                old_d = min(abs(old_head[0] - fx) + abs(old_head[1] - fy) for fx, fy in self.food_positions)
                rewards[aid] += self.distance_shaping * (old_d - new_d)

            if will_eat.get(aid, False):
                agent.score      += 1
                agent.food_eaten += 1
                agent.update_speed() # <--- Math executes here
                rewards[aid]     += self.food_reward
                eaten_food.add(new_head)
                info[aid]["ate_food"] = True
            else:
                agent.body.pop()

        # ── 6. Refill food ─────────
        if eaten_food:
            self.food_positions = [f for f in self.food_positions if f not in eaten_food]
            self._refill_food()

        # ── 7. End conditions ─────────
        alive = [a for a in self.agents.values() if a.alive]

        if len(alive) == 1:
            self.done   = True
            self.winner = alive[0].agent_id
            rewards[alive[0].agent_id] += self.win_reward
        elif len(alive) == 0:
            self.done   = True
            self.winner = None

        obs = self._observations() if build_obs else {}
        dones = {aid: (self.done or not a.alive) for aid, a in self.agents.items()}
        dones["__all__"] = self.done

        return obs, rewards, dones, info

    # ──────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────

    def _observations(self) -> Dict[int, dict]:
        obs = {}
        for aid, agent in self.agents.items():
            obs[aid] = {
                "grid":       self._build_grid(aid),
                "direction":  int(agent.direction),
                "speed":      agent.current_speed() / 100.0,
                "speed_credit": agent.speed_credits,
                "score":      agent.score,
                "alive":      agent.alive,
                "head":       agent.head,
                "food":       list(self.food_positions),
                "step_count": self.step_count,
            }
        return obs

    def _build_grid(self, viewer_id: int) -> np.ndarray:
        self._obs_buf[:] = 0.0

        for fx, fy in self.food_positions:
            self._obs_buf[4, fy, fx] = 1.0

        for aid, agent in self.agents.items():
            if not agent.alive or not agent.body:
                continue

            is_self  = (aid == viewer_id)
            head     = agent.body[0]
            body_len = len(agent.body)

            head_ch = 0 if is_self else 2
            self._obs_buf[head_ch, head[1], head[0]] = 1.0

            if body_len > 0:
                body_ch = 1 if is_self else 3
                order_ch = 5 if is_self else 6
                
                # Draw the head's gradient value
                self._obs_buf[order_ch, head[1], head[0]] = 1.0 
                
                # Native Python loop — hyper-fast for arrays < 50 items
                for i in range(1, body_len):
                    bx, by = agent.body[i]
                    self._obs_buf[body_ch, by, bx] = 1.0
                    self._obs_buf[order_ch, by, bx] = (body_len - i) / body_len

        return self._obs_buf.copy()

    def _spawn_food(self) -> Optional[Tuple[int, int]]:
        occupied = set(self.food_positions)
        for a in self.agents.values():
            occupied.update(a.body)
        mask = np.zeros(self.W * self.H, dtype=bool)
        for x, y in occupied:
            mask[y * self.W + x] = True
        empty = np.where(~mask)[0]
        if len(empty) == 0:
            return None
        idx = self.rng.choice(empty.tolist())
        return (idx % self.W, idx // self.W)

    def _refill_food(self):
        while len(self.food_positions) < self.num_food:
            food = self._spawn_food()
            if food is None:
                break
            self.food_positions.append(food)

    def _score_winner(self) -> Optional[int]:
        scores = {aid: a.score for aid, a in self.agents.items()}
        best   = max(scores.values())
        top    = [aid for aid, s in scores.items() if s == best]
        return top[0] if len(top) == 1 else None

    # ──────────────────────────────────────────────────────────
    # Optional pygame renderer
    # ──────────────────────────────────────────────────────────

    def render(self, cell_size: int = 28, padding: int = 8):
        try:
            import pygame
        except ImportError:
            raise ImportError("pip install pygame to use render()")

        if not hasattr(self, "_pg_init"):
            pygame.init()
            self._pg_init = True
            self._cs      = cell_size
            self._pad     = padding
            self._screen  = pygame.display.set_mode((
                self.W * cell_size + 2 * padding,
                self.H * cell_size + 2 * padding + 72,
            ))
            pygame.display.set_caption("Snake RL")
            self._font = pygame.font.SysFont(None, 22)

        cs, pad, scr = self._cs, self._pad, self._screen

        def cell(x, y):
            return pygame.Rect(pad + x * cs, pad + y * cs, cs - 1, cs - 1)

        scr.fill((30, 30, 30))
        pygame.draw.rect(
            scr, (70, 70, 70),
            pygame.Rect(pad, pad, self.W * cs, self.H * cs), 1
        )

        for fx, fy in self.food_positions:
            pygame.draw.rect(scr, (60, 200, 60), cell(fx, fy))

        palette = {
            0: ((220, 220, 220), (70, 170, 150)),
            1: ((220, 100,  90), (160,  55,  55)),
        }
        for aid, agent in self.agents.items():
            hc, bc = palette.get(aid, ((180, 180, 180), (90, 90, 90)))
            for i, (x, y) in enumerate(agent.body):
                pygame.draw.rect(scr, hc if i == 0 else bc, cell(x, y))

        iy = pad + self.H * cs + 6
        for aid, agent in self.agents.items():
            t = self._font.render(
                f"P{aid+1}  score={agent.score}  alive={agent.alive}",
                True, (190, 190, 190),
            )
            scr.blit(t, (pad, iy + aid * 24))

        if self.done:
            msg = "Draw" if self.winner is None else f"Winner: P{self.winner+1}"
            t   = self._font.render(msg, True, (255, 215, 60))
            scr.blit(t, (pad, iy + 50))

        pygame.display.flip()
        try:
            pygame.event.pump()
        except Exception:
            pass

    def close_render(self):
        if hasattr(self, "_pg_init"):
            import pygame
            pygame.quit()
            del self._pg_init