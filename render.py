"""
render_agent.py
===============
Watch a trained Snake agent play.
SPEEFD
Supports:
  • Local rendering via pygame window
  • Google Colab rendering via virtual display (Xvfb) + video recording
  • Three opponent modes: random, self (same checkpoint), vs another checkpoint

Usage
─────
  # Locally:
  python render_agent.py --checkpoint final_model.pt --opponent self --episodes 5

  # In Colab (run as a cell):
  from render_agent import render
  render(
      checkpoint  = "final_model.pt",
      opponent    = "self",            # "random" | "self" | path to .pt file
      episodes    = 5,
      colab       = True,              # record to video instead of pygame window
      video_path  = "replay.mp4",
  )
"""

import argparse
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn

from snake_env import MultiAgentSnakeEnv, Action
from ppo_snake_core import ActorCritic, PPOConfig


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_network(checkpoint_path: str, device: torch.device) -> tuple:
    """Load ActorCritic + PPOConfig from a checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct config — checkpoints saved by train_final.py include config_dict
    cfg_dict = ckpt.get("config_dict", {})
    # Strip non-PPOConfig keys that might have crept in
    import dataclasses
    valid_keys = {f.name for f in dataclasses.fields(PPOConfig)}
    cfg_dict   = {k: v for k, v in cfg_dict.items() if k in valid_keys}
    config     = PPOConfig(**cfg_dict)

    network = ActorCritic(
        grid_h=config.grid_height,
        grid_w=config.grid_width,
    ).to(device)
    network.load_state_dict(ckpt["network"])
    network.eval()
    return network, config


def _get_action(network, obs: dict, device: torch.device, W: int, H: int) -> int:
    """Run one forward pass and return the greedy action (no sampling)."""
    grid = torch.tensor(obs["grid"], dtype=torch.float32, device=device).unsqueeze(0)

    d_oh = torch.zeros(4, device=device)
    d_oh[obs["direction"]] = 1.0
    speed  = torch.tensor([obs["speed"]], dtype=torch.float32, device=device)
    credit = torch.tensor([obs["speed_credit"]], dtype=torch.float32, device=device)
    hx, hy = obs["head"]
    walls  = torch.tensor([
        hx / max(W - 1, 1),
        (W - 1 - hx) / max(W - 1, 1),
        hy / max(H - 1, 1),
        (H - 1 - hy) / max(H - 1, 1),
    ], device=device)
    scalars = torch.cat([d_oh, speed, credit, walls]).unsqueeze(0)

    with torch.no_grad():
        logits, _ = network(grid, scalars)
    return int(logits.argmax(dim=-1).item())


# ─────────────────────────────────────────────────────────────────────────────
# Colab virtual display setup
# ─────────────────────────────────────────────────────────────────────────────

def _setup_colab_display():
    """Start Xvfb virtual display for headless rendering in Colab."""
    try:
        import subprocess
        subprocess.Popen(
            ["Xvfb", ":99", "-screen", "0", "1280x720x24"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.environ["DISPLAY"] = ":99"
        time.sleep(0.5)
        print("[Render] Virtual display started on :99")
    except FileNotFoundError:
        print("[Render] Xvfb not found — installing...")
        os.system("apt-get install -y xvfb > /dev/null 2>&1")
        _setup_colab_display()


def _frames_to_video(frames: list, video_path: str, fps: int = 10):
    """Save a list of numpy RGB frames to an mp4 video."""
    try:
        import cv2
        H, W, _ = frames[0].shape
        writer  = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (W, H),
        )
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"[Render] Video saved to {video_path}  ({len(frames)} frames, {fps} fps)")
    except ImportError:
        print("[Render] cv2 not found — saving as GIF instead")
        try:
            from PIL import Image
            gif_path = video_path.replace(".mp4", ".gif")
            imgs = [Image.fromarray(f) for f in frames]
            imgs[0].save(
                gif_path,
                save_all=True,
                append_images=imgs[1:],
                duration=int(1000 / fps),
                loop=0,
            )
            print(f"[Render] GIF saved to {gif_path}")
        except ImportError:
            print("[Render] Neither cv2 nor PIL available — cannot save video")


# ─────────────────────────────────────────────────────────────────────────────
# Main render function
# ─────────────────────────────────────────────────────────────────────────────

def render(
    checkpoint:  str   = "final_model.pt",
    opponent:    str   = "self",
    episodes:    int   = 5,
    fps:         int   = 10,
    colab:       bool  = False,
    video_path:  str   = "replay.mp4",
    cell_size:   int   = 28,
) -> None:
    """
    Watch the trained agent play.

    Parameters
    ----------
    checkpoint : str
        Path to the .pt checkpoint to render.
    opponent : str
        "random"         — agent-1 picks random actions
        "self"           — agent-1 uses the same checkpoint
        "/path/to/b.pt"  — agent-1 uses a different checkpoint
    episodes : int
        Number of episodes to play.
    fps : int
        Frames per second (local window speed / video fps).
    colab : bool
        If True, use virtual display + record to video.
        If False, open a pygame window directly.
    video_path : str
        Output path for the video (Colab mode only).
    cell_size : int
        Pygame cell size in pixels (local mode only).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load agent ────────────────────────────────────────────────────
    if not os.path.exists(checkpoint):
        print(f"[Render] Checkpoint not found: {checkpoint}")
        return

    print(f"[Render] Loading {checkpoint} ...")
    agent, config = _load_network(checkpoint, device)
    W, H = config.grid_width, config.grid_height
    speed_mode = getattr(config, "speed_mode", True)
    if speed_mode:
        print("[Render] Speed mode: ON — faster snakes act more per step")
    else:
        print("[Render] Speed mode: OFF")

    # ── Load opponent ─────────────────────────────────────────────────
    if opponent == "random":
        opp_network = None
        print("[Render] Opponent: random")
    elif opponent == "self":
        opp_network = agent   # same weights, separate forward calls
        print("[Render] Opponent: self-play (same checkpoint)")
    elif os.path.exists(opponent):
        opp_network, _ = _load_network(opponent, device)
        print(f"[Render] Opponent: {opponent}")
    else:
        print(f"[Render] Opponent path not found: {opponent} — falling back to random")
        opp_network = None

    # ── Environment ───────────────────────────────────────────────────
    env = MultiAgentSnakeEnv(
        grid_width       = W,
        grid_height      = H,
        survival_reward  = config.survival_reward,
        food_reward      = config.food_reward,
        death_penalty    = config.death_penalty,
        win_reward       = config.win_reward,
        distance_shaping = config.distance_shaping,
        speed_mode       = speed_mode,
        seed             = 42,
    )

    # ── Colab: virtual display setup ──────────────────────────────────
    if colab:
        _setup_colab_display()

    import pygame
    pygame.init()
    screen_w = W * cell_size + 16
    screen_h = H * cell_size + 16 + 72
    screen   = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Snake RL — Replay")
    clock    = pygame.time.Clock()

    all_frames  = []   # for Colab video
    ep_rewards  = []
    ep_lengths  = []
    ep_winners  = []

    print(f"\n{'─'*50}")
    print(f"  Rendering {episodes} episodes")
    print(f"  Grid: {W}×{H}   FPS: {fps}")
    print(f"{'─'*50}")

    for ep in range(episodes):
        obs  = env.reset()
        done = False
        ep_r = {0: 0.0, 1: 0.0}
        steps = 0

        while not done:
            # Handle pygame quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Agent-0 action
            a0 = _get_action(agent, obs[0], device, W, H)

            # Agent-1 action
            if opp_network is None:
                a1 = random.randint(0, 3)
            else:
                a1 = _get_action(opp_network, obs[1], device, W, H)

            actions = {0: a0, 1: a1}

            # ── The Environment now natively handles Speed Mode internally ──
            # 1 step() call = 1 global physics tick = 1 rendered frame.
            obs, rews, dones, _ = env.step(actions)
            
            ep_r[0] += rews.get(0, 0.0)
            ep_r[1] += rews.get(1, 0.0)
            done     = dones.get("__all__", False)

            # Render frame
            env.render(cell_size=cell_size)
            pygame.display.flip()
            
            if colab:
                frame = pygame.surfarray.array3d(screen)
                all_frames.append(np.transpose(frame, (1, 0, 2)))
            
            clock.tick(fps)
            steps += 1
        winner = env.winner
        ep_rewards.append(ep_r[0])
        ep_lengths.append(steps)
        ep_winners.append(winner)

        result = "Draw" if winner is None else (
            "Agent wins!" if winner == 0 else "Opponent wins"
        )
        print(
            f"  Episode {ep+1:2d}/{episodes}  |  "
            f"agent reward: {ep_r[0]:+.2f}  "
            f"opp reward: {ep_r[1]:+.2f}  "
            f"steps: {steps:4d}  |  {result}"
        )

    pygame.quit()

    # ── Save video (Colab) ─────────────────────────────────────────────
    if colab and all_frames:
        _frames_to_video(all_frames, video_path, fps=fps)
        # Display inline in Colab
        try:
            from IPython.display import Video, display as ipy_display
            ipy_display(Video(video_path, embed=True))
        except Exception:
            pass

    # ── Summary ───────────────────────────────────────────────────────
    wins  = sum(1 for w in ep_winners if w == 0)
    draws = sum(1 for w in ep_winners if w is None)
    print(f"\n{'─'*50}")
    print(f"  Results over {episodes} episodes:")
    print(f"    Agent wins  : {wins}  ({100*wins/episodes:.0f}%)")
    print(f"    Draws       : {draws}")
    print(f"    Losses      : {episodes - wins - draws}")
    print(f"    Mean reward : {np.mean(ep_rewards):+.3f}")
    print(f"    Mean length : {np.mean(ep_lengths):.0f} steps")
    print(f"{'─'*50}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a trained Snake agent")
    parser.add_argument("--checkpoint", default="final_checkpoints/step_055M.pt",
                        help="Path to checkpoint .pt file")
    parser.add_argument("--opponent",   default="self",
                        help="'random', 'self', or path to opponent .pt file")
    parser.add_argument("--episodes",   type=int, default=5)
    parser.add_argument("--fps",        type=int, default=10)
    parser.add_argument("--colab",      action="store_true",
                        help="Use virtual display + record video (for Colab)")
    parser.add_argument("--video",      default="replay.mp4",
                        help="Output video path (Colab mode only)")
    parser.add_argument("--cell-size",  type=int, default=28,
                        help="Pygame cell size in pixels (local mode only)")
    args = parser.parse_args(['--cell-size','40'])

    render(
        checkpoint = args.checkpoint,
        opponent   = args.opponent,
        episodes   = args.episodes,
        fps        = args.fps,
        colab      = args.colab,
        video_path = args.video,
        cell_size  = args.cell_size,
    )
