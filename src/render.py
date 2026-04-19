"""
Watch a trained Snake agent play back a set of episodes.

Works locally via a pygame window and in Colab via a virtual display
with video export. Supports random opponents, self-play, or a separate checkpoint.
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


def _load_network(checkpoint_path: str, device: torch.device) -> tuple:
    """
    Loads an ActorCritic network and its PPOConfig from a checkpoint.

    Args:
        checkpoint_path (str): path to a .pt file saved by StatefulSnakeTrainer.save_state.
        device (torch.device): compute device for the loaded network.

    Returns:
        tuple: (network, config)
            network (ActorCritic): loaded network set to eval mode.
            config (PPOConfig): config stored alongside the weights.

    Example:
        import torch
        net, cfg = _load_network("outputs/models/final_model_base.pt", torch.device("cpu"))
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg_dict = ckpt.get("config_dict", {})
    import dataclasses
    valid_keys = {f.name for f in dataclasses.fields(PPOConfig)}
    cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid_keys}
    config = PPOConfig(**cfg_dict)

    network = ActorCritic(
        grid_h=config.grid_height,
        grid_w=config.grid_width,
    ).to(device)
    network.load_state_dict(ckpt["network"])
    network.eval()
    return network, config


def _get_action(
        network,
        obs: dict,
        device: torch.device,
        W: int,
        H: int) -> int:
    """
    Runs one greedy forward pass through the network.

    Args:
        network (ActorCritic): the policy network.
        obs (dict): observation dict from MultiAgentSnakeEnv with keys
            grid, direction, speed, speed_credit, head.
        device (torch.device): device for tensor operations.
        W (int): grid width, used to compute wall distance features.
        H (int): grid height, used to compute wall distance features.

    Returns:
        int: action index (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT) with the highest logit.
    """
    grid = torch.tensor(
        obs["grid"],
        dtype=torch.float32,
        device=device).unsqueeze(0)

    d_oh = torch.zeros(4, device=device)
    d_oh[obs["direction"]] = 1.0
    speed = torch.tensor([obs["speed"]], dtype=torch.float32, device=device)
    credit = torch.tensor([obs["speed_credit"]],
                          dtype=torch.float32, device=device)
    hx, hy = obs["head"]
    walls = torch.tensor([
        hx / max(W - 1, 1),
        (W - 1 - hx) / max(W - 1, 1),
        hy / max(H - 1, 1),
        (H - 1 - hy) / max(H - 1, 1),
    ], device=device)
    scalars = torch.cat([d_oh, speed, credit, walls]).unsqueeze(0)

    with torch.no_grad():
        logits, _ = network(grid, scalars)
    return int(logits.argmax(dim=-1).item())


def _setup_colab_display():
    """
    Starts a virtual Xvfb display for headless rendering in Colab.

    Sets the DISPLAY environment variable to :99 and sleeps briefly so
    the server is ready before pygame initializes. Installs Xvfb via apt
    if it is not already present. No return value.
    """
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
        print("[Render] Xvfb not found, installing...")
        os.system("apt-get install -y xvfb > /dev/null 2>&1")
        _setup_colab_display()


def _frames_to_video(frames: list, video_path: str, fps: int = 10):
    """
    Saves a list of RGB frames to a video file on disk.

    Falls back to GIF if cv2 is not available.

    Args:
        frames (list of np.ndarray): RGB images of shape (H, W, 3).
        video_path (str): output path; should end in .mp4.
        fps (int): playback speed in frames per second. Default 10.

    Example:
        frames = [np.zeros((720, 1280, 3), dtype=np.uint8)]
        _frames_to_video(frames, "replay.mp4", fps=15)
    """
    try:
        import cv2
        H, W, _ = frames[0].shape
        writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (W, H),
        )
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(
            f"[Render] Video saved to {video_path}  ({len(frames)} frames, {fps} fps)")
    except ImportError:
        print("[Render] cv2 not found, saving as GIF instead")
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
            print("[Render] Neither cv2 nor PIL available, cannot save video")


def render(
    checkpoint: str = "outputs/models/final_model_base_pompHPO500.pt",
    opponent: str = "self",
    episodes: int = 5,
    fps: int = 10,
    colab: bool = False,
    video_path: str = "replay.mp4",
    cell_size: int = 28,
) -> None:
    """
    Watches the trained agent play a set number of episodes and prints results.

    Args:
        checkpoint (str): path to the .pt checkpoint to load.
        opponent (str): "random" for random actions, "self" for the same checkpoint,
            or a file path to a different .pt checkpoint.
        episodes (int): number of episodes to play. Default 5.
        fps (int): frames per second for the pygame window or output video. Default 10.
        colab (bool): if True, use a virtual display and record to video.
            If False, open a pygame window directly. Default False.
        video_path (str): output path for the video in Colab mode. Default "replay.mp4".
        cell_size (int): pygame cell size in pixels for local mode. Default 28.

    Example:
        render(checkpoint="outputs/models/final_model_base.pt", episodes=3, fps=15)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint):
        print(f"[Render] Checkpoint not found: {checkpoint}")
        return

    print(f"[Render] Loading {checkpoint} ...")
    agent, config = _load_network(checkpoint, device)
    W, H = config.grid_width, config.grid_height
    speed_mode = getattr(config, "speed_mode", True)
    if speed_mode:
        print("[Render] Speed mode: ON, faster snakes act more per step")
    else:
        print("[Render] Speed mode: OFF")

    if opponent == "random":
        opp_network = None
        print("[Render] Opponent: random")
    elif opponent == "self":
        opp_network = agent
        print("[Render] Opponent: self-play (same checkpoint)")
    elif os.path.exists(opponent):
        opp_network, _ = _load_network(opponent, device)
        print(f"[Render] Opponent: {opponent}")
    else:
        print(
            f"[Render] Opponent path not found: {opponent}, falling back to random")
        opp_network = None

    env = MultiAgentSnakeEnv(
        grid_width=W,
        grid_height=H,
        survival_reward=config.survival_reward,
        food_reward=config.food_reward,
        death_penalty=config.death_penalty,
        win_reward=config.win_reward,
        distance_shaping=config.distance_shaping,
        speed_mode=speed_mode,
        seed=42,
    )

    if colab:
        _setup_colab_display()

    import pygame
    pygame.init()
    screen_w = W * cell_size + 16
    screen_h = H * cell_size + 16 + 72
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Snake RL, Replay")
    clock = pygame.time.Clock()

    all_frames = []
    ep_rewards = []
    ep_lengths = []
    ep_winners = []

    print(f"\n{'_'*50}")
    print(f"  Rendering {episodes} episodes")
    print(f"  Grid: {W}x{H}   FPS: {fps}")
    print(f"{'_'*50}")

    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_r = {0: 0.0, 1: 0.0}
        steps = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            a0 = _get_action(agent, obs[0], device, W, H)

            if opp_network is None:
                a1 = random.randint(0, 3)
            else:
                a1 = _get_action(opp_network, obs[1], device, W, H)

            actions = {0: a0, 1: a1}

            # One step call equals one global physics tick and one rendered
            # frame
            obs, rews, dones, _ = env.step(actions)

            ep_r[0] += rews.get(0, 0.0)
            ep_r[1] += rews.get(1, 0.0)
            done = dones.get("__all__", False)

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

    if colab and all_frames:
        _frames_to_video(all_frames, video_path, fps=fps)
        try:
            from IPython.display import Video, display as ipy_display
            ipy_display(Video(video_path, embed=True))
        except Exception:
            pass

    wins = sum(1 for w in ep_winners if w == 0)
    draws = sum(1 for w in ep_winners if w is None)
    print(f"\n{'_'*50}")
    print(f"  Results over {episodes} episodes:")
    print(f"    Agent wins  : {wins}  ({100*wins/episodes:.0f}%)")
    print(f"    Draws       : {draws}")
    print(f"    Losses      : {episodes - wins - draws}")
    print(f"    Mean reward : {np.mean(ep_rewards):+.3f}")
    print(f"    Mean length : {np.mean(ep_lengths):.0f} steps")
    print(f"{'_'*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render a trained Snake agent")
    parser.add_argument(
        "--checkpoint",
        default="outputs/models/final_model_base_pompHPO500.pt",
        help="Path to checkpoint .pt file")
    parser.add_argument("--opponent", default="self",
                        help="'random', 'self', or path to opponent .pt file")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--colab",
        action="store_true",
        help="Use virtual display and record video (for Colab)")
    parser.add_argument("--video", default="replay.mp4",
                        help="Output video path (Colab mode only)")
    parser.add_argument("--cell-size", type=int, default=28,
                        help="Pygame cell size in pixels (local mode only)")
    args = parser.parse_args()

    render(
        checkpoint=args.checkpoint,
        opponent=args.opponent,
        episodes=args.episodes,
        fps=args.fps,
        colab=args.colab,
        video_path=args.video,
        cell_size=args.cell_size,
    )
