"""
train_final.py
==============
Master Execution Script for the Snake PPO Architecture Benchmarks.
Supports both Fully Observable (MDP) and Partially Observable (POMDP) evaluations.
Enforces strict A/B testing with hardcoded global determinism and A100 scaling.
"""

import argparse
import json
import hashlib
import os
import sys
import time
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from IPython import display

from async_vec_env import close_all_cached


# ─────────────────────────────────────────────────────────────────────────────
# ★ STRICT DETERMINISM FOR A/B TESTING ★
# ─────────────────────────────────────────────────────────────────────────────
def set_global_seed(seed: int):
    """Enforces strict determinism across all internal random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[System] Global seed securely locked to {seed}")

# ─────────────────────────────────────────────────────────────────────────────
# ★  SWITCH HERE  ★
# ─────────────────────────────────────────────────────────────────────────────
MODE = "epiplexity_pomp"   # "base" | "epiplexity" | "base_pomp" | "epiplexity_pomp"

# HARDCODED SEED FOR A/B TEST PARITY
FIXED_SEED = 0

EPIPLEXITY_GRU_HIDDEN       = 512
EPIPLEXITY_BPTT_SEQ_LEN     = 16
EPIPLEXITY_CNN_BOTTLENECK   = 256
EPIPLEXITY_ENTROPY_FLOOR    = 0.10
EPIPLEXITY_ENTROPY_BOOST    = 0.02
EPIPLEXITY_IDM_COEF         = 0.10  

TARGET_STEPS           = 500_000_000
PRINT_EVERY            = 200_000
CHECKPOINT_EVERY       = 5_000_000
LIVE_PLOT_STEP_INTERVAL = 2_000_000
PLOT_SAVE_EVERY        = 2_000_000
PLOT_SNAPSHOT_DIR      = "training_plots"

# ── DYNAMIC ROUTING MAP ──
_PATHS = {
    "base": {"params": "best_base_params.json", "final_model": "final_model_base.pt", "checkpoint_dir": "final_checkpoints_base", "plot_prefix": "base_"},
    "epiplexity": {"params": "best_epiplexity_params.json", "final_model": "final_model_epi_HPO500.pt", "checkpoint_dir": "final_checkpoints_epi", "plot_prefix": "epi_"},
    "base_pomp": {"params": "best_base_params.json", "final_model": "final_model_base_pompHPO500.pt", "checkpoint_dir": "final_checkpoints_base_pomp432", "plot_prefix": "base_pompHPO500_"},
    "epiplexity_pomp": {"params": "best_epiplexity_params.json", "final_model": "final_model_epi_pomp.pt", "checkpoint_dir": "final_checkpoints_epi_pomp", "plot_prefix": "epi_pompHPO500_"},
}

def _build_trainer(mode: str, best_params: dict, device: torch.device, seed: int):
    # ── A100 MIG (20GB VRAM) CONSTRAINTS ──
    # Scaled to perfectly saturate 1/4th of an A100's SMs while fitting in a 20GB slice
    mig20_params = {
        "num_envs": 512,            # Dropped to 512 to reduce static buffer footprint
        "rollout_steps": 256,       # Buffer size = 131,072 transitions
        "clip_eps": 0.20,           
        "update_epochs": 4,         
        "minibatch_size": 16384,    # Drastically reduced to prevent 6.75GB backprop spikes
        "env_backend": "torch",
        "torch_env_compile": True
    }
    
    safe_params = best_params.copy()
    safe_params.update(mig20_params)

    # Route to the correct Recurrent files
    if "epiplexity" in mode:
        if mode == "epiplexity_pomp":
            from ppo_snake_epiplexity_POMP import EpiplexityConfig, EpiplexityTrainer
        else:
            from ppo_snake_epiplexity import EpiplexityConfig, EpiplexityTrainer
        
        safe_params.update({
            "lr": 1e-4,                 # Tuned for the 262k transition buffer
            "entropy_coef_final": 0.001,
        })
        
        cfg = EpiplexityConfig.from_dict(safe_params)
        cfg.seed = seed
        cfg.gru_hidden = EPIPLEXITY_GRU_HIDDEN
        cfg.bptt_seq_len = EPIPLEXITY_BPTT_SEQ_LEN
        cfg.feat_dim = EPIPLEXITY_CNN_BOTTLENECK
        cfg.entropy_floor = EPIPLEXITY_ENTROPY_FLOOR
        cfg.entropy_boost = EPIPLEXITY_ENTROPY_BOOST
        cfg.idm_coef = EPIPLEXITY_IDM_COEF
        
        return EpiplexityTrainer(cfg, device), cfg

    # Route to the correct Feedforward files
    else:
        if mode == "base_pomp":
            from ppo_snake_core_POMP import PPOConfig, StatefulSnakeTrainer
        else:
            from ppo_snake_core import PPOConfig, StatefulSnakeTrainer
            
        safe_params.update({
            "lr": 2e-4,               # Tuned for the 262k transition buffer
        })
            
        cfg = PPOConfig.from_dict(safe_params)
        cfg.seed = seed
        return StatefulSnakeTrainer(cfg, device), cfg

def _make_fig(mode: str):
    if "epiplexity" in mode:
        fig, axes = plt.subplots(2, 3, figsize=(17, 8))
        titles = ["Mean Episode Reward", "Policy Loss", "Value Loss", "Entropy", "IDM Loss", "Episode Length"]
    else:
        fig, axes = plt.subplots(2, 3, figsize=(17, 8))
        titles = ["Mean Episode Reward", "Policy Loss", "Value Loss", "Entropy", "Episode Length", "Steps / sec"]
        
    labels = {
        "base": "Base PPO (Fully Observable)", 
        "epiplexity": "Epiplexity PPO (Fully Observable)", 
        "base_pomp": "Base PPO (POMDP Fog of War)",
        "epiplexity_pomp": "Epiplexity PPO (POMDP Fog of War)"
    }
    
    fig.suptitle(f"Snake PPO — Final Training Run  [{labels.get(mode, mode)}]", fontsize=13, fontweight="bold")
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title, fontsize=10); ax.set_xlabel("Steps")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M")); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, list(axes.flat)

def _should_live_plot() -> bool:
    if not sys.stdout.isatty(): return False
    try: return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except Exception: return False

def _update_plot(fig, axes, metrics, target_steps, mode: str, show: bool = True):
    if fig is None or axes is None: return
    steps = metrics["steps"]
    
    if "epiplexity" in mode:
        keys, colors, ylabels = ["reward", "pi_loss", "v_loss", "entropy", "idm_loss", "ep_len"], ["#2196F3", "#F44336", "#FF9800", "#4CAF50", "#9C27B0", "#607D8B"], ["Avg reward (last 100 eps)", "Loss", "Loss", "Nats", "CE Loss", "Steps"]
    else:
        keys, colors, ylabels = ["reward", "pi_loss", "v_loss", "entropy", "ep_len", "sps"], ["#2196F3", "#F44336", "#FF9800", "#4CAF50", "#9C27B0", "#607D8B"], ["Avg reward (last 100 eps)", "Loss", "Loss", "Nats", "Steps", "Steps/sec"]

    for ax, key, color, ylabel in zip(axes, keys, colors, ylabels):
        ax.clear(); ax.set_xlabel("Steps"); ax.set_ylabel(ylabel, fontsize=8)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M")); ax.grid(True, alpha=0.3)
        y = metrics.get(key, [])
        if len(steps) > 1 and len(y) > 1:
            ax.plot(steps, y, color=color, linewidth=1.2, alpha=0.4)
            w = min(50, max(10, len(y) // 10))
            if len(y) >= w: ax.plot(steps[w - 1:], np.convolve(y, np.ones(w) / w, mode="valid"), color=color, linewidth=2.0)
        if key == "reward":
            ax.set_title(f"Mean Episode Reward  |  {min(100, 100 * max(steps, default=0) / target_steps):.1f}% complete  |  best: {max(y, default=float('nan')):.3f}", fontsize=9)
        else: ax.set_title(key.replace("_", " ").title(), fontsize=10)
    fig.tight_layout()
    if show: display.clear_output(wait=True); display.display(fig)

def _make_callback(target_steps, print_every, checkpoint_every, checkpoint_dir, plot_prefix, live_plot, fig, axes, start_step, start_time, mode):
    state = {"last_print": start_step, "last_checkpoint": start_step, "last_plot_save": start_step, "t0": start_time}
    def callback(trainer):
        steps, metrics = trainer.total_steps, trainer._metrics
        if steps - state["last_print"] >= print_every:
            elapsed, steps_done = time.time() - state["t0"], steps - start_step
            sps, remaining = steps_done / max(elapsed, 1), (target_steps - steps) / max(steps_done / max(elapsed, 1), 1)
            
            line = (f"  [{steps/1e6:5.2f}M / {target_steps/1e6:.0f}M steps]  ep_len: {metrics['ep_len'][-1] if metrics.get('ep_len') else 0.0:5.1f}  "
                    f"reward: {float(np.mean(trainer.ep_rews)) if trainer.ep_rews else float('nan'):+.3f}  "
                    f"pi_loss: {metrics['pi_loss'][-1]:+.4f}  v_loss: {metrics['v_loss'][-1]:.4f}  entropy: {metrics['entropy'][-1]:.3f}  ")
            if "epiplexity" in mode and metrics.get("idm_loss"):
                line += f"idm_loss: {metrics['idm_loss'][-1]:.4f}  "
            line += f"sps: {sps:,.0f}  ETA: {remaining/3600:.1f}h"
            print(line, flush=True)
            state["last_print"] = steps
            
        if steps - state["last_checkpoint"] >= checkpoint_every:
            os.makedirs(checkpoint_dir, exist_ok=True); ckpt_path = f"{checkpoint_dir}/step_{steps // 1_000_000:03d}M.pt"
            trainer.save_state(ckpt_path); print(f"  [Checkpoint] Saved → {ckpt_path}"); state["last_checkpoint"] = steps
        if steps - state["last_plot_save"] >= PLOT_SAVE_EVERY and metrics["steps"]:
            os.makedirs(PLOT_SNAPSHOT_DIR, exist_ok=True); snap_path = f"{PLOT_SNAPSHOT_DIR}/{plot_prefix}step_{steps // 1_000_000:03d}M.png"
            if fig is not None and axes is not None: _update_plot(fig, axes, metrics, target_steps, mode, show=live_plot); fig.savefig(snap_path, dpi=150, bbox_inches="tight")
            else: snap_fig, snap_axes = _make_fig(mode); _update_plot(snap_fig, snap_axes, metrics, target_steps, mode, show=False); snap_fig.savefig(snap_path, dpi=150, bbox_inches="tight"); plt.close(snap_fig)
            print(f"  [Plot] Saved → {snap_path}"); state["last_plot_save"] = steps
        if live_plot and fig is not None and axes is not None and len(metrics["steps"]) % max(1, LIVE_PLOT_STEP_INTERVAL // (trainer.config.num_envs * trainer.config.rollout_steps)) == 0:
            try: _update_plot(fig, axes, metrics, target_steps, mode, show=True)
            except Exception: pass
    return callback

def run(target_steps=TARGET_STEPS, checkpoint_every=CHECKPOINT_EVERY, print_every=PRINT_EVERY, custom_params_path=None, mode=MODE) -> None:
    # 1. Engage global determinism BEFORE loading networks or environments
    set_global_seed(FIXED_SEED)

    paths = _PATHS[mode]
    
    # Resolve the correct JSON file 
    target_params = custom_params_path if custom_params_path else paths["params"]
    
    print(f"[Run] Mode: {mode.upper()}  |  Reading parameters from: {target_params}", flush=True)
    with open(target_params) as f: 
        best_params = json.load(f)["hyperparameters"]
    
    config_id = hashlib.md5(json.dumps(best_params, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Run] Device: {device}", flush=True)
    
    # 2. Build the correct trainer based on MODE
    trainer, cfg = _build_trainer(mode, best_params, device, FIXED_SEED)
    print(f"[Run] env_backend={cfg.env_backend}", flush=True)

    dehb_ckpt = f"dehb_checkpoints/trial_{config_id}/state.pt"
    try:
        if os.path.exists(paths["final_model"]): trainer.load_state(paths["final_model"]); print(f"Resuming from: {paths['final_model']}")
        elif os.path.exists(dehb_ckpt): trainer.load_state(dehb_ckpt); print(f"Resuming from DEHB checkpoint: {dehb_ckpt}")
        else: print("Starting from scratch (no checkpoint found)")
    except Exception as e: print(f"[Warn] Could not load checkpoint ({e}) — starting from scratch")

    start_step = trainer.total_steps
    print(f"Starting at step {start_step:,} / {target_steps:,}\nSteps remaining: {target_steps - start_step:,}\n" + "-" * 60)
    if start_step >= target_steps: return

    fig, axes = (_make_fig(mode) if _should_live_plot() else (None, None))
    if _should_live_plot(): plt.ion()
    trainer.enable_tracking(progress_cb=_make_callback(target_steps, print_every, checkpoint_every, paths["checkpoint_dir"], paths["plot_prefix"], _should_live_plot(), fig, axes, start_step, time.time(), mode))

    t0 = time.time()
    try:
        try: final_reward = trainer.train_chunk(target_steps)
        except KeyboardInterrupt: print("\n[Stopped by user]"); final_reward = float(np.mean(trainer.ep_rews)) if trainer.ep_rews else float("nan")
        trainer.save_state(paths["final_model"])
        print(f"\n{'=' * 60}\nTraining complete  [{mode.upper()}]\n  Total steps : {trainer.total_steps:,}\n  Final reward: {final_reward:.4f}\n  Wall time   : {(time.time() - t0) / 3600:.2f} hours\n  Saved to    : {paths['final_model']}\n{'=' * 60}")
        if fig is None or axes is None: fig, axes = _make_fig(mode)
        plt.ioff(); _update_plot(fig, axes, trainer._metrics, target_steps, mode, show=False); curve_path = f"{paths['plot_prefix']}training_curve.png"
        fig.savefig(curve_path, dpi=150, bbox_inches="tight"); print(f"Plot saved to {curve_path}")
    finally: trainer.close(); close_all_cached()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Snake PPO agent")
    parser.add_argument("--mode", default=MODE, choices=["base", "epiplexity", "base_pomp", "epiplexity_pomp"])
    parser.add_argument("--steps", type=int, default=TARGET_STEPS)
    parser.add_argument("--params", default=None, help="Optional: Override JSON parameter file path manually")
    parser.add_argument("--print-every", type=int, default=PRINT_EVERY)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    args = parser.parse_args()
    
    run(args.steps, args.checkpoint_every, args.print_every, args.params, args.mode)