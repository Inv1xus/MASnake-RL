"""DEHB sweep for the Epiplexity POMDP trainer.

This script defines the search space, evaluates one DEHB trial at a time,
and writes the best result to JSON at the end of the run.
"""

import os
import torch
import ConfigSpace as CS
import time
import json
import hashlib
import collections
import numpy as np
from pathlib import Path

from dehb import DEHB
from async_vec_env import close_all_cached

from ppo_snake_epiplexity_POMP import EpiplexityConfig, EpiplexityTrainer

torch.serialization.add_safe_globals([collections.deque])

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT_DIR / "configs"
DEHB_DATA_DIR = ROOT_DIR / "data" / "dehb" / "epi"
DEHB_CKPT_DIR = ROOT_DIR / "outputs" / "checkpoints" / "dehb_epi"


def get_epiplexity_snake_space():
    """
    Builds and returns the ConfigSpace for Epiplexity POMDP hyperparameter optimization.

    Extends the base space with IDM coefficient, auxiliary loss weight, entropy
    floor, and entropy boost to cover all recurrent trainer knobs.

    Returns:
        CS.ConfigurationSpace: the search space with 19 hyperparameters.

    Example:
        cs = get_epiplexity_snake_space()
        print(cs)
    """
    cs = CS.ConfigurationSpace()
    cs.add([
        CS.UniformFloatHyperparameter("lr", 1e-5, 3e-4, log=True),
        CS.UniformFloatHyperparameter("lr_min", 1e-7, 1e-5, log=True),
        CS.UniformFloatHyperparameter("gamma", 0.90, 0.9999),
        CS.UniformFloatHyperparameter("gae_lambda", 0.80, 1.0),
        CS.UniformFloatHyperparameter("clip_eps", 0.10, 0.30),
        CS.UniformFloatHyperparameter("idm_coef", 0.05, 0.50),
        CS.UniformFloatHyperparameter("aux_loss_coef", 0.10, 1.00),
        CS.UniformFloatHyperparameter("entropy_floor", 0.01, 0.15),
        CS.UniformFloatHyperparameter("entropy_boost", 0.01, 0.05),
        CS.UniformFloatHyperparameter("entropy_coef", 0.01, 0.20),
        CS.UniformFloatHyperparameter("entropy_coef_final", 1e-4, 0.01),
        CS.UniformFloatHyperparameter("food_reward", 0.50, 5.00),
        CS.UniformFloatHyperparameter("survival_reward", -0.10, 0.01),
        CS.UniformFloatHyperparameter("death_penalty", -2.00, -0.10),
        CS.UniformFloatHyperparameter("value_loss_coef", 0.25, 1.00),
        CS.UniformFloatHyperparameter("win_reward", 0.10, 2.00),
        CS.UniformFloatHyperparameter("distance_shaping", 0.00, 0.10),
        CS.UniformFloatHyperparameter("opponent_pool_frac", 0.20, 0.80),
        CS.UniformIntegerHyperparameter("update_epochs", 2, 8),
    ])
    return cs


def dehb_objective(config, fidelity, **kwargs):
    """
    Trains one Epiplexity trial at the requested fidelity and returns a fitness payload.

    Loads any existing checkpoint for this config, trains until the fidelity step
    budget is reached, saves the checkpoint, and returns the negated combined score
    so DEHB minimizes it.

    Args:
        config (CS.Configuration): hyperparameter assignment sampled by DEHB.
        fidelity (float): step budget for this trial, between min and max fidelity.
        **kwargs: unused extra arguments passed by the DEHB scheduler.

    Returns:
        dict: keys "fitness" (float, lower is better) and "cost" (float, wall seconds).

    Example:
        result = dehb_objective(config, fidelity=10_000_000)
        print(result["fitness"])
    """
    config_dict = dict(config)
    config_str = json.dumps(config_dict, sort_keys=True)
    config_id = hashlib.md5(config_str.encode("utf-8")).hexdigest()[:8]

    checkpoint_dir = DEHB_CKPT_DIR / f"trial_{config_id}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "state.pt"

    trial_seed = int(config_id, 16) % 100_000

    # Keep infrastructure knobs fixed so DEHB spends budget on policy quality
    safe_params = dict(config_dict)
    safe_params.update({
        "gru_hidden": 512,
        "bptt_seq_len": 16,
        "feat_dim": 256,
        "env_backend": "torch",
        "use_compile": False,
        "torch_env_compile": False,
        "num_envs": 512,
        "minibatch_size": 16384,
        "rollout_steps": 256
    })

    ppo_cfg = EpiplexityConfig(**safe_params)
    ppo_cfg.seed = trial_seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = EpiplexityTrainer(ppo_cfg, device)

    if checkpoint_path.exists():
        trainer.load_state(str(checkpoint_path))

    target_steps = int(fidelity)
    start_time = time.time()
    print(
        f"\n>>> [EPI Trial {config_id}] Starting Fidelity: {target_steps:,} steps",
        flush=True)

    try:
        final_reward = trainer.train_chunk(target_steps)

        if hasattr(trainer, "ep_lens") and len(trainer.ep_lens) > 0:
            final_ep_len = float(np.mean(trainer.ep_lens))
        else:
            final_ep_len = 0.0

        trainer.save_state(str(checkpoint_path))
    except Exception as e:
        import traceback
        print(f"Error in Trial {config_id}: {e}")
        traceback.print_exc()
        final_reward, final_ep_len = -2.0, 0.0
    finally:
        trainer.close()

    execution_cost = time.time() - start_time

    combined_score = final_reward + (final_ep_len / 5.0)
    fitness = -combined_score

    print(
        f">>> [EPI Trial {config_id}] Finished. Rew: {final_reward:.2f} | Len: {final_ep_len:.1f} | Fitness: {fitness:.4f}",
        flush=True)

    return {"fitness": fitness, "cost": execution_cost}


if __name__ == "__main__":
    print("\n[Diagnostic] Script loaded. Bypassing OS buffer...", flush=True)

    DEHB_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    DEHB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("[Diagnostic] Directories verified. Loading config space...", flush=True)

    cs = get_epiplexity_snake_space()
    print(
        "[Diagnostic] Initializing DEHB optimizer... (If the script freezes here, it is an NFS/Dask network block)",
        flush=True)

    dehb = DEHB(
        f=dehb_objective,
        cs=cs,
        dimensions=len(list(cs.values())),
        min_fidelity=10_000_000,
        max_fidelity=60_000_000,
        n_workers=1,
        resume=False,
        output_path=str(DEHB_DATA_DIR),
    )

    print("\n" + "=" * 60)
    print("EPIPLEXITY HPO STARTING... (Ctrl+C to stop and get results)", flush=True)
    print("=" * 60)

    try:
        dehb.run(fevals=50)
    except KeyboardInterrupt:
        print("\n\nUser stop detected. Finalizing results...", flush=True)
    finally:
        print("Closing cached worker processes...", flush=True)
        close_all_cached()

    if dehb.inc_config is not None:
        best_params = dict(dehb.vector_to_configspace(dehb.inc_config))
        best_reward = -dehb.inc_score

        results = {"best_reward": best_reward, "hyperparameters": best_params}
        with open(CONFIG_DIR / "best_epiplexity_params.json", "w") as f:
            json.dump(results, f, indent=4)

        print("\n" + "=" * 60)
        print(f"BEST FITNESS SCORE: {best_reward:.4f}")
        print("BEST HYPERPARAMETERS:")
        for k, v in best_params.items():
            print(f"  {k:25}: {v}")
        print("=" * 60)
    else:
        print("No trials completed.")
