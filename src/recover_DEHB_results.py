"""Recover best DEHB trials by parsing SLURM logs and checkpoints.

Useful when a run was interrupted but trial logs and checkpoint folders were
already written to disk.
"""

import os
import glob
import re
import torch
import json
import collections
from pathlib import Path

# Required for PyTorch 2.0+ secure loading
torch.serialization.add_safe_globals([collections.deque])

ROOT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT_DIR / "outputs" / "logs"
CONFIG_DIR = ROOT_DIR / "configs"
DEHB_BASE_CKPT_PREFIX = ROOT_DIR / "outputs" / \
    "checkpoints" / "dehb_base" / "trial"
DEHB_EPI_CKPT_PREFIX = ROOT_DIR / "outputs" / \
    "checkpoints" / "dehb_epi" / "trial"


def recover_from_slurm_logs():
    """
    Scans SLURM log files to find the best BASE and EPI trials and export their configs.

    Reads every .txt and .out file in the log directory, parses the per-trial
    summary lines, picks the winner for each model family, then loads the
    corresponding checkpoint and writes a cleaned JSON to the configs directory.

    Example:
        recover_from_slurm_logs()
        # writes configs/best_base_params.json and configs/best_epiplexity_params.json
    """
    print("Scanning directory for SLURM output logs (*.txt, *.out)...")
    log_files = glob.glob(str(LOG_DIR / "*.txt")) + \
        glob.glob(str(LOG_DIR / "*.out"))

    if not log_files:
        print("Error: No .txt or .out log files found in the current directory.")
        return

    # Match the per-trial summary line printed by DEHB objective calls
    pattern = re.compile(
        r">>> \[(BASE|EPI) Trial ([a-f0-9]+)\] Finished\. Rew: ([-\.\d]+) \| Len: ([-\.\d]+) \| Fitness: ([-\.\d]+)")

    best_base = {"fitness": float('inf'), "id": None, "rew": None}
    best_epi = {"fitness": float('inf'), "id": None, "rew": None}

    # Pass 1: parse logs and keep the best trial ID per model family
    for file in log_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    match = pattern.search(line)
                    if match:
                        model_type = match.group(1)
                        trial_id = match.group(2)
                        rew = float(match.group(3))
                        fitness = float(match.group(5))

                        # A fitness of 2.0 is the sentinel for a failed trial
                        if fitness == 2.0:
                            continue

                        if model_type == "BASE" and fitness < best_base["fitness"]:
                            best_base = {
                                "fitness": fitness, "id": trial_id, "rew": rew}
                        elif model_type == "EPI" and fitness < best_epi["fitness"]:
                            best_epi = {
                                "fitness": fitness, "id": trial_id, "rew": rew}
        except Exception as e:
            print(f"Could not read {file}: {e}")

    # Pass 2: pull the final hyperparameters from the winning checkpoint
    def extract_checkpoint(model_name, best_data, ckpt_dir_prefix, out_json):
        """
        Loads the winning checkpoint for one model family and exports its config.

        Args:
            model_name (str): display name used in print statements, e.g. "BASE POMP".
            best_data (dict): dict with keys "id" (str trial hash), "fitness" (float),
                and "rew" (float) from the log parsing pass.
            ckpt_dir_prefix (str): filesystem prefix; the trial folder is
                formed as prefix + "_" + trial_id.
            out_json (str): path to write the extracted hyperparameters as JSON.
        """
        print("\n" + "=" * 60)
        if best_data["id"] is None:
            print(f"No completed {model_name} trials found in the text logs.")
            return

        trial_id = best_data["id"]
        fitness = best_data["fitness"]
        rew = best_data["rew"]

        ckpt_path = os.path.join(f"{ckpt_dir_prefix}_{trial_id}", "state.pt")

        print(f"[{model_name}] Winning Trial ID: {trial_id}")
        print(f"[{model_name}] Winning Reward:   {rew:.2f} (Fitness: {fitness:.4f})")

        if not os.path.exists(ckpt_path):
            print(f"Error: Checkpoint {ckpt_path} is missing from the disk!")
            return

        try:
            # Load to CPU to avoid CUDA memory mapping issues on the login node
            ckpt = torch.load(
                ckpt_path,
                map_location="cpu",
                weights_only=False)
            config_dict = ckpt.get("config_dict")

            if not config_dict:
                print("Error: 'config_dict' was not found inside the state.pt file.")
                return

            # Filter out hardware and backend settings for a cleaner output
            ignore_keys = [
                "env_backend",
                "torch_env_compile",
                "num_envs",
                "minibatch_size",
                "rollout_steps",
                "use_compile",
                "use_amp",
                "env_device",
                "compile_thread_cap",
                "reward_norm_warmup_steps",
                "speed_mode",
                "total_timesteps",
                "max_steps_per_episode"]
            clean_params = {
                k: v for k,
                v in config_dict.items() if k not in ignore_keys}

            results = {
                "best_reward": rew,
                "fitness_score": fitness,
                "hyperparameters": clean_params
            }
            with open(out_json, "w") as f:
                json.dump(results, f, indent=4)

            print(f"Successfully extracted hyperparameters from checkpoint!")
            print(f"Saved to '{out_json}'")

        except Exception as e:
            print(f"Failed to load checkpoint: {e}")

    extract_checkpoint(
        "BASE POMP", best_base, str(DEHB_BASE_CKPT_PREFIX), str(
            CONFIG_DIR / "best_base_params.json"))
    extract_checkpoint(
        "EPIPLEXITY POMP", best_epi, str(DEHB_EPI_CKPT_PREFIX), str(
            CONFIG_DIR / "best_epiplexity_params.json"))


if __name__ == "__main__":
    recover_from_slurm_logs()
