import os
import glob
import re
import torch
import json
import collections

# Required for PyTorch 2.0+ secure loading
torch.serialization.add_safe_globals([collections.deque])

def recover_from_slurm_logs():
    print("Scanning directory for SLURM output logs (*.txt, *.out)...")
    log_files = glob.glob("*.txt") + glob.glob("*.out")
    
    if not log_files:
        print("Error: No .txt or .out log files found in the current directory.")
        return

    # Regex to match the manual print statements we injected
    pattern = re.compile(r">>> \[(BASE|EPI) Trial ([a-f0-9]+)\] Finished\. Rew: ([-\.\d]+) \| Len: ([-\.\d]+) \| Fitness: ([-\.\d]+)")

    best_base = {"fitness": float('inf'), "id": None, "rew": None}
    best_epi  = {"fitness": float('inf'), "id": None, "rew": None}

    # 1. Parse all text logs to find the absolute best Trial IDs
    for file in log_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    match = pattern.search(line)
                    if match:
                        model_type = match.group(1) # BASE or EPI
                        trial_id   = match.group(2)
                        rew        = float(match.group(3))
                        fitness    = float(match.group(5))

                        # Skip crashed trials (-2.0 penalty)
                        if fitness == 2.0: continue 

                        if model_type == "BASE" and fitness < best_base["fitness"]:
                            best_base = {"fitness": fitness, "id": trial_id, "rew": rew}
                        elif model_type == "EPI" and fitness < best_epi["fitness"]:
                            best_epi = {"fitness": fitness, "id": trial_id, "rew": rew}
        except Exception as e:
            print(f"Could not read {file}: {e}")

    # 2. Extract configurations directly from the PyTorch Checkpoints
    def extract_checkpoint(model_name, best_data, ckpt_dir_prefix, out_json):
        print("\n" + "=" * 60)
        if best_data["id"] is None:
            print(f"No completed {model_name} trials found in the text logs.")
            return

        trial_id = best_data["id"]
        fitness  = best_data["fitness"]
        rew      = best_data["rew"]
        
        ckpt_path = os.path.join(f"{ckpt_dir_prefix}_{trial_id}", "state.pt")
        
        print(f"[{model_name}] Winning Trial ID: {trial_id}")
        print(f"[{model_name}] Winning Reward:   {rew:.2f} (Fitness: {fitness:.4f})")
        
        if not os.path.exists(ckpt_path):
            print(f"Error: Checkpoint {ckpt_path} is missing from the disk!")
            return
        
        try:
            # Load strictly to CPU to prevent CUDA memory mapping issues on the login node
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            config_dict = ckpt.get("config_dict")
            
            if not config_dict:
                print("Error: 'config_dict' was not found inside the state.pt file.")
                return
            
            # Clean up the output to only show the relevant DEHB parameters we searched for
            # (Filters out hardware/backend settings for clarity)
            ignore_keys = ["env_backend", "torch_env_compile", "num_envs", "minibatch_size", "rollout_steps", "use_compile", "use_amp", "env_device", "compile_thread_cap", "reward_norm_warmup_steps", "speed_mode", "total_timesteps", "max_steps_per_episode"]
            clean_params = {k: v for k, v in config_dict.items() if k not in ignore_keys}

            results = {
                "best_reward": rew, 
                "fitness_score": fitness, 
                "hyperparameters": clean_params
            }
            with open(out_json, "w") as f:
                json.dump(results, f, indent=4)
                
            print(f"🏆 Successfully pried hyperparameters from PyTorch checkpoint!")
            print(f"Saved to '{out_json}'")
            
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")

    extract_checkpoint("BASE POMP", best_base, "dehb_checkpoints_base/trial", "best_base_params.json")
    extract_checkpoint("EPIPLEXITY POMP", best_epi, "dehb_checkpoints_epi/trial", "best_epiplexity_params.json")

if __name__ == "__main__":
    recover_from_slurm_logs()