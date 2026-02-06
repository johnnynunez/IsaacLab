# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate a trained Cosmos Policy in an Isaac Lab environment.

This script loads a trained Cosmos Policy model and runs closed-loop rollouts
in an Isaac Lab simulation environment, collecting success metrics and
optionally recording videos.

**Prerequisites**:
    1. cosmos-policy package installed: ``pip install -e cosmos-policy``
    2. A trained Cosmos Policy checkpoint.
    3. An Isaac Lab task with camera observations registered.

**Usage**::

    # Evaluate a trained model
    python scripts/imitation_learning/cosmos_policy/play.py \\
        --task Isaac-Lift-Franka-v0 \\
        --checkpoint_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \\
        --config_name libero_config \\
        --dataset_stats_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json \\
        --primary_image_key front_cam \\
        --proprio_key joint_pos \\
        --task_instruction "Pick up the cube." \\
        --num_envs 1 \\
        --num_episodes 10

    # With video recording
    python scripts/imitation_learning/cosmos_policy/play.py \\
        --task Isaac-Lift-Franka-v0 \\
        --checkpoint_path /path/to/checkpoint \\
        --config_name your_config \\
        --dataset_stats_path /path/to/stats.json \\
        --primary_image_key front_cam \\
        --proprio_key joint_pos \\
        --video \\
        --num_episodes 5
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate a Cosmos Policy in an Isaac Lab environment.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during rollout.")
parser.add_argument("--video_length", type=int, default=500, help="Max video length in steps.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--task", type=str, required=True, help="Isaac Lab task name.")

# Cosmos-specific arguments
parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to Cosmos Policy checkpoint.")
parser.add_argument("--config_name", type=str, default="", help="Experiment config name.")
parser.add_argument("--config_file", type=str, default="", help="Optional config file override.")
parser.add_argument(
    "--dataset_stats_path", type=str, required=True,
    help="Path to dataset statistics JSON for action un-normalization."
)
parser.add_argument("--t5_embeddings_path", type=str, default="", help="Path to pre-computed T5 embeddings.")
parser.add_argument("--task_instruction", type=str, default="Perform the task.", help="Language instruction.")
parser.add_argument("--primary_image_key", type=str, default="front_cam", help="Primary camera obs key.")
parser.add_argument("--wrist_image_key", type=str, default="", help="Wrist camera obs key (optional).")
parser.add_argument("--secondary_image_key", type=str, default="", help="Secondary camera obs key (optional).")
parser.add_argument("--proprio_key", type=str, default="joint_pos", help="Proprioception obs key.")
parser.add_argument("--obs_group", type=str, default="policy", help="Observation group name.")
parser.add_argument("--image_size", type=int, default=224, help="Image size for Cosmos model.")
parser.add_argument("--action_horizon", type=int, default=1, help="Action steps to execute per prediction.")
parser.add_argument("--num_denoising_steps", type=int, default=5, help="Denoising steps for action sampling.")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of evaluation episodes.")
parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--suite", type=str, default="isaaclab", help="Evaluation suite identifier.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras for VLA policy
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging
import os
import time
from datetime import datetime
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.cosmos_policy import CosmosPolicyEnvWrapper

logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    """Evaluate a Cosmos Policy in an Isaac Lab environment."""

    # ---- Import Cosmos Policy ----
    try:
        from cosmos_policy.experiments.robot.cosmos_utils import (
            get_action,
            get_model,
            init_t5_text_embeddings_cache,
            load_dataset_stats,
        )
    except ImportError as e:
        print(
            "\n[ERROR] cosmos-policy is not installed. Please install it:\n"
            "  cd cosmos-policy && pip install -e .\n"
        )
        raise e

    # ---- Create environment ----
    print(f"[INFO] Creating environment: {args_cli.task}")
    env = gym.make(
        args_cli.task,
        cfg=None,
        render_mode="rgb_array" if args_cli.video else None,
    )

    # Wrap for video recording
    log_dir = os.path.join("logs", "cosmos_policy", "eval", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during evaluation.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # ---- Wrap environment for Cosmos Policy ----
    env_wrapper = CosmosPolicyEnvWrapper(
        env,
        primary_image_key=args_cli.primary_image_key,
        wrist_image_key=args_cli.wrist_image_key,
        secondary_image_key=args_cli.secondary_image_key,
        proprio_key=args_cli.proprio_key,
        obs_group=args_cli.obs_group,
        image_size=args_cli.image_size,
    )

    # ---- Load Cosmos Policy model ----
    print(f"[INFO] Loading Cosmos Policy model from: {args_cli.checkpoint_path}")

    # Build a config namespace for get_model
    eval_cfg = SimpleNamespace(
        ckpt_path=args_cli.checkpoint_path,
        config=args_cli.config_name,
        config_file=args_cli.config_file if args_cli.config_file else None,
        suite=args_cli.suite,
    )
    model, model_config = get_model(eval_cfg)
    print("[INFO] Cosmos Policy model loaded successfully.")

    # Load dataset statistics for action un-normalization
    print(f"[INFO] Loading dataset stats from: {args_cli.dataset_stats_path}")
    dataset_stats = load_dataset_stats(args_cli.dataset_stats_path)

    # Initialize T5 text embeddings
    if args_cli.t5_embeddings_path:
        init_t5_text_embeddings_cache(args_cli.t5_embeddings_path)

    # ---- Run rollout episodes ----
    episode_rewards = []
    episode_lengths = []
    start_time = time.time()

    for ep in range(args_cli.num_episodes):
        cosmos_obs_list, info = env_wrapper.reset()
        episode_reward = 0.0
        step_count = 0
        done = False

        while not done and step_count < args_cli.max_steps:
            # Process each environment instance
            all_actions = []
            for env_idx in range(env_wrapper.num_envs):
                obs = cosmos_obs_list[env_idx]

                # Get action from Cosmos Policy
                with torch.inference_mode():
                    result = get_action(
                        cfg=eval_cfg,
                        model=model,
                        dataset_stats=dataset_stats,
                        obs=obs,
                        task_label_or_embedding=args_cli.task_instruction,
                        seed=args_cli.seed,
                        num_denoising_steps_action=args_cli.num_denoising_steps,
                    )

                # Extract action chunk
                if isinstance(result, dict) and "actions" in result:
                    action_chunk = result["actions"]
                elif isinstance(result, list):
                    action_chunk = np.array(result)
                elif isinstance(result, np.ndarray):
                    action_chunk = result
                else:
                    action_chunk = np.array(result)

                all_actions.append(action_chunk)

            # Execute action steps
            for t in range(args_cli.action_horizon):
                action_tensors = []
                for env_idx in range(env_wrapper.num_envs):
                    chunk = all_actions[env_idx]
                    if chunk.ndim >= 2:
                        step_action = chunk[min(t, len(chunk) - 1)]
                    else:
                        step_action = chunk
                    action_tensors.append(torch.from_numpy(step_action).float())

                action_batch = torch.stack(action_tensors).to(env_wrapper.device)
                cosmos_obs_list, reward, terminated, truncated, info = env_wrapper.step(action_batch)
                episode_reward += reward.mean().item()
                step_count += 1
                done = (terminated | truncated).any().item()
                if done:
                    break

        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        print(f"  Episode {ep + 1}/{args_cli.num_episodes}: reward={episode_reward:.3f}, length={step_count}")

    # ---- Print summary ----
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("Cosmos Policy Evaluation Summary")
    print("=" * 60)
    print(f"  Task:            {args_cli.task}")
    print(f"  Checkpoint:      {args_cli.checkpoint_path}")
    print(f"  Episodes:        {args_cli.num_episodes}")
    print(f"  Avg reward:      {np.mean(episode_rewards):.3f} +/- {np.std(episode_rewards):.3f}")
    print(f"  Avg length:      {np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f}")
    print(f"  Total time:      {elapsed:.1f}s")
    print(f"  Log directory:   {log_dir}")
    print("=" * 60)

    # Close environment
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
