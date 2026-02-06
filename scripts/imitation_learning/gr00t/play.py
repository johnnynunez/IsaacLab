# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate a finetuned GR00T VLA policy in an Isaac Lab environment.

This script loads a finetuned GR00T model and runs closed-loop rollouts in an
Isaac Lab simulation environment, collecting success metrics and optionally
recording videos.

**Prerequisites**:
    1. Isaac-GR00T package installed: ``pip install -e Isaac-GR00T``
    2. A finetuned GR00T checkpoint.
    3. An Isaac Lab task with camera observations registered.

**Usage**::

    # Evaluate a finetuned model
    python scripts/imitation_learning/gr00t/play.py \\
        --task Isaac-Lift-Franka-v0 \\
        --model_path /path/to/finetuned_checkpoint \\
        --embodiment_tag new_embodiment \\
        --video_keys front_cam \\
        --state_keys joint_pos \\
        --action_key joint_pos \\
        --task_instruction "Pick up the cube and place it on the target." \\
        --num_envs 1 \\
        --num_episodes 10

    # With video recording
    python scripts/imitation_learning/gr00t/play.py \\
        --task Isaac-Lift-Franka-v0 \\
        --model_path /path/to/finetuned_checkpoint \\
        --embodiment_tag new_embodiment \\
        --video_keys front_cam \\
        --state_keys joint_pos \\
        --action_key joint_pos \\
        --video \\
        --num_episodes 5
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate a GR00T VLA policy in an Isaac Lab environment.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during rollout.")
parser.add_argument("--video_length", type=int, default=500, help="Max video length in steps.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--task", type=str, required=True, help="Isaac Lab task name (e.g., Isaac-Lift-Franka-v0).")

# GR00T-specific arguments
parser.add_argument("--model_path", type=str, required=True, help="Path to finetuned GR00T checkpoint.")
parser.add_argument("--embodiment_tag", type=str, required=True, help="GR00T embodiment tag.")
parser.add_argument("--task_instruction", type=str, default="Perform the task.", help="Language instruction.")
parser.add_argument("--video_keys", type=str, nargs="+", default=["front_cam"], help="Camera observation keys.")
parser.add_argument("--state_keys", type=str, nargs="+", default=["joint_pos"], help="State observation keys.")
parser.add_argument("--action_key", type=str, default="joint_pos", help="Action key in GR00T output.")
parser.add_argument("--action_horizon", type=int, default=1, help="Action steps to execute per prediction.")
parser.add_argument("--obs_group", type=str, default="policy", help="Observation group name.")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of evaluation episodes.")
parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")

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

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.gr00t import Gr00tEnvWrapper

logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    """Evaluate a GR00T policy in an Isaac Lab environment."""

    # ---- Import GR00T ----
    try:
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy
    except ImportError as e:
        print(
            "\n[ERROR] Isaac-GR00T is not installed. Please install it:\n"
            "  cd Isaac-GR00T && pip install -e .\n"
        )
        raise e

    # ---- Create environment ----
    print(f"[INFO] Creating environment: {args_cli.task}")
    env = gym.make(
        args_cli.task,
        cfg=None,  # Use default env config from registry
        render_mode="rgb_array" if args_cli.video else None,
    )

    # Wrap for video recording
    log_dir = os.path.join("logs", "gr00t", "eval", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
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

    # ---- Wrap environment for GR00T ----
    env_wrapper = Gr00tEnvWrapper(
        env,
        video_obs_keys=args_cli.video_keys,
        state_obs_keys=args_cli.state_keys,
        task_instruction=args_cli.task_instruction,
        obs_group=args_cli.obs_group,
    )

    # ---- Load GR00T policy ----
    print(f"[INFO] Loading GR00T model from: {args_cli.model_path}")
    embodiment_tag = EmbodimentTag(args_cli.embodiment_tag)
    policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=args_cli.model_path,
        device=args_cli.device if args_cli.device else "cuda:0",
    )
    print("[INFO] GR00T policy loaded successfully.")

    # ---- Run rollout episodes ----
    episode_rewards = []
    episode_lengths = []
    start_time = time.time()

    for ep in range(args_cli.num_episodes):
        obs, info = env_wrapper.reset()
        episode_reward = 0.0
        step_count = 0
        done = False

        while not done and step_count < args_cli.max_steps:
            # Get action from GR00T policy
            with torch.inference_mode():
                action_dict, policy_info = policy.get_action(obs)

            # Convert GR00T action to Isaac Lab tensor
            action_tensor = Gr00tEnvWrapper.gr00t_action_to_tensor(
                action_dict,
                action_key=args_cli.action_key,
                action_horizon=args_cli.action_horizon,
                device=str(env_wrapper.device),
            )

            # Execute action steps
            if args_cli.action_horizon > 1:
                # Multi-step execution from action chunk
                for t in range(min(args_cli.action_horizon, action_tensor.shape[0] if action_tensor.ndim > 1 else 1)):
                    act = action_tensor[t] if action_tensor.ndim > 1 else action_tensor
                    # Expand to match num_envs if needed
                    if act.ndim == 1:
                        act = act.unsqueeze(0).expand(env_wrapper.num_envs, -1)
                    obs, reward, terminated, truncated, info = env_wrapper.step(act)
                    episode_reward += reward.mean().item()
                    step_count += 1
                    done = (terminated | truncated).any().item()
                    if done:
                        break
            else:
                # Single-step execution
                if action_tensor.ndim == 1:
                    action_tensor = action_tensor.unsqueeze(0).expand(env_wrapper.num_envs, -1)
                obs, reward, terminated, truncated, info = env_wrapper.step(action_tensor)
                episode_reward += reward.mean().item()
                step_count += 1
                done = (terminated | truncated).any().item()

        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        print(f"  Episode {ep + 1}/{args_cli.num_episodes}: reward={episode_reward:.3f}, length={step_count}")

    # ---- Print summary ----
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("GR00T Evaluation Summary")
    print("=" * 60)
    print(f"  Task:            {args_cli.task}")
    print(f"  Model:           {args_cli.model_path}")
    print(f"  Embodiment:      {args_cli.embodiment_tag}")
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
