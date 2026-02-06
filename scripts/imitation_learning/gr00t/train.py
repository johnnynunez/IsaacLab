# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to finetune a GR00T N1.6 VLA model on Isaac Lab demonstration data.

This script provides an Isaac Lab-style CLI entry point for finetuning GR00T.
It wraps the GR00T finetuning pipeline so that you can launch it from the Isaac Lab
project root following the standard convention.

**Prerequisites**:
    1. Isaac-GR00T package installed: ``pip install -e Isaac-GR00T``
    2. A demonstration dataset in LeRobot V2 format collected from Isaac Lab.
    3. (Optional) A custom modality config if using a new embodiment.

**Usage**::

    # Basic finetuning on a dataset
    python scripts/imitation_learning/gr00t/train.py \\
        --base_model_path nvidia/GR00T-N1.5-3B \\
        --dataset_path /path/to/lerobot_dataset \\
        --embodiment_tag new_embodiment \\
        --output_dir logs/gr00t/finetune \\
        --max_steps 10000

    # With custom modality config and W&B logging
    python scripts/imitation_learning/gr00t/train.py \\
        --base_model_path nvidia/GR00T-N1.5-3B \\
        --dataset_path /path/to/lerobot_dataset \\
        --embodiment_tag my_robot \\
        --modality_config_path /path/to/my_modality_config.py \\
        --use_wandb \\
        --wandb_project isaaclab-gr00t \\
        --max_steps 20000

    # Multi-GPU training
    torchrun --nproc_per_node=4 scripts/imitation_learning/gr00t/train.py \\
        --base_model_path nvidia/GR00T-N1.5-3B \\
        --dataset_path /path/to/lerobot_dataset \\
        --embodiment_tag new_embodiment \\
        --num_gpus 4 \\
        --global_batch_size 256
"""

import argparse
import os
import sys
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description="Finetune a GR00T N1.6 VLA model on Isaac Lab demonstration data."
    )

    # -- Model --
    parser.add_argument(
        "--base_model_path", type=str, required=True,
        help="Path to the pretrained GR00T base model (local dir or HuggingFace repo ID)."
    )
    parser.add_argument(
        "--embodiment_tag", type=str, required=True,
        help="Embodiment tag (e.g., 'new_embodiment', 'libero_panda')."
    )
    parser.add_argument(
        "--modality_config_path", type=str, default=None,
        help="Optional path to a custom modality configuration Python file."
    )

    # -- Data --
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Path to the LeRobot-format dataset directory."
    )
    parser.add_argument("--shard_size", type=int, default=50, help="Episodes per data shard.")
    parser.add_argument("--episode_sampling_rate", type=float, default=1.0, help="Fraction of episodes per epoch.")
    parser.add_argument("--num_shards_per_epoch", type=int, default=None, help="Limit shards per epoch.")

    # -- Training --
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum training steps.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Peak learning rate.")
    parser.add_argument("--global_batch_size", type=int, default=64, help="Effective global batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="LR warmup fraction.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs.")
    parser.add_argument("--dataloader_num_workers", type=int, default=8, help="Data loader workers per GPU.")

    # -- Tunable components --
    parser.add_argument("--tune_llm", action="store_true", default=False, help="Finetune LLM backbone.")
    parser.add_argument("--tune_visual", action="store_true", default=False, help="Finetune vision encoder.")
    parser.add_argument("--tune_projector", action="store_true", default=True, help="Finetune projector.")
    parser.add_argument("--tune_diffusion_model", action="store_true", default=True, help="Finetune diffusion head.")

    # -- Augmentation --
    parser.add_argument("--state_dropout_prob", type=float, default=0.0, help="State dropout probability.")
    parser.add_argument("--random_rotation_angle", type=float, default=0.0, help="Max random rotation (degrees).")

    # -- Checkpointing --
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--save_steps", type=int, default=500, help="Checkpoint interval (steps).")
    parser.add_argument("--save_total_limit", type=int, default=5, help="Max checkpoints to keep.")

    # -- Logging --
    parser.add_argument("--use_wandb", action="store_true", default=False, help="Enable W&B logging.")
    parser.add_argument("--wandb_project", type=str, default="isaaclab-gr00t-finetune", help="W&B project name.")
    parser.add_argument("--experiment_name", type=str, default="gr00t_finetune", help="Experiment name.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    # Set default output directory with timestamp
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.output_dir = os.path.join("logs", "gr00t", args.experiment_name, timestamp)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] GR00T finetuning output directory: {args.output_dir}")

    # ---- Import GR00T ----
    try:
        from gr00t.configs.base_config import get_default_config
        from gr00t.configs.finetune_config import FinetuneConfig
        from gr00t.experiment.experiment import run
    except ImportError as e:
        print(
            "\n[ERROR] Isaac-GR00T is not installed. Please install it:\n"
            "  cd Isaac-GR00T && pip install -e .\n"
        )
        raise e

    # Load modality config if provided
    if args.modality_config_path is not None:
        from pathlib import Path
        import importlib

        path = Path(args.modality_config_path)
        if path.exists() and path.suffix == ".py":
            sys.path.append(str(path.parent))
            importlib.import_module(path.stem)
            print(f"[INFO] Loaded custom modality config: {path}")
        else:
            raise FileNotFoundError(f"Modality config not found: {args.modality_config_path}")

    # ---- Build GR00T config ----
    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [args.dataset_path],
                        "mix_ratio": 1.0,
                        "embodiment_tag": args.embodiment_tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    # Model settings
    config.model.tune_llm = args.tune_llm
    config.model.tune_visual = args.tune_visual
    config.model.tune_projector = args.tune_projector
    config.model.tune_diffusion_model = args.tune_diffusion_model
    config.model.state_dropout_prob = args.state_dropout_prob
    config.model.random_rotation_angle = args.random_rotation_angle
    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.eagle_collator = True
    config.model.model_name = "nvidia/Eagle-Block2A-2B-v2"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    # Training settings
    config.training.start_from_checkpoint = args.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = args.global_batch_size
    config.training.dataloader_num_workers = args.dataloader_num_workers
    config.training.learning_rate = args.learning_rate
    config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.training.output_dir = args.output_dir
    config.training.save_steps = args.save_steps
    config.training.save_total_limit = args.save_total_limit
    config.training.num_gpus = args.num_gpus
    config.training.use_wandb = args.use_wandb
    config.training.max_steps = args.max_steps
    config.training.weight_decay = args.weight_decay
    config.training.warmup_ratio = args.warmup_ratio
    config.training.wandb_project = args.wandb_project

    # Data settings
    config.data.shard_size = args.shard_size
    config.data.episode_sampling_rate = args.episode_sampling_rate
    if args.num_shards_per_epoch is not None:
        config.data.num_shards_per_epoch = args.num_shards_per_epoch

    # ---- Launch training ----
    print("[INFO] Starting GR00T finetuning...")
    print(f"  Base model:      {args.base_model_path}")
    print(f"  Dataset:         {args.dataset_path}")
    print(f"  Embodiment:      {args.embodiment_tag}")
    print(f"  Max steps:       {args.max_steps}")
    print(f"  Batch size:      {args.global_batch_size}")
    print(f"  Learning rate:   {args.learning_rate}")
    print(f"  GPUs:            {args.num_gpus}")
    print(f"  Output:          {args.output_dir}")

    run(config)
    print("[INFO] GR00T finetuning complete.")


if __name__ == "__main__":
    main()
