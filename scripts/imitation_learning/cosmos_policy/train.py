# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train a Cosmos Policy model on Isaac Lab demonstration data.

This script provides an Isaac Lab-style CLI entry point for Cosmos Policy training.
Under the hood it calls the Cosmos Policy distributed training pipeline via
``torchrun``.

**Prerequisites**:
    1. cosmos-policy package installed: ``pip install -e cosmos-policy``
    2. A demonstration dataset collected from Isaac Lab.
    3. A Cosmos Policy experiment config file (or use the provided Isaac Lab default).

**Usage**::

    # Single-GPU training
    python scripts/imitation_learning/cosmos_policy/train.py \\
        --config cosmos_policy/config/experiment/your_config.py \\
        --checkpoint_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \\
        --dataset_path /path/to/dataset \\
        --output_dir logs/cosmos_policy/train

    # Multi-GPU training (recommended)
    torchrun --nproc_per_node=4 scripts/imitation_learning/cosmos_policy/train.py \\
        --config cosmos_policy/config/experiment/your_config.py \\
        --checkpoint_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \\
        --dataset_path /path/to/dataset \\
        --num_gpus 4

    # Quick dry-run to validate config
    python scripts/imitation_learning/cosmos_policy/train.py \\
        --config cosmos_policy/config/experiment/your_config.py \\
        --dryrun

**Note**:
    Cosmos Policy uses a Python-based LazyConfig system (similar to Detectron2).
    The ``--config`` flag points to a Python file that defines the full training
    configuration. Additional overrides can be passed after ``--`` as
    ``path.key=value`` pairs.
"""

import argparse
import os
import sys
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description="Train a Cosmos Policy model on Isaac Lab demonstration data."
    )

    # -- Core --
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the Cosmos Policy experiment config Python file."
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default="",
        help="Path to a pretrained checkpoint or HuggingFace repo ID."
    )
    parser.add_argument(
        "--dataset_path", type=str, default="",
        help="Path to training dataset (overrides config)."
    )
    parser.add_argument(
        "--dataset_stats_path", type=str, default="",
        help="Path to dataset statistics JSON."
    )

    # -- Training --
    parser.add_argument("--max_steps", type=int, default=None, help="Override max training steps.")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override per-GPU batch size.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs.")

    # -- Output --
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--experiment_name", type=str, default="cosmos_policy_train", help="Experiment name.")

    # -- Logging --
    parser.add_argument("--use_wandb", action="store_true", default=False, help="Enable W&B logging.")
    parser.add_argument("--wandb_project", type=str, default="isaaclab-cosmos-policy", help="W&B project name.")

    # -- Misc --
    parser.add_argument("--dryrun", action="store_true", default=False, help="Dry run (validate config only).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Capture remaining args as config overrides
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, default=None,
        help="Config overrides in 'path.key=value' format."
    )

    args = parser.parse_args()

    # Set default output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.output_dir = os.path.join("logs", "cosmos_policy", args.experiment_name, timestamp)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] Cosmos Policy training output directory: {args.output_dir}")

    # ---- Import Cosmos Policy ----
    try:
        from cosmos_policy._src.imaginaire.config import Config, load_config, pretty_print_overrides
        from cosmos_policy._src.imaginaire.lazy_config import LazyConfig, instantiate
        from cosmos_policy._src.imaginaire.serialization import to_yaml
        from cosmos_policy._src.imaginaire.utils import distributed
        from cosmos_policy._src.imaginaire.utils.context_managers import (
            data_loader_init,
            distributed_init,
            model_init,
        )
        from cosmos_policy._src.imaginaire.utils.launch import log_reproducible_setup
    except ImportError as e:
        print(
            "\n[ERROR] cosmos-policy is not installed. Please install it:\n"
            "  cd cosmos-policy && pip install -e .\n"
        )
        raise e

    # ---- Build config overrides ----
    config_overrides = list(args.opts) if args.opts else []

    # Apply CLI overrides to the LazyConfig system
    if args.output_dir:
        config_overrides.append(f"job.path_local={args.output_dir}")
    if args.max_steps is not None:
        config_overrides.append(f"trainer.max_iter={args.max_steps}")
    if args.learning_rate is not None:
        config_overrides.append(f"optimizer.lr={args.learning_rate}")
    if args.batch_size is not None:
        config_overrides.append(f"dataloader_train.batch_size={args.batch_size}")

    # ---- Load config ----
    print(f"[INFO] Loading Cosmos Policy config from: {args.config}")
    config = load_config(args.config, config_overrides, enable_one_logger=True)

    # ---- Dry run ----
    if args.dryrun:
        from loguru import logger as logging

        logging.info(
            "Config:\n"
            + config.pretty_print(use_color=True)
            + "\n"
            + pretty_print_overrides(config_overrides, use_color=True)
        )
        os.makedirs(config.job.path_local, exist_ok=True)
        try:
            to_yaml(config, f"{config.job.path_local}/config.yaml")
        except Exception:
            LazyConfig.save_yaml(config, f"{config.job.path_local}/config.yaml")
        print(f"[INFO] Dry run complete. Config saved to: {config.job.path_local}/config.yaml")
        return

    # ---- Launch training ----
    print("[INFO] Starting Cosmos Policy training...")
    print(f"  Config:          {args.config}")
    print(f"  Checkpoint:      {args.checkpoint_path or '(from config)'}")
    print(f"  Dataset:         {args.dataset_path or '(from config)'}")
    print(f"  GPUs:            {args.num_gpus}")
    print(f"  Output:          {args.output_dir}")

    from megatron.core import parallel_state
    from torch.utils.data import DataLoader, DistributedSampler

    # Initialize distributed environment
    with distributed_init():
        distributed.init()

    # Validate and freeze config
    config.validate()
    config.freeze()

    # Create trainer
    trainer = config.trainer.type(config)

    # Create a mock args namespace for log_reproducible_setup
    mock_args = argparse.Namespace(config=args.config, opts=config_overrides, dryrun=False)
    log_reproducible_setup(config, mock_args)

    # Initialize model
    with model_init():
        model = instantiate(config.model)

    # Create dataloaders
    with data_loader_init():
        dataset = instantiate(config.dataloader_train.dataset)
        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=parallel_state.get_data_parallel_world_size(),
            rank=parallel_state.get_data_parallel_rank(),
            shuffle=True,
            seed=args.seed,
        )
        dataloader_train = DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=config.dataloader_train.batch_size,
            drop_last=config.dataloader_train.drop_last,
            num_workers=config.dataloader_train.num_workers,
            persistent_workers=config.dataloader_train.persistent_workers,
            pin_memory=config.dataloader_train.pin_memory,
            pin_memory_device=config.dataloader_train.pin_memory_device,
            timeout=config.dataloader_train.timeout,
        )

        dataloader_val = None
        if config.trainer.run_validation:
            dataset_val = instantiate(config.dataloader_val.dataset)
            sampler_val = DistributedSampler(
                dataset=dataset_val,
                num_replicas=parallel_state.get_data_parallel_world_size(),
                rank=parallel_state.get_data_parallel_rank(),
                shuffle=False,
                seed=args.seed,
            )
            dataloader_val = DataLoader(
                dataset=dataset_val,
                sampler=sampler_val,
                batch_size=config.dataloader_val.batch_size,
                drop_last=config.dataloader_val.drop_last,
                num_workers=config.dataloader_val.num_workers,
                persistent_workers=config.dataloader_val.persistent_workers,
                pin_memory=config.dataloader_val.pin_memory,
                pin_memory_device=config.dataloader_val.pin_memory_device,
                timeout=config.dataloader_val.timeout,
            )

    # Start training
    trainer.train(model, dataloader_train, dataloader_val)
    print("[INFO] Cosmos Policy training complete.")


if __name__ == "__main__":
    main()
