# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for Cosmos Policy integration with Isaac Lab.

These configs follow the Isaac Lab ``@configclass`` pattern and map to the
parameters expected by the Cosmos Policy training and evaluation pipelines.
"""

from __future__ import annotations

from dataclasses import MISSING, field
from typing import Literal

from isaaclab.utils import configclass


@configclass
class CosmosPolicyTrainCfg:
    """Configuration for training a Cosmos Policy model on Isaac Lab demonstration data.

    Cosmos Policy uses a distributed training pipeline via ``torchrun`` with a
    Python-based LazyConfig system. This config captures the key parameters that
    are passed to the training script.
    """

    # -- Model --
    config_file: str = MISSING
    """Path to the Cosmos Policy experiment config Python file.

    Example: ``"cosmos_policy/config/experiment/libero_config.py"``
    """

    checkpoint_path: str = ""
    """Path to a pretrained Cosmos-Predict2 checkpoint or HuggingFace repo ID.

    Example: ``"nvidia/Cosmos-Policy-LIBERO-Predict2-2B"`` or a local directory.
    Leave empty to train from scratch (not recommended).
    """

    # -- Training --
    max_steps: int = 50000
    """Maximum number of training steps."""

    learning_rate: float = 1e-4
    """Peak learning rate for the optimizer."""

    batch_size: int = 8
    """Per-GPU batch size."""

    num_gpus: int = 1
    """Number of GPUs to use for distributed training."""

    num_workers: int = 8
    """Number of data-loader workers per GPU."""

    gradient_accumulation_steps: int = 1
    """Number of gradient accumulation steps."""

    # -- Data --
    dataset_path: str = MISSING
    """Path to the training dataset directory."""

    dataset_stats_path: str = ""
    """Path to dataset statistics JSON file for action normalization.

    Can be a local path or HuggingFace path like
    ``"nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json"``.
    """

    suite: str = "isaaclab"
    """Name of the evaluation / data suite.

    This determines the camera layout and observation mapping.
    Use ``"isaaclab"`` for Isaac Lab environments.
    """

    # -- Image settings --
    image_size: int = 224
    """Image size expected by the Cosmos model (images are resized to this)."""

    # -- Checkpointing --
    output_dir: str = "logs/cosmos_policy/train"
    """Directory where training outputs (checkpoints, logs) are saved."""

    save_steps: int = 1000
    """Save a checkpoint every N training steps."""

    # -- Logging --
    use_wandb: bool = False
    """Whether to log training metrics to Weights & Biases."""

    wandb_project: str = "isaaclab-cosmos-policy"
    """W&B project name."""

    experiment_name: str = "cosmos_policy_train"
    """Human-readable name for this experiment."""

    seed: int = 42
    """Random seed for reproducibility."""

    device: str = "cuda:0"
    """Device for training (single-GPU only; multi-GPU uses torchrun)."""


@configclass
class CosmosPolicyEvalCfg:
    """Configuration for evaluating (rolling out) a Cosmos Policy inside Isaac Lab."""

    checkpoint_path: str = MISSING
    """Path to the trained Cosmos Policy checkpoint or HuggingFace repo ID.

    Example: ``"nvidia/Cosmos-Policy-LIBERO-Predict2-2B"`` or a local directory.
    """

    config_name: str = ""
    """Experiment config name for the model (passed to ``load_model_from_checkpoint``).

    If empty, the config is inferred from the checkpoint directory.
    """

    config_file: str = ""
    """Optional path to a config file override."""

    dataset_stats_path: str = MISSING
    """Path to dataset statistics JSON for action un-normalization.

    Can be a local path or HuggingFace path.
    """

    t5_embeddings_path: str = ""
    """Path to pre-computed T5 text embeddings pickle file.

    If empty, T5 embeddings are computed on-the-fly (requires T5 model).
    """

    suite: str = "isaaclab"
    """Name of the evaluation suite. Use ``"isaaclab"`` for Isaac Lab environments."""

    device: str = "cuda:0"
    """Device to run inference on."""

    task_instruction: str = "Perform the task."
    """Natural-language task instruction for the Cosmos model."""

    action_horizon: int = 1
    """Number of action steps to execute from each predicted action chunk."""

    num_denoising_steps: int = 5
    """Number of denoising steps for action diffusion sampling."""

    # -- Observation keys --
    primary_image_key: str = "front_cam"
    """Isaac Lab observation key for the primary (third-person) camera image."""

    wrist_image_key: str = ""
    """Isaac Lab observation key for the wrist camera image (optional)."""

    secondary_image_key: str = ""
    """Isaac Lab observation key for a secondary camera image (optional)."""

    proprio_key: str = "joint_pos"
    """Isaac Lab observation key for proprioception / state vector."""

    obs_group: str = "policy"
    """Observation group name to read from the environment."""

    # -- Image settings --
    image_size: int = 224
    """Image size expected by the Cosmos model."""

    # -- Evaluation --
    experiment_name: str = "cosmos_policy_eval"
    """Name for this evaluation run."""

    seed: int = 42
    """Random seed."""

    num_episodes: int = 10
    """Number of rollout episodes for evaluation."""

    max_steps_per_episode: int = 500
    """Maximum number of steps per evaluation episode."""

    video_record: bool = False
    """Whether to record rollout videos."""
