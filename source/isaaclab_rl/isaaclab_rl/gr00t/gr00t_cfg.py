# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for Isaac-GR00T integration with Isaac Lab.

These configs follow the Isaac Lab ``@configclass`` pattern and map to the
parameters expected by the GR00T finetuning and evaluation pipelines.
"""

from __future__ import annotations

from dataclasses import MISSING, field
from typing import Literal

from isaaclab.utils import configclass


@configclass
class Gr00tFinetuneRunnerCfg:
    """Configuration for finetuning a GR00T N1.6 model on Isaac Lab demonstration data.

    This maps to the fields of ``gr00t.configs.finetune_config.FinetuneConfig``
    and ``gr00t.configs.base_config.Config``.
    """

    # -- Model --
    base_model_path: str = MISSING
    """Path to the pretrained GR00T base model (local path or HuggingFace repo ID).

    Example: ``"nvidia/GR00T-N1.5-3B"`` or a local directory.
    """

    embodiment_tag: str = MISSING
    """Embodiment tag that identifies the robot morphology.

    Must correspond to a registered :class:`gr00t.data.embodiment_tags.EmbodimentTag` value,
    e.g. ``"new_embodiment"`` for a custom embodiment or ``"libero_panda"`` for LIBERO Panda.
    """

    modality_config_path: str | None = None
    """Optional path to a custom modality configuration Python file.

    If your robot requires a new embodiment with custom observation/action modalities,
    provide the path to the config file that registers them.
    """

    # -- Training --
    max_steps: int = 10000
    """Maximum number of training steps."""

    learning_rate: float = 2e-5
    """Peak learning rate for the optimizer."""

    global_batch_size: int = 64
    """Effective global batch size across all GPUs."""

    gradient_accumulation_steps: int = 1
    """Number of gradient accumulation steps."""

    weight_decay: float = 1e-4
    """Weight decay for the AdamW optimizer."""

    warmup_ratio: float = 0.05
    """Fraction of total steps used for learning-rate warmup."""

    num_gpus: int = 1
    """Number of GPUs to use for training."""

    dataloader_num_workers: int = 8
    """Number of data-loader workers per GPU."""

    # -- Tunable components --
    tune_llm: bool = False
    """Whether to unfreeze and finetune the LLM backbone layers."""

    tune_visual: bool = False
    """Whether to unfreeze and finetune the vision encoder."""

    tune_projector: bool = True
    """Whether to unfreeze and finetune the vision-language projector."""

    tune_diffusion_model: bool = True
    """Whether to unfreeze and finetune the diffusion action head."""

    # -- Augmentation --
    state_dropout_prob: float = 0.0
    """Probability of dropping state inputs during training for robustness."""

    random_rotation_angle: float = 0.0
    """Maximum random rotation angle (degrees) for image augmentation."""

    color_jitter_params: list[float] | None = None
    """Parameters for color jitter augmentation ``[brightness, contrast, saturation, hue]``."""

    # -- Data --
    dataset_path: str = MISSING
    """Path to the LeRobot-format dataset directory used for finetuning."""

    shard_size: int = 50
    """Number of episodes per data shard for efficient streaming."""

    episode_sampling_rate: float = 1.0
    """Fraction of episodes to use per epoch (1.0 = all episodes)."""

    num_shards_per_epoch: int | None = None
    """If set, limits the number of shards loaded per epoch."""

    # -- Checkpointing --
    output_dir: str = "logs/gr00t/finetune"
    """Directory where training outputs (checkpoints, logs) are saved."""

    save_steps: int = 500
    """Save a checkpoint every N training steps."""

    save_total_limit: int = 5
    """Maximum number of checkpoints to keep on disk."""

    # -- Logging --
    use_wandb: bool = False
    """Whether to log training metrics to Weights & Biases."""

    wandb_project: str = "isaaclab-gr00t-finetune"
    """W&B project name."""

    experiment_name: str = "gr00t_finetune"
    """Human-readable name for this experiment."""

    seed: int = 42
    """Random seed for reproducibility."""

    device: str = "cuda:0"
    """Device for training (only relevant for single-GPU; multi-GPU uses all available)."""


@configclass
class Gr00tEvalCfg:
    """Configuration for evaluating (rolling out) a GR00T policy inside Isaac Lab."""

    model_path: str = MISSING
    """Path to the finetuned GR00T model checkpoint directory."""

    embodiment_tag: str = MISSING
    """Embodiment tag that identifies the robot morphology."""

    device: str | int = "cuda:0"
    """Device to run inference on."""

    task_instruction: str = "Perform the task."
    """Natural-language task instruction fed to the VLA model."""

    action_horizon: int = 1
    """Number of future action steps to execute from each prediction chunk.

    GR00T predicts a chunk of future actions; this controls how many
    steps to actually execute before re-querying the model.
    """

    video_keys: list[str] = field(default_factory=lambda: ["front_cam"])
    """List of camera observation keys from the Isaac Lab environment.

    These are mapped to the GR00T ``video`` modality.
    """

    state_keys: list[str] = field(default_factory=lambda: ["joint_pos"])
    """List of state observation keys from the Isaac Lab environment.

    These are mapped to the GR00T ``state`` modality.
    """

    action_key: str = "joint_pos"
    """The action key name in GR00T's output to use as the environment action."""

    experiment_name: str = "gr00t_eval"
    """Name for this evaluation run (used in logging directories)."""

    seed: int = 42
    """Random seed."""

    num_episodes: int = 10
    """Number of rollout episodes to run for evaluation."""

    max_steps_per_episode: int = 500
    """Maximum number of steps per evaluation episode."""

    video_record: bool = False
    """Whether to record rollout videos."""
