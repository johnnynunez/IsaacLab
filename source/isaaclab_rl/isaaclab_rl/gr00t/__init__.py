# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrappers and utilities for integrating Isaac-GR00T (N1.6) VLA models with Isaac Lab.

Isaac-GR00T is a Vision-Language-Action (VLA) foundation model that uses a
diffusion transformer action head on top of a vision-language backbone.
This module provides:

- :class:`Gr00tEnvWrapper`: Wraps Isaac Lab environments to provide observations
  in the format expected by GR00T (video, state, language).
- :class:`Gr00tFinetuneRunnerCfg`: Configuration class for GR00T finetuning.
- :class:`Gr00tEvalCfg`: Configuration class for GR00T evaluation / rollout.

Reference:
    https://github.com/NVIDIA-Omniverse/Isaac-GR00T
"""

from .gr00t_cfg import Gr00tEvalCfg, Gr00tFinetuneRunnerCfg
from .gr00t_env_wrapper import Gr00tEnvWrapper

__all__ = [
    "Gr00tEnvWrapper",
    "Gr00tFinetuneRunnerCfg",
    "Gr00tEvalCfg",
]
