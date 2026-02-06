# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrappers and utilities for integrating Cosmos Policy with Isaac Lab.

Cosmos Policy is a diffusion-based world-model policy fine-tuned from
Cosmos-Predict2 that jointly predicts actions, future observations, and values.
This module provides:

- :class:`CosmosPolicyEnvWrapper`: Wraps Isaac Lab environments to provide
  observations in the format expected by Cosmos Policy (images, proprio, text).
- :class:`CosmosPolicyTrainCfg`: Configuration class for Cosmos Policy training.
- :class:`CosmosPolicyEvalCfg`: Configuration class for Cosmos Policy evaluation.

Reference:
    https://github.com/nvidia-cosmos/cosmos-policy
"""

from .cosmos_cfg import CosmosPolicyEvalCfg, CosmosPolicyTrainCfg
from .cosmos_env_wrapper import CosmosPolicyEnvWrapper

__all__ = [
    "CosmosPolicyEnvWrapper",
    "CosmosPolicyTrainCfg",
    "CosmosPolicyEvalCfg",
]
