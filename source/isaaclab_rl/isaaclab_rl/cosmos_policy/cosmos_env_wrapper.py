# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment wrapper that adapts Isaac Lab environments for Cosmos Policy inference.

The wrapper translates Isaac Lab observations into the dict format expected by
``cosmos_policy.experiments.robot.cosmos_utils.get_action()``:

- ``primary_image``: Third-person camera image ``(H, W, 3)`` uint8.
- ``wrist_image``: Wrist camera image ``(H, W, 3)`` uint8 (optional).
- ``secondary_image``: Secondary camera image ``(H, W, 3)`` uint8 (optional).
- ``proprio``: Proprioceptive state vector ``(D,)`` float32.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv


class CosmosPolicyEnvWrapper(gym.Wrapper):
    """Wraps an Isaac Lab environment so that a Cosmos Policy can consume its observations.

    The wrapper does the following on each ``step`` / ``reset``:

    1. Collects RGB camera images and resizes them to the Cosmos model's expected size.
    2. Collects proprioceptive state vectors.
    3. Returns a flat observation dict per environment instance, suitable for
       ``cosmos_utils.get_action()``.

    After the Cosmos Policy returns actions, the wrapper:

    4. Un-normalizes and converts them to a ``torch.Tensor`` for ``env.step()``.

    This wrapper is designed for single-environment rollouts. For batched environments,
    the caller iterates over the batch dimension.

    Args:
        env: An Isaac Lab gymnasium environment (already created via ``gym.make``).
        primary_image_key: Observation key for the primary (third-person) camera.
        wrist_image_key: Observation key for the wrist camera (empty string to skip).
        secondary_image_key: Observation key for a secondary camera (empty string to skip).
        proprio_key: Observation key for proprioception / state.
        obs_group: The observation group name to pull data from.
        image_size: Target image size for the Cosmos model (square resize).

    Example:
        >>> import gymnasium as gym
        >>> env = gym.make("Isaac-Lift-Franka-v0", cfg=env_cfg)
        >>> wrapped = CosmosPolicyEnvWrapper(
        ...     env,
        ...     primary_image_key="front_cam",
        ...     proprio_key="joint_pos",
        ...     image_size=224,
        ... )
        >>> obs, info = wrapped.reset()
        >>> # obs is a list[dict], one per env instance
    """

    def __init__(
        self,
        env: gym.Env,
        primary_image_key: str = "front_cam",
        wrist_image_key: str = "",
        secondary_image_key: str = "",
        proprio_key: str = "joint_pos",
        obs_group: str = "policy",
        image_size: int = 224,
    ):
        super().__init__(env)

        # Validate that the underlying environment is an Isaac Lab env
        if not isinstance(env.unwrapped, (ManagerBasedRLEnv, DirectRLEnv)):
            raise ValueError(
                "CosmosPolicyEnvWrapper requires an Isaac Lab environment "
                f"(ManagerBasedRLEnv or DirectRLEnv). Got: {type(env.unwrapped)}"
            )

        self.primary_image_key = primary_image_key
        self.wrist_image_key = wrist_image_key
        self.secondary_image_key = secondary_image_key
        self.proprio_key = proprio_key
        self.obs_group = obs_group
        self.image_size = image_size
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device

    def _tensor_to_numpy_image(self, tensor: torch.Tensor, env_idx: int = 0) -> np.ndarray:
        """Convert a single-environment image tensor to ``(H, W, 3)`` uint8 numpy array.

        Handles both ``(B, H, W, C)`` and ``(B, C, H, W)`` layouts.
        Resizes to ``self.image_size``.
        """
        arr = tensor[env_idx].detach().cpu().numpy()

        # Handle channel-first
        if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[0] < arr.shape[1]:
            arr = np.transpose(arr, (1, 2, 0))

        # Keep only RGB
        arr = arr[..., :3]

        # Convert to uint8
        if arr.dtype in (np.float32, np.float64):
            arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)

        # Resize to target image size
        img = Image.fromarray(arr)
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        return np.array(img)

    def _tensor_to_numpy_proprio(self, tensor: torch.Tensor, env_idx: int = 0) -> np.ndarray:
        """Convert a single-environment state tensor to ``(D,)`` float32 numpy array."""
        return tensor[env_idx].detach().cpu().float().numpy().astype(np.float32)

    def _build_cosmos_obs(self, obs_dict: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert Isaac Lab observation dict to a list of Cosmos Policy observation dicts.

        Args:
            obs_dict: Raw observation dictionary from Isaac Lab.

        Returns:
            List of observation dicts (one per environment instance).
        """
        # Resolve the observation group
        if isinstance(obs_dict, dict) and self.obs_group in obs_dict:
            group_obs = obs_dict[self.obs_group]
        else:
            group_obs = obs_dict

        if isinstance(group_obs, torch.Tensor):
            group_obs = {"state": group_obs}

        cosmos_obs_list = []
        for env_idx in range(self.num_envs):
            obs = {}

            # Primary image (required)
            if self.primary_image_key and self.primary_image_key in group_obs:
                obs["primary_image"] = self._tensor_to_numpy_image(group_obs[self.primary_image_key], env_idx)
            else:
                raise KeyError(
                    f"Primary image key '{self.primary_image_key}' not found in observations. "
                    f"Available keys: {list(group_obs.keys())}"
                )

            # Wrist image (optional)
            if self.wrist_image_key and self.wrist_image_key in group_obs:
                obs["wrist_image"] = self._tensor_to_numpy_image(group_obs[self.wrist_image_key], env_idx)

            # Secondary image (optional)
            if self.secondary_image_key and self.secondary_image_key in group_obs:
                obs["secondary_image"] = self._tensor_to_numpy_image(group_obs[self.secondary_image_key], env_idx)

            # Proprioception (required)
            if self.proprio_key and self.proprio_key in group_obs:
                obs["proprio"] = self._tensor_to_numpy_proprio(group_obs[self.proprio_key], env_idx)
            else:
                raise KeyError(
                    f"Proprioception key '{self.proprio_key}' not found in observations. "
                    f"Available keys: {list(group_obs.keys())}"
                )

            cosmos_obs_list.append(obs)

        return cosmos_obs_list

    def reset(self, **kwargs) -> tuple[list[dict[str, Any]], dict]:
        """Reset the environment and return observations in Cosmos Policy format."""
        obs_dict, info = self.env.reset(**kwargs)
        cosmos_obs = self._build_cosmos_obs(obs_dict)
        return cosmos_obs, info

    def step(self, action: torch.Tensor):
        """Step the environment and return observations in Cosmos Policy format.

        Args:
            action: Action tensor of shape ``(num_envs, action_dim)``.

        Returns:
            Tuple of ``(obs_list, reward, terminated, truncated, info)`` where
            ``obs_list`` is a list of Cosmos Policy observation dicts.
        """
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        cosmos_obs = self._build_cosmos_obs(obs_dict)
        return cosmos_obs, reward, terminated, truncated, info

    @staticmethod
    def cosmos_action_to_tensor(
        action_chunk: np.ndarray,
        action_horizon: int = 1,
        device: str = "cuda:0",
    ) -> torch.Tensor:
        """Convert a Cosmos Policy action chunk to an Isaac Lab action tensor.

        Cosmos Policy returns action chunks of shape ``(action_chunk_size, action_dim)``.
        This extracts the first ``action_horizon`` steps.

        Args:
            action_chunk: Action chunk from ``get_action()``, shape ``(T, D)``.
            action_horizon: Number of steps to return from the chunk.
            device: Target torch device.

        Returns:
            Action tensor of shape ``(D,)`` (if horizon=1) or ``(T, D)``.
        """
        action = action_chunk[:action_horizon]
        action_tensor = torch.from_numpy(action).float().to(device)
        if action_horizon == 1:
            action_tensor = action_tensor.squeeze(0)  # (D,)
        return action_tensor
