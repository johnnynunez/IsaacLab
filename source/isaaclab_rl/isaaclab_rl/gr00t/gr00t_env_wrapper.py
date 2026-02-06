# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment wrapper that adapts Isaac Lab environments for GR00T VLA policy inference.

The wrapper translates Isaac Lab's observation dictionary into the nested
``{video: {}, state: {}, language: {}}`` format expected by
:class:`gr00t.policy.gr00t_policy.Gr00tPolicy`.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv


class Gr00tEnvWrapper(gym.Wrapper):
    """Wraps an Isaac Lab environment so that a GR00T policy can consume its observations.

    The wrapper does the following on each ``step`` / ``reset``:

    1. Collects RGB camera images from the environment and formats them
       as ``np.uint8`` arrays with shape ``(B, T, H, W, C)``.
    2. Collects proprioceptive / state vectors and formats them as
       ``np.float32`` arrays with shape ``(B, T, D)``.
    3. Attaches a language instruction as ``list[list[str]]`` with shape ``(B, 1)``.
    4. Returns the observation in GR00T's expected nested-dict format.

    After the GR00T policy returns an action chunk, the wrapper:

    5. Extracts the first ``action_horizon`` steps from the predicted action.
    6. Converts them to a ``torch.Tensor`` for Isaac Lab's ``env.step()``.

    Args:
        env: An Isaac Lab gymnasium environment (already created via ``gym.make``).
        video_obs_keys: Observation group keys that contain camera images.
            Each key should map to a tensor of shape ``(num_envs, H, W, C)``
            or ``(num_envs, C, H, W)`` (will be transposed automatically).
        state_obs_keys: Observation group keys that contain proprioceptive state.
            Each key should map to a tensor of shape ``(num_envs, D)``.
        task_instruction: The natural-language instruction for the GR00T model.
        obs_group: The observation group name to pull data from (default: ``"policy"``).

    Example:
        >>> import gymnasium as gym
        >>> env = gym.make("Isaac-Lift-Franka-v0", cfg=env_cfg)
        >>> wrapped = Gr00tEnvWrapper(
        ...     env,
        ...     video_obs_keys=["front_cam"],
        ...     state_obs_keys=["joint_pos"],
        ...     task_instruction="Pick up the cube and place it on the target.",
        ... )
        >>> obs, info = wrapped.reset()
        >>> # obs is a nested dict ready for Gr00tPolicy.get_action(obs)
    """

    def __init__(
        self,
        env: gym.Env,
        video_obs_keys: list[str],
        state_obs_keys: list[str],
        task_instruction: str = "Perform the task.",
        obs_group: str = "policy",
    ):
        super().__init__(env)

        # Validate that the underlying environment is an Isaac Lab env
        if not isinstance(env.unwrapped, (ManagerBasedRLEnv, DirectRLEnv)):
            raise ValueError(
                "Gr00tEnvWrapper requires an Isaac Lab environment (ManagerBasedRLEnv or DirectRLEnv). "
                f"Got: {type(env.unwrapped)}"
            )

        self.video_obs_keys = video_obs_keys
        self.state_obs_keys = state_obs_keys
        self.task_instruction = task_instruction
        self.obs_group = obs_group
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device

    def _tensor_to_numpy_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert a batched image tensor to a numpy array in (B, T, H, W, C) uint8 format.

        Handles both ``(B, H, W, C)`` and ``(B, C, H, W)`` inputs.
        """
        arr = tensor.detach().cpu().numpy()
        if arr.ndim == 4:
            # Detect channel-first layout: (B, C, H, W) where C is 3 or 4
            if arr.shape[1] in (3, 4) and arr.shape[1] < arr.shape[2]:
                arr = np.transpose(arr, (0, 2, 3, 1))  # -> (B, H, W, C)
            # Add temporal dimension -> (B, 1, H, W, C)
            arr = arr[:, np.newaxis, :, :, :]
        elif arr.ndim == 5:
            # Already (B, T, H, W, C) or (B, T, C, H, W)
            if arr.shape[2] in (3, 4) and arr.shape[2] < arr.shape[3]:
                arr = np.transpose(arr, (0, 1, 3, 4, 2))
        else:
            raise ValueError(f"Expected 4D or 5D image tensor, got shape {arr.shape}")

        # Keep only RGB channels
        arr = arr[..., :3]

        # Scale to uint8 if needed
        if arr.dtype in (np.float32, np.float64):
            arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)

        return arr

    def _tensor_to_numpy_state(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert a batched state tensor to (B, T, D) float32 format."""
        arr = tensor.detach().cpu().float().numpy()
        if arr.ndim == 2:
            # (B, D) -> (B, 1, D)
            arr = arr[:, np.newaxis, :]
        elif arr.ndim != 3:
            raise ValueError(f"Expected 2D or 3D state tensor, got shape {arr.shape}")
        return arr.astype(np.float32)

    def _build_gr00t_obs(self, obs_dict: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Convert Isaac Lab observation dict to GR00T nested observation format.

        Args:
            obs_dict: The raw observation dictionary from Isaac Lab. For manager-based envs
                this is ``{"policy": {"obs_key": tensor, ...}, ...}``. For direct envs it
                may be ``{"policy": tensor}``.

        Returns:
            Nested dict with ``video``, ``state``, and ``language`` keys.
        """
        # Resolve the observation group
        if isinstance(obs_dict, dict) and self.obs_group in obs_dict:
            group_obs = obs_dict[self.obs_group]
        else:
            group_obs = obs_dict

        # If group_obs is a single tensor (direct env), it will be treated as state
        if isinstance(group_obs, torch.Tensor):
            group_obs = {"state": group_obs}

        # Build video observations
        video = {}
        for key in self.video_obs_keys:
            if key in group_obs:
                video[key] = self._tensor_to_numpy_image(group_obs[key])
            else:
                raise KeyError(
                    f"Video observation key '{key}' not found in environment observations. "
                    f"Available keys: {list(group_obs.keys())}"
                )

        # Build state observations
        state = {}
        for key in self.state_obs_keys:
            if key in group_obs:
                state[key] = self._tensor_to_numpy_state(group_obs[key])
            else:
                raise KeyError(
                    f"State observation key '{key}' not found in environment observations. "
                    f"Available keys: {list(group_obs.keys())}"
                )

        # Build language observations
        language = {
            "task": [[self.task_instruction] for _ in range(self.num_envs)]
        }

        return {"video": video, "state": state, "language": language}

    def reset(self, **kwargs) -> tuple[dict[str, dict[str, Any]], dict]:
        """Reset the environment and return observations in GR00T format."""
        obs_dict, info = self.env.reset(**kwargs)
        gr00t_obs = self._build_gr00t_obs(obs_dict)
        return gr00t_obs, info

    def step(self, action: torch.Tensor):
        """Step the environment and return observations in GR00T format.

        Args:
            action: Action tensor of shape ``(num_envs, action_dim)`` to apply.

        Returns:
            Tuple of ``(obs, reward, terminated, truncated, info)`` where ``obs``
            is in GR00T nested-dict format.
        """
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        gr00t_obs = self._build_gr00t_obs(obs_dict)
        return gr00t_obs, reward, terminated, truncated, info

    @staticmethod
    def gr00t_action_to_tensor(
        action_dict: dict[str, np.ndarray],
        action_key: str,
        action_horizon: int = 1,
        device: str = "cuda:0",
    ) -> torch.Tensor:
        """Convert a GR00T action dict to an Isaac Lab action tensor.

        GR00T returns actions as ``{action_key: np.ndarray(B, T, D)}``.
        This extracts the first ``action_horizon`` steps and returns them as a
        ``torch.Tensor`` suitable for ``env.step()``.

        Args:
            action_dict: Action dictionary from ``Gr00tPolicy.get_action()``.
            action_key: The key in ``action_dict`` to use.
            action_horizon: Number of time-steps to return from the chunk.
            device: Target torch device.

        Returns:
            Action tensor of shape ``(B, D)`` (if horizon=1) or ``(B, T, D)``.
        """
        action = action_dict[action_key]  # (B, T, D)
        action = action[:, :action_horizon, :]
        action_tensor = torch.from_numpy(action).float().to(device)
        if action_horizon == 1:
            action_tensor = action_tensor.squeeze(1)  # (B, D)
        return action_tensor
