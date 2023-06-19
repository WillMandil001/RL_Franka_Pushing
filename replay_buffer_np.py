# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lightweight in-memory replay buffer.

Adapted from https://github.com/ikostrikov/jaxrl, 
https://github.com/google-research/google-research/tree/master/xirl,
https://github.com/facebookresearch/drqv2


"""
import collections
import numpy as np
import torch

Batch = collections.namedtuple(
    "Batch", ["obses", "actions", "rewards", "next_obses", "masks"])
TensorType = torch.Tensor

class ReplayBuffer:
  """Buffer to store environment transitions."""

  def __init__(
      self,
      obs_shape,
      obs_frame_stack,
      nstep,
      discount,
      action_shape,
      batch_size,
      num_eps,
      ep_len,
      device,
  ):
    """Constructor.

    Args:
      obs_shape: The dimensions of the observation space.
      action_shape: The dimensions of the action space
      capacity: The maximum length of the replay buffer.
      device: The torch device wherein to return sampled transitions.
    """
    self.obs_shape = obs_shape
    self.obs_frame_stack = obs_frame_stack
    self.nstep = nstep
    self.discount = discount
    self.num_eps = num_eps
    self.ep_len = ep_len
    self.batch_size = batch_size
    self.device = device

    # Full buffer
    obs_dtype = np.uint8
    self.obses = self._empty_arr(obs_shape, obs_dtype) 
    self.actions = self._empty_arr(action_shape, np.float32)
    self.rewards = self._empty_arr((1,), np.float32)
    self.masks = self._empty_arr((1,), np.float32)
    # Temporary buffer to store current episode
    self.current_ep_obs = np.zeros((self.ep_len+1, *obs_shape), dtype=obs_dtype)
    self.current_ep_actions = np.zeros((self.ep_len+1, *action_shape), dtype=np.float32)
    self.current_ep_rewards = np.zeros((self.ep_len+1, 1), dtype=np.float32)
    self.current_ep_masks = np.zeros((self.ep_len+1, 1), dtype=np.float32)
    # Temporary arrays for frame stack function
    self.sampled_obses = np.zeros((batch_size, obs_frame_stack*obs_shape[0], obs_shape[-2], obs_shape[-1]), dtype=obs_dtype)
    self.sampled_next_obses = np.zeros((batch_size, obs_frame_stack*obs_shape[0], obs_shape[-2], obs_shape[-1]), dtype=obs_dtype)
    self.sampled_frames = np.zeros((self.obs_frame_stack, *obs_shape), dtype=obs_dtype)
    # Counters
    self.ep_step_counter = 0
    self.ep_counter = 0

  def _empty_arr(self, shape, dtype):
    """Creates an empty array of specified shape and type."""
    return np.zeros((self.num_eps, self.ep_len+1, *shape), dtype=dtype)

  def _to_tensor(self, arr):
    """Convert an ndarray to a torch Tensor and move it to the device."""
    return torch.as_tensor(arr, device=self.device)

  def insert(
      self,
      obs,
      action,
      reward,
      mask,
  ):
    
    """Insert an episode transition into the buffer."""
    # Add the transition to the current episode
    self.current_ep_obs[self.ep_step_counter] = obs
    self.current_ep_actions[self.ep_step_counter] = action
    self.current_ep_rewards[self.ep_step_counter] = reward
    self.current_ep_masks[self.ep_step_counter] = mask
    self.ep_step_counter +=1
    # If we are at the end of the episode, add the episode to the buffer
    if self.ep_step_counter == self.ep_len:
      self.ep_step_counter = 0
      if self.ep_counter < self.num_eps:
        self.obses[self.ep_counter] = self.current_ep_obs
        self.actions[self.ep_counter] = self.current_ep_actions
        self.rewards[self.ep_counter] = self.current_ep_rewards
        self.masks[self.ep_counter] = self.current_ep_masks
        self.ep_counter = self.ep_counter + 1
      else:
        # If we have filled the buffer, roll the buffer and add the episode to the end
        self.obses = np.roll(self.obses, -1, axis=0)
        self.actions = np.roll(self.actions, -1, axis=0)
        self.rewards = np.roll(self.rewards, -1, axis=0)
        self.masks = np.roll(self.masks, -1, axis=0)
        # Add newest episode to the end
        self.obses[-1] = self.current_ep_obs
        self.actions[-1] = self.current_ep_actions
        self.rewards[-1] = self.current_ep_rewards
        self.masks[-1] = self.current_ep_masks

  def frame_stack(self, obses, ep_idxs, step_idxs, frame_stack, storage):
    """Stacks frames from the buffer."""
    for i, idx in enumerate(step_idxs):
        # If the index is greater than the frame stack, we can just take the previous frame_stack frames
        if idx >= frame_stack -1 :
            storage[i] = np.concatenate(obses[ep_idxs[i], idx-frame_stack+1:idx+1], axis=0)
        else:
          # Otherwise, we need to pad the beginning with the initial observation of that episode
          full = idx + 1 # e.g if idx = 0 we have 1 frame, if idx = 1 we have 2 frames etc
          obs_zero = obses[ep_idxs[i], 0] # initial observation to pad with
          needed = frame_stack - full # number of frames we need to pad
          self.sampled_frames[-full:] = obses[ep_idxs[i], idx-full+1:idx+1] # Add frames we have to end of sampled frames
          for j in range(needed): # Pad the beginning of sampled frames with initial observation
              self.sampled_frames[j] = obs_zero
          storage[i] = np.concatenate(self.sampled_frames, axis=0)
    return storage

  def sample(self, batch_size):
    """Sample an episode transition from the buffer."""
    ep_idxs = np.random.randint(low=0, high=self.ep_counter, size=(batch_size,))
    step_idxs = np.random.randint(low=0, high=self.ep_len-self.nstep+1, size=(batch_size,)) + 1
    obs_step_idxs = step_idxs -1
    
    if self.obs_frame_stack > 1:
        obses = self.frame_stack(self.obses, ep_idxs, obs_step_idxs, self.obs_frame_stack, self.sampled_obses)
        next_obses = self.frame_stack(self.obses, ep_idxs, step_idxs, self.obs_frame_stack, self.sampled_next_obses)
    else:
        obses = self.obses[ep_idxs, obs_step_idxs]
        next_obses = self.obses[ep_idxs, step_idxs]

  

    return Batch(
        obses=self._to_tensor(obses),
        actions=self._to_tensor(self.actions[ep_idxs, step_idxs]),
        rewards=self._to_tensor(self.rewards[ep_idxs, step_idxs]),
        next_obses=self._to_tensor(next_obses),
        masks=self._to_tensor(self.masks[ep_idxs, step_idxs]),
    )

  def __len__(self):
    return self.ep_counter