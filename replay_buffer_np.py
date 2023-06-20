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
    "Batch", ["embs", "states", "actions", "rewards", "next_embs", "next_states", "masks"])
TensorType = torch.Tensor

class ReplayBuffer:
  """Buffer to store environment transitions."""

  def __init__(
      self,
      embs_shape,
      state_shape,
      obs_frame_stack,
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
    self.embs_shape = embs_shape
    self.state_shape = state_shape
    self.obs_frame_stack = obs_frame_stack
    self.num_eps = num_eps
    self.ep_len = ep_len
    self.batch_size = batch_size
    self.device = device

    # Full buffer
    obs_dtype = np.float32
    self.embs = self._empty_arr(embs_shape, obs_dtype) 
    self.states = self._empty_arr(state_shape, obs_dtype)
    self.actions = self._empty_arr(action_shape, np.float32)
    self.rewards = self._empty_arr((1,), np.float32)
    self.masks = self._empty_arr((1,), np.float32)
    # Temporary buffer to store current episode
    self.current_ep_embs = np.zeros((self.ep_len+1, *embs_shape), dtype=obs_dtype)
    self.current_ep_states = np.zeros((self.ep_len+1, *state_shape), dtype=obs_dtype)
    self.current_ep_actions = np.zeros((self.ep_len+1, *action_shape), dtype=np.float32)
    self.current_ep_rewards = np.zeros((self.ep_len+1, 1), dtype=np.float32)
    self.current_ep_masks = np.zeros((self.ep_len+1, 1), dtype=np.float32)
    # Temporary arrays for frame stack function
    self.sampled_embs = np.zeros((batch_size, obs_frame_stack*embs_shape[0], *embs_shape[1:]), dtype=obs_dtype)
    self.sampled_next_embs = np.zeros((batch_size, obs_frame_stack*embs_shape[0], *embs_shape[1:]), dtype=obs_dtype)
    self.temp_embs = np.zeros((self.obs_frame_stack, *embs_shape), dtype=obs_dtype)
    self.sampled_states = np.zeros((batch_size, obs_frame_stack*state_shape[0], *state_shape[1:]), dtype=obs_dtype)
    self.sampled_next_states = np.zeros((batch_size, obs_frame_stack*state_shape[0], *state_shape[1:]), dtype=obs_dtype)
    self.temp_states = np.zeros((self.obs_frame_stack, *state_shape), dtype=obs_dtype)
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
      embs,
      states,
      action,
      reward,
      mask,
  ):
    
    """Insert an episode transition into the buffer."""
    # Add the transition to the current episode
    self.current_ep_embs[self.ep_step_counter] = embs
    self.current_ep_states[self.ep_step_counter] = states
    self.current_ep_actions[self.ep_step_counter] = action
    self.current_ep_rewards[self.ep_step_counter] = reward
    self.current_ep_masks[self.ep_step_counter] = mask
    self.ep_step_counter +=1
    # If we are at the end of the episode, add the episode to the buffer
    if self.ep_step_counter == self.ep_len:
      self.ep_step_counter = 0
      if self.ep_counter < self.num_eps:
        self.embs[self.ep_counter] = self.current_ep_embs
        self.states[self.ep_counter] = self.current_ep_states
        self.actions[self.ep_counter] = self.current_ep_actions
        self.rewards[self.ep_counter] = self.current_ep_rewards
        self.masks[self.ep_counter] = self.current_ep_masks
        self.ep_counter = self.ep_counter + 1
      else:
        # If we have filled the buffer, roll the buffer and add the episode to the end
        self.embs = np.roll(self.obses, -1, axis=0)
        self.states = np.roll(self.states, -1, axis=0)
        self.actions = np.roll(self.actions, -1, axis=0)
        self.rewards = np.roll(self.rewards, -1, axis=0)
        self.masks = np.roll(self.masks, -1, axis=0)
        # Add newest episode to the end
        self.embs[-1] = self.current_ep_embs
        self.states[-1] = self.current_ep_states
        self.actions[-1] = self.current_ep_actions
        self.rewards[-1] = self.current_ep_rewards
        self.masks[-1] = self.current_ep_masks

  def frame_stack(self, obses, ep_idxs, step_idxs, frame_stack, storage, temp):
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
          temp[-full:] = obses[ep_idxs[i], idx-full+1:idx+1] # Add frames we have to end of sampled frames
          for j in range(needed): # Pad the beginning of sampled frames with initial observation
              temp[j] = obs_zero
          storage[i] = np.concatenate(temp, axis=0)
    return storage

  def sample(self, batch_size):
    """Sample an episode transition from the buffer."""
    ep_idxs = np.random.randint(low=0, high=self.ep_counter-1, size=(batch_size,))
    step_idxs = np.random.randint(low=0, high=self.ep_len, size=(batch_size,)) + 1
    obs_step_idxs = step_idxs -1
    
    if self.obs_frame_stack > 1:
        embs = self.frame_stack(self.embs, ep_idxs, obs_step_idxs, self.obs_frame_stack, self.sampled_embs, self.temp_embs)
        next_embs = self.frame_stack(self.embs, ep_idxs, step_idxs, self.obs_frame_stack, self.sampled_next_embs, self.temp_embs)
        states = self.frame_stack(self.states, ep_idxs, obs_step_idxs, self.obs_frame_stack, self.sampled_states, self.temp_states)
        next_states = self.frame_stack(self.states, ep_idxs, step_idxs, self.obs_frame_stack, self.sampled_next_states, self.temp_states)
    else:
        embs = self.embs[ep_idxs, obs_step_idxs]
        next_embs = self.embs[ep_idxs, step_idxs]
        states = self.states[ep_idxs, obs_step_idxs]
        next_states = self.states[ep_idxs, step_idxs]

    return Batch(
        embs=self._to_tensor(embs),
        states=self._to_tensor(states),
        actions=self._to_tensor(self.actions[ep_idxs, step_idxs]),
        rewards=self._to_tensor(self.rewards[ep_idxs, step_idxs]),
        next_embs=self._to_tensor(next_embs),
        next_states=self._to_tensor(next_states),
        masks=self._to_tensor(self.masks[ep_idxs, step_idxs]),
    )

  def __len__(self):
    return self.ep_counter