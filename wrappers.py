import gymnasium as gym
from gymnasium.spaces import Box, Dict
import numpy as np
from collections import deque
import cv2
import imageio
import os

class FrameStack(gym.Wrapper):
    def __init__(self, env, num_frames=3):
        self._env = env
        self._num_frames = num_frames
        self._embedding_shape = env.observation_space['embeddings'].shape
        self._state_shape = env.observation_space['state'].shape
        self._embedding_frames = deque([], maxlen=num_frames)
        self._state_frames = deque([], maxlen=num_frames)
        self.observation_space = Dict({"state": Box(low=-np.inf, high=np.inf, shape=(num_frames, *self._state_shape), dtype=np.float32),
                                       "embeddings": Box(low=-np.inf, high=np.inf, shape=(num_frames, *self._embedding_shape), dtype=np.float32)})
        self.action_space = env.action_space


    def step(self, action):

        obs, reward, terminated, truncated, info = self._env.step(action)
        state = obs['state']
        embeddings = obs['embeddings']
        self._state_frames.append(state)
        self._embedding_frames.append(embeddings)
        stacked_states = np.array(list(self._state_frames))
        stacked_embeddings = np.array(list(self._embedding_frames))
        obs['state'] = stacked_states
        obs['embeddings'] = stacked_embeddings

        return obs, reward, terminated, truncated, info
    
    def reset(self):
        obs, info = self._env.reset()
        state = obs['state']
        embeddings = obs['embeddings']
        for _ in range(self._num_frames):
            self._state_frames.append(state)
            self._embedding_frames.append(embeddings)
        stacked_states = np.array(list(self._state_frames))
        stacked_embeddings = np.array(list(self._embedding_frames))
        obs['state'] = stacked_states
        obs['embeddings'] = stacked_embeddings
        
        return obs, info
    
class CustomObservation(gym.ObservationWrapper):
  """Resize the observation to a given resolution"""
  def __init__(self, env, crop_resolution=None, resize_resolution=None):
    super().__init__(env)
    if isinstance(resize_resolution, int):
      resize_resolution = (resize_resolution, resize_resolution)
    if isinstance(crop_resolution, int):
      crop_resolution = (crop_resolution, crop_resolution)
    if crop_resolution is None:
      self.crop_resolution = env.observation_space["pixels"].shape[:2]
    if resize_resolution is None:
      self.resize_resolution = env.observation_space["pixels"].shape[:2]
    self.crop_resolution = crop_resolution
    self.resize_resolution = resize_resolution
    self._state_shape = env.observation_space['state'].shape
    self.observation_space = Dict({"state": Box(low=-np.inf, high=np.inf, shape=self._state_shape, dtype=np.float32),
                                       "pixels": Box(low=0, high=255, shape=(*self.resize_resolution, 3), dtype=np.uint8)})
    
  def observation(self, observation):
    if self.crop_resolution is not None:
      if observation["pixels"].shape[:2] != self.crop_resolution:
        center = observation["pixels"].shape
        x = center[1]/2 - self.crop_resolution[1]/2
        y = center[0]/2 - self.crop_resolution[0]/2
        observation["pixels"] = observation[int(y):int(y+self.crop_resolution[0]), int(x):int(x+self.crop_resolution[1])]
    if self.resize_resolution is not None:
      if observation["pixels"].shape[:2] != self.resize_resolution:
        observation["pixels"] = cv2.resize(
            observation["pixels"],
            dsize=self.resize_resolution,
            interpolation=cv2.INTER_CUBIC,
        )
    return observation



class VideoRecorder(gym.Wrapper):
  """Wrapper for rendering and saving rollouts to disk.
  Reference: https://github.com/ikostrikov/jaxrl/
  """

  def __init__(
      self,
      env,
      save_dir,
      crop_resolution,
      resize_resolution,
      fps = 60,
  ):
    super().__init__(env)

    self.save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(resize_resolution, int):
      self.resize_resolution = (resize_resolution, resize_resolution)
    if isinstance(crop_resolution, int):
      self.crop_resolution = (crop_resolution, crop_resolution)

    self.resize_h, self.resize_w = self.resize_resolution
    self.crop_h, self.crop_w = self.crop_resolution
    self.fps = fps
    self.enabled = True
    self.current_episode = 0
    self.frames = []

  def step(self, action):
    frame = self.env.render()
    if self.crop_resolution is not None:
      # Crop
      if frame.shape[:2] != (self.crop_h, self.crop_w):
        center = frame.shape
        x = center[1]/2 - self.crop_w/2
        y = center[0]/2 - self.crop_h/2
        frame = frame[int(y):int(y+self.crop_h), int(x):int(x+self.crop_w)]
    if self.resize_resolution is not None:
      if frame.shape[:2] != (self.resize_h, self.resize_w):
        frame = cv2.resize(
            frame,
            dsize=(self.resize_h, self.resize_w),
            interpolation=cv2.INTER_CUBIC,
        )
    # Save
    self.frames.append(frame)
    observation, reward, terminated, truncated, info = self.env.step(action)
    if terminated or truncated:
      filename = os.path.join(self.save_dir, f"{self.current_episode}.mp4")
      imageio.mimsave(filename, self.frames, fps=self.fps)
      self.frames = []
      self.current_episode += 1
    return observation, reward, terminated, truncated, info
  
class ActionRepeat(gym.Wrapper):
  """Repeat the agent's action N times in the environment.
  Reference: https://github.com/ikostrikov/jaxrl/
  """

  def __init__(self, env, repeat):
    """Constructor.
    Args:
      env: A gym env.
      repeat: The number of times to repeat the action per single underlying env
        step.
    """
    super().__init__(env)

    assert repeat > 1, "repeat should be greater than 1."
    self._repeat = repeat

  def step(self, action):
    total_reward = 0.0
    for _ in range(self._repeat):
      observation, reward, terminated, truncated, info = self.env.step(action)
      total_reward += reward
      if terminated or truncated:
        break
    return observation, total_reward, terminated, truncated, info
