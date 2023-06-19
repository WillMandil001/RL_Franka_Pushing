import gymnasium as gym
import numpy as np
from collections import deque

class FrameStack(gym.Wrapper):
    def __init__(self, env, num_frames=3):
        self._env = env
        self._num_frames = num_frames
        self._pixel_shape = env.observation_space['pixels'].shape
        self._state_shape = env.observation_space['state'].shape
        self._pixel_frames = deque([], maxlen=num_frames)
        self._state_frames = deque([], maxlen=num_frames)

        self.observation_space['state'].shape = (num_frames, *self._env.obersevation_space['state'].shape)
        self.observation_space['pixels'].shape = (num_frames, *self._env.obersevation_space['pixels'].shape)


    def step(self, action):

        obs, reward, terminated, truncated, info = self._env.step(action)
        state = obs['state']
        pixels = obs['pixels']
        self._state_frames.append(state)
        self._pixel_frames.append(pixels)
        stacked_states = np.concatenate(list(self._state_frames), axis=0)
        stacked_pixels = np.concatenate(list(self._pixel_frames), axis=0)

        return obs.update({'state': stacked_states, 'pixels': stacked_pixels}), reward, terminated, truncated, info
    
    def reset(self):
        obs, info = self._env.reset()
        state = obs['state']
        pixels = obs['pixels']
        for _ in range(self._num_frames):
            self._state_frames.append(obs)
            self._pixel_frames.append(pixels)
        stacked_states = np.concatenate(list(self._state_frames), axis=0)
        stacked_pixels = np.concatenate(list(self._pixel_frames), axis=0)
        
        return obs.update({'state': state, 'pixels': pixels}), info
