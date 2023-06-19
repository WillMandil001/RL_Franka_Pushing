import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import collections
from tqdm.auto import tqdm
import psutil
import cv2
from sac import SAC
from replay_buffer_np import ReplayBuffer
import gymnasium as gym
from gymnasium.wrappers import PixelObservationWrapper, RecordEpisodeStatistics
from wrappers import ActionRepeat, FrameStack, VideoRecorder, CustomObservation

class dinov2_obs(gym.ObservationWrapper):
  # Pass image observation through first resnet layers to reduce number of inferences
  def __init__(self, env):
    super().__init__(env)
    self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    self.model =torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(self.device)
    self.model.eval()
    for param in self.model.parameters():
            param.requires_grad = False
    self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
    self.observation_space = gym.spaces.Box(
        low= -np.inf,
        high= np.inf,
        shape=(1, self.model.embed_dim),
    )

  def observation(self, obs):
    obs = self.transform(obs)
    obs = obs.unsqueeze(0).to(self.device)
    features = self.model(obs)
    return features.cpu().numpy()

class Workspace:
  def __init__(self):
    self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    self.num_gpus = torch.cuda.device_count()
    cwd = os.getcwd()
    workdir = Path.cwd()
    self.work_dir = workdir
    tb_path = os.path.join(cwd, 'tb')
    cp_path = os.path.join(cwd, 'checkpoints')
    os.makedirs(tb_path, exist_ok=True)
    os.makedirs(cp_path, exist_ok=True)
    self.writer = SummaryWriter(log_dir=tb_path)
    self.frame_stack = 3
    self.action_repeat = 2
    self._global_step = 0
    self._global_episode = 0
    self.ep_len = 500
  
    self.setup()

  def setup(self):
    self.env = self.create_environment(name='cheetah_run', frame_stack=self.frame_stack, action_repeat=self.action_repeat)
    self.eval_env = self.create_environment(name='cheetah_run', frame_stack=self.frame_stack, action_repeat=self.action_repeat, record=True)
    self.policy = SAC(self.device)
    # create replay buffer

    self.buffer = ReplayBuffer(obs_shape = (3, self.env.observation_spec().shape[-2], self.env.observation_spec().shape[-1]),
                        obs_frame_stack=self.frame_stack,
                        nstep=self.policy.nstep,
                        discount = self.policy.discount,
                        action_shape=self.env.action_spec().shape,
                        batch_size=self.policy.batch_size,
                        num_eps=self.policy.capacity//self.ep_len,
                        ep_len=self.ep_len,
                        device=self.policy.device,
                        )
    
  def create_environment(name='FetchReachDense-v3', frame_stack=3, action_repeat=2, record=False):
    # Names = 'FrankaKitchen-v1', 'FetchReachDense-v3', 'HalfCheetah-v4'
    env = gym.make(name, render_mode='rgb_array')
    if record:
      env = VideoRecorder(env, save_dir="./videos", crop_resolution=224, resize_resolution=224)
    if action_repeat > 1:
      env = ActionRepeat(env, action_repeat)
    env = PixelObservationWrapper(env)
    env = CustomObservation(env, crop_resolution=224, resize_resolution=224)
    env = dinov2_obs(env)
    env = FrameStack(env, frame_stack)
    env = RecordEpisodeStatistics(env)
    return env

  @property
  def global_step(self):
      return self._global_step

  @property
  def global_episode(self):
      return self._global_episode

  def evaluate(
    policy,
    env,
    num_episodes,
  ):
    """Evaluate the policy and dump rollout videos to disk."""
    policy.eval()
    stats = collections.defaultdict(list)
    total_reward = 0
    for j in range(num_episodes):
      observation, info = env.reset()
      observation = np.asarray(observation)
      terminated = False
      truncated = False
      while not (terminated or truncated):
        action = policy.act(np.asarray(observation[:, 0, :]).flatten(), sample=False)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
      end_reward = reward
      for k, v in info["episode"].items():
        stats[k].append(v)
      stats["end_reward"].append(end_reward)
      stats["episode_reward"].append(total_reward)
    for k, v in stats.items():
      stats[k] = np.mean(v)
    return stats
  
  def train(self):
    try:
      episode_step, episode_reward = 0, 0
      observation, _ = self.env.reset()
      embs = observation['embeddings']
      states = observation['state']
      action = self.env.action_space.sample()
      reward = -1
      mask = 1.0
      terminated = 0
      truncated = 0
      
      self.buffer.insert(observation, action, reward, mask)
      for i in tqdm(range(self.policy.num_train_steps)):
        if terminated or truncated:
          if terminated:
            mask = 0.0
          else: mask = 1.0
          self._global_episode += 1
          self.writer.add_scalar("episode end reward", reward, i)
          self.writer.add_scalar("episode return", episode_reward, i)
          # Reset env
          obs, _ = self.env.reset()
          episode_step = 0
          episode_reward = 0
          self.buffer.insert(obs, action, reward, mask)

        # Evaluate
        if i % self.policy.eval_frequency == 0:
          eval_stats = self.eval()
          for k, v in eval_stats.items():
            self.writer.add_scalar(f"eval {k}", v, i)

        # Sample action
        with torch.no_grad():
          action = self.policy.act(observation, eval_mode=False)

        # Update agent
        if i >= self.policy.num_seed_steps:
          train_info = self.policy.update(self.buffer, i)
          if i % self.policy.log_frequency == 0:
            if train_info is not None:
              for k, v in train_info.items():
                self.writer.add_scalar(k, v, i)
            ram_usage = psutil.virtual_memory().percent
            self.writer.add_scalar("ram usage", ram_usage, i)

        # Take env step
        obs, reward, terminated, truncated, info = self.env.step(action)
        embs = obs['embeddings']
        states = obs['state']
        if terminated:
          mask=0.0
        else: 
          mask=1.0
        
        episode_reward += reward
        episode_step += 1
        self._global_step += 1
        self.buffer.insert(obs, action, reward, mask)

    except KeyboardInterrupt:
      print("Caught keyboard interrupt. Saving before quitting.")

    finally:
      print(f"done?")  # pylint: disable=undefined-loop-variable

def main():
  workspace = Workspace()
  workspace.train()

if __name__ == "__main__":
  main()
