import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.optim import AdamW


"""## Polyak average"""
def soft_update_params(
    net,
    target_net,
    tau,
):
  for param, target_param in zip(net.parameters(), target_net.parameters()):
    val = tau * param.data + (1 - tau) * target_param.data
    target_param.data.copy_(val)

"""## Orthogonal Init function"""
def orthogonal_init(m):
  """Orthogonal init for Conv2D and Linear layers."""
  if isinstance(m, nn.Linear):
    nn.init.orthogonal_(m.weight.data)
    if hasattr(m.bias, "data"):
      m.bias.data.fill_(0.0)

"""## MLP Function"""
def mlp(
    input_dim,
    hidden_dim,
    output_dim,
    hidden_depth,
    dropout=0.0,
    output_mod = None,
):
  """Construct an MLP module."""
  if hidden_depth == 0:
    mods = [nn.Linear(input_dim, output_dim)]
  elif dropout > 0.0:
    mods = [nn.Linear(input_dim, hidden_dim), nn.Dropout(p=dropout), nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True)]
    for _ in range(hidden_depth - 1):
      mods += [nn.Linear(hidden_dim, hidden_dim), nn.Dropout(p=dropout), nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True)]
    mods += [nn.Linear(hidden_dim, output_dim)]
  else:
    mods = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True)]
    for _ in range(hidden_depth - 1):
      mods += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True)]
    mods += [nn.Linear(hidden_dim, output_dim)]
  if output_mod is not None:
    mods += [output_mod]
  trunk = nn.Sequential(*mods)
  return trunk


"""## Create Deep Q Networks (Critics)"""

class Critic(nn.Module):
  """Critic module."""

  def __init__(
      self,
      obs_dim,
      action_dim,
      hidden_dim,
      hidden_depth,
  ):
    super().__init__()

    self.model = mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth, dropout=0.00)
    self.apply(orthogonal_init)

  def forward(self, obs_action):
    return self.model(obs_action)

class MultiCritic(nn.Module):
  """DoubleCritic module."""

  def __init__(
      self,
      emb_dim,
      feature_dim,
      state_dim,
      action_dim,
      hidden_dim,
      hidden_depth,
  ):
    super().__init__()

    self.trunk = nn.Sequential(nn.Linear(emb_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

    self.critic1 = Critic(feature_dim + state_dim, action_dim, hidden_dim, hidden_depth)
    self.critic2 = Critic(feature_dim + state_dim, action_dim, hidden_dim, hidden_depth)

  def forward(self, emb, state, action):
    emb = self.trunk(emb)
    obs_action = torch.cat([emb, state, action], dim=-1)
    return self.critic1(obs_action), self.critic2(obs_action)

"""## Squashed Normal"""

class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
  """A tanh-squashed Normal distribution."""

  def __init__(self, loc, scale):
    self.loc = loc
    self.scale = scale

    self.base_dist = pyd.Normal(loc, scale)
    transforms = [pyd.TanhTransform(cache_size=1)]
    super().__init__(self.base_dist, transforms)

  @property
  def mean(self):
    mu = self.loc
    for tr in self.transforms:
      mu = tr(mu)
    return mu

"""## Create Policy Network"""

class DiagGaussianActor(nn.Module):
  """A torch.distributions implementation of a diagonal Gaussian policy."""

  def __init__(
      self,
      emb_dim,
      feature_dim,
      state_dim,
      action_dim,
      hidden_dim,
      hidden_depth,
      log_std_bounds,
  ):
    super().__init__()

    self.log_std_bounds = log_std_bounds
    self.trunk = self.trunk = nn.Sequential(nn.Linear(emb_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())
    self.policy = mlp(feature_dim + state_dim, hidden_dim, 2 * action_dim, hidden_depth, dropout=0.00)

    self.apply(orthogonal_init)

  def forward(self, emb, state):
    emb = self.trunk(emb)
    obs = torch.cat([emb, state], dim=-1)
    mu, log_std = self.policy(obs).chunk(2, dim=-1)

    # Constrain log_std inside [log_std_min, log_std_max].
    log_std = torch.tanh(log_std)
    log_std_min, log_std_max = self.log_std_bounds
    log_std_range = log_std_max - log_std_min
    log_std = log_std_min + 0.5 * log_std_range * (log_std + 1)

    std = log_std.exp()
    return SquashedNormal(mu, std)

"""## SAC Algorithm"""

class SAC(nn.Module):
  """Soft-Actor-Critic."""

  def __init__(self, device, env):
    super().__init__()
    # Hyperparametes
    self.device = device
    self.env = env
    self.capacity = 1_000_000
    self.num_train_steps = 1_000_000
    self.num_seed_steps = 4000
    self.num_eval_episodes = 5
    self.eval_frequency = 10_000
    self.checkpoint_frequency = 20_000
    self.log_frequency = 1_000
    self.batch_size = 256
    self.lr = 1e-4
    self.hidden_dim = 1024
    self.hidden_depth = 2
    self.gamma = 0.99
    self.loss_fn = F.smooth_l1_loss
    self.optim = AdamW
    self.samples_per_epoch = 1_000
    self.tau = 0.005
    self.epsilon = 0.05
    self.init_temp = 0.1
    self.alpha_betas = [0.9, 0.999]
    self.alpha_lr = self.lr
    self.learnable_temperature = True
    self.discount = 0.99
    # env things
    self.emb_dim = self.env.observation_space['embeddings'].shape[-1]*3
    self.state_dim = self.env.observation_space['state'].shape[-1]*3
    self.action_dim = self.env.action_space.shape[0]
    self.action_range = [float(env.action_space.low.min()),
      float(env.action_space.high.max()),]
    # Actor
    self.actor_update_frequency = 1
    self.actor_lr = self.lr
    self.actor_betas = [0.9, 0.999]
    self.log_std_bounds = [-5, 2]
    # Critic
    self.critic_update_frequency = 1
    self.critic_target_update_frequency = 2
    self.critic_lr = self.lr
    self.critic_betas = [0.9, 0.999]
    self.critic_tau = 0.005  
    

    self.feature_dim = 50

    self.utd = 1


    # Initialise Critics and actor
    self.critic = MultiCritic(self.emb_dim, self.feature_dim, self.state_dim,  self.action_dim,
                               self.hidden_dim, self.hidden_depth).to(self.device)
    self.critic_target = MultiCritic(self.emb_dim, self.feature_dim, self.state_dim,  self.action_dim,
                               self.hidden_dim, self.hidden_depth).to(self.device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    
    self.actor = DiagGaussianActor(self.emb_dim, self.feature_dim, self.state_dim, self.action_dim,
                                    self.hidden_dim, self.hidden_depth, self.log_std_bounds).to(self.device)


    self.log_alpha = nn.Parameter(
        torch.as_tensor(np.log(self.init_temp), device=self.device),
        requires_grad=True,
    )

    # Set target entropy to -|A|.
    self.target_entropy = -self.action_dim/2

    # Optimizers.

    self.critic_optimizer = self.optim(
        self.critic.parameters(),
        lr=self.critic_lr,
        betas=self.critic_betas,
    )

    self.actor_optimizer = self.optim(
        self.actor.parameters(),
        lr=self.actor_lr,
        betas=self.actor_betas,
    )
    
    self.log_alpha_optimizer = self.optim(
        [self.log_alpha],
        lr=self.alpha_lr,
        betas=self.alpha_betas,
    )

    self.train()
    self.critic_target.train()

  def train(self, training = True):
    self.training = training
    self.actor.train(training)
    self.critic.train(training)

  @property
  def alpha(self):
    return self.log_alpha.exp()

  @torch.no_grad()
  def act(self, emb, state, sample = False):
    emb = torch.as_tensor(emb, device=self.device)
    state = torch.as_tensor(state, device=self.device)
    dist = self.actor(emb, state)
    action = dist.sample() if sample else dist.mean
    action = action.clamp(*self.action_range)
    return action.cpu().numpy()

  def update_critic(
      self,
      emb,
      state,
      action,
      reward,
      next_emb,
      next_state,
      mask,
  ):
    with torch.no_grad():
      dist = self.actor(next_emb, next_state)
      next_action = dist.rsample()
      log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
      target_q1, target_q2 = self.critic_target(next_emb, next_state, next_action)
      target_v = (
          torch.min(target_q1, target_q2) - self.alpha.detach() * log_prob)
      target_q = reward + (mask * self.discount * target_v)

    # Get current Q estimates.
    current_q1, current_q2 = self.critic(emb, state, action)
    critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
        current_q2, target_q)

    # Optimize the critic.
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

    return {"critic_loss": critic_loss}

  def update_actor_and_alpha(
      self,
      emb,
      state,
  ):
    dist = self.actor(emb, state)
    action = dist.rsample()
    log_prob = dist.log_prob(action).sum(-1, keepdim=True)
    actor_q1, actor_q2 = self.critic(emb, state, action)

    actor_q = torch.min(actor_q1, actor_q2)
    actor_loss = (self.alpha.detach() * log_prob - actor_q).mean()
    actor_info = {
        "actor_loss": actor_loss,
        "entropy": -log_prob.mean(),
    }

    # Optimize the actor.
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()

    # Optimize the temperature.
    alpha_info = {}
    if self.learnable_temperature:
      self.log_alpha_optimizer.zero_grad()
      alpha_loss = (self.alpha *
                    (-log_prob - self.target_entropy).detach()).mean()
      alpha_loss.backward()
      self.log_alpha_optimizer.step()
      alpha_info["temperature_loss"] = alpha_loss
      alpha_info["temperature"] = self.alpha

    return actor_info, alpha_info

  def update(
      self,
      replay_buffer,
      step,
  ):
    
    batch_info = {}
    critic_info = {}
    actor_info = {}
    alpha_info = {}

    if step % self.critic_update_frequency == 0:
      for i in range(self.utd):
        embs, states, action, reward, next_embs, next_states, mask = replay_buffer.sample(self.batch_size)
        critic_info = self.update_critic(embs, states, action, reward, next_embs, next_states, mask)
        actor_info, alpha_info = self.update_actor_and_alpha(embs, states)
        if i % self.critic_target_update_frequency == 0:
          soft_update_params(self.critic, self.critic_target, self.critic_tau)
      batch_info = {"batch_reward": reward.mean()}

    # if step % self.actor_update_frequency == 0:
    #   embs, states, action, reward, next_embs, next_states, mask = replay_buffer.sample(self.batch_size)
    #   actor_info, alpha_info = self.update_actor_and_alpha(embs, states)

    return {**batch_info, **critic_info, **actor_info, **alpha_info}

  def optim_dict(self):
    return {
        "actor_optimizer": self.actor_optimizer,
        "log_alpha_optimizer": self.log_alpha_optimizer,
        "critic_optimizer": self.critic_optimizer,
    }