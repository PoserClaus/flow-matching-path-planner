"""The velocity network.

Do not rename attributes inside these classes (obs_encoder, phi, net): saved
weights are keyed by those names, so renaming would break old checkpoints.
"""

import torch
import torch.nn as nn


class ObstacleSetEncoder(nn.Module):
  """Per-obstacle MLP + masked max/mean pooling (order and count invariant)."""

  def __init__(self, hidden=128, out_dim=128):
    super().__init__()
    # per obstacle: [cx, cy, w, h, cx - goal_x, cy - goal_y, present]
    self.phi = nn.Sequential(
        nn.Linear(7, hidden),
        nn.Mish(),
        nn.Linear(hidden, hidden),
        nn.Mish(),
        nn.Linear(hidden, out_dim),
    )

  def forward(self, obs, goal_rel):
    K = obs.shape[1]
    g = goal_rel[:, None, :].expand(-1, K, -1)
    feat_in = torch.cat([obs[..., 0:4], obs[..., 0:2] - g, obs[..., 4:5]], dim=-1)
    h = self.phi(feat_in)  # [B, K, D]

    mask = obs[..., 4:5]  # [B, K, 1], 1 = real obstacle
    pooled_max = h.masked_fill(mask < 0.5, -1e4).max(dim=1).values
    pooled_max = pooled_max * (mask.sum(dim=1) > 0).float()  # no obstacles -> 0
    pooled_mean = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return torch.cat([pooled_max, pooled_mean], dim=-1)  # [B, 2D]


class FlowMatchingRotate(nn.Module):

  def __init__(self, n_points=40, hidden_dim=512, obs_dim=128):
    super().__init__()
    self.obs_encoder = ObstacleSetEncoder(hidden=128, out_dim=obs_dim)
    traj_dim = 2 * n_points
    in_dim = traj_dim + 1 + 2 + 2 * obs_dim  # x_t, t, goal_rel, obstacle context
    self.net = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.Mish(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Mish(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Mish(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Mish(),
        nn.Linear(hidden_dim, traj_dim),  # velocity v_t
    )

  def forward(self, x_t, t, goal_rel, obstacles):
    ctx = self.obs_encoder(obstacles, goal_rel)
    return self.net(torch.cat([x_t, t, goal_rel, ctx], dim=-1))


def build_model(settings):
  """Network with the sizes from the settings (used for training and loading)."""
  rep, model = settings.representation, settings.model
  return FlowMatchingRotate(
      n_points=rep.n_points, hidden_dim=model.hidden_dim, obs_dim=model.obs_dim
  )
