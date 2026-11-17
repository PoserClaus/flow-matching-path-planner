"""Generating paths with a trained model (used by evaluation and the viewer)."""

import numpy as np
import torch

from fm.data import build_condition
from fm.source import candidate_sources


@torch.no_grad()
def sample_paths(model, x0, goal_rel, obstacles, rep, num_steps=20):
  """Euler integration of the flow from x0. Returns [B, n_points, 2], relative frame."""
  B = goal_rel.shape[0]
  x = x0
  dt = 1.0 / num_steps
  for i in range(num_steps):
    t = torch.full((B, 1), i * dt, device=goal_rel.device)
    x = x + model(x, t, goal_rel, obstacles) * dt
  return x.reshape(B, rep.n_points, 2)


@torch.no_grad()
def generate_candidates(model, agent_pos, goal, obstacles, settings,
                        n_candidates=7, num_steps=20):
  """n candidate paths in WORLD coordinates, shape [n, n_points, 2].

  For source kind "bend" the candidates differ in the sideways bend of the
  source line (so they cover both sides of an obstacle); for "gaussian" they
  are independent noise draws. Choose among them with your own cost.
  """
  rep, src = settings.representation, settings.source
  device = next(model.parameters()).device
  goal_rel, obs = build_condition(agent_pos, goal, obstacles, rep)
  g = torch.from_numpy(goal_rel).to(device).unsqueeze(0).repeat(n_candidates, 1)
  o = torch.from_numpy(obs).to(device).unsqueeze(0).repeat(n_candidates, 1, 1)
  x0 = candidate_sources(g, rep, src)
  rel = sample_paths(model, x0, g, o, rep, num_steps).cpu().numpy()
  return rel * rep.coord_scale + np.asarray(agent_pos, dtype=np.float32)
