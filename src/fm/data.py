"""Data pipeline: expert episodes -> training samples (random fixed-horizon crops).

Every sample is "the next horizon_px of the expert path" from a random point on
a random episode, with the GLOBAL goal as condition. If the remaining path is
shorter than the horizon, the missing points are placed on the goal. Everything
is expressed relative to the agent and divided by rep.coord_scale.

All functions take the settings groups they need (`rep`, `aug`) as arguments, so
nothing here depends on global constants.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_episodes(path):
  if not Path(path).exists():
    raise FileNotFoundError(f"'{path}' not found! Run generate_data.py first.")
  try:
    raw = torch.load(path, weights_only=False)
  except TypeError:  # older torch versions have no weights_only argument
    raw = torch.load(path)

  episodes = []
  for item in raw:
    cond = item["condition"]
    if "obstacles" not in cond:
      raise KeyError(
          "Dataset has no 'obstacles'. Generate it with generate_data.py."
      )
    xy = np.asarray(item["trajectory"], dtype=np.float32)
    seg = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    episodes.append({
        "path": xy,
        "cum": np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float32),
        "goal": np.asarray(cond["goal"], dtype=np.float32),
        "obs": np.asarray(cond["obstacles"], dtype=np.float32).reshape(-1, 4),
    })
  return episodes


def crop_path(ep, s0, rep):
  """rep.n_points points covering rep.horizon_px of arc length from arc position s0.

  Past the end of the path the points stay at the goal (padding).
  """
  s = np.minimum(
      s0 + np.linspace(0.0, rep.horizon_px, rep.n_points), ep["cum"][-1]
  )
  x = np.interp(s, ep["cum"], ep["path"][:, 0])
  y = np.interp(s, ep["cum"], ep["path"][:, 1])
  return np.stack([x, y], axis=1).astype(np.float32)


def build_condition(agent_pos, goal, obstacles, rep):
  """Condition in the agent-relative frame.

  obstacles: iterable of (x, y, w, h) (top-left corner) or pygame.Rect.
  Returns goal_rel (2,) and obs (max_obs, 5) with rows [cx, cy, w, h, present].
  Obstacles are sorted nearest first; extra ones beyond max_obs are dropped.
  """
  a = np.asarray(agent_pos, dtype=np.float32)
  obs = np.asarray([tuple(o) for o in obstacles], dtype=np.float32).reshape(-1, 4)
  centers = obs[:, 0:2] + obs[:, 2:4] / 2.0 - a
  order = np.argsort(np.hypot(centers[:, 0], centers[:, 1]))[:rep.max_obs]

  feat = np.zeros((rep.max_obs, 5), dtype=np.float32)
  for row, i in enumerate(order):
    feat[row] = [
        centers[i, 0] / rep.coord_scale,
        centers[i, 1] / rep.coord_scale,
        obs[i, 2] / rep.coord_scale,
        obs[i, 3] / rep.coord_scale,
        1.0,
    ]
  goal_rel = (np.asarray(goal, dtype=np.float32) - a) / rep.coord_scale
  return goal_rel, feat


def make_sample(ep, rng, rep, aug, augment=True):
  """One training sample: (obstacles [K,5], goal_rel [2], x1 [2N])."""
  total = float(ep["cum"][-1])
  s0 = 0.0 if rng.random() < aug.p_start else rng.random() * total
  pts = crop_path(ep, s0, rep)
  agent = pts[0].copy()

  if augment and rng.random() < aug.p_offpath:
    delta = rng.normal(0.0, aug.offpath_std_px, size=2)
    norm = float(np.linalg.norm(delta))
    if norm > aug.offpath_max_px:
      delta *= aug.offpath_max_px / norm
    arc = np.arange(rep.n_points) * rep.spacing_px
    fade = np.clip(1.0 - arc / max(aug.rejoin_px, 1e-6), 0.0, 1.0)[:, None]
    pts = pts + delta[None, :] * fade  # starts at the shifted agent, rejoins path
    agent = agent + delta

  goal_rel, obs = build_condition(agent, ep["goal"], ep["obs"], rep)
  rel = (pts - agent) / rep.coord_scale

  if augment and aug.mirror:  # mirror the whole relative world
    for axis in (0, 1):
      if rng.random() < 0.5:
        rel[:, axis] *= -1.0
        goal_rel[axis] *= -1.0
        obs[:, axis] *= -1.0
  return (
      obs.astype(np.float32),
      goal_rel.astype(np.float32),
      rel.reshape(-1).astype(np.float32),
  )


class CropDataset(Dataset):
  """Draws a fresh random crop for every access (or a fixed cache)."""

  def __init__(self, episodes, crops_per_episode, augment, seed, rep, aug,
               cache=False):
    self.eps = episodes
    self.k = crops_per_episode
    self.augment = augment
    self.rep = rep
    self.aug = aug
    self.rng = np.random.default_rng(seed)
    self.cache = None
    if cache:
      self.cache = [self._make(i) for i in range(len(self))]

  def __len__(self):
    return len(self.eps) * self.k

  def _make(self, idx):
    obs, goal_rel, x1 = make_sample(
        self.eps[idx % len(self.eps)], self.rng, self.rep, self.aug,
        self.augment,
    )
    return torch.from_numpy(obs), torch.from_numpy(goal_rel), torch.from_numpy(x1)

  def __getitem__(self, idx):
    if self.cache is not None:
      return self.cache[idx]
    return self._make(idx)
