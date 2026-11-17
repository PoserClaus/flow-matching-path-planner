"""Small geometry helpers shared by training, evaluation and the simulation."""

import numpy as np


def collision_count(pts, obs_xywh, margin):
  """Number of points inside an obstacle inflated by `margin` px.

  obs_xywh: iterable of (x, y, w, h) with (x, y) the top-left corner.
  """
  pts = np.asarray(pts, dtype=np.float32)
  hit = np.zeros(len(pts), dtype=bool)
  for x, y, w, h in np.asarray(obs_xywh, dtype=np.float32).reshape(-1, 4):
    hit |= (
        (pts[:, 0] > x - margin) & (pts[:, 0] < x + w + margin)
        & (pts[:, 1] > y - margin) & (pts[:, 1] < y + h + margin)
    )
  return int(hit.sum())


def path_length(pts):
  pts = np.asarray(pts, dtype=np.float32)
  return float(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])).sum())
