"""Source distributions of the flow (the x0 that paths are generated from).

kind="bend":     straight line towards the goal + one smooth sideways bend
                 + small per-point jitter.
kind="line":     straight line towards the goal + small per-point jitter, and
                 nothing else (no bend). The only randomness is the jitter.
kind="gaussian": plain Gaussian noise, x0 ~ N(0, gaussian_std^2) per coordinate.
                 The first point is NOT pinned to the agent here; the viewer's
                 postprocessing pins the start of the final path.

Training draws a random x0 per sample (`training_source`). At inference,
`candidate_sources` returns one x0 per candidate: for "bend" the bends are
spread over [-cand_bend_max_px, +cand_bend_max_px], for "line" and "gaussian"
they are independent noise draws (for "line" these differ only by the small
jitter, so the candidates are nearly identical).

To add a new source: add a branch in `training_source` and `candidate_sources`
and its name to SOURCE_KINDS in settings.py.
"""

import math

import torch


def bend_source(goal_rel, bend, rep, jitter):
  """Straight line towards the goal + smooth sideways bend + jitter.

  goal_rel: [B, 2] (normalized), bend: [B] (normalized, signed).
  Returns [B, 2 * n_points].
  """
  B = goal_rel.shape[0]
  N = rep.n_points
  dist = goal_rel.norm(dim=-1, keepdim=True).clamp_min(1e-6)  # [B, 1]
  direction = goal_rel / dist  # [B, 2]
  perp = torch.stack([-direction[:, 1], direction[:, 0]], dim=-1)  # [B, 2]

  k = torch.linspace(0.0, 1.0, N, device=goal_rel.device)  # [N]
  reach = torch.minimum(k[None, :] * (rep.horizon_px / rep.coord_scale), dist)
  line = direction[:, None, :] * reach[..., None]  # [B, N, 2]

  bump = torch.sin(math.pi * k)  # 0 at both ends, 1 in the middle
  line = line + bend[:, None, None] * bump[None, :, None] * perp[:, None, :]

  if jitter > 0:
    noise = torch.randn_like(line) * jitter
    noise[:, 0] = 0.0  # the first point stays exactly at the agent
    line = line + noise
  return line.reshape(B, 2 * N)


def line_source(goal_rel, rep, jitter):
  """Straight line towards the goal + jitter (the bend source without a bend)."""
  zero_bend = torch.zeros(goal_rel.shape[0], device=goal_rel.device)
  return bend_source(goal_rel, zero_bend, rep, jitter)


def gaussian_source(goal_rel, rep, std):
  """Plain Gaussian noise. Returns [B, 2 * n_points]."""
  B = goal_rel.shape[0]
  return torch.randn(B, 2 * rep.n_points, device=goal_rel.device) * std


def training_source(goal_rel, rep, src):
  """One random x0 per training sample."""
  if src.kind == "line":
    return line_source(goal_rel, rep, src.jitter_std)
  if src.kind == "gaussian":
    return gaussian_source(goal_rel, rep, src.gaussian_std)
  if src.kind == "bend":
    B = goal_rel.shape[0]
    bend = torch.randn(B, device=goal_rel.device) * (
        src.bend_std_px / rep.coord_scale
    )
    return bend_source(goal_rel, bend, rep, src.jitter_std)
  raise ValueError(f"Unknown source kind {src.kind!r}")


def candidate_sources(goal_rel, rep, src):
  """x0 for each of the n candidates (n = goal_rel.shape[0]) at inference."""
  if src.kind == "line":
    return line_source(goal_rel, rep, src.jitter_std)
  if src.kind == "gaussian":
    return gaussian_source(goal_rel, rep, src.gaussian_std)
  if src.kind == "bend":
    n = goal_rel.shape[0]
    if n <= 1:
      bends = torch.zeros(n, device=goal_rel.device)
    else:
      bends = torch.linspace(
          -src.cand_bend_max_px, src.cand_bend_max_px, n,
          device=goal_rel.device,
      ) / rep.coord_scale
    return bend_source(goal_rel, bends, rep, src.jitter_std)
  raise ValueError(f"Unknown source kind {src.kind!r}")