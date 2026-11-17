"""Closed-loop simulation: the model proposes paths, the car drives them.

No drawing in here, so it can also run headless (benchmark). The viewer shows
a Simulation; stepping it forward is `sim.step()`.

At every replan the model produces several candidate paths. The best one is
chosen by a simple cost (length, distance to the goal, obstacle hits,
consistency with the current plan) and blended into the path the car is
currently following. If the path ahead would hit an obstacle, the plan is
replaced immediately (no blending) and the car slows down while the collision
is still predicted close by.
"""

import math
import random
from dataclasses import dataclass

import numpy as np

import generate_data
from car import CarRotate
from fm.sampling import generate_candidates
from geometry import collision_count, path_length
from planner import resample_by_arclength


@dataclass
class SimParams:
  """Settings of the control loop around the model (not part of the model)."""

  n_candidates: int = 7
  ode_steps: int = 20
  replan_dist: float = 12.0  # replan after driving this far
  blend_len: float = 40.0  # arc length over which the old plan fades into the new
  check_horizon: float = 120.0  # path ahead is checked for collisions this far
  safety_margin: float = 20.0  # obstacle inflation for the path check
  collision_penalty: float = 50.0  # cost per path point inside an inflated obstacle
  consistency_w: float = 5.0  # weight for staying close to the current plan
  slow_factor: float = 0.4  # speed factor while a collision is predicted close by
  resample_spacing: float = 6.0
  goal_tol: float = 15.0
  max_frames: int = 2500


# ------------------------------------------------------------------
# Path processing and selection
# ------------------------------------------------------------------
def smooth_interior(pts, window=5):
  n = len(pts)
  out = pts.copy()
  h = window // 2
  for i in range(1, n - 1):
    out[i] = pts[max(0, i - h):min(n, i + h + 1)].mean(axis=0)
  return out


def postprocess(raw, agent, goal, rep, params):
  """Smooth, pin the start to the agent (and the end to the goal if near)."""
  pts = smooth_interior(np.asarray(raw, dtype=float))
  n = len(pts)
  agent = np.asarray(agent, dtype=float)
  goal = np.asarray(goal, dtype=float)

  pts += (agent - pts[0]) * np.linspace(1.0, 0.0, n)[:, None]
  if np.linalg.norm(goal - agent) < rep.horizon_px * 0.9:
    pts += (goal - pts[-1]) * (np.linspace(0.0, 1.0, n)[:, None] ** 2)
  return resample_by_arclength(pts, params.resample_spacing)


def path_cost(path, obs_t, goal, params, old_rem=None):
  """Estimated total cost: length + distance to goal + penalties (lower = better)."""
  cost = path_length(path) + float(np.hypot(*(path[-1] - np.asarray(goal))))
  cost += params.collision_penalty * collision_count(
      path, obs_t, params.safety_margin
  )
  if old_rem is not None:
    old = np.asarray(old_rem, dtype=float)
    head = path[:10]
    d = np.hypot(
        head[:, None, 0] - old[None, :, 0], head[:, None, 1] - old[None, :, 1]
    ).min(axis=1)
    cost += params.consistency_w * float(d.mean())
  return cost


def blend_paths(old_rem, new, params):
  """Near the agent follow the old plan, further away follow the new one."""
  spacing = params.resample_spacing
  old = resample_by_arclength(np.asarray(old_rem, dtype=float), spacing)
  n = len(new)
  k = np.arange(n)
  w_start = np.clip(k * spacing / params.blend_len, 0.0, 1.0)
  w_end = np.clip(1.0 - (n - 1 - k) * spacing / params.blend_len, 0.0, 1.0)
  w = np.maximum(w_start, w_end)[:, None]
  old_k = old[np.minimum(k, len(old) - 1)]
  return (1.0 - w) * old_k + w * new


def path_ahead(path, idx, dist):
  """Points of the path from idx on, up to `dist` px of arc length."""
  p = path[idx:]
  seg = np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))
  cum = np.concatenate([[0.0], np.cumsum(seg)])
  return p[cum <= dist]


# ------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------
class Simulation:

  def __init__(self, model, settings, params=None, verbose=True):
    self.model = model
    self.settings = settings
    self.params = params or SimParams()
    self.verbose = verbose
    self.reset_stats()
    self.new_world()

  def reset_stats(self):
    self.episodes = 0
    self.clean = 0  # reached the goal without any body collision
    self.collided_eps = 0
    self.timeouts = 0
    self.last_result = ""

  def new_world(self, episode=None):
    """Load a given (obstacles, start, goal, expert_path) or draw a random one."""
    ep = episode
    while ep is None:
      ep = generate_data.generate_episode(spacing=3.0)
    self.obstacles, self.start, self.goal, self.expert = ep
    self.obs_t = [tuple(o) for o in self.obstacles]
    self.reset_episode()

  def reset_episode(self):
    """Same world again, new random start heading."""
    self.car = CarRotate(
        self.start[0], self.start[1], random.uniform(-math.pi, math.pi)
    )
    self.car.goal = self.goal  # slow down towards the real goal, not the path end
    self.car.path_history = [(self.car.x, self.car.y)]
    self.base_speed = self.car.max_speed
    self.path = None
    self.cands, self.best = [], 0
    self.trail = [(self.car.x, self.car.y)]
    self.dist_since_plan = 0.0
    self.frames = 0
    self.replans = 0
    self.collided = False
    self.status = None  # None while running, then "reached" or "timeout"

  def replan(self, agent, urgent):
    p = self.params
    old_rem = None
    if self.path is not None:
      old_rem = np.vstack([agent, self.path[self.car.path_idx:]])

    cands = generate_candidates(
        self.model, agent, self.goal, self.obs_t, self.settings,
        p.n_candidates, p.ode_steps,
    )
    rep = self.settings.representation
    paths = [postprocess(c, agent, self.goal, rep, p) for c in cands]
    costs = [path_cost(q, self.obs_t, self.goal, p, old_rem) for q in paths]
    best = int(np.argmin(costs))

    new = paths[best]
    if old_rem is not None and not urgent:
      new = blend_paths(old_rem, new, p)

    self.cands, self.best = paths, best
    self.path = new
    self.car.path = new
    self.car.path_idx = 0
    self.dist_since_plan = 0.0
    self.replans += 1

  def finish(self, status):
    self.status = status
    self.episodes += 1
    if self.collided:
      self.collided_eps += 1
    if status == "timeout":
      self.timeouts += 1
    elif not self.collided:
      self.clean += 1
    self.last_result = status + (" + collision" if self.collided else "")
    if self.verbose:
      print(
          f"episode {self.episodes}: {self.last_result} after {self.frames} "
          f"frames ({self.replans} replans) | clean {self.clean}/{self.episodes}"
      )

  def step(self):
    """Advance one frame. Returns the status (None while running)."""
    if self.status is not None:
      return self.status

    p = self.params
    car = self.car
    agent = np.array([car.x, car.y], dtype=float)

    urgent = self.path is not None and collision_count(
        path_ahead(self.path, car.path_idx, p.check_horizon), self.obs_t,
        p.safety_margin,
    ) > 0
    if (
        self.path is None
        or urgent
        or self.dist_since_plan >= p.replan_dist
        or car.path_idx >= len(self.path) - 2
    ):
      self.replan(agent, urgent)

    blocked = collision_count(
        path_ahead(self.path, car.path_idx, 40.0), self.obs_t, 4.0
    ) > 0
    car.max_speed = self.base_speed * (p.slow_factor if blocked else 1.0)

    x0, y0 = car.x, car.y
    car.follow_path()
    self.dist_since_plan += math.hypot(car.x - x0, car.y - y0)
    if math.hypot(car.x - self.trail[-1][0], car.y - self.trail[-1][1]) > 3.0:
      self.trail.append((car.x, car.y))

    self.frames += 1
    if car.collides(self.obstacles):
      self.collided = True
    if math.hypot(car.x - self.goal[0], car.y - self.goal[1]) < p.goal_tol:
      self.finish("reached")
    elif self.frames > p.max_frames:
      self.finish("timeout")
    return self.status
