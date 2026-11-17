"""Shortest-path planner for a car that can rotate on the spot.

Because heading is not a constraint, we plan in plain (x, y):
  1. inflate every rectangle by a clearance (car radius + margin),
  2. build a visibility graph over start, goal and the inflated corners,
  3. run Dijkstra for the exact shortest path,
  4. round the corners (quadratic Bezier, limited cut depth),
  5. verify the rounded path and resample it at uniform spacing.

Coordinates are screen coordinates (y points down).
Headings elsewhere in the project use the same convention: atan2(dy, dx), radians.
"""

import heapq
import math

import numpy as np
import pygame
from config import HEIGHT, WIDTH

CAR_RADIUS = 25.0  # circumscribed radius of the 44 x 22 car (~24.6 px)
CLEARANCES = (32.0, 26.0, 20.0, 14.0)  # tried in this order (widest first)
BOUNDARY_MARGIN = 10.0  # waypoints must stay this far inside the screen
EPS = 1e-9


# ------------------------------------------------------------------
# Geometry helpers (rects are tuples: left, top, right, bottom)
# ------------------------------------------------------------------
def _inflate(o, c):
  return (o.left - c, o.top - c, o.right + c, o.bottom + c)


def _inside_open(p, r):
  """True if p lies strictly inside rect r (the boundary does not count)."""
  return r[0] + EPS < p[0] < r[2] - EPS and r[1] + EPS < p[1] < r[3] - EPS


def _segment_hits(a, b, r):
  """True if segment a-b passes through the open interior of rect r.

  Touching the boundary or sliding along an edge does not count, so paths may
  run along the edges of the inflated rectangles.
  """
  dx, dy = b[0] - a[0], b[1] - a[1]
  t0, t1 = 0.0, 1.0
  for p, q in (
      (-dx, a[0] - r[0]),
      (dx, r[2] - a[0]),
      (-dy, a[1] - r[1]),
      (dy, r[3] - a[1]),
  ):
    if abs(p) < EPS:
      if q <= EPS:
        return False
    else:
      t = q / p
      if p < 0:
        t0 = max(t0, t)
      else:
        t1 = min(t1, t)
      if t1 - t0 <= EPS:
        return False
  return True


def _free(a, b, rects):
  return not any(_segment_hits(a, b, r) for r in rects)


# ------------------------------------------------------------------
# Visibility graph + Dijkstra
# ------------------------------------------------------------------
def _shortest_path(start, goal, rects):
  nodes = [tuple(map(float, start)), tuple(map(float, goal))]
  for r in rects:
    for corner in ((r[0], r[1]), (r[2], r[1]), (r[2], r[3]), (r[0], r[3])):
      if not (
          BOUNDARY_MARGIN <= corner[0] <= WIDTH - BOUNDARY_MARGIN
          and BOUNDARY_MARGIN <= corner[1] <= HEIGHT - BOUNDARY_MARGIN
      ):
        continue
      if any(_inside_open(corner, o) for o in rects):
        continue
      nodes.append(corner)

  n = len(nodes)
  dist = [math.inf] * n
  prev = [-1] * n
  dist[0] = 0.0
  heap = [(0.0, 0)]
  while heap:
    d, u = heapq.heappop(heap)
    if d > dist[u]:
      continue
    if u == 1:
      break
    for v in range(n):
      if v == u:
        continue
      w = math.hypot(nodes[u][0] - nodes[v][0], nodes[u][1] - nodes[v][1])
      if d + w < dist[v] and _free(nodes[u], nodes[v], rects):
        dist[v] = d + w
        prev[v] = u
        heapq.heappush(heap, (dist[v], v))

  if math.isinf(dist[1]):
    return None
  way, k = [], 1
  while k != -1:
    way.append(nodes[k])
    k = prev[k]
  return way[::-1]


# ------------------------------------------------------------------
# Smoothing and resampling
# ------------------------------------------------------------------
def _round_corners(pts, max_cut, n_arc=8):
  """Replace each corner by a quadratic Bezier; cut depth is limited by max_cut."""
  pts = np.asarray(pts, dtype=float)
  if len(pts) < 3 or max_cut <= 0:
    return pts
  t = np.linspace(0.0, 1.0, n_arc)[:, None]
  out = [pts[0]]
  for i in range(1, len(pts) - 1):
    p0, c, p2 = pts[i - 1], pts[i], pts[i + 1]
    l1, l2 = np.linalg.norm(p0 - c), np.linalg.norm(p2 - c)
    if l1 < 1e-6 or l2 < 1e-6:
      continue
    d = min(max_cut, 0.5 * l1, 0.5 * l2)
    a = c + (p0 - c) / l1 * d
    b = c + (p2 - c) / l2 * d
    out.extend((1 - t) ** 2 * a + 2 * t * (1 - t) * c + t**2 * b)
  out.append(pts[-1])
  return np.array(out)


def resample_by_arclength(pts, spacing):
  """Uniform spacing along the path; keeps the exact first and last point."""
  pts = np.asarray(pts, dtype=float)
  seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
  cum = np.concatenate([[0.0], np.cumsum(seg)])
  total = cum[-1]
  if total < 1e-6:
    return pts[[0, -1]]
  s = np.arange(0.0, total, spacing)
  if len(s) > 1 and total - s[-1] < spacing * 0.3:
    s = s[:-1]
  s = np.append(s, total)
  return np.stack(
      [np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])], axis=1
  )


def _path_clear(path, rects):
  for i in range(len(path) - 1):
    if not _free(path[i], path[i + 1], rects):
      return False
  return True


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def plan_path(start, goal, obstacles, spacing=3.0, clearances=CLEARANCES):
  """Returns an (N, 2) array with uniformly spaced points, or None if no path.

  Tries the widest clearance first and falls back to narrower ones, so paths
  keep as much distance to the obstacles as the world allows.
  """
  raw = [pygame.Rect(o) for o in obstacles]
  for c in clearances:
    rects = [_inflate(o, c) for o in raw]
    if any(_inside_open(start, r) or _inside_open(goal, r) for r in rects):
      continue
    way = _shortest_path(start, goal, rects)
    if way is None:
      continue
    # The rounded path may cut into the inflated zone, so verify it against a
    # slightly smaller clearance and reduce the rounding if needed.
    check = [_inflate(o, max(0.7 * c, 12.0)) for o in raw]
    for cut in (20.0, 10.0, 5.0, 0.0):
      smooth = _round_corners(way, cut)
      path = resample_by_arclength(smooth, spacing)
      if _path_clear(path, check):
        return path
  return None


def path_headings(path):
  """Tangent direction at each path point (radians, screen convention)."""
  d = np.gradient(np.asarray(path, dtype=float), axis=0)
  return np.arctan2(d[:, 1], d[:, 0])


def path_length(path):
  p = np.asarray(path, dtype=float)
  return float(np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1])).sum())