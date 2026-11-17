import math

import numpy as np
import pygame
from config import BLUE


class CarRotate:
  """Car that can rotate on the spot and follows a planned path.

  Heading convention: radians, atan2(dy, dx) in screen coordinates (y down),
  so heading 0 points right and positive angles turn clockwise on screen.
  This is the same convention as the verifier (sin/cos of the agent angle).
  """

  def __init__(self, x, y, heading=0.0):
    self.x = float(x)
    self.y = float(y)
    self.heading = float(heading)
    self.width = 44
    self.height = 22

    self.speed = 0.0
    self.max_speed = 4.5
    self.acceleration = 0.15
    self.deceleration = 0.35
    self.rotate_speed = math.radians(5.0)  # max turn per frame (also at speed 0)
    self.turn_in_place_angle = math.radians(40.0)  # above this error: stop and turn
    self.lookahead = 20.0

    self.path = None
    self.path_idx = 0
    self.path_history = []
    self.goal = None  # optional real goal; the path may only cover part of the way

  # ------------------------------------------------------------------
  def set_path(self, path):
    self.path = np.asarray(path, dtype=float)
    self.path_idx = 0
    self.path_history = [(self.x, self.y)]

  def _lookahead_point(self):
    p = self.path
    acc = 0.0
    for j in range(self.path_idx, len(p) - 1):
      acc += math.hypot(p[j + 1][0] - p[j][0], p[j + 1][1] - p[j][1])
      if acc >= self.lookahead:
        return p[j + 1]
    return p[-1]

  def distance_to_goal(self):
    gx, gy = self.goal if self.goal is not None else self.path[-1]
    return math.hypot(gx - self.x, gy - self.y)

  def reached(self, tol=15.0):
    return self.path is not None and self.distance_to_goal() < tol

  # ------------------------------------------------------------------
  def follow_path(self):
    """One simulation step: turn towards the lookahead point, drive if aligned."""
    if self.path is None:
      return

    # closest path point, searched only forward so the car never jumps back
    end = min(len(self.path), self.path_idx + 15)
    seg = self.path[self.path_idx:end]
    self.path_idx += int(
        np.argmin(np.hypot(seg[:, 0] - self.x, seg[:, 1] - self.y))
    )

    tx, ty = self._lookahead_point()
    desired = math.atan2(ty - self.y, tx - self.x)
    err = (desired - self.heading + math.pi) % (2 * math.pi) - math.pi

    # rotation works at any speed, including zero
    self.heading += max(-self.rotate_speed, min(self.rotate_speed, err))
    self.heading = (self.heading + math.pi) % (2 * math.pi) - math.pi

    # target speed: zero while badly misaligned, slower near the goal
    if abs(err) > self.turn_in_place_angle:
      target_speed = 0.0
    else:
      align = 1.0 - abs(err) / self.turn_in_place_angle
      target_speed = self.max_speed * (0.3 + 0.7 * align)
    target_speed = min(target_speed, 0.12 * self.distance_to_goal() + 0.6)

    if self.speed < target_speed:
      self.speed = min(target_speed, self.speed + self.acceleration)
    else:
      self.speed = max(target_speed, self.speed - self.deceleration)

    self.x += self.speed * math.cos(self.heading)
    self.y += self.speed * math.sin(self.heading)

    last = self.path_history[-1]
    if math.hypot(self.x - last[0], self.y - last[1]) > 4.0:
      self.path_history.append((self.x, self.y))

  # ------------------------------------------------------------------
  def footprint_points(self, step=4.0):
    """Points along the outline of the rotated car body."""
    hw, hh = self.width / 2, self.height / 2
    xs = np.arange(-hw, hw + 0.1, step)
    ys = np.arange(-hh, hh + 0.1, step)
    local = (
        [(x, -hh) for x in xs]
        + [(x, hh) for x in xs]
        + [(-hw, y) for y in ys]
        + [(hw, y) for y in ys]
    )
    c, s = math.cos(self.heading), math.sin(self.heading)
    return [
        (self.x + px * c - py * s, self.y + px * s + py * c)
        for px, py in local
    ]

  def collides(self, obstacles):
    for px, py in self.footprint_points():
      for obs in obstacles:
        if obs.collidepoint(int(px), int(py)):
          return True
    return False

  # ------------------------------------------------------------------
  def draw(self, surface):
    car_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
    pygame.draw.rect(car_surface, BLUE, (0, 0, self.width, self.height), 4)
    pygame.draw.rect(
        car_surface, (255, 255, 255), (self.width - 10, 4, 6, self.height - 8)
    )
    # pygame rotates counter-clockwise; our heading is clockwise on screen
    rotated = pygame.transform.rotate(car_surface, -math.degrees(self.heading))
    rect = rotated.get_rect(center=(self.x, self.y))
    surface.blit(rotated, rect.topleft)