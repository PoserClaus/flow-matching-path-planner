import math
import pygame
from config import BLUE, LASER_GREEN, RED


class Car:

  def __init__(self, x, y):
    self.x = x
    self.y = y
    self.width = 44
    self.height = 22
    self.angle = 0
    self.speed = 0
    self.max_speed = 4.5
    self.acceleration = 0.1
    self.steering_speed = 3.0
    self.path_history = []
    self.ray_endpoints = []

  def get_lidar_readings(self, obstacles, max_range=150):
    ray_angles = [-60, -30, 0, 30, 60]
    readings = []
    self.ray_endpoints = []

    for angle_offset in ray_angles:
      ray_angle = math.radians(self.angle + angle_offset)
      distance = max_range
      hit_point = (
          self.x + max_range * math.cos(ray_angle),
          self.y - max_range * math.sin(ray_angle),
      )

      for step in range(8, max_range, 4):
        check_x = self.x + step * math.cos(ray_angle)
        check_y = self.y - step * math.sin(ray_angle)
        point_rect = pygame.Rect(check_x - 2, check_y - 2, 4, 4)

        hit = False
        for obs in obstacles:
          if obs.colliderect(point_rect):
            hit = True
            break

        if hit:
          distance = step
          hit_point = (check_x, check_y)
          break

      self.ray_endpoints.append(hit_point)
      readings.append(distance / max_range)

    return readings

  def expert_drive(self, goal_x, goal_y, obstacles):
    # 1. Goal Attraction
    dx = goal_x - self.x
    dy = self.y - goal_y
    target_angle = math.degrees(math.atan2(dy, dx))
    angle_diff = (target_angle - self.angle + 180) % 360 - 180

    # 2. LiDAR Obstacle Repulsion (Tighter tuning)
    lidar = self.get_lidar_readings(obstacles, max_range=150)

    d0 = (1.0 - lidar[0]) ** 2.0
    d1 = (1.0 - lidar[1]) ** 2.0
    d2 = (1.0 - lidar[2]) ** 2.0
    d3 = (1.0 - lidar[3]) ** 2.0
    d4 = (1.0 - lidar[4]) ** 2.0

    left_danger = d0 * 2.0 + d1 * 1.0
    right_danger = d4 * 2.0 + d3 * 1.0

    if d2 > 0.2:
      clearance_difference = lidar[3] - lidar[1]
      if abs(clearance_difference) < 0.05:
        if angle_diff > 0:
          right_danger += d2 * 2.5
        else:
          left_danger += d2 * 2.5
      else:
        if clearance_difference > 0:
          left_danger += d2 * 2.5
        else:
          right_danger += d2 * 2.5

    avoidance_steering = (left_danger - right_danger) * 55.0
    total_steering_error = angle_diff + avoidance_steering

    if total_steering_error > 2:
      self.angle += self.steering_speed
    elif total_steering_error < -2:
      self.angle -= self.steering_speed

    target_speed = self.max_speed * (0.4 + 0.6 * lidar[2])
    if lidar[2] < 0.25:
      target_speed = 1.2

    if self.speed < target_speed:
      self.speed = min(target_speed, self.speed + self.acceleration)
    elif self.speed > target_speed:
      self.speed = max(target_speed, self.speed - self.acceleration)

    rad = math.radians(self.angle)
    self.x += self.speed * math.cos(rad)
    self.y -= self.speed * math.sin(rad)

    # Record history path
    if (
        len(self.path_history) == 0
        or math.hypot(
            self.x - self.path_history[-1][0], self.y - self.path_history[-1][1]
        )
        > 10
    ):
      self.path_history.append((self.x, self.y))
      if len(self.path_history) > 1200:
        self.path_history.pop(0)

  def draw(self, surface):
    # Draw LiDAR lasers
    for pt in self.ray_endpoints:
      pygame.draw.line(surface, LASER_GREEN, (self.x, self.y), pt, 1)
      pygame.draw.circle(surface, RED, (int(pt[0]), int(pt[1])), 3)

    # Draw Car Body
    car_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
    pygame.draw.rect(car_surface, BLUE, (0, 0, self.width, self.height), 4)
    pygame.draw.rect(
        car_surface, (255, 255, 255), (self.width - 10, 4, 6, self.height - 8)
    )

    rotated_surface = pygame.transform.rotate(car_surface, self.angle)
    new_rect = rotated_surface.get_rect(center=(self.x, self.y))
    surface.blit(rotated_surface, new_rect.topleft)