import random
import pygame
import math
from config import WIDTH, HEIGHT

# Start and goal keep at least this distance to every obstacle, so the car can
# rotate on the spot at the start and the planner's widest clearance (32 px)
# always applies at both ends.
FREE_MARGIN = 34

def generate_random_obstacles():
  """Generiert eine zufällige Anordnung von Hindernissen pro Episode."""
  obstacles = []
  num_obstacles = random.randint(2, 4)  # 2 bis 4 Hindernisse pro Welt
  for _ in range(num_obstacles):
    while True:
      w = random.randint(50, 100)
      h = random.randint(80, 160)
      x = random.randint(150, WIDTH - 250)
      y = random.randint(50, HEIGHT - 160 - 50)
      rect = pygame.Rect(x, y, w, h)

      # Überlappung mit anderen Hindernissen dieser Welt vermeiden
      overlap = any(rect.colliderect(o) for o in obstacles)
      if not overlap:
        obstacles.append(rect)
        break
  return obstacles

def sample_start_goal(obstacles, margin=FREE_MARGIN, min_dist=300):
  inflated = [o.inflate(2 * margin, 2 * margin) for o in obstacles]
  while True:
    sx = random.randint(50, WIDTH - 50)
    sy = random.randint(50, HEIGHT - 50)
    gx = random.randint(50, WIDTH - 50)
    gy = random.randint(50, HEIGHT - 50)
    if any(r.collidepoint(sx, sy) for r in inflated):
      continue
    if any(r.collidepoint(gx, gy) for r in inflated):
      continue
    if math.hypot(sx - gx, sy - gy) > min_dist:
      return sx, sy, gx, gy

def get_local_occupancy_grid(
    car_x, car_y, obstacles, grid_size=32, cell_size=8
):
  """Erzeugt das lokale Occupancy Grid basierend auf den zufälligen Hindernissen."""
  grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
  half_extent = (grid_size * cell_size) // 2
  origin_x = car_x - half_extent
  origin_y = car_y - half_extent

  for r in range(grid_size):
    for c in range(grid_size):
      cell_world_x = origin_x + c * cell_size
      cell_world_y = origin_y + r * cell_size
      cell_rect = pygame.Rect(
          cell_world_x, cell_world_y, cell_size, cell_size
      )

      is_obstacle = any(cell_rect.colliderect(obs) for obs in obstacles)
      out_of_bounds = (
          cell_world_x < 0
          or cell_world_x > WIDTH
          or cell_world_y < 0
          or cell_world_y > HEIGHT
      )

      if is_obstacle or out_of_bounds:
        grid[r][c] = 1

  return grid