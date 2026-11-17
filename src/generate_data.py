import math
import random
from pathlib import Path
import pygame
import torch
from config import HEIGHT, WIDTH, DATA_DIR
from planner import path_headings, plan_path
from world import generate_random_obstacles, sample_start_goal, get_local_occupancy_grid

DATASET_PATH = DATA_DIR / "expert_dataset_rotate.pt"

def generate_episode(spacing=3.0):
  """Returns (obstacles, start, goal, path) or None if no path exists."""
  obstacles = generate_random_obstacles()
  sx, sy, gx, gy = sample_start_goal(obstacles)
  path = plan_path((sx, sy), (gx, gy), obstacles, spacing=spacing)
  if path is None:
    return None
  return obstacles, (sx, sy), (gx, gy), path


def generate_dataset(num_episodes=5000, spacing=3.0):
  dataset = []
  print(f"Generating {num_episodes} planned expert paths...")
  for episode in range(num_episodes):
    ep = generate_episode(spacing)
    if ep is not None:
      obstacles, start, goal, path = ep
      grid = get_local_occupancy_grid(
          start[0], start[1], obstacles
      )
      # The car can rotate in place, so the start heading does not change the
      # path. It is stored anyway (radians, screen convention) for compatibility.
      start_angle = random.uniform(-math.pi, math.pi)
      dataset.append({
          "condition": {
              "start": start,
              "goal": goal,
              "occupancy_grid": grid,
              "start_angle": start_angle,
              # (x, y, w, h) tuples: needed to compute grids for cropped samples
              "obstacles": [tuple(o) for o in obstacles],
          },
          "trajectory": [tuple(p) for p in path.tolist()],  # uniform spacing
          "headings": path_headings(path).tolist(),  # tangent per point
      })
    if (episode + 1) % 500 == 0:
      print(f"{episode + 1}/{num_episodes} episodes, {len(dataset)} valid paths")

  torch.save(dataset, DATASET_PATH)
  print(f"Saved {len(dataset)} paths to '{DATASET_PATH}'")


if __name__ == "__main__":
  generate_dataset(5000)