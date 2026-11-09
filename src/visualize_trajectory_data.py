import math
import os
import random
import sys
import pygame
import torch
from config import FPS, GRAY, HEIGHT, WHITE, WIDTH
from environment import get_default_obstacles

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(
    "Dataset Inspector - Limited & Clean Waypoint View"
)
clock = pygame.time.Clock()


def draw_arrow(surface, color, p1, p2, size=4):
  """Draws a small directional arrowhead between two points."""
  angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
  left_angle = angle + math.radians(150)
  right_angle = angle - math.radians(150)

  p3 = (p2[0] + size * math.cos(left_angle), p2[1] + size * math.sin(left_angle))
  p4 = (
      p2[0] + size * math.cos(right_angle),
      p2[1] + size * math.sin(right_angle),
  )
  pygame.draw.polygon(surface, color, [p2, p3, p4])


def visualize_dataset(dataset_path="expert_dataset.pt", max_trajectories=15):
  """Loads the dataset and visualizes a limited number of trajectories

  for clean inspection.
  """
  if not os.path.exists(dataset_path):
    print(f"Error: '{dataset_path}' not found! Run data_generator.py first.")
    sys.exit()

  full_dataset = torch.load(dataset_path)
  obstacles = get_default_obstacles()

  # Limit the number of trajectories to plot for visual clarity
  num_to_show = min(max_trajectories, len(full_dataset))
  dataset = random.sample(full_dataset, num_to_show)

  print(
      f"Loaded {len(full_dataset)} total trajectories. Displaying a random"
      f" sample of {num_to_show}."
  )

  running = True
  while running:
    clock.tick(FPS)

    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False

    # Render screen
    screen.fill(WHITE)

    # 1. Draw Environment Obstacles
    for obs in obstacles:
      pygame.draw.rect(screen, GRAY, obs, border_radius=6)

    # 2. Draw the selected subset of trajectories
    for item in dataset:
      traj = item["trajectory"]

      for i in range(len(traj) - 1):
        p1 = traj[i]
        p2 = traj[i + 1]

        # Draw connecting line segment
        pygame.draw.line(screen, (130, 170, 230), p1, p2, 1)

        # Draw a clear dot for every exact waypoint
        pygame.draw.circle(screen, (30, 100, 200), (int(p1[0]), int(p1[1])), 3)

        # Draw directional arrowheads periodically
        if i % 6 == 0 and math.hypot(p2[0] - p1[0], p2[1] - p1[1]) > 2:
          draw_arrow(screen, (10, 50, 150), p1, p2, size=4)

      # Draw start point (Greenish) and endpoint (Red) for each path
      if len(traj) > 0:
        pygame.draw.circle(
            screen, (40, 180, 80), (int(traj[0][0]), int(traj[0][1])), 5
        )
        pygame.draw.circle(
            screen, (200, 50, 50), (int(traj[-1][0]), int(traj[-1][1])), 5
        )

    pygame.display.flip()

  pygame.quit()
  sys.exit()


if __name__ == "__main__":
  # You can easily change how many trajectories are displayed right here:
  visualize_dataset(dataset_path="expert_dataset.pt", max_trajectories=300)