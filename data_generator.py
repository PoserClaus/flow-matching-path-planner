import math
import random
import pygame
import torch
from car import Car
from environment import get_default_obstacles
from config import WIDTH, HEIGHT

# Initialize pygame library backend (needed for rect math)
pygame.init()


def generate_dataset(num_episodes=500):
  obstacles = get_default_obstacles()
  dataset = []

  print(
      f"Generating {num_episodes} expert trajectories in Turbo Headless"
      " Mode..."
  )

  for episode in range(num_episodes):
    start_x, start_y = get_valid_start_position(obstacles)
    goal_x, goal_y = 800, 300

    car = Car(start_x, start_y)
    car.path_history = []

    max_steps = 400
    success = False

    for _ in range(max_steps):
      car.expert_drive(goal_x, goal_y, obstacles)

      if math.hypot(car.x - goal_x, car.y - goal_y) < 40:
        success = True
        break

      if car.x < 0 or car.x > WIDTH or car.y < 0 or car.y > HEIGHT:
        break

    if success and len(car.path_history) > 10:
      # initial_lidar = car.get_lidar_readings(obstacles, max_range=150)
      condition = {
          "start": (start_x, start_y),
          "goal": (goal_x, goal_y),
          # "lidar": initial_lidar,
      }
      trajectory = car.path_history.copy()
      dataset.append({"condition": condition, "trajectory": trajectory})

  print(f"Successfully collected {len(dataset)} valid expert paths!")
  torch.save(dataset, "expert_dataset.pt")
  print("Dataset saved successfully to 'expert_dataset.pt'!")


def get_valid_start_position(obstacles):
  """Loops until it finds a random (x, y) coordinate

  that does not collide with any obstacle.
  """
  while True:
    start_x = random.randint(50, WIDTH - 50)
    start_y = random.randint(50, HEIGHT - 50)

    # Approximate bounding box for the car to ensure full clearance
    car_rect = pygame.Rect(start_x - 15, start_y - 20, 30, 40)

    collision = False
    for obs in obstacles:
      if obs.colliderect(car_rect):
        collision = True
        break

    if not collision:
      return start_x, start_y

if __name__ == "__main__":
  generate_dataset(3000)