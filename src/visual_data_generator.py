import math
import random
import sys
import pygame
from car import Car
from config import FPS, GREEN, GRAY, HEIGHT, WHITE, WIDTH
from environment import get_default_obstacles

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Visual Expert Simulator")
clock = pygame.time.Clock()

obstacles = get_default_obstacles()
goal_x, goal_y = 800, 300


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


# Initialize car at a valid random starting position
start_x, start_y = get_valid_start_position(obstacles)
car = Car(start_x, start_y)

running = True
while running:
  clock.tick(FPS)

  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False

  # Expert autonomous step
  car.expert_drive(goal_x, goal_y, obstacles)

  # Reset if goal reached or car goes out of bounds
  if (
      math.hypot(car.x - goal_x, car.y - goal_y) < 40
      or car.x < 0
      or car.x > WIDTH
      or car.y < 0
      or car.y > HEIGHT
  ):
    start_x, start_y = get_valid_start_position(obstacles)
    car.x, car.y = start_x, start_y
    car.angle = 0
    car.speed = 0
    car.path_history.clear()
    goal_x, goal_y = 800, 300  # Keep goal fixed

  # Render graphics
  screen.fill(WHITE)

  # Goal
  pygame.draw.circle(screen, GREEN, (goal_x, goal_y), 25)

  # Obstacles
  for obs in obstacles:
    pygame.draw.rect(screen, GRAY, obs, border_radius=6)

  # Breadcrumb trail
  for pt in car.path_history:
    pygame.draw.circle(screen, (100, 150, 250), (int(pt[0]), int(pt[1])), 3)

  # Car & LiDAR
  car.draw(screen)

  pygame.display.flip()

pygame.quit()
sys.exit()