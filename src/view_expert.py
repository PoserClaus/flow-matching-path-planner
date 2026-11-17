import math
import random
import sys

import generate_data
import pygame
from car import CarRotate
from config import FPS, GRAY, GREEN, HEIGHT, WHITE, WIDTH
from planner import CLEARANCES, path_length

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Rotate-capable expert: visibility graph + path following")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 22)

GOAL_PAUSE_FRAMES = 60
MAX_FRAMES = 3000


def new_episode():
  while True:
    ep = generate_data.generate_episode(spacing=3.0)
    if ep is not None:
      break
  obstacles, start, goal, path = ep
  car = CarRotate(start[0], start[1], random.uniform(-math.pi, math.pi))
  car.set_path(path)
  return obstacles, goal, path, car


obstacles, goal, path, car = new_episode()
frames = 0
pause = 0
collided = False
show_clearance = False

running = True
while running:
  clock.tick(FPS)
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False
    elif event.type == pygame.KEYDOWN:
      if event.key == pygame.K_ESCAPE:
        running = False
      elif event.key == pygame.K_SPACE:
        obstacles, goal, path, car = new_episode()
        frames, pause, collided = 0, 0, False
      elif event.key == pygame.K_i:
        show_clearance = not show_clearance

  # --- update ---
  if pause > 0:
    pause -= 1
    if pause == 0:
      obstacles, goal, path, car = new_episode()
      frames, collided = 0, False
  else:
    car.follow_path()
    frames += 1
    if car.collides(obstacles):
      collided = True
    if car.reached() or frames > MAX_FRAMES:
      pause = GOAL_PAUSE_FRAMES

  # --- draw ---
  screen.fill(WHITE)
  pygame.draw.circle(screen, GREEN, (int(goal[0]), int(goal[1])), 25)

  for obs in obstacles:
    pygame.draw.rect(screen, GRAY, obs, border_radius=6)
    if show_clearance:
      pygame.draw.rect(
          screen,
          (200, 120, 120),
          obs.inflate(2 * CLEARANCES[0], 2 * CLEARANCES[0]),
          1,
          border_radius=10,
      )

  for pt in path[::2]:  # planned path
    pygame.draw.circle(screen, (0, 160, 220), (int(pt[0]), int(pt[1])), 2)
  for pt in car.path_history:  # driven trail
    pygame.draw.circle(screen, (220, 80, 80), (int(pt[0]), int(pt[1])), 2)

  car.draw(screen)

  info = (
      f"path {path_length(path):.0f}px   speed {car.speed:.1f}   "
      f"[SPACE] new   [I] clearance   [ESC] quit"
  )
  screen.blit(font.render(info, True, (60, 60, 60)), (10, 8))
  if collided:
    screen.blit(font.render("COLLISION", True, (200, 30, 30)), (10, 30))

  pygame.display.flip()

pygame.quit()
sys.exit()