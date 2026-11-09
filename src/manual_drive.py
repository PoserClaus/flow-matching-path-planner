import math
import random
import sys
import pygame
from config import FPS, GREEN, GRAY, HEIGHT, WHITE, WIDTH
from environment import get_default_obstacles

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(
    "Manual Car Control - Arrow Keys / WASD to Drive, [R] to Reset"
)
clock = pygame.time.Clock()


class ManualCar:

  def __init__(self, x, y):
    self.x = float(x)
    self.y = float(y)
    self.theta = 0.0  # Heading angle in radians
    self.v = 0.0  # Current velocity

    # Vehicle parameters
    self.max_speed = 5.0
    self.max_reverse_speed = -2.0
    self.acceleration = 0.15
    self.friction = 0.05
    self.steering_speed = 0.04
    self.radius = 12  # Collision radius

  def update(self, keys, obstacles):
    # 1. Acceleration / Braking input
    if keys[pygame.K_UP] or keys[pygame.K_w]:
      self.v = min(self.v + self.acceleration, self.max_speed)
    elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
      self.v = max(self.v - self.acceleration, self.max_reverse_speed)
    else:
      # Apply natural friction/slowdown
      if self.v > 0:
        self.v = max(0.0, self.v - self.friction)
      elif self.v < 0:
        self.v = min(0.0, self.v + self.friction)

    # 2. Steering input (only turn if the car is moving)
    if abs(self.v) > 0.1:
      turn_factor = self.v / self.max_speed
      if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        self.theta -= self.steering_speed * turn_factor
      if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        self.theta += self.steering_speed * turn_factor

    # 3. Update position based on velocity and heading
    new_x = self.x + self.v * math.cos(self.theta)
    new_y = self.y + self.v * math.sin(self.theta)

    # 4. Boundary & Obstacle Collision Check
    car_rect = pygame.Rect(new_x - self.radius, new_y - self.radius, 24, 24)

    # Screen boundaries check
    in_bounds = 0 <= new_x <= WIDTH and 0 <= new_y <= HEIGHT
    # Obstacle collision check
    collided = any(car_rect.colliderect(obs) for obs in obstacles)

    if in_bounds and not collided:
      self.x = new_x
      self.y = new_y
    else:
      # Stop velocity upon collision or wall hit
      self.v = 0.0

  def draw(self, surface):
    # Draw car body as a circle with a heading indicator line
    pos = (int(self.x), int(self.y))
    pygame.draw.circle(surface, (40, 120, 220), pos, self.radius)

    # Heading indicator tip
    tip_x = self.x + (self.radius + 6) * math.cos(self.theta)
    tip_y = self.y + (self.radius + 6) * math.sin(self.theta)
    pygame.draw.line(surface, (20, 40, 80), pos, (int(tip_x), int(tip_y)), 3)


# Setup Environment and Goal
obstacles = get_default_obstacles()
fixed_goal = (800, 300)


def get_random_start(obstacles):
  while True:
    sx = random.randint(80, 250)
    sy = random.randint(50, HEIGHT - 50)
    rect = pygame.Rect(sx - 15, sy - 20, 30, 40)
    if not any(obs.colliderect(rect) for obs in obstacles):
      return sx, sy


start_x, start_y = get_random_start(obstacles)
car = ManualCar(start_x, start_y)

# Main Simulation Loop
running = True
while running:
  clock.tick(FPS)

  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False
    elif event.type == pygame.KEYDOWN:
      if event.key == pygame.K_r:
        # Reset car position
        sx, sy = get_random_start(obstacles)
        car.x, car.y = sx, sy
        car.v = 0.0
        car.theta = 0.0

  # Get continuous keyboard states
  keys = pygame.key.get_pressed()
  car.update(keys, obstacles)

  # Check if goal reached
  distance_to_goal = math.hypot(car.x - fixed_goal[0], car.y - fixed_goal[1])
  goal_reached = distance_to_goal < 25

  # Drawing
  screen.fill(WHITE)

  # Draw Goal
  pygame.draw.circle(screen, GREEN, fixed_goal, 25)

  # Draw Obstacles
  for obs in obstacles:
    pygame.draw.rect(screen, GRAY, obs, border_radius=6)

  # Draw Car
  car.draw(screen)

  # Status Text overlay
  if goal_reached:
    font = pygame.font.SysFont(None, 36)
    text = font.render(
        "Goal Reached! Press [R] to restart.", True, (40, 180, 40)
    )
    screen.blit(text, (WIDTH // 2 - 180, 40))

  pygame.display.flip()

pygame.quit()
sys.exit()