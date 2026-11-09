import os
import random
import sys
import pygame
import torch
from config import FPS, GREEN, GRAY, HEIGHT, WHITE, WIDTH
from environment import get_default_obstacles

# Import the model directly from your training script! No code duplication.
from train_flow_matching import FlowMatchingMLP

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flow Matching Geometric Path Verifier")
clock = pygame.time.Clock()

# ==========================================
# 1. LOAD MODEL
# ==========================================
model_path = "flow_matching_model.pt"
if not os.path.exists(model_path):
  print(f"Error: '{model_path}' not found! Run train_flow_matching.py first.")
  sys.exit()

model = FlowMatchingMLP(condition_dim=4, trajectory_dim=80, hidden_dim=256)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()
print("Flow Matching model loaded successfully for verification!")


# ==========================================
# 2. ODE SAMPLING FUNCTION (Record All Steps)
# ==========================================
@torch.no_grad()
def generate_path_steps(start, goal, num_steps=30):
  c_vector = [
      start[0] / WIDTH,
      start[1] / HEIGHT,
      goal[0] / WIDTH,
      goal[1] / HEIGHT,
  ]
  c = torch.tensor(c_vector, dtype=torch.float32).unsqueeze(0)

  # Correct scaling vector for 40 points (80 values total: alternating X, Y)
  scale_vector = torch.tensor([WIDTH, HEIGHT], dtype=torch.float32).repeat(40)

  # Straight-line prior initialization with correct aspect ratio
  line_pts = torch.linspace(0, 1, 40).unsqueeze(1)
  start_t = torch.tensor(start, dtype=torch.float32)
  goal_t = torch.tensor(goal, dtype=torch.float32)
  straight_line = (1.0 - line_pts) * start_t + line_pts * goal_t

  # Normalize x_t correctly using WIDTH for X and HEIGHT for Y
  x_t = straight_line.flatten().unsqueeze(0) / scale_vector

  dt = 1.0 / num_steps
  trajectory_steps = []

  # Save initial straight-line prior (t = 0.0)
  initial_traj = x_t.clone().squeeze(0).reshape(40, 2)
  initial_traj[:, 0] *= WIDTH
  initial_traj[:, 1] *= HEIGHT
  trajectory_steps.append(initial_traj.tolist())

  # Step-by-step ODE integration loop
  for i in range(num_steps):
    t_val = i * dt
    t = torch.full((1, 1), t_val, dtype=torch.float32)
    v_pred = model(x_t, t, c)
    x_t = x_t + v_pred * dt

    # Save current intermediate state
    step_traj = x_t.clone().squeeze(0).reshape(40, 2)
    step_traj[:, 0] *= WIDTH
    step_traj[:, 1] *= HEIGHT
    trajectory_steps.append(step_traj.tolist())

  return trajectory_steps


# ==========================================
# 3. SIMULATION & ANIMATION LOOP
# ==========================================
obstacles = get_default_obstacles()
fixed_goal = (800, 300)


def get_random_start():
  while True:
    sx = random.randint(80, WIDTH - 50)
    sy = random.randint(50, HEIGHT - 50)
    rect = pygame.Rect(sx - 15, sy - 20, 30, 40)
    if not any(obs.colliderect(rect) for obs in obstacles):
      return (sx, sy)


# Initialize starts and precompute all animation steps for each path
test_starts = [get_random_start() for _ in range(8)]
current_path_steps = [
    generate_path_steps(start, fixed_goal) for start in test_starts
]

anim_step = 0
max_steps = 20
frame_counter = 0

running = True
while running:
  clock.tick(FPS)

  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False
    elif event.type == pygame.KEYDOWN:
      if event.key == pygame.K_SPACE:
        # Generate new random starts and reset animation progress
        test_starts = [get_random_start() for _ in range(8)]
        current_path_steps = [
            generate_path_steps(start, fixed_goal) for start in test_starts
        ]
        anim_step = 0
        frame_counter = 0

  # Advance animation step every few frames to control speed
  frame_counter += 1
  if frame_counter >= 3:  # Adjust speed here (higher = slower animation)
    frame_counter = 0
    if anim_step < max_steps:
      anim_step += 1

  screen.fill(WHITE)

  # Draw Goal
  pygame.draw.circle(screen, GREEN, fixed_goal, 25)

  # Draw Obstacles
  for obs in obstacles:
    pygame.draw.rect(screen, GRAY, obs, border_radius=6)

  # Draw Start Points and Current Morphing Paths
  for start, path_steps in zip(test_starts, current_path_steps):
    pygame.draw.circle(screen, (220, 50, 50), (int(start[0]), int(start[1])), 6)

    # Get the path points corresponding to the current animation step
    current_pts = path_steps[anim_step]
    for pt in current_pts:
      pygame.draw.circle(screen, (0, 160, 220), (int(pt[0]), int(pt[1])), 4)

  pygame.display.set_caption(
      f"Flow Matching Verifier [Step {anim_step}/{max_steps}] - Press [SPACE]"
      " to reset"
  )
  pygame.display.flip()

pygame.quit()
sys.exit()