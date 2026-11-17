"""Visual check of a trained planner on random worlds.

Examples (run from src/):
  python viewer.py                      # newest run in runs/
  python viewer.py --run gaussian       # a specific run
  python viewer.py --seed 3             # reproducible worlds

What you see:
  thin colored lines  the candidate paths from the last replan
  blue dots           the path the car is following now
  red dots            where the car has driven
  grey line           the expert planner's path (for comparison, key E)

Keys: SPACE new world | R retry same world | TAB next run (same world) |
      P pause | E expert path | C candidates | ESC quit

TAB cycles through all runs in runs/ on the SAME world, so you can compare how
different models handle the same situation. The counters restart on every switch.
"""

import argparse
import random
import sys

import numpy as np
import pygame
import torch

from config import FPS, GRAY, GREEN, HEIGHT, WHITE, WIDTH
from fm import runs
from sim import SimParams, Simulation

PAUSE_FRAMES = 60  # how long the result of a finished episode stays visible

CAND_COLORS = [
    (230, 120, 120), (230, 170, 100), (215, 205, 90), (120, 200, 130),
    (100, 190, 210), (130, 140, 230), (190, 130, 220),
]


def _pts(arr, step=1):
  return [(int(x), int(y)) for x, y in arr[::step]]


def draw(screen, sim, run_name, font, show_cands, show_expert, paused):
  screen.fill(WHITE)
  pygame.draw.circle(screen, GREEN, (int(sim.goal[0]), int(sim.goal[1])), 25)
  for obs in sim.obstacles:
    pygame.draw.rect(screen, GRAY, obs, border_radius=6)

  if show_expert:
    pygame.draw.lines(screen, (185, 185, 185), False, _pts(sim.expert, 3), 3)

  if show_cands:
    for k, p in enumerate(sim.cands):
      if k != sim.best and len(p) > 1:
        pygame.draw.lines(
            screen, CAND_COLORS[k % len(CAND_COLORS)], False, _pts(p), 1
        )

  if sim.path is not None:
    for x, y in _pts(sim.path[sim.car.path_idx:]):
      pygame.draw.circle(screen, (0, 120, 220), (x, y), 3)
  for x, y in _pts(sim.trail):
    pygame.draw.circle(screen, (220, 80, 80), (x, y), 2)

  sim.car.draw(screen)

  lines = [
      f"run {run_name}   source {sim.settings.source.kind}",
      f"clean {sim.clean}/{sim.episodes}   body collisions {sim.collided_eps}"
      f"   timeouts {sim.timeouts}",
      f"replans {sim.replans}   speed {sim.car.speed:.1f}"
      + ("   [PAUSED]" if paused else ""),
  ]
  for i, text in enumerate(lines):
    screen.blit(font.render(text, True, (50, 50, 50)), (10, 8 + 20 * i))
  if sim.collided:
    screen.blit(font.render("COLLISION", True, (200, 30, 30)), (10, 68))
  elif sim.status is not None:
    screen.blit(font.render(sim.last_result, True, (30, 130, 30)), (10, 68))
  help_text = (
      "[SPACE] new world  [R] retry  [TAB] next run  [P] pause  [E] expert  "
      "[C] candidates  [ESC]"
  )
  screen.blit(font.render(help_text, True, (90, 90, 90)), (10, HEIGHT - 24))


def parse_args():
  p = argparse.ArgumentParser(description="Visualize a trained planner.")
  p.add_argument("--run", default=None, help="run name in runs/ (default: newest)")
  p.add_argument("--candidates", type=int, default=None,
                 help="number of candidate paths per replan (default 7)")
  p.add_argument("--seed", type=int, default=None, help="seed for the worlds")
  return p.parse_args()


def main():
  args = parse_args()

  names = runs.list_runs()
  if not names:
    sys.exit(
        f"No runs found in '{runs.RUNS_DIR}'.\n"
        "Train one:   python train.py --name my_run\n"
        "Or keep old weights:   python -m fm.runs adopt "
        "<path/to/old_weights.pt> my_old_run"
    )
  current = args.run or runs.latest_run()
  if current not in names:
    sys.exit(f"Run '{current}' not found. Available runs: {names}")

  if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

  pygame.init()
  screen = pygame.display.set_mode((WIDTH, HEIGHT))
  clock = pygame.time.Clock()
  font = pygame.font.SysFont(None, 22)

  loaded = {}

  def get_run(name):
    if name not in loaded:
      loaded[name] = runs.load_run(name)
      print(f"Loaded run '{name}'.")
    return loaded[name]

  params = SimParams()
  if args.candidates is not None:
    params.n_candidates = args.candidates

  model, settings = get_run(current)
  sim = Simulation(model, settings, params)
  pygame.display.set_caption(f"Flow matching planner - {current}")
  show_cands, show_expert, paused = True, True, False
  countdown = 0  # frames until the next world after an episode ended

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
          sim.new_world()
        elif event.key == pygame.K_r:
          sim.reset_episode()
        elif event.key == pygame.K_TAB and len(names) > 1:
          current = names[(names.index(current) + 1) % len(names)]
          sim.model, sim.settings = get_run(current)
          sim.reset_stats()
          sim.reset_episode()  # same world, other model
          pygame.display.set_caption(f"Flow matching planner - {current}")
        elif event.key == pygame.K_p:
          paused = not paused
        elif event.key == pygame.K_e:
          show_expert = not show_expert
        elif event.key == pygame.K_c:
          show_cands = not show_cands

    if not paused:
      if sim.status is None:
        sim.step()
        if sim.status is not None:
          countdown = PAUSE_FRAMES
      else:
        countdown -= 1
        if countdown <= 0:
          sim.new_world()
    draw(screen, sim, current, font, show_cands, show_expert, paused)
    pygame.display.flip()

  pygame.quit()
  sys.exit()


if __name__ == "__main__":
  main()
