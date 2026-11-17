"""Closed-loop benchmark: compare trained runs on identical test worlds.

Examples (run from src/):
  python evaluate.py                                  # all runs in runs/
  python evaluate.py --runs bend gaussian             # only these runs
  python evaluate.py --runs bend gaussian --worlds 100   # quicker, less precise
  python evaluate.py --baseline bend --force          # re-run, ignore cached results

How it works
  * Test worlds: `--worlds` (default 300) random worlds are generated once with a
    fixed seed and saved to data/test_worlds.json. Every run is tested on exactly
    these worlds, each with a fixed random start heading. The file is reused until
    you change --worlds or --seed (or pass --regenerate).
  * Each run is driven through sim.py headless: the model proposes candidate
    paths, the best one is chosen, the car drives, and this repeats until the
    goal is reached, the car touches an obstacle (it keeps driving, but the world
    counts as a collision) or the time runs out.
  * Results of each run are cached in runs/<name>/benchmark.json and reused as
    long as the weights, the test worlds and the simulation parameters are
    unchanged.

Table columns
  success      reached the goal without ever touching an obstacle (with a 95%
               confidence interval)
  collisions   worlds in which the car touched an obstacle at any time
  timeouts     worlds in which the goal was not reached in time
  length/exp   driven distance / expert path length   (median over successes)
  frames       time to the goal                       (median over successes)
  replans      number of replans                      (median over successes)

The paired comparison counts worlds where one run succeeds and the baseline
does not, and vice versa. With few worlds, differences of a few percentage
points are within the noise; the sign test p-value tells you if a difference
is likely real.
"""

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict

import numpy as np
import pygame
import torch

import generate_data
from config import DATA_DIR
from fm import runs
from fm.settings import Settings
from geometry import path_length
from sim import SimParams, Simulation

TEST_WORLDS_FILE = DATA_DIR / "test_worlds.json"
BENCHMARK_FILE = "benchmark.json"  # per run, inside runs/<name>/
CSV_FILE = "benchmark.csv"  # summary table, inside runs/


# ------------------------------------------------------------------
# Test worlds
# ------------------------------------------------------------------
def make_test_worlds(n, seed):
  random.seed(seed)
  worlds = []
  while len(worlds) < n:
    ep = generate_data.generate_episode(spacing=3.0)
    if ep is None:  # no path in this random world
      continue
    obstacles, start, goal, path = ep
    worlds.append({
        "obstacles": [[o.x, o.y, o.w, o.h] for o in obstacles],
        "start": [float(start[0]), float(start[1])],
        "goal": [float(goal[0]), float(goal[1])],
        "expert": np.asarray(path, dtype=float).tolist(),
        "heading": random.uniform(-math.pi, math.pi),
    })
  return worlds


def load_test_worlds(n, seed, regenerate=False):
  if TEST_WORLDS_FILE.exists() and not regenerate:
    data = json.loads(TEST_WORLDS_FILE.read_text())
    if data.get("seed") == seed and data.get("n") == n:
      return data
    print(
        f"'{TEST_WORLDS_FILE.name}' has seed {data.get('seed')} / "
        f"{data.get('n')} worlds; regenerating for seed {seed} / {n} worlds."
    )
  print(f"Generating {n} test worlds (seed {seed})...")
  data = {"seed": seed, "n": n, "worlds": make_test_worlds(n, seed)}
  TEST_WORLDS_FILE.parent.mkdir(parents=True, exist_ok=True)
  TEST_WORLDS_FILE.write_text(json.dumps(data))
  return data


def world_to_episode(w):
  obstacles = [pygame.Rect(*o) for o in w["obstacles"]]
  return obstacles, tuple(w["start"]), tuple(w["goal"]), np.asarray(w["expert"])


# ------------------------------------------------------------------
# Running one model
# ------------------------------------------------------------------
def run_model(name, worlds, seed, params, device):
  """Drives one run through all worlds; returns one result dict per world."""
  model, settings = runs.load_run(name, device)
  sim = Simulation(model, settings, params, verbose=False)
  results = []
  t0 = time.time()
  for i, w in enumerate(worlds):
    # Same seeds for every model: same noise draws, same simulation randomness.
    random.seed(seed + i)
    np.random.seed(seed + i)
    torch.manual_seed(seed + i)

    sim.new_world(world_to_episode(w))
    sim.car.heading = w["heading"]  # fixed start heading per world
    car = sim.car
    driven = 0.0
    while sim.status is None:
      x0, y0 = car.x, car.y
      sim.step()
      driven += math.hypot(car.x - x0, car.y - y0)

    results.append({
        "world": i,
        "status": sim.status,  # "reached" or "timeout"
        "collided": bool(sim.collided),
        "frames": sim.frames,
        "replans": sim.replans,
        "driven_px": driven,
        "expert_px": path_length(w["expert"]),
    })

    if (i + 1) % 10 == 0 or i + 1 == len(worlds):
      done = sum(r["status"] == "reached" and not r["collided"] for r in results)
      elapsed = time.time() - t0
      eta = elapsed / (i + 1) * (len(worlds) - i - 1)
      print(
          f"  {name}: {i + 1}/{len(worlds)} worlds | success so far "
          f"{done / (i + 1):.1%} | {elapsed:.0f}s elapsed, ~{eta:.0f}s left"
      )
  return results


def cached_or_run(name, test_set, worlds, seed, params, device, force):
  model_mtime = (runs.run_dir(name) / runs.MODEL_FILE).stat().st_mtime
  path = runs.run_dir(name) / BENCHMARK_FILE
  key = {
      "test_set": test_set,
      "sim_params": asdict(params),
      "model_mtime": model_mtime,
  }
  if path.exists() and not force:
    try:
      cached = json.loads(path.read_text())
      if all(cached.get(k) == v for k, v in key.items()):
        print(f"  {name}: using cached results ({len(cached['results'])} worlds)")
        return cached["results"]
    except (json.JSONDecodeError, KeyError):
      pass  # unreadable cache: just run again

  results = run_model(name, worlds, seed, params, device)
  path.write_text(json.dumps({**key, "results": results}))
  return results


# ------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------
def is_success(r):
  return r["status"] == "reached" and not r["collided"]


def wilson_interval(k, n, z=1.96):
  """95% confidence interval for a rate k/n."""
  if n == 0:
    return 0.0, 0.0
  p = k / n
  denom = 1 + z * z / n
  center = (p + z * z / (2 * n)) / denom
  half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
  return max(0.0, center - half), min(1.0, center + half)


def sign_test_p(wins, losses):
  """Two-sided exact sign test: are wins and losses likely equally frequent?"""
  n = wins + losses
  if n == 0:
    return 1.0
  k = min(wins, losses)
  tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
  return min(1.0, 2 * tail)


def median_or_nan(values):
  return float(np.median(values)) if values else float("nan")


def summarize(results):
  n = len(results)
  ok = [r for r in results if is_success(r)]
  lo, hi = wilson_interval(len(ok), n)
  return {
      "n": n,
      "success": len(ok) / n,
      "success_lo": lo,
      "success_hi": hi,
      "collisions": sum(r["collided"] for r in results) / n,
      "timeouts": sum(r["status"] == "timeout" for r in results) / n,
      "length_ratio": median_or_nan(
          [r["driven_px"] / max(r["expert_px"], 1e-6) for r in ok]
      ),
      "frames": median_or_nan([r["frames"] for r in ok]),
      "replans": median_or_nan([r["replans"] for r in ok]),
  }


def describe_changes(settings, baseline):
  changes = settings.changes(baseline)  # {key: (baseline_value, this_value)}
  if not changes:
    return "(same settings)"
  return "; ".join(f"{k}: {a}->{b}" for k, (a, b) in changes.items())


# ------------------------------------------------------------------
# Output
# ------------------------------------------------------------------
def fmt(x, spec):
  return "n/a" if x != x else format(x, spec)  # x != x is True only for NaN


def print_table(rows):
  header = ["run", "success (95% CI)", "collisions", "timeouts", "length/exp",
            "frames", "replans", "changes vs baseline"]
  body = []
  for r in rows:
    s = r["summary"]
    body.append([
        r["name"],
        f"{s['success']:.1%} ({s['success_lo']:.0%}-{s['success_hi']:.0%})",
        f"{s['collisions']:.1%}",
        f"{s['timeouts']:.1%}",
        fmt(s["length_ratio"], ".2f"),
        fmt(s["frames"], ".0f"),
        fmt(s["replans"], ".0f"),
        r["changes"],
    ])
  widths = [max(len(header[i]), *(len(b[i]) for b in body)) for i in range(len(header))]
  line = "  ".join("{:<" + str(w) + "}" for w in widths)
  print()
  print(line.format(*header))
  print("  ".join("-" * w for w in widths))
  for b in body:
    print(line.format(*b))
  print(
      f"\n{rows[0]['summary']['n']} worlds. length/exp, frames and replans are "
      "medians over successful worlds."
  )


def print_paired(rows, all_results, baseline):
  others = [r["name"] for r in rows if r["name"] != baseline]
  if not others:
    return
  base_ok = [is_success(r) for r in all_results[baseline]]
  print(f"\nPaired comparison against '{baseline}' (same worlds):")
  for name in others:
    ok = [is_success(r) for r in all_results[name]]
    wins = sum(a and not b for a, b in zip(ok, base_ok))
    losses = sum(b and not a for a, b in zip(ok, base_ok))
    same = len(ok) - wins - losses
    p = sign_test_p(wins, losses)
    verdict = "difference likely real" if p < 0.05 else "within the noise"
    p_text = "< 0.001" if p < 0.001 else f"= {p:.3f}"
    print(
        f"  {name}: succeeds where baseline fails in {wins} worlds, fails where "
        f"baseline succeeds in {losses}, same in {same}  "
        f"(sign test p {p_text}, {verdict})"
    )


def save_csv(rows):
  path = runs.RUNS_DIR / CSV_FILE
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run", "n_worlds", "success", "success_ci_low", "success_ci_high",
                "collisions", "timeouts", "length_ratio_median",
                "frames_median", "replans_median", "changes_vs_baseline"])
    for r in rows:
      s = r["summary"]
      w.writerow([
          r["name"], s["n"], f"{s['success']:.4f}", f"{s['success_lo']:.4f}",
          f"{s['success_hi']:.4f}", f"{s['collisions']:.4f}",
          f"{s['timeouts']:.4f}", fmt(s["length_ratio"], ".4f"),
          fmt(s["frames"], ".1f"), fmt(s["replans"], ".1f"), r["changes"],
      ])
  return path


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def parse_args():
  p = argparse.ArgumentParser(description="Closed-loop benchmark of trained runs.")
  p.add_argument("--runs", nargs="+", default=None,
                 help="run names (default: all runs in runs/)")
  p.add_argument("--baseline", default=None,
                 help="run the others are compared against (default: first run)")
  p.add_argument("--worlds", type=int, default=300, help="number of test worlds")
  p.add_argument("--seed", type=int, default=1234, help="seed of the test worlds")
  p.add_argument("--candidates", type=int, default=None,
                 help="candidate paths per replan (default 7)")
  p.add_argument("--device", default="cpu")
  p.add_argument("--regenerate", action="store_true",
                 help="create new test worlds even if the file matches")
  p.add_argument("--force", action="store_true",
                 help="ignore cached results and run everything again")
  return p.parse_args()


def main():
  args = parse_args()

  names = args.runs or runs.list_runs()
  if not names:
    sys.exit(f"No runs found in '{runs.RUNS_DIR}'. Train one with train.py first.")
  missing = [n for n in names if not runs.has_model(n)]
  if missing:
    sys.exit(f"Unknown runs: {missing}. Available: {runs.list_runs()}")
  baseline = args.baseline or names[0]
  if baseline not in names:
    sys.exit(f"Baseline '{baseline}' is not among the runs: {names}")
  order = [baseline] + sorted(n for n in names if n != baseline)

  params = SimParams()
  if args.candidates is not None:
    params.n_candidates = args.candidates

  data = load_test_worlds(args.worlds, args.seed, args.regenerate)
  worlds = data["worlds"]
  test_set = {"seed": data["seed"], "n": data["n"]}

  baseline_settings = Settings.load(runs.run_dir(baseline) / runs.SETTINGS_FILE)
  all_results, rows = {}, []
  for name in order:
    print(f"Run '{name}':")
    all_results[name] = cached_or_run(
        name, test_set, worlds, args.seed, params, args.device, args.force
    )
    settings = Settings.load(runs.run_dir(name) / runs.SETTINGS_FILE)
    rows.append({
        "name": name,
        "summary": summarize(all_results[name]),
        "changes": "(baseline)" if name == baseline
        else describe_changes(settings, baseline_settings),
    })

  print_table(rows)
  print_paired(rows, all_results, baseline)
  print(f"\nSaved: {save_csv(rows)} and runs/<name>/{BENCHMARK_FILE}")


if __name__ == "__main__":
  main()