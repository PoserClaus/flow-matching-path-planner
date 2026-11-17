"""Train a flow matching planner. One call = one run in runs/<name>/.

Examples (run from src/):
  python train.py --name bend_baseline
  python train.py --name gaussian --set source.kind=gaussian
  python train.py --name short --epochs 20 --set training.lr=1e-3

All settings and their defaults are in fm/settings.py. The settings used are
saved next to the weights, so the viewer and the benchmark always use a model
with the settings it was trained with.

Checkpoints (EMA weights) are written every `evaluation.eval_every` epochs and
at the end. Stopping with Ctrl+C keeps the last checkpoint.
"""

import argparse
import copy
import sys

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from fm import runs
from fm.data import CropDataset, load_episodes
from fm.evaluation import evaluate
from fm.model import build_model
from fm.settings import Settings
from fm.source import training_source


def fm_loss(model, obs, goal_rel, x1, settings):
  x0 = training_source(goal_rel, settings.representation, settings.source)
  t = torch.rand(x1.shape[0], 1, device=x1.device)
  x_t = (1.0 - t) * x0 + t * x1
  v_pred = model(x_t, t, goal_rel, obs)
  return ((v_pred - (x1 - x0)) ** 2).mean()


@torch.no_grad()
def validation_loss(model, loader, device, settings):
  model.eval()
  total, n = 0.0, 0
  for obs, goal_rel, x1 in loader:
    obs, goal_rel, x1 = obs.to(device), goal_rel.to(device), x1.to(device)
    total += fm_loss(model, obs, goal_rel, x1, settings).item() * x1.shape[0]
    n += x1.shape[0]
  return total / max(n, 1)


@torch.no_grad()
def ema_update(ema, model, decay):
  for p_ema, p in zip(ema.parameters(), model.parameters()):
    p_ema.mul_(decay).add_(p.detach(), alpha=1.0 - decay)


def train_model(settings):
  rep, aug, tr, ev = (
      settings.representation, settings.augmentation, settings.training,
      settings.evaluation,
  )
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  np.random.seed(tr.seed)
  torch.manual_seed(tr.seed)

  print("Loading dataset...")
  episodes = load_episodes(runs.DATASET_PATH)
  perm = np.random.default_rng(tr.seed).permutation(len(episodes))
  n_val = max(1, int(tr.val_fraction * len(episodes)))
  val_eps = [episodes[i] for i in perm[:n_val]]
  train_eps = [episodes[i] for i in perm[n_val:]]
  lengths = [float(e["cum"][-1]) for e in episodes]
  print(
      f"{len(train_eps)} train / {len(val_eps)} val worlds, median path "
      f"{np.median(lengths):.0f}px, horizon {rep.horizon_px:.0f}px, "
      f"source '{settings.source.kind}', device {device}"
  )

  train_ds = CropDataset(
      train_eps, tr.crops_per_episode, augment=True, seed=tr.seed, rep=rep,
      aug=aug,
  )
  val_ds = CropDataset(
      val_eps, tr.val_crops_per_episode, augment=False, seed=tr.seed + 1,
      rep=rep, aug=aug, cache=True,
  )
  loader = DataLoader(
      train_ds, batch_size=tr.batch_size, shuffle=True, drop_last=True
  )
  val_loader = DataLoader(val_ds, batch_size=512)

  model = build_model(settings).to(device)
  ema = copy.deepcopy(model).eval()
  for p in ema.parameters():
    p.requires_grad_(False)

  optimizer = optim.AdamW(model.parameters(), lr=tr.lr, weight_decay=1e-4)
  total_steps = tr.epochs * len(loader)
  scheduler = optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=total_steps, eta_min=1e-5
  )
  print(
      f"Training {sum(p.numel() for p in model.parameters()):,} parameters for "
      f"{tr.epochs} epochs ({len(loader)} steps each)..."
  )

  runs.save_settings(settings.name, settings)  # recorded even if interrupted
  history = []
  step = 0
  for epoch in range(tr.epochs):
    model.train()
    total, n = 0.0, 0
    for obs, goal_rel, x1 in loader:
      obs, goal_rel, x1 = obs.to(device), goal_rel.to(device), x1.to(device)
      loss = fm_loss(model, obs, goal_rel, x1, settings)
      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      scheduler.step()

      step += 1
      ema_update(ema, model, min(tr.ema_decay, (1 + step) / (10 + step)))
      total += loss.item() * x1.shape[0]
      n += x1.shape[0]

    log_now = (epoch + 1) % 5 == 0 or epoch == 0
    eval_now = (epoch + 1) % ev.eval_every == 0
    if log_now or eval_now:
      entry = {
          "epoch": epoch + 1,
          "train_loss": total / n,
          "val_loss": validation_loss(ema, val_loader, device, settings),
      }
      if log_now:
        print(
            f"Epoch [{epoch + 1}/{tr.epochs}] train {entry['train_loss']:.5f} | "
            f"val {entry['val_loss']:.5f}"
        )
      if eval_now:
        entry["eval"] = evaluate(ema, val_eps, settings)
        runs.save_model(settings.name, ema, settings)
        history.append(entry)
        runs.save_history(settings.name, history)
      else:
        history.append(entry)

  runs.save_model(settings.name, ema, settings)
  runs.save_history(settings.name, history)
  print(
      f"\nTraining complete! EMA model saved to "
      f"'{runs.run_dir(settings.name)}'."
  )


def parse_args():
  p = argparse.ArgumentParser(description="Train a flow matching planner.")
  p.add_argument("--name", default=None,
                 help="run name (folder under runs/); default: 'default'")
  p.add_argument("--set", action="append", default=[], dest="overrides",
                 metavar="group.key=value",
                 help="override a setting, e.g. --set source.kind=gaussian "
                      "(can be repeated)")
  p.add_argument("--epochs", type=int, default=None,
                 help="shortcut for --set training.epochs=N")
  p.add_argument("--overwrite", action="store_true",
                 help="allow overwriting an existing run of the same name")
  return p.parse_args()


if __name__ == "__main__":
  args = parse_args()
  settings = Settings()
  overrides = list(args.overrides)
  if args.epochs is not None:
    overrides.append(f"training.epochs={args.epochs}")
  if args.name:
    overrides.append(f"name={args.name}")
  settings.override(overrides)

  if runs.has_model(settings.name) and not args.overwrite:
    sys.exit(
        f"Run '{settings.name}' already exists in '{runs.RUNS_DIR}'. Pick "
        f"another --name or pass --overwrite."
    )
  print(f"Run '{settings.name}' | changes vs defaults: {settings.changes()}")
  train_model(settings)
