"""Where things live: the dataset and one folder per training run.

runs/<name>/
  model.pt         EMA weights
  settings.json    the settings the model was trained with (needed to use it)
  history.json     losses and evaluation results over the epochs

Files are written atomically (temp file, then rename), so a viewer that loads
a model while training saves a checkpoint never sees a half-written file.

Command line (run from src/):
  python -m fm.runs list
  python -m fm.runs adopt <weights.pt> <name>    # give old weights a run folder
"""

import json
import os
import shutil
import sys
from pathlib import Path

import torch

from config import DATA_DIR
from fm.model import build_model
from fm.settings import Settings

DATASET_PATH = DATA_DIR / "expert_dataset_rotate.pt"
RUNS_DIR = DATA_DIR.parent / "runs"

MODEL_FILE = "model.pt"
SETTINGS_FILE = "settings.json"
HISTORY_FILE = "history.json"


def run_dir(name) -> Path:
  return RUNS_DIR / name


def has_model(name) -> bool:
  return (run_dir(name) / MODEL_FILE).exists()


def _atomic_write(path: Path, writer):
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_name(path.name + ".tmp")
  writer(tmp)
  os.replace(tmp, path)


def save_settings(name, settings):
  _atomic_write(run_dir(name) / SETTINGS_FILE, lambda p: settings.save(p))


def save_model(name, model, settings):
  """Weights + settings together, so a checkpoint is always usable."""
  _atomic_write(
      run_dir(name) / MODEL_FILE, lambda p: torch.save(model.state_dict(), p)
  )
  save_settings(name, settings)


def save_history(name, history):
  _atomic_write(
      run_dir(name) / HISTORY_FILE,
      lambda p: p.write_text(json.dumps(history, indent=2)),
  )


def list_runs():
  """Names of all runs that have weights and settings, alphabetically."""
  if not RUNS_DIR.exists():
    return []
  return sorted(
      d.name for d in RUNS_DIR.iterdir()
      if (d / MODEL_FILE).exists() and (d / SETTINGS_FILE).exists()
  )


def latest_run():
  """The run whose weights were written most recently, or None."""
  names = list_runs()
  if not names:
    return None
  return max(names, key=lambda n: (run_dir(n) / MODEL_FILE).stat().st_mtime)


def load_run(name, device="cpu"):
  """Returns (model in eval mode, settings) for a run."""
  d = run_dir(name)
  if not (d / MODEL_FILE).exists():
    raise FileNotFoundError(f"No model in '{d}'. Existing runs: {list_runs()}")
  settings = Settings.load(d / SETTINGS_FILE)
  model = build_model(settings)
  model.load_state_dict(torch.load(d / MODEL_FILE, map_location=device))
  model.to(device)
  model.eval()
  return model, settings


def adopt_weights(src_path, name, settings=None):
  """Copy existing weights into a new run folder (default settings unless given)."""
  if has_model(name):
    raise FileExistsError(f"Run '{name}' already exists.")
  settings = settings or Settings()
  settings.name = name
  d = run_dir(name)
  d.mkdir(parents=True, exist_ok=True)
  shutil.copyfile(src_path, d / MODEL_FILE)
  save_settings(name, settings)
  return d


if __name__ == "__main__":
  args = sys.argv[1:]
  if args[:1] == ["list"]:
    for n in list_runs():
      print(n)
  elif args[:1] == ["adopt"] and len(args) == 3:
    print(f"Created '{adopt_weights(args[1], args[2])}' with default settings.")
  else:
    print(__doc__)
