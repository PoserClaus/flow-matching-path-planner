# Trajectory Planning (Flow Matching)

Learns paths from sensor data: **LiDAR scan + goal → path**.
Uses the datasets from `Training_Set_Generation`.

## Setup (once)

```
cd flow-matching-path-planner
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install numpy matplotlib
```

## Usage

From `src/Trajectory_Planning`, with the venv:

```
python Train.py                          # uses configs/default.json
python Train.py --config configs/x.json
python Dashboard.py                      # learning curve + validation of the newest run
python Dashboard.py runs/<run> --save dashboard.png
python Overview.py                       # compare all runs: settings that differ + metrics
```

Result: `runs/<name>_<time>/model.pt`, `config.json`, `log.csv`.

The dashboard plans 8 paths for every start/goal pair of the validation scenes and reports:
collision rate, path length ÷ shortest expert path, goal error (m), distance to the closest expert path (m).

## Files

| File | Purpose |
|---|---|
| `Models/Base_Model.py` | Interface every model has: `model(x_t, t, cond) → velocity`. |
| `Models/MLP.py` | Simple MLP. All inputs (path, time, goal, raw LiDAR) concatenated. |
| `Models/__init__.py` | List of models. Add new models here. |
| `Flow_Matching.py` | Loss and sampling. Works with any model. |
| `Dataset.py` | Loads the dataset, splits train/val by scene, optional normalisation. |
| `Train.py` | Training loop. |
| `Dashboard.py` | Learning curve, validation metrics and example scenes for one run. Saves `metrics.json`. |
| `Overview.py` | Table of all runs (data config, training config, metrics). Writes `runs/overview.csv`. |

## Config

| Key | Meaning |
|---|---|
| `name` | Name of the run folder. |
| `dataset` | Dataset file, relative to this folder. |
| `inputs` | Sensors used as input, e.g. `["lidar"]`. |
| `normalize` | `true`: scale values to roughly ±1. `false`: raw values in metres. |
| `val_fraction` | Share of scenes used for validation. |
| `model.type` | Which model (name from `Models/__init__.py`). |
| `model.params` | Parameters of that model, e.g. `hidden`, `n_layers` for the MLP. |
| `training.steps` | Number of training steps. |
| `training.batch_size` | Samples per step. |
| `training.lr` | Learning rate (constant, Adam). |
| `training.eval_every` | Print train and validation loss every n steps. |

## Adding a model

1. Create `Models/My_Model.py`: subclass `BaseModel`, set `name`, write `__init__(spec, ...)` and `forward(x_t, t, cond)`.
2. Add it to `MODELS` in `Models/__init__.py`.
3. In the config: `"model": {"type": "my_model", "params": {...}}`.
