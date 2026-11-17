# Training Set Generation

Generates training data: random scenes, random start/goal pairs, several different paths per pair (A*),
LiDAR scan at the start.

## Open the app

The app lets you set the config, generate a dataset with one click and explore it.

**Once** (from the repo root `flow-matching-path-planner`):

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

**Start the app** (from the repo root):

```
cd src\Training_Set_Generation
..\..\.venv\Scripts\streamlit run App.py
```

The browser opens automatically (otherwise go to http://localhost:8501). Stop with `Ctrl+C` in the terminal.

In the app:
1. **Sidebar:** pick a config as starting point, change values, click **Datensatz generieren**.
2. **Dashboard:** opens the new dataset automatically. Older datasets can be picked at the top.
   - *Trajektorien:* pick scene and pair, show/hide single paths, the A* grid and the LiDAR scan.
   - *Übersicht:* paths per pair, path lengths, table of all pairs.
   - *Config:* the config the dataset was generated with.
   - *Daten:* stored arrays with shape and example values.

Datasets are saved as `Test_Sets/<date>_<time>.npz`, nothing gets overwritten.

## Without the app

Command line, config file and all parameters: see [configs/README.md](configs/README.md).
