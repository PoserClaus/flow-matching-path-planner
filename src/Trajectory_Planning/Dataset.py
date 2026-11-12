"""
Dataset.py — Datensatz aus Training_Set_Generation laden und fürs Training aufbereiten.

    1. .npz laden (Format: Training_Set_Generation/Dataset_Format.py)
    2. Train/Val nach Szene trennen (Samples einer Szene sind sich zu ähnlich)
    3. Optional normieren (normalize=True), Faktoren nur aus den Trainingsdaten:
         Trajektorie und Ziel ÷ pos_scale  (Streuung der Trajektorien)
         Sensoren             ÷ größter Wert (LiDAR: 8 m -> 0..1)
       Ohne Normierung sind alle Faktoren 1 (Rohwerte in Metern).
    4. Batches nach Gewicht ziehen: kurze Wege eines Start/Ziel-Paars kommen öfter dran.
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "Training_Set_Generation"))

from Dataset_Format import load_dataset  # noqa: E402


class TrajectoryData:
    def __init__(self, path, inputs, val_fraction=0.2, normalize=True, seed=0, device="cpu"):
        d = load_dataset(path)
        self.inputs = inputs

        # 2. Train/Val nach Szene
        scenes = np.random.default_rng(seed).permutation(np.unique(d["scene_id"]))
        n_val = max(1, round(len(scenes) * val_fraction))
        is_val = np.isin(d["scene_id"], scenes[:n_val])
        self.n_scenes = (len(scenes) - n_val, n_val)
        self.raw = d                              # Rohdaten (Szenen, Startposen, ...) für die Validierung
        self.val_idx = np.flatnonzero(is_val)     # self.val[...][j] gehört zu raw[...][val_idx[j]]

        # 3. Normierung
        train = ~is_val
        self.norm = {
            "pos_scale": float(d["traj"][train].std()) if normalize else 1.0,
            "obs_scale": {k: float(d[f"obs/{k}"][train].max()) if normalize else 1.0 for k in inputs},
        }

        self.spec = {
            "n_points": d["traj"].shape[1],
            "goal_dim": 2,
            "obs": {k: list(d[f"obs/{k}"].shape[1:]) for k in inputs},
        }
        self.train = self._to_tensors(d, train, device)
        self.val = self._to_tensors(d, is_val, device)

    def _to_tensors(self, d, mask, device):
        def t(x):
            return torch.tensor(x, dtype=torch.float32, device=device)

        cond = {"goal": t(d["goal_rel"][mask, :2] / self.norm["pos_scale"])}
        for k in self.inputs:
            cond[k] = t(d[f"obs/{k}"][mask] / self.norm["obs_scale"][k])
        return {
            "x1": t(d["traj"][mask] / self.norm["pos_scale"]),
            "cond": cond,
            "weight": t(d["weight"][mask]),
        }

    def batch(self, split, batch_size):
        """Zufälliger Batch (x1, cond) aus "train" oder "val", gezogen nach Gewicht."""
        data = self.train if split == "train" else self.val
        i = torch.multinomial(data["weight"], batch_size, replacement=True)
        return data["x1"][i], {k: v[i] for k, v in data["cond"].items()}
