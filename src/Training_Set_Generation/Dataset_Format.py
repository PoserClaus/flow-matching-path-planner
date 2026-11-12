"""
Dataset_Format.py — Speicherformat des Trainingsdatensatzes (.npz, nur numpy).

Ein Sample = ein Weg des Solvers für ein Start/Ziel-Paar in einer Szene.
Hat ein Paar mehrere Wege, entstehen mehrere Samples mit gleicher Beobachtung
(multimodal), ihre Gewichte summieren sich pro Paar zu 1.

Arrays (M = Anzahl Samples, N = Punkte pro Trajektorie):
    obs/<sensor>   [M, ...]   Sensormessung an der Startpose, z. B. obs/lidar [M, 360]
    goal_rel       [M, 3]     Ziel im Fahrzeugframe des Starts (x, y, yaw)
    traj           [M, N, 2]  Weg im Fahrzeugframe des Starts, gleichmäßig nach Bogenlänge,
                              traj[:, 0] = (0, 0), traj[:, -1] = Ziel
    weight         [M]        Gewicht des Wegs innerhalb seines Paars (Summe pro Paar = 1)
    length         [M]        Weglänge in m
    start          [M, 3]     Startpose im Weltframe (zum Plotten / Zurückrechnen)
    goal           [M, 3]     Zielpose im Weltframe
    scene_id       [M]        Index in scenes
    pair_id        [M]        Laufende Nummer des Start/Ziel-Paars (global)

Zusätzlich (als JSON-Text):
    scenes         Liste von Scene.to_dict()  -> Scene.from_dict()
    meta           Config, Beschreibung von Szenengenerator, Solver, Sensoren, Version

Wichtig fürs Training: Train/Val nach scene_id trennen, nicht nach Sample.
"""

from __future__ import annotations

import json

import numpy as np

FORMAT_VERSION = 1


def save_dataset(path: str, arrays: dict[str, np.ndarray], scenes: list[dict], meta: dict) -> None:
    meta = {**meta, "format_version": FORMAT_VERSION}
    np.savez_compressed(
        path,
        **arrays,
        scenes=np.array(json.dumps(scenes)),
        meta=np.array(json.dumps(meta, default=_to_json)),
    )


def load_dataset(path: str) -> dict:
    """Lädt den Datensatz. Rückgabe: Arrays wie oben, plus 'scenes' (Liste) und 'meta' (dict)."""
    with np.load(path, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    data["scenes"] = json.loads(str(data["scenes"]))
    data["meta"] = json.loads(str(data["meta"]))
    return data


def _to_json(obj):
    """numpy-Typen in JSON-Typen umwandeln (z. B. np.float64 in Configs)."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Nicht JSON-serialisierbar: {type(obj)}")
