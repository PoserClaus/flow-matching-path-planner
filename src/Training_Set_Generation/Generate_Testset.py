"""
Generate_Testset.py — Trainingsdatensatz erzeugen.

Ablauf pro Szene:
    1. Szene würfeln (Szenengenerator aus der Config)
    2. pairs_per_scene Start/Ziel-Paare zufällig in der Szene verteilen
    3. Pro Paar: Solver sucht mehrere verschiedene Wege (gewichtet nach Länge)
    4. Pro Paar: Sensoren messen an der Startpose (Fahrzeug steht)
    5. Pro Weg ein Sample: (Sensordaten, Ziel relativ, ganzer Weg relativ, Gewicht)

Bewusst einfach: ganze Wege statt Ausschnitten, keine Fahrbarkeitsprüfung.
Das Format steht in Dataset_Format.py.

Aufruf (aus dem Ordner Training_Set_Generation):
    python Generate_Testset.py                                   # configs/default.json
    python Generate_Testset.py --config configs/default.json --out Test_Sets/train.npz
    python Generate_Testset.py --n-scenes 5                      # schneller Test
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np

from Car_Model import CarConfig, VehicleState, world_to_vehicle, wrap_angle
from Dataset_Format import save_dataset
from Scenes import SCENES
from Sensors import SENSORS
from Solvers import SOLVERS
from Solvers.Base_Solver import resample_polyline

HERE = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────────────
# Bausteine aus der Config
# ─────────────────────────────────────────────────────────────────────────────
def build(config: dict):
    """Szenengenerator, Solver und Sensoren aus der Config erzeugen."""
    car = CarConfig(**config.get("car", {}))
    scene_gen = SCENES[config["scene"]["type"]](**config["scene"].get("params", {}))
    solver = SOLVERS[config["solver"]["type"]](car, **config["solver"].get("params", {}))
    sensors = [SENSORS[s["type"]](**s.get("params", {})) for s in config["sensors"]]
    names = [s.name for s in sensors]
    if len(set(names)) != len(names):
        raise ValueError(f"Sensornamen müssen eindeutig sein: {names}")
    return car, scene_gen, solver, sensors


def sample_pair(scene, rng, min_dist: float, min_clearance: float, max_tries: int = 200):
    """Zufällige Start- und Zielpose mit Freiraum und Mindestabstand. None, wenn nichts gefunden."""
    xmin, xmax, ymin, ymax = scene.bounds
    m = min_clearance

    def free_point():
        for _ in range(max_tries):
            p = rng.uniform([xmin + m, ymin + m], [xmax - m, ymax - m])
            if scene.clearance(p) >= m:
                return p
        return None

    start_xy = free_point()
    if start_xy is None:
        return None
    for _ in range(max_tries):
        goal_xy = free_point()
        if goal_xy is not None and np.hypot(*(goal_xy - start_xy)) >= min_dist:
            heading = float(np.arctan2(*(goal_xy - start_xy)[::-1]))
            start = np.array([*start_xy, rng.uniform(-np.pi, np.pi)])
            goal = np.array([*goal_xy, heading])
            return start, goal
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Generierung
# ─────────────────────────────────────────────────────────────────────────────
def generate(config: dict, verbose: bool = True):
    car, scene_gen, solver, sensors = build(config)
    rng = np.random.default_rng(config["seed"])
    pairs_cfg = config["pairs"]
    n_points = config["trajectory"]["n_points"]

    samples = {k: [] for k in ("goal_rel", "traj", "weight", "length", "start", "goal", "scene_id", "pair_id")}
    obs = {s.name: [] for s in sensors}
    scenes = []
    pair_id = 0
    n_pairs_failed = 0
    t0 = time.time()

    for i in range(config["n_scenes"]):
        scene = scene_gen.sample(rng)
        if scene is None:
            print(f"  Szene {i}: kein gültiger Aufbau gefunden, übersprungen")
            continue
        scene_id = len(scenes)
        scenes.append(scene.to_dict())
        n_samples_scene = 0

        for _ in range(pairs_cfg["per_scene"]):
            pair = sample_pair(scene, rng, pairs_cfg["min_dist"], pairs_cfg["min_clearance"])
            if pair is None:
                n_pairs_failed += 1
                continue
            start, goal = pair
            solutions = solver.solve(scene, start, goal)
            if not solutions:
                n_pairs_failed += 1
                continue

            # Sensoren an der Startpose (Fahrzeug steht)
            state = VehicleState.standing(start)
            measurements = {}
            for s in sensors:
                s.reset(rng)
                measurements[s.name] = s.observe(scene, state, rng)

            goal_rel = np.array([*world_to_vehicle(goal[:2], start), wrap_angle(goal[2] - start[2])])
            for sol in solutions:
                traj_world = resample_polyline(sol.waypoints, n=n_points)
                samples["traj"].append(world_to_vehicle(traj_world, start))
                samples["goal_rel"].append(goal_rel)
                samples["weight"].append(sol.weight)
                samples["length"].append(sol.length)
                samples["start"].append(start)
                samples["goal"].append(goal)
                samples["scene_id"].append(scene_id)
                samples["pair_id"].append(pair_id)
                for name, z in measurements.items():
                    obs[name].append(z)
                n_samples_scene += 1
            pair_id += 1

        if verbose:
            print(f"  Szene {i + 1}/{config['n_scenes']}: {len(scene.boxes)} Boxen, "
                  f"{n_samples_scene} Samples  ({time.time() - t0:.0f} s)")

    arrays = {
        "goal_rel": np.array(samples["goal_rel"], dtype=np.float32).reshape(-1, 3),
        "traj": np.array(samples["traj"], dtype=np.float32).reshape(-1, n_points, 2),
        "weight": np.array(samples["weight"], dtype=np.float32),
        "length": np.array(samples["length"], dtype=np.float32),
        "start": np.array(samples["start"], dtype=np.float32).reshape(-1, 3),
        "goal": np.array(samples["goal"], dtype=np.float32).reshape(-1, 3),
        "scene_id": np.array(samples["scene_id"], dtype=np.int32),
        "pair_id": np.array(samples["pair_id"], dtype=np.int32),
    }
    for s in sensors:
        arrays[f"obs/{s.name}"] = np.array(obs[s.name], dtype=s.spec().dtype).reshape(-1, *s.spec().shape)

    meta = {
        "config": config,
        "car": car.__dict__,
        "scene_generator": scene_gen.describe(),
        "solver": solver.describe(),
        "sensors": [s.describe() for s in sensors],
        "n_pairs": pair_id,
        "n_pairs_failed": n_pairs_failed,
    }
    return arrays, scenes, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trainingsdatensatz erzeugen.")
    parser.add_argument("--config", default=os.path.join(HERE, "configs", "default.json"))
    parser.add_argument("--out", default=None, help="Ausgabedatei (.npz). Default: aus der Config")
    parser.add_argument("--n-scenes", type=int, default=None, help="Anzahl Szenen überschreiben")
    parser.add_argument("--seed", type=int, default=None, help="Seed überschreiben")
    parser.add_argument("--overwrite", action="store_true", help="Vorhandene Ausgabedatei ersetzen")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    config = copy.deepcopy(config)
    if args.n_scenes is not None:
        config["n_scenes"] = args.n_scenes
    if args.seed is not None:
        config["seed"] = args.seed
    out = args.out or os.path.join(HERE, config["out"])
    if os.path.exists(out) and not args.overwrite:
        raise SystemExit(f"{out} existiert schon.\nAnderen \"out\"-Namen in der Config setzen "
                         f"oder --overwrite anhängen, um die Datei zu ersetzen.")

    print(f"Config: {args.config}")
    arrays, scenes, meta = generate(config)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    save_dataset(out, arrays, scenes, meta)

    n = len(arrays["weight"])
    print(f"\nGespeichert: {out}")
    print(f"  {len(scenes)} Szenen, {meta['n_pairs']} Paare ({meta['n_pairs_failed']} ohne Lösung), {n} Samples")
    for k, v in arrays.items():
        print(f"  {k:12s} {str(v.shape):16s} {v.dtype}")
