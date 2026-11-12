"""
Dashboard.py — Training und Validierung eines Laufs anzeigen.

    oben links   Lernkurve (train / val loss aus log.csv)
    oben rechts  Metriken auf den Val-Szenen (nie im Training gesehen)
    darunter     Beispiel-Szenen: Experten-Wege grau, Netz-Wege blau (rot = Kollision)

Validierung: Für jedes Start/Ziel-Paar der Val-Szenen plant das Netz --samples Wege.
    Kollisionsrate        Anteil Wege, bei denen der Roboterkreis ein Hindernis berührt
    Weglänge              Länge des Netz-Wegs ÷ kürzester Experten-Weg  (1.0 = gleich lang)
    Zielfehler            Abstand Wegende -> Ziel (m)
    Abstand zum Experten  mittlerer Punktabstand zum ähnlichsten Experten-Weg (m)

Aufruf (aus dem Ordner Trajectory_Planning, mit der venv):
    python Dashboard.py                           # neuester Lauf in runs/
    python Dashboard.py runs/mlp_20261112-181538
    python Dashboard.py --save dashboard.png
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from Dataset import HERE, TrajectoryData   # setzt auch den Pfad zu Training_Set_Generation
from Flow_Matching import sample
from Models import MODELS

from Car_Model import vehicle_to_world             # noqa: E402  (aus Training_Set_Generation)
from Scenes import Scene                           # noqa: E402
from Scenes.Plot_Scene import plot_scene           # noqa: E402
from Solvers.Base_Solver import polyline_length, resample_polyline  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Laden
# ─────────────────────────────────────────────────────────────────────────────
def load_run(run_dir, device):
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    ckpt = torch.load(os.path.join(run_dir, "model.pt"), map_location=device, weights_only=True)
    model = MODELS[ckpt["model_type"]](ckpt["spec"], **ckpt["model_params"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    data = TrajectoryData(os.path.join(HERE, config["dataset"]), config["inputs"],
                          config["val_fraction"], config["normalize"], config["seed"], device)
    log = np.genfromtxt(os.path.join(run_dir, "log.csv"), delimiter=",", names=True, ndmin=1)
    return model, data, log, ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Validierung
# ─────────────────────────────────────────────────────────────────────────────
def validate(model, data, n_samples):
    d = data.raw
    radius = d["meta"]["car"]["radius"]
    pair_of = d["pair_id"][data.val_idx]
    pairs, first = np.unique(pair_of, return_index=True)      # first: Position im Val-Set

    # Netz planen lassen: jedes Paar n_samples-mal
    js = torch.as_tensor(np.repeat(first, n_samples), device=data.val["x1"].device)
    cond = {k: v[js] for k, v in data.val["cond"].items()}
    paths = sample(model, cond, data.spec["n_points"]) * data.norm["pos_scale"]
    paths = paths.cpu().numpy().reshape(len(pairs), n_samples, -1, 2)          # [P, K, N, 2] in m

    results = []
    for pair, j, pair_paths in zip(pairs, first, paths):
        i = data.val_idx[j]                                                     # Index in den Rohdaten
        scene = Scene.from_dict(d["scenes"][d["scene_id"][i]])
        start = d["start"][i].astype(float)
        experts = d["traj"][d["pair_id"] == pair]
        shortest = d["length"][d["pair_id"] == pair].min()

        r = {"idx": i, "paths": pair_paths, "experts": experts, "collision": [], "length_ratio": [],
             "goal_error": [], "expert_dist": []}
        for p in pair_paths:
            world = resample_polyline(vehicle_to_world(p, start), spacing=0.05)
            r["collision"].append(bool(np.any(scene.collides(world, radius))))
            r["length_ratio"].append(polyline_length(p) / shortest)
            r["goal_error"].append(float(np.linalg.norm(p[-1] - d["goal_rel"][i, :2])))
            r["expert_dist"].append(float(np.linalg.norm(p - experts, axis=-1).mean(axis=1).min()))
        results.append(r)

    metrics = {k: float(np.mean([r[k] for r in results]))
               for k in ("collision", "length_ratio", "goal_error", "expert_dist")}
    return metrics, results


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────────────────────────────────────
def draw(run_dir, log, metrics, results, data, ckpt, n_samples, n_plot):
    d = data.raw
    fig = plt.figure(figsize=(15, 9.5))
    grid = fig.add_gridspec(3, 3, height_ratios=[1, 1.15, 1.15])
    fig.suptitle(f"{os.path.basename(run_dir)}  ·  {ckpt['model_type']}  ·  "
                 f"{sum(v.numel() for v in ckpt['state_dict'].values()):,} Parameter", fontsize=12)

    # Lernkurve
    ax = fig.add_subplot(grid[0, :2])
    ax.plot(log["step"], log["train_loss"], "-o", ms=3, label="train")
    ax.plot(log["step"], log["val_loss"], "-o", ms=3, label="val")
    ax.set_xlabel("Trainingsschritt")
    ax.set_ylabel("Flow-Matching-Loss")
    ax.set_title("Training", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend()

    # Metriken
    ax = fig.add_subplot(grid[0, 2])
    ax.axis("off")
    rows = [
        ("Kollisionsrate", f"{metrics['collision']:.0%}"),
        ("Weglänge / kürzester Experte", f"{metrics['length_ratio']:.2f}"),
        ("Zielfehler", f"{metrics['goal_error']:.2f} m"),
        ("Abstand zum Experten", f"{metrics['expert_dist']:.2f} m"),
    ]
    ax.set_title(f"Validierung: {len(results)} Paare × {n_samples} Wege\n"
                 f"({data.n_scenes[1]} Szenen, nie im Training gesehen)", fontsize=10)
    for k, (name, value) in enumerate(rows):
        y = 0.8 - k * 0.22
        ax.text(0.02, y, name, fontsize=11, transform=ax.transAxes)
        ax.text(0.98, y, value, fontsize=14, fontweight="bold", ha="right", transform=ax.transAxes)

    # Beispiel-Szenen: je ein Paar aus verschiedenen Szenen
    _, one_per_scene = np.unique([d["scene_id"][r["idx"]] for r in results], return_index=True)
    for k, r in enumerate([results[j] for j in one_per_scene[:n_plot]]):
        ax = fig.add_subplot(grid[1 + k // 3, k % 3])
        i = r["idx"]
        scene = Scene.from_dict(d["scenes"][d["scene_id"][i]])
        start = d["start"][i].astype(float)
        scene.start, scene.goal = start, d["goal"][i].astype(float)
        plot_scene(ax, scene, robot_radius=d["meta"]["car"]["radius"],
                   title=f"Kollision {np.mean(r['collision']):.0%} · Zielfehler {np.mean(r['goal_error']):.2f} m")
        for e in r["experts"]:
            w = vehicle_to_world(e, start)
            ax.plot(w[:, 0], w[:, 1], color="#9e9e9e", lw=4, alpha=0.6, zorder=2.5)
        for p, hit in zip(r["paths"], r["collision"]):
            w = vehicle_to_world(p, start)
            ax.plot(w[:, 0], w[:, 1], color="#d32f2f" if hit else "#1565c0", lw=1, alpha=0.8, zorder=3.5)

    fig.legend(handles=[plt.Line2D([], [], color="#9e9e9e", lw=4, label="Experte (A*)"),
                        plt.Line2D([], [], color="#1565c0", lw=1.5, label="Netz"),
                        plt.Line2D([], [], color="#d32f2f", lw=1.5, label="Netz, Kollision")],
               loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", nargs="?", help="Lauf-Ordner (Default: neuester in runs/)")
    parser.add_argument("--samples", type=int, default=8, help="Wege pro Start/Ziel-Paar")
    parser.add_argument("--n-plot", type=int, default=6, help="Beispiel-Szenen im Dashboard (max. 6)")
    parser.add_argument("--save", help="Als Bild speichern statt Fenster öffnen")
    args = parser.parse_args()

    run_dir = args.run or max(glob.glob(os.path.join(HERE, "runs", "*")), key=os.path.getmtime)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    model, data, log, ckpt = load_run(run_dir, device)
    metrics, results = validate(model, data, args.samples)
    print(f"Lauf: {run_dir}")
    for k, v in metrics.items():
        print(f"  {k:14s} {v:.3f}")
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:   # für Overview.py
        json.dump({**metrics, "final_train_loss": float(log["train_loss"][-1]),
                   "final_val_loss": float(log["val_loss"][-1])}, f, indent=2)

    fig = draw(run_dir, log, metrics, results, data, ckpt, args.samples, min(args.n_plot, 6))
    if args.save:
        fig.savefig(args.save, dpi=100, bbox_inches="tight")
        print(f"Gespeichert: {args.save}")
    else:
        plt.show()
