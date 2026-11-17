"""
Dashboard_Testset.py — Überblick über einen erzeugten Datensatz in einem Fenster.

    links         Config, mit der der Datensatz erzeugt wurde
    Mitte oben    Inhalt: Anzahl Szenen/Paare/Samples, jedes Array mit Form und erstem Wert
    Mitte unten   Wie viele verschiedene Wege der Solver pro Paar gefunden hat
    rechts        Beispiel-Trajektorien aus verschiedenen Szenen

Aufruf (aus dem Ordner Training_Set_Generation):
    python Dashboard_Testset.py                              # neuester Datensatz in Test_Sets/
    python Dashboard_Testset.py Test_Sets/2026-11-17_14-32-05.npz
    python Dashboard_Testset.py --save dashboard.png
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from Car_Model import vehicle_to_world
from Dataset_Format import load_dataset, newest_dataset
from Scenes import Scene
from Scenes.Plot_Scene import plot_scene
from Solvers import Solution


def config_text(data: dict) -> str:
    return json.dumps(data["meta"]["config"], indent=2)


def data_text(data: dict) -> str:
    meta = data["meta"]
    n_pairs = len(np.unique(data["pair_id"]))
    lines = [
        f"Szenen:  {len(data['scenes'])}",
        f"Paare:   {n_pairs}   (ohne Lösung: {meta['n_pairs_failed']})",
        f"Samples: {len(data['weight'])}   (1 Sample = 1 Weg)",
        "",
        "Arrays (Form · erstes Sample):",
    ]
    for key, v in data.items():
        if not isinstance(v, np.ndarray):
            continue
        first = np.round(v[0].reshape(-1)[:3].astype(float), 2).tolist()
        more = " …" if v[0].size > 3 else ""
        lines.append(f"  {key:10s} {str(v.shape):14s} {first}{more}")
    return "\n".join(lines)


def plot_example(ax, data: dict, pair: int):
    """Ein Start/Ziel-Paar mit allen Wegen in der Weltansicht."""
    idx = np.flatnonzero(data["pair_id"] == pair)
    i0 = idx[0]
    scene = Scene.from_dict(data["scenes"][data["scene_id"][i0]])
    start = data["start"][i0].astype(float)
    scene.start, scene.goal = start, data["goal"][i0].astype(float)
    sols = [Solution(vehicle_to_world(data["traj"][i], start), float(data["length"][i]), float(data["weight"][i]))
            for i in idx]
    plot_scene(ax, scene, solutions=sols, robot_radius=data["meta"]["car"]["radius"],
               title=f"Szene {data['scene_id'][i0]} · {len(idx)} Wege")


def dashboard(data: dict, title: str, n_examples: int = 4):
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1.3, 1.5, 1.5])

    # Config
    ax = fig.add_subplot(gs[:, 0])
    ax.axis("off")
    ax.set_title("Config", loc="left", fontweight="bold")
    ax.text(0, 1, config_text(data), va="top", family="monospace", fontsize=7, transform=ax.transAxes)

    # Inhalt
    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    ax.set_title("Inhalt", loc="left", fontweight="bold")
    ax.text(0, 1, data_text(data), va="top", family="monospace", fontsize=7.5, transform=ax.transAxes)

    # Wege pro Paar
    ax = fig.add_subplot(gs[1, 1])
    _, paths_per_pair = np.unique(data["pair_id"], return_counts=True)
    values, counts = np.unique(paths_per_pair, return_counts=True)
    ax.bar(values, counts, color="#1f77b4")
    ax.set_xticks(values)
    ax.set_xlabel("verschiedene Wege pro Paar")
    ax.set_ylabel("Anzahl Paare")
    ax.set_title("Wege pro Paar", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Beispiele: je ein Paar aus verschiedenen Szenen
    _, first_of_scene = np.unique(data["scene_id"], return_index=True)
    examples = data["pair_id"][first_of_scene][:n_examples]
    for k, pair in enumerate(examples):
        plot_example(fig.add_subplot(gs[k // 2, 2 + k % 2]), data, int(pair))

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dashboard für einen Datensatz.")
    parser.add_argument("path", nargs="?", help="Datensatz (Default: neuester in Test_Sets/)")
    parser.add_argument("--save", help="Als Bild speichern statt Fenster öffnen")
    args = parser.parse_args()

    path = args.path or newest_dataset()
    data = load_dataset(path)
    meta = data["meta"]
    fig = dashboard(data, f"{path}   ·   erzeugt {meta.get('created', '?')} mit {meta.get('config_file', '?')}")
    if args.save:
        fig.savefig(args.save, dpi=110, bbox_inches="tight")
        print(f"Gespeichert: {args.save}")
    else:
        plt.show()
