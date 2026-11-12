"""
Plot_Testset.py — Erzeugten Datensatz anschauen.

Jedes Start/Ziel-Paar wird zweimal gezeigt:
    links   Weltansicht: Szene mit allen Wegen des Paars (Dicke = Gewicht)
    rechts  Fahrzeugansicht = was das Netz sieht: LiDAR-Punkte, Ziel und Wege
            im Fahrzeugframe des Starts (x nach vorne, y nach links)

Aufruf (aus dem Ordner Training_Set_Generation):
    python Plot_Testset.py Test_Sets/default.npz                 # durchblättern
    python Plot_Testset.py Test_Sets/default.npz --pdf all.pdf   # alle Szenen als PDF zum Scrollen
    python Plot_Testset.py Test_Sets/default.npz --save check.png --n 4 --seed 1   # Stichprobe als Bild

Tasten im Viewer:
    ← / →   vorheriges / nächstes Paar
    ↑ / ↓   vorherige / nächste Szene
    Pos1 / Ende   erstes / letztes Paar
    G       A*-Raster ein-/ausblenden

Das A*-Raster zeigt, wo der Fahrzeugmittelpunkt NICHT hin darf (rot:
Hindernis + Fahrzeugradius + Sicherheitsabstand). Abschalten mit --no-grid.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from Car_Model import CarConfig, vehicle_to_world
from Dataset_Format import load_dataset
from Scenes import Scene
from Scenes.Plot_Scene import COLOR_GOAL, COLOR_START, PATH_COLORS, label_pose, plot_scene
from Sensors import Lidar
from Solvers import SOLVERS, Solution


def plot_pair(ax_world, ax_vehicle, data: dict, pair: int, lidar: Lidar | None, solver=None):
    idx = np.flatnonzero(data["pair_id"] == pair)
    i0 = idx[0]
    scene = Scene.from_dict(data["scenes"][data["scene_id"][i0]])
    start, goal = data["start"][i0].astype(float), data["goal"][i0].astype(float)
    scene.start, scene.goal = start, goal          # nur fürs Zeichnen

    # Weltansicht
    sols = [Solution(vehicle_to_world(data["traj"][i], start), float(data["length"][i]), float(data["weight"][i]))
            for i in idx]
    grid = solver.occupancy_grid(scene) if solver is not None and hasattr(solver, "occupancy_grid") else None
    plot_scene(ax_world, scene, solutions=sols, grid=grid, robot_radius=data["meta"]["car"]["radius"],
               title=f"Paar {pair} · Szene {data['scene_id'][i0]} · {len(idx)} Wege")

    # Fahrzeugansicht
    if lidar is not None and "obs/lidar" in data:
        r = data["obs/lidar"][i0].astype(float)
        hit = r < lidar.cfg.max_range - 1e-6
        a = lidar.beam_angles
        pts = np.stack([lidar.cfg.mount_x + r * np.cos(a), lidar.cfg.mount_y + r * np.sin(a)], -1)
        ax_vehicle.scatter(pts[hit, 0], pts[hit, 1], s=3, color="#d81b60", label="LiDAR")
    for k, i in enumerate(idx):
        t = data["traj"][i]
        ax_vehicle.plot(t[:, 0], t[:, 1], "-o", ms=2, color=PATH_COLORS[k % len(PATH_COLORS)],
                        lw=1.0 + 4.0 * data["weight"][i])
    g = data["goal_rel"][i0]
    ax_vehicle.arrow(g[0], g[1], 0.4 * np.cos(g[2]), 0.4 * np.sin(g[2]), width=0.04,
                     head_width=0.15, color=COLOR_GOAL, length_includes_head=True)
    ax_vehicle.arrow(0, 0, 0.5, 0, width=0.04, head_width=0.15, color=COLOR_START, length_includes_head=True)
    label_pose(ax_vehicle, 0, 0, "Start", COLOR_START)
    label_pose(ax_vehicle, g[0], g[1], "Ziel", COLOR_GOAL)
    ax_vehicle.set_aspect("equal")
    ax_vehicle.grid(alpha=0.3)
    ax_vehicle.tick_params(labelsize=7)
    ax_vehicle.set_title("Fahrzeugframe (Netz-Sicht)", fontsize=9)


def _solver_from_meta(meta: dict):
    """Solver mit genau den Einstellungen, mit denen der Datensatz erzeugt wurde."""
    return SOLVERS[meta["solver"]["name"]](CarConfig(**meta["car"]), **meta["solver"]["params"])


def _lidar_from_meta(meta: dict) -> Lidar | None:
    for s in meta["sensors"]:
        if s["name"] == "lidar":
            return Lidar(**s["params"])
    return None


class Viewer:
    """Interaktives Durchblättern: eine Seite pro Start/Ziel-Paar."""

    def __init__(self, data: dict, lidar: Lidar | None, title: str, solver=None):
        self.data, self.lidar, self.title = data, lidar, title
        self.solver, self.show_grid = solver, solver is not None
        self.pairs = np.unique(data["pair_id"])
        first = {p: np.flatnonzero(data["pair_id"] == p)[0] for p in self.pairs}
        self.pair_scene = np.array([data["scene_id"][first[p]] for p in self.pairs])
        self.scenes = np.unique(self.pair_scene)
        self.k = 0

        # Pfeiltasten und Pos1 gehören dem Viewer, nicht der Matplotlib-Navigation
        for name, keys in (("keymap.back", ("left",)), ("keymap.forward", ("right",)),
                           ("keymap.home", ("home",))):
            plt.rcParams[name] = [k for k in plt.rcParams[name] if k not in keys]

        self.fig, (self.ax_world, self.ax_vehicle) = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.draw()

    def on_key(self, event):
        n = len(self.pairs)
        if event.key == "right":
            self.k = min(self.k + 1, n - 1)
        elif event.key == "left":
            self.k = max(self.k - 1, 0)
        elif event.key in ("down", "up"):
            j = int(np.searchsorted(self.scenes, self.pair_scene[self.k]))
            j = min(j + 1, len(self.scenes) - 1) if event.key == "down" else max(j - 1, 0)
            self.k = int(np.flatnonzero(self.pair_scene == self.scenes[j])[0])
        elif event.key == "home":
            self.k = 0
        elif event.key == "end":
            self.k = n - 1
        elif event.key == "g" and self.solver is not None:
            self.show_grid = not self.show_grid
        else:
            return
        self.draw()

    def draw(self):
        scene = self.pair_scene[self.k]
        in_scene = np.flatnonzero(self.pair_scene == scene)
        pos = int(np.flatnonzero(in_scene == self.k)[0])
        j = int(np.searchsorted(self.scenes, scene))
        self.ax_world.clear()
        self.ax_vehicle.clear()
        plot_pair(self.ax_world, self.ax_vehicle, self.data, int(self.pairs[self.k]), self.lidar,
                  self.solver if self.show_grid else None)
        self.fig.suptitle(
            f"{self.title}\nSzene {j + 1}/{len(self.scenes)}  ·  Paar {pos + 1}/{len(in_scene)}  "
            f"·  gesamt {self.k + 1}/{len(self.pairs)}      [← → Paar   ↑ ↓ Szene   G Raster]",
            fontsize=10)
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()


def export_pdf(path: str, data: dict, lidar: Lidar | None, title: str, solver=None):
    """Alle Szenen in eine PDF: eine Seite pro Szene, ihre Paare untereinander."""
    pairs = np.unique(data["pair_id"])
    pair_scene = np.array([data["scene_id"][np.flatnonzero(data["pair_id"] == p)[0]] for p in pairs])
    scenes = np.unique(pair_scene)
    with PdfPages(path) as pdf:
        for j, scene in enumerate(scenes):
            scene_pairs = pairs[pair_scene == scene]
            fig, axes = plt.subplots(len(scene_pairs), 2, figsize=(10, 3.6 * len(scene_pairs)), squeeze=False)
            for row, pair in zip(axes, scene_pairs):
                plot_pair(row[0], row[1], data, int(pair), lidar, solver)
            fig.suptitle(f"{title}  ·  Szene {j + 1}/{len(scenes)}", fontsize=11)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            print(f"  Seite {j + 1}/{len(scenes)}")
    print(f"Gespeichert: {path} ({len(scenes)} Seiten)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Datensatz anschauen.")
    parser.add_argument("path", nargs="?", default="Test_Sets/default.npz")
    parser.add_argument("--pdf", help="Alle Szenen in diese PDF exportieren (eine Seite pro Szene)")
    parser.add_argument("--save", help="Stichprobe von --n Paaren als Bild speichern")
    parser.add_argument("--no-grid", action="store_true", help="A*-Raster nicht einblenden")
    parser.add_argument("--n", type=int, default=3, help="Anzahl Paare für --save")
    parser.add_argument("--seed", type=int, default=0, help="Zufallsauswahl für --save")
    args = parser.parse_args()

    data = load_dataset(args.path)
    lidar = _lidar_from_meta(data["meta"])
    solver = None if args.no_grid else _solver_from_meta(data["meta"])
    pairs = np.unique(data["pair_id"])
    title = (f"{args.path}  ·  {len(data['weight'])} Samples, {len(pairs)} Paare, "
             f"{len(data['scenes'])} Szenen")

    if args.pdf:
        export_pdf(args.pdf, data, lidar, title, solver)
    elif args.save:
        chosen = np.random.default_rng(args.seed).choice(pairs, size=min(args.n, len(pairs)), replace=False)
        fig, axes = plt.subplots(len(chosen), 2, figsize=(10, 3.6 * len(chosen)), squeeze=False)
        for row, pair in zip(axes, chosen):
            plot_pair(row[0], row[1], data, int(pair), lidar, solver)
        fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        fig.savefig(args.save, dpi=110, bbox_inches="tight")
        print(f"Gespeichert: {args.save}")
    else:
        viewer = Viewer(data, lidar, title, solver)
        plt.show()
