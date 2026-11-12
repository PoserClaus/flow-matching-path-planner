"""
Plot_Scene.py — Szenen visualisieren.

Als Skript: zeigt ein Raster zufälliger Szenen eines Generators.
    python Plot_Scene.py                                   # Standard-Generator, 3x3 Szenen
    python Plot_Scene.py random_boxes_random_start --n 6 --seed 3
    python Plot_Scene.py --lidar                           # LiDAR-Scan vom Start einzeichnen
    python Plot_Scene.py --clearance                       # Freiraum als Farbverlauf
    python Plot_Scene.py --paths                           # A*-Wege einzeichnen (Dicke = Gewicht)
    python Plot_Scene.py --save scenes.png                 # als Bild speichern statt anzeigen
    python Plot_Scene.py --set n_boxes=4,8 low_box_prob=0.3   # Parameter überschreiben

Als Modul: plot_scene(ax, scene, ...) zeichnet eine Szene in eine Achse.
    Wird auch von Plot_Testset.py wiederverwendet.
"""

from __future__ import annotations

import argparse
import ast
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Patch, Polygon

# Als Skript gestartet: übergeordneten Ordner (Training_Set_Generation) importierbar machen
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Car_Model import VehicleState  # noqa: E402
from Scenes import SCENES, Scene  # noqa: E402
from Sensors import Lidar  # noqa: E402
from Solvers import SOLVERS  # noqa: E402

# Farben (LAUNCH-unabhängig, gut unterscheidbar, auch in Graustufen)
COLOR_BOX = "#4a90d9"
COLOR_LOW_BOX = "#f5a623"
COLOR_WALL = "#555555"
COLOR_START = "#2e7d32"
COLOR_GOAL = "#c62828"
COLOR_LIDAR = "#d81b60"
COLOR_BLOCKED = "#e57373"
PATH_COLORS = ["#6a1b9a", "#00838f", "#ef6c00", "#2e7d32", "#ad1457", "#37474f"]


def plot_scene(
    ax,
    scene: Scene,
    robot_radius: float | None = 0.3,
    lidar: Lidar | None = None,
    lidar_pose=None,
    show_clearance: bool = False,
    low_height: float = 0.2,
    title: str | None = None,
    solutions=None,
    grid=None,
):
    """Zeichnet eine Szene in die Achse ax.

    robot_radius:   Fahrzeugkreis an Start und Ziel (None = nicht zeichnen)
    lidar:          optionaler LiDAR, dessen Scan eingezeichnet wird
    lidar_pose:     Pose für den Scan (Default: Startpose)
    show_clearance: Freiraum (Abstand zum nächsten Hindernis) als Farbverlauf
    low_height:     Boxen unter dieser Höhe werden als "niedrig" markiert
    solutions:      Liste von Solver-Lösungen, als Linien (Dicke = Gewicht)
    grid:           Belegungsraster des Solvers (solver.occupancy_grid(scene)):
                    Zelllinien + gesperrte Zellen (Hindernis + Radius + Sicherheitsabstand)
    """
    xmin, xmax, ymin, ymax = scene.bounds

    if show_clearance:
        gx, gy = np.meshgrid(np.linspace(xmin, xmax, 160), np.linspace(ymin, ymax, 120))
        c = scene.clearance(np.stack([gx.ravel(), gy.ravel()], -1)).reshape(gx.shape)
        im = ax.imshow(np.clip(c, 0, None), extent=(xmin, xmax, ymin, ymax), origin="lower",
                       cmap="Greens", alpha=0.5, zorder=0)
        plt.colorbar(im, ax=ax, fraction=0.03, label="Freiraum [m]")

    # Raster des Solvers
    if grid is not None:
        xe, ye = grid.edges()
        blocked = (~grid.free).reshape(grid.H, grid.W)
        ax.pcolormesh(xe, ye, np.ma.masked_where(~blocked, blocked.astype(float)),
                      cmap=ListedColormap([COLOR_BLOCKED]), alpha=0.35, zorder=0.5, shading="flat")
        line = dict(color="#9e9e9e", lw=0.25, alpha=0.5, zorder=0.4)
        ax.vlines(xe, ye[0], ye[-1], **line)
        ax.hlines(ye, xe[0], xe[-1], **line)

    # Boxen und Wände
    corners = scene.box_corners(include_walls=True)
    for i, poly in enumerate(corners):
        is_wall = i >= len(scene.boxes)
        if is_wall:
            color = COLOR_WALL
        else:
            color = COLOR_LOW_BOX if scene.all_boxes[i, 5] < low_height else COLOR_BOX
        ax.add_patch(Polygon(poly, closed=True, facecolor=color, edgecolor="black",
                             linewidth=0.5, alpha=0.9, zorder=2))

    # LiDAR-Scan
    if lidar is not None:
        pose = scene.start if lidar_pose is None else np.asarray(lidar_pose)
        state = VehicleState.standing(pose)
        ranges = lidar.observe(scene, state, np.random.default_rng(0)).astype(float)
        pts = lidar.beam_endpoints(ranges, pose)
        origin = lidar.mount_pose(pose)[:2]
        for p in pts:
            ax.plot([origin[0], p[0]], [origin[1], p[1]], color=COLOR_LIDAR, lw=0.3, alpha=0.35, zorder=1)
        hits = ranges < lidar.cfg.max_range - 1e-6
        ax.scatter(pts[hits, 0], pts[hits, 1], s=3, color=COLOR_LIDAR, zorder=3)

    # Wege des Solvers
    for k, sol in enumerate(solutions or []):
        xy = sol.waypoints
        ax.plot(xy[:, 0], xy[:, 1], color=PATH_COLORS[k % len(PATH_COLORS)],
                lw=1.0 + 4.0 * sol.weight, alpha=0.85, zorder=3,
                label=f"{sol.length:.1f} m · w={sol.weight:.2f}")
    handles, _ = ax.get_legend_handles_labels()
    if grid is not None:
        handles.append(Patch(facecolor=COLOR_BLOCKED, alpha=0.35,
                             label=f"A*-Sperrzone (Raster {grid.res * 100:.0f} cm)"))
    if handles:
        ax.legend(handles=handles, fontsize=6, loc="upper right")

    # Start und Ziel als Pfeil mit Fahrzeugkreis
    for pose, color, label in ((scene.start, COLOR_START, "Start"), (scene.goal, COLOR_GOAL, "Ziel")):
        _draw_pose(ax, pose, color, robot_radius, label)

    ax.set_xlim(xmin - 0.3, xmax + 0.3)
    ax.set_ylim(ymin - 0.3, ymax + 0.3)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    if title:
        ax.set_title(title, fontsize=9)


def _draw_pose(ax, pose, color, radius, label):
    x, y, yaw = pose
    if radius:
        ax.add_patch(Circle((x, y), radius, facecolor=color, alpha=0.25, edgecolor=color, zorder=4))
    length = radius * 1.6 if radius else 0.4
    ax.arrow(x, y, length * np.cos(yaw), length * np.sin(yaw), width=0.04,
             head_width=0.15, color=color, length_includes_head=True, zorder=5, label=label)
    label_pose(ax, x, y, label, color)


def label_pose(ax, x, y, text, color):
    """Beschriftung (z. B. "Start", "Ziel") direkt unter einer Pose."""
    ax.annotate(text, (x, y), xytext=(0, -16), textcoords="offset points", ha="center", va="top",
                fontsize=8, fontweight="bold", color=color, zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8))


def preview_generator(generator, n: int = 9, seed: int = 0, solver=None, **plot_kwargs):
    """Würfelt n Szenen und zeigt sie in einem Raster. Gibt die Figure zurück.
    Mit solver werden für jede Szene die Wege Start -> Ziel eingezeichnet."""
    rng = np.random.default_rng(seed)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)

    for k, ax in enumerate(axes.ravel()):
        if k >= n:
            ax.axis("off")
            continue
        scene = generator.sample(rng)
        if scene is None:
            ax.text(0.5, 0.5, "keine gültige Szene\n(max_tries erreicht)",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue
        solutions = solver.solve(scene, scene.start, scene.goal) if solver else None
        title = f"#{k}  ·  {len(scene.boxes)} Boxen"
        if solver:
            title += f"  ·  {len(solutions)} Wege"
        plot_scene(ax, scene, title=title, solutions=solutions, **plot_kwargs)

    fig.suptitle(f"{generator.name}  (seed={seed})", fontsize=12)
    fig.tight_layout()
    return fig


def _parse_overrides(items: list[str]) -> dict:
    """'n_boxes=4,8 low_box_prob=0.3'  ->  {'n_boxes': (4, 8), 'low_box_prob': 0.3}"""
    out = {}
    for item in items:
        key, value = item.split("=", 1)
        out[key] = ast.literal_eval(value)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zufällige Szenen eines Generators anzeigen.")
    parser.add_argument("scene", nargs="?", default="random_boxes_random_start",
                        choices=sorted(SCENES), help="Name des Generators")
    parser.add_argument("--n", type=int, default=9, help="Anzahl Szenen")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lidar", action="store_true", help="LiDAR-Scan vom Start einzeichnen")
    parser.add_argument("--clearance", action="store_true", help="Freiraum als Farbverlauf")
    parser.add_argument("--paths", nargs="?", const="astar", choices=sorted(SOLVERS),
                        help="Wege mit diesem Solver einzeichnen (Default: astar)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Generator-Parameter überschreiben")
    parser.add_argument("--save", help="Bild speichern statt anzeigen")
    args = parser.parse_args()

    generator = SCENES[args.scene](**_parse_overrides(args.set))
    fig = preview_generator(
        generator, n=args.n, seed=args.seed,
        lidar=Lidar(n_beams=180) if args.lidar else None,
        show_clearance=args.clearance,
        solver=SOLVERS[args.paths]() if args.paths else None,
    )
    if args.save:
        fig.savefig(args.save, dpi=110, bbox_inches="tight")
        print(f"Gespeichert: {args.save}")
    else:
        plt.show()