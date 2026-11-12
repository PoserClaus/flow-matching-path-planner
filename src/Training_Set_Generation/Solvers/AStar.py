"""
AStar.py — A* auf einem Belegungsraster, mehrere verschiedene Wege per Strafkosten.

Ablauf:
    1. Raster über die bounds legen. Eine Zelle ist frei, wenn ihr Mittelpunkt
       mindestens radius + safety_margin von jedem Hindernis entfernt ist
       (= Hindernisse um das Fahrzeug aufgeblasen, Fahrzeug wird zum Punkt).
    2. A* mit 8er-Nachbarschaft. Diagonalen nur, wenn beide Nachbarzellen frei
       sind (keine Ecken schneiden). Kosten = Weglänge * (1 + Strafe der Zelle).
    3. Glätten: Rasterweg per Sichtlinie abkürzen (String Pulling). Die
       Sichtlinie wird exakt gegen die Szene geprüft, nicht gegen das Raster.
    4. Für den nächsten Weg die Zellen in der Nähe des gefundenen Wegs
       verteuern und erneut suchen. So weicht A* auf andere Wege aus.
       Ob ein neuer Weg verschieden genug ist, entscheidet die Basisklasse.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .Base_Solver import BaseSolver, SolverConfig, resample_polyline

_SQRT2 = float(np.sqrt(2.0))
_NEIGHBORS = [(-1, -1, _SQRT2), (-1, 0, 1.0), (-1, 1, _SQRT2), (0, -1, 1.0),
              (0, 1, 1.0), (1, -1, _SQRT2), (1, 0, 1.0), (1, 1, _SQRT2)]


@dataclass
class AStarConfig(SolverConfig):
    resolution: float = 0.05           # m, Kantenlänge einer Rasterzelle
    penalty_radius: float = 0.4        # m, Zellen so nah am letzten Weg werden verteuert
    penalty_weight: float = 2.0        # Zusatzkosten pro Treffer (1 = doppelt so teuer)
    max_iterations: int = 15           # höchstens so viele A*-Läufe pro Start/Ziel


class AStar(BaseSolver):
    """A* mit iterativen Strafkosten für alternative Wege."""

    Config = AStarConfig
    name = "astar"

    def occupancy_grid(self, scene) -> "_Grid":
        """Das Belegungsraster, auf dem A* in dieser Szene sucht (z. B. zum Plotten)."""
        return _Grid(scene, self.cfg.resolution, self.inflation)

    # ── Kandidaten ──────────────────────────────────────────────────────────
    def _candidates(self, scene, start, goal) -> Iterator[np.ndarray]:
        grid = _Grid(scene, self.cfg.resolution, self.inflation)
        s = grid.snap(start[:2], self._line_of_sight, scene)
        g = grid.snap(goal[:2], self._line_of_sight, scene)
        if s is None or g is None:
            return

        penalty = np.zeros(grid.n_cells)
        for _ in range(self.cfg.max_iterations):
            cells = self._astar(grid, penalty, s, g)
            if cells is None:
                return
            raw = np.vstack([start[:2], grid.centers[cells], goal[:2]])
            path = self._shortcut(scene, raw)
            yield path

            # Zellen nahe am gefundenen Weg verteuern
            pts = resample_polyline(path, self.cfg.resolution)
            near = np.zeros(grid.n_cells, dtype=bool)
            for chunk in np.array_split(pts, max(1, len(pts) // 64)):
                d = np.linalg.norm(grid.centers[:, None, :] - chunk[None, :, :], axis=-1)
                near |= d.min(axis=1) < self.cfg.penalty_radius
            penalty[near] += self.cfg.penalty_weight

    # ── A* ──────────────────────────────────────────────────────────────────
    def _astar(self, grid: "_Grid", penalty: np.ndarray, s: int, g: int):
        """Kürzester Weg (mit Strafkosten) als Liste von Zellindizes, None wenn keiner existiert."""
        W, H, res = grid.W, grid.H, grid.res
        free = grid.free.tolist()
        cost = (1.0 + penalty).tolist()
        gy, gx = divmod(g, W)

        g_score = [np.inf] * grid.n_cells
        came = [-1] * grid.n_cells
        closed = [False] * grid.n_cells
        g_score[s] = 0.0
        sy, sx = divmod(s, W)
        heap = [(res * ((sx - gx) ** 2 + (sy - gy) ** 2) ** 0.5, 0.0, s)]

        while heap:
            _, gs, u = heapq.heappop(heap)
            if closed[u]:
                continue
            if u == g:
                path = [u]
                while came[path[-1]] != -1:
                    path.append(came[path[-1]])
                return path[::-1]
            closed[u] = True
            uy, ux = divmod(u, W)
            for dy, dx, step in _NEIGHBORS:
                vy, vx = uy + dy, ux + dx
                if not (0 <= vy < H and 0 <= vx < W):
                    continue
                v = vy * W + vx
                if not free[v] or closed[v]:
                    continue
                if dy and dx and not (free[uy * W + vx] and free[vy * W + ux]):
                    continue
                ng = gs + step * res * cost[v]
                if ng < g_score[v]:
                    g_score[v] = ng
                    came[v] = u
                    h = res * ((vx - gx) ** 2 + (vy - gy) ** 2) ** 0.5   # zulässig, da cost >= 1
                    heapq.heappush(heap, (ng + h, ng, v))
        return None

    # ── Glätten ─────────────────────────────────────────────────────────────
    def _line_of_sight(self, scene, a: np.ndarray, b: np.ndarray) -> bool:
        """True, wenn der Fahrzeugmittelpunkt geradlinig von a nach b fahren kann."""
        n = max(int(np.ceil(np.hypot(*(b - a)) / (self.cfg.resolution / 2))), 1) + 1
        pts = a + np.linspace(0.0, 1.0, n)[:, None] * (b - a)
        return bool(np.all(scene.clearance(pts) >= self.inflation - 1e-9)
                    and np.all(scene.in_bounds(pts, margin=self.inflation - 1e-9)))

    def _shortcut(self, scene, raw: np.ndarray) -> np.ndarray:
        """Greedy String Pulling: von jedem Punkt so weit wie möglich geradeaus."""
        out = [raw[0]]
        i, n = 0, len(raw)
        while i < n - 1:
            j = i + 1
            while j + 1 < n and self._line_of_sight(scene, raw[i], raw[j + 1]):
                j += 1
            out.append(raw[j])
            i = j
        return np.array(out)


class _Grid:
    """Belegungsraster über den bounds der Szene. Zellindex = iy * W + ix."""

    def __init__(self, scene, res: float, inflation: float):
        xmin, xmax, ymin, ymax = scene.bounds
        xs = np.arange(xmin + res / 2, xmax, res)
        ys = np.arange(ymin + res / 2, ymax, res)
        self.W, self.H, self.res = len(xs), len(ys), res
        self.origin = np.array([xs[0], ys[0]])
        gx, gy = np.meshgrid(xs, ys)
        self.centers = np.stack([gx.ravel(), gy.ravel()], -1)
        self.free = (scene.clearance(self.centers) >= inflation) & scene.in_bounds(self.centers, margin=inflation)
        self.n_cells = len(self.centers)

    def edges(self):
        """Zellgrenzen in x und y ([W+1], [H+1]), z. B. für pcolormesh."""
        xe = self.origin[0] - self.res / 2 + np.arange(self.W + 1) * self.res
        ye = self.origin[1] - self.res / 2 + np.arange(self.H + 1) * self.res
        return xe, ye

    def snap(self, p: np.ndarray, line_of_sight, scene, search: int = 3):
        """Nächstgelegene freie Zelle, die von p aus geradlinig erreichbar ist."""
        ix, iy = np.round((p - self.origin) / self.res).astype(int)
        best, best_d = None, np.inf
        for dy in range(-search, search + 1):
            for dx in range(-search, search + 1):
                cx, cy = ix + dx, iy + dy
                if not (0 <= cx < self.W and 0 <= cy < self.H):
                    continue
                k = cy * self.W + cx
                d = np.hypot(*(self.centers[k] - p))
                if self.free[k] and d < best_d and line_of_sight(scene, p, self.centers[k]):
                    best, best_d = k, d
        return best
