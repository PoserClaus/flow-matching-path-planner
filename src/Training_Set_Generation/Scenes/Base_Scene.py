"""
Base_Scene.py — Grundbausteine für alle Szenen.

Enthält:
    Scene           Eine fertige, statische Welt (Boxen, Wände, Start, Ziel).
                    Beantwortet die Abfragen von Sensoren und Solver.
    SceneConfig     Gemeinsame Parameter aller Szenengeneratoren.
    SceneGenerator  Basisklasse, von der jeder Szenengenerator erbt.

Konventionen:
    - Einheiten SI (m, rad), alles im Weltframe.
    - Box: (x, y, yaw, length, width, height)
           x, y = Mittelpunkt, length entlang der lokalen x-Achse.
    - Hindernisse sind ausschließlich Quader. Wände = dünne Quader,
      L-Formen = zwei Quader, Engstellen = zwei Quader mit Lücke.
    - bounds = (xmin, xmax, ymin, ymax) ist der befahrbare Bereich.
      Mit walls=True stehen direkt außerhalb davon vier Wände.
    - Das Fahrzeug wird als Kreis mit Radius r betrachtet (Prüfung im Solver).
    - Die Szene ist statisch.

Abfragen (der Vertrag zu Sensoren und Solver):
    raycast_2d(origin, angles, max_range, height)  ->  Abstände je Strahl
    clearance(points)                              ->  Abstand zum nächsten Hindernis
    collides(position, radius)                     ->  berührt der Fahrzeugkreis etwas?
    in_bounds(points, margin)                      ->  im befahrbaren Bereich?

Neue Szene schreiben:
    1. Config-Dataclass anlegen, die von SceneConfig erbt
    2. Klasse von SceneGenerator ableiten, Config und name setzen
    3. _sample_once(rng) implementieren
    4. Optional is_valid() erweitern (super().is_valid(scene) mit aufrufen)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

BOX_DIM = 6  # x, y, yaw, length, width, height


# ─────────────────────────────────────────────────────────────────────────────
# Szene
# ─────────────────────────────────────────────────────────────────────────────
class Scene:
    """Statische Welt aus quaderförmigen Hindernissen."""

    def __init__(
        self,
        boxes,
        start,
        goal,
        bounds,
        walls: bool = True,
        wall_thickness: float = 0.1,
        wall_height: float = 1.0,
        name: str = "",
    ):
        self.boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, BOX_DIM)
        self.start = np.asarray(start, dtype=np.float64).reshape(3)
        self.goal = np.asarray(goal, dtype=np.float64).reshape(3)
        self.bounds = tuple(float(b) for b in bounds)
        self.walls = walls
        self.wall_thickness = wall_thickness
        self.wall_height = wall_height
        self.name = name

        xmin, xmax, ymin, ymax = self.bounds
        if not (xmin < xmax and ymin < ymax):
            raise ValueError(f"Ungültige bounds {self.bounds}: es muss xmin < xmax und ymin < ymax gelten")

        self.wall_boxes = self._build_walls() if walls else np.zeros((0, BOX_DIM))
        # Alle Hindernisse zusammen: Grundlage für sämtliche Abfragen
        self.all_boxes = np.vstack([self.boxes, self.wall_boxes])
        self._corners = self._compute_corners(self.all_boxes)

    # ── Aufbau ──────────────────────────────────────────────────────────────
    def _build_walls(self) -> np.ndarray:
        """Vier Wände direkt außerhalb der bounds. Ecken überlappen, damit es keine Lücken gibt."""
        xmin, xmax, ymin, ymax = self.bounds
        t, h = self.wall_thickness, self.wall_height
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        span_x, span_y = xmax - xmin + 2 * t, ymax - ymin + 2 * t
        return np.array([
            [xmin - t / 2, cy, 0.0, t, span_y, h],      # links
            [xmax + t / 2, cy, 0.0, t, span_y, h],      # rechts
            [cx, ymin - t / 2, 0.0, span_x, t, h],      # unten
            [cx, ymax + t / 2, 0.0, span_x, t, h],      # oben
        ])

    @staticmethod
    def _compute_corners(boxes: np.ndarray) -> np.ndarray:
        """Eckpunkte im Weltframe, Shape [N, 4, 2], gegen den Uhrzeigersinn."""
        if len(boxes) == 0:
            return np.zeros((0, 4, 2))
        x, y, yaw, hl, hw = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3] / 2, boxes[:, 4] / 2
        local = np.stack([
            np.stack([hl, hw], -1), np.stack([-hl, hw], -1),
            np.stack([-hl, -hw], -1), np.stack([hl, -hw], -1),
        ], axis=1)                                                  # [N, 4, 2]
        c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        world_x = c * local[..., 0] - s * local[..., 1] + x[:, None]
        world_y = s * local[..., 0] + c * local[..., 1] + y[:, None]
        return np.stack([world_x, world_y], -1)

    def box_corners(self, include_walls: bool = True) -> np.ndarray:
        """Eckpunkte der Boxen [N, 4, 2], z. B. zum Plotten."""
        return self._corners if include_walls else self._corners[: len(self.boxes)]

    # ── Abfrage für Sensoren ────────────────────────────────────────────────
    def raycast_2d(self, origin, angles, max_range: float, height: float = 0.0) -> np.ndarray:
        """Horizontales Raycasting in einer Ebene.

        origin:  [2]  Startpunkt der Strahlen (x, y) im Weltframe
        angles:  [R]  Strahlrichtungen im Weltframe (rad)
        height:  Höhe der Scanebene; Boxen mit height <= dieser Höhe werden nicht getroffen
        return:  [R]  Abstand zum ersten Treffer, max_range ohne Treffer

        Für jeden Strahl o + t*d und jede Kante a + u*e (e = b - a) gilt:
            t = ((a - o) x e) / (d x e),   u = ((a - o) x d) / (d x e)
        Treffer, wenn t >= 0 und 0 <= u <= 1. Abstand = kleinstes gültiges t.
        """
        origin = np.asarray(origin, dtype=np.float64)
        angles = np.asarray(angles, dtype=np.float64)
        result = np.full(angles.shape, float(max_range))

        visible = self.all_boxes[:, 5] > height
        if not np.any(visible):
            return result

        corners = self._corners[visible]                       # [N, 4, 2]
        a = corners.reshape(-1, 2)                             # Kantenanfang [M, 2]
        e = np.roll(corners, -1, axis=1).reshape(-1, 2) - a    # Kantenvektor [M, 2]
        d = np.stack([np.cos(angles), np.sin(angles)], -1)     # Strahlrichtung [R, 2]
        ao = a - origin                                        # [M, 2]

        denom = _cross(d[:, None, :], e[None, :, :])           # [R, M]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = _cross(ao[None, :, :], e[None, :, :]) / denom
            u = _cross(ao[None, :, :], d[:, None, :]) / denom
        hit = (np.abs(denom) > 1e-12) & (t >= 0) & (u >= 0) & (u <= 1)
        t = np.where(hit, t, np.inf).min(axis=1)
        return np.minimum(result, t)

    # ── Abfragen für Solver und Generator ───────────────────────────
    def clearance(self, points) -> np.ndarray:
        """Vorzeichenbehafteter Abstand zum nächsten Hindernis (inkl. Wände).

        points:  [2] oder [K, 2]
        return:  Skalar bzw. [K]. Positiv = frei, negativ = im Hindernis.
                 inf, wenn es keine Hindernisse gibt.

        Pro Box: Punkt ins Box-Frame drehen, dann
            q = |p'| - (length/2, width/2)
            d = || max(q, 0) || + min(max(q_x, q_y), 0)
        """
        pts = np.asarray(points, dtype=np.float64)
        single = pts.ndim == 1
        pts = pts.reshape(-1, 2)

        if len(self.all_boxes) == 0:
            out = np.full(len(pts), np.inf)
            return float(out[0]) if single else out

        b = self.all_boxes
        dx = pts[:, None, 0] - b[None, :, 0]                   # [K, N]
        dy = pts[:, None, 1] - b[None, :, 1]
        c, s = np.cos(b[:, 2]), np.sin(b[:, 2])
        px = c * dx + s * dy                                   # Punkt im Box-Frame
        py = -s * dx + c * dy
        qx = np.abs(px) - b[:, 3] / 2
        qy = np.abs(py) - b[:, 4] / 2
        outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
        inside = np.minimum(np.maximum(qx, qy), 0)
        out = (outside + inside).min(axis=1)                   # [K]
        return float(out[0]) if single else out

    def in_bounds(self, points, margin: float = 0.0) -> np.ndarray | bool:
        """True, wenn der Punkt mindestens margin Abstand zum Rand der bounds hat."""
        pts = np.asarray(points, dtype=np.float64)
        single = pts.ndim == 1
        pts = pts.reshape(-1, 2)
        xmin, xmax, ymin, ymax = self.bounds
        ok = ((pts[:, 0] >= xmin + margin) & (pts[:, 0] <= xmax - margin)
              & (pts[:, 1] >= ymin + margin) & (pts[:, 1] <= ymax - margin))
        return bool(ok[0]) if single else ok

    def collides(self, position, radius: float) -> np.ndarray | bool:
        """True, wenn der Fahrzeugkreis ein Hindernis berührt oder die bounds verlässt.
        position: [2] oder [K, 2] (eine Pose (x, y, yaw) wird ebenfalls akzeptiert)."""
        pts = np.asarray(position, dtype=np.float64)
        pts = pts[..., :2]
        hit = (np.asarray(self.clearance(pts)) < radius) | ~np.asarray(self.in_bounds(pts, margin=radius))
        return bool(hit) if hit.ndim == 0 else hit

    # ── Metadaten ───────────────────────────────────────────────────────────
    def describe(self) -> dict:
        return {
            "name": self.name,
            "n_boxes": int(len(self.boxes)),
            "bounds": list(self.bounds),
            "walls": self.walls,
            "wall_thickness": self.wall_thickness,
            "wall_height": self.wall_height,
        }

    def to_dict(self) -> dict:
        """Vollständige, JSON-taugliche Beschreibung (zum Speichern im Datensatz)."""
        return {
            "name": self.name,
            "boxes": self.boxes.tolist(),
            "start": self.start.tolist(),
            "goal": self.goal.tolist(),
            "bounds": list(self.bounds),
            "walls": self.walls,
            "wall_thickness": self.wall_thickness,
            "wall_height": self.wall_height,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Scene":
        return cls(d["boxes"], d["start"], d["goal"], d["bounds"], walls=d["walls"],
                   wall_thickness=d["wall_thickness"], wall_height=d["wall_height"], name=d["name"])


def _cross(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """2D-Kreuzprodukt (z-Komponente), broadcastfähig."""
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SceneConfig:
    """Parameter, die JEDER Szenengenerator hat. Eigene Configs erben hiervon."""

    bounds: tuple = (0.0, 8.0, -3.0, 3.0)   # befahrbarer Bereich: xmin, xmax, ymin, ymax (m)
    walls: bool = True                      # Wände um den Arbeitsbereich (Labor)
    wall_thickness: float = 0.1             # m
    wall_height: float = 1.0                # m
    min_clearance: float = 0.4              # Mindestabstand von Start/Ziel zu Hindernissen (m)
    max_tries: int = 100                    # Versuche, bis eine gültige Szene gefunden ist


class SceneGenerator(ABC):
    """Basisklasse für alle Szenengeneratoren.

    Unterklassen setzen:
        Config = MeineConfig      (Dataclass, erbt von SceneConfig)
        name   = "meine_szene"
    und implementieren:
        _sample_once(rng) -> Scene | None
    Zum Erzeugen der Scene am besten self.make_scene(...) benutzen,
    dann werden bounds und Wände aus der Config automatisch übernommen.
    """

    Config = SceneConfig
    name: str = "base"

    def __init__(self, cfg: Optional[SceneConfig] = None, **overrides):
        self.cfg = cfg if cfg is not None else self.Config(**overrides)

    def sample(self, rng: np.random.Generator) -> Scene | None:
        """Würfelt so lange, bis eine gültige Szene entsteht. None nach max_tries Fehlversuchen."""
        for _ in range(self.cfg.max_tries):
            scene = self._sample_once(rng)
            if scene is not None and self.is_valid(scene):
                scene.name = self.name
                return scene
        return None

    @abstractmethod
    def _sample_once(self, rng: np.random.Generator) -> Scene | None:
        """Einen Versuch würfeln. NUR rng benutzen, nie np.random direkt!"""

    def make_scene(self, boxes, start, goal) -> Scene:
        """Scene mit bounds und Wand-Einstellungen aus der Config erzeugen."""
        return Scene(
            boxes, start, goal,
            bounds=self.cfg.bounds,
            walls=self.cfg.walls,
            wall_thickness=self.cfg.wall_thickness,
            wall_height=self.cfg.wall_height,
            name=self.name,
        )

    def is_valid(self, scene: Scene) -> bool:
        """Grundprüfung: Start und Ziel liegen frei und mit Abstand im befahrbaren Bereich.
        Ob die Szene lösbar ist, prüft später der Solver."""
        m = self.cfg.min_clearance
        for pose in (scene.start, scene.goal):
            if not scene.in_bounds(pose[:2], margin=m):
                return False
            if scene.clearance(pose[:2]) < m:
                return False
        return True

    def describe(self) -> dict:
        return {"class": type(self).__name__, "name": self.name, "params": asdict(self.cfg)}