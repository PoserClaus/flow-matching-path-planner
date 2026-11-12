"""
Random_Boxes_Random_Start_Random_Target.py — Zufällige Boxen, zufälliger Start, zufälliges Ziel.

Ablauf eines Versuchs (_sample_once):
    1. Start würfeln: Position im befahrbaren Bereich, beliebige Ausrichtung
    2. Ziel würfeln:  mit Mindestabstand zum Start
    3. Boxen würfeln: Anzahl, Größe, Drehung und Höhe zufällig,
                      ohne Überlappung untereinander

Zusätzliche Prüfung (is_valid):
    - Grundprüfung der Basisklasse (Start/Ziel frei und im Bereich)
    - Optional: Mindestens eine Box blockiert die direkte Linie Start -> Ziel,
      damit das Netz nicht hauptsächlich Geradeausfahren lernt.

Ob die Szene wirklich lösbar ist, prüft später der Solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .Base_Scene import Scene, SceneConfig, SceneGenerator


@dataclass
class RandomBoxesRandomStartConfig(SceneConfig):
    """Parameter der Szene. Alle Längen in m, Winkel in rad."""

    # Boxen
    n_boxes: tuple = (2, 6)                 # min/max Anzahl (beide inklusive)
    box_length: tuple = (0.3, 1.0)          # min/max Länge
    box_width: tuple = (0.3, 1.0)           # min/max Breite
    box_height: tuple = (0.3, 0.8)          # min/max Höhe
    low_box_prob: float = 0.0               # Anteil Boxen unterhalb der LiDAR-Ebene
    low_box_height: tuple = (0.05, 0.15)    # Höhe dieser niedrigen Boxen
    min_box_gap: float = 0.6                # Mindestlücke zwischen Boxen (≈ Fahrzeugbreite)
    box_tries: int = 50                     # Versuche pro Box, einen freien Platz zu finden

    # Start und Ziel
    min_start_goal_dist: float = 4.0        # Mindestabstand Start -> Ziel
    goal_yaw_spread: float = np.pi / 4      # Zielausrichtung = Richtung Start->Ziel ± diesem Wert

    # Interessante Szenen erzwingen
    require_blocking: bool = True           # direkte Linie Start->Ziel muss blockiert sein
    corridor_half_width: float = 0.3        # halbe Breite des Korridors (≈ Fahrzeugradius)


class RandomBoxesRandomStart(SceneGenerator):
    """Zufällige Boxen, zufällige Start- und Zielpose."""

    Config = RandomBoxesRandomStartConfig
    name = "random_boxes_random_start"

    # ── Ein Versuch ─────────────────────────────────────────────────────────
    def _sample_once(self, rng: np.random.Generator) -> Scene | None:
        start_xy = self._random_point(rng)
        goal_xy = self._random_point(rng)
        delta = goal_xy - start_xy
        if np.linalg.norm(delta) < self.cfg.min_start_goal_dist:
            return None                                       # neuer Versuch

        heading = np.arctan2(delta[1], delta[0])
        start = np.array([*start_xy, rng.uniform(-np.pi, np.pi)])
        goal_yaw = heading + rng.uniform(-self.cfg.goal_yaw_spread, self.cfg.goal_yaw_spread)
        goal = np.array([*goal_xy, _wrap(goal_yaw)])

        boxes = self._sample_boxes(rng, keep_free=[start_xy, goal_xy])
        return self.make_scene(boxes, start, goal)

    def _random_point(self, rng) -> np.ndarray:
        xmin, xmax, ymin, ymax = self.cfg.bounds
        m = self.cfg.min_clearance
        return rng.uniform([xmin + m, ymin + m], [xmax - m, ymax - m])

    def _sample_boxes(self, rng, keep_free) -> np.ndarray:
        """Boxen nacheinander platzieren. Eine Box, die keinen Platz findet, entfällt."""
        cfg = self.cfg
        n_target = rng.integers(cfg.n_boxes[0], cfg.n_boxes[1], endpoint=True)
        xmin, xmax, ymin, ymax = cfg.bounds
        boxes = []

        for _ in range(n_target):
            for _ in range(cfg.box_tries):
                length = rng.uniform(*cfg.box_length)
                width = rng.uniform(*cfg.box_width)
                if rng.random() < cfg.low_box_prob:
                    height = rng.uniform(*cfg.low_box_height)
                else:
                    height = rng.uniform(*cfg.box_height)
                radius = 0.5 * np.hypot(length, width)        # Umkreis der Box
                x = rng.uniform(xmin + radius, xmax - radius)
                y = rng.uniform(ymin + radius, ymax - radius)
                candidate = np.array([x, y, rng.uniform(-np.pi, np.pi), length, width, height])

                if self._fits(candidate, radius, boxes, keep_free):
                    boxes.append(candidate)
                    break

        return np.array(boxes).reshape(-1, 6)

    def _fits(self, box, radius, boxes, keep_free) -> bool:
        """Konservativ über Umkreise: Abstand zu anderen Boxen und zu Start/Ziel."""
        for p in keep_free:
            if np.hypot(box[0] - p[0], box[1] - p[1]) < radius + self.cfg.min_clearance:
                return False
        for other in boxes:
            r_other = 0.5 * np.hypot(other[3], other[4])
            if np.hypot(box[0] - other[0], box[1] - other[1]) < radius + r_other + self.cfg.min_box_gap:
                return False
        return True

    # ── Prüfung ─────────────────────────────────────────────────────────────
    def is_valid(self, scene: Scene) -> bool:
        if not super().is_valid(scene):
            return False
        if self.cfg.require_blocking and not self.line_blocked(scene):
            return False
        return True

    def line_blocked(self, scene: Scene) -> bool:
        """True, wenn eine Box (nicht Wand) den Korridor Start -> Ziel schneidet."""
        if len(scene.boxes) == 0:
            return False
        t = np.linspace(0.0, 1.0, 100)[:, None]
        line = scene.start[:2] + t * (scene.goal[:2] - scene.start[:2])          # [100, 2]
        only_boxes = Scene(scene.boxes, scene.start, scene.goal, scene.bounds, walls=False)
        return bool(np.any(only_boxes.clearance(line) < self.cfg.corridor_half_width))


def _wrap(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi