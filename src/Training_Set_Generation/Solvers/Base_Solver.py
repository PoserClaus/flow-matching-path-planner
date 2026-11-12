"""
Base_Solver.py — Basisklasse für alle Solver.

Ein Solver bekommt eine Szene sowie Start- und Zielpose und liefert MEHRERE
kollisionsfreie, ausreichend verschiedene Wege, gewichtet nach ihrer Länge.
Ein Weg ist eine Polylinie für den Fahrzeugmittelpunkt (Fahrzeug = Kreis).
Fahrbarkeit (Kinematik) wird noch NICHT geprüft.

Arbeitsteilung:
    Unterklasse  _candidates()   schlägt Wege vor (Polylinien), kürzeste zuerst
    Basisklasse  solve()         filtert und gewichtet:
        1. Weg länger als max_length_ratio * kürzester Weg  -> Suche beenden
        2. Weg zu ähnlich zu einem akzeptierten Weg          -> verwerfen
           (symmetrischer Hausdorff-Abstand < min_path_distance)
        3. Gewicht w_i = exp(-(L_i - L_min) / length_tau), normiert auf Summe 1

Neuen Solver schreiben:
    1. Config-Dataclass anlegen, die von SolverConfig erbt
    2. Klasse von BaseSolver ableiten, Config und name setzen
    3. _candidates(scene, start, goal) als Generator implementieren
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Iterator, Optional

import numpy as np

from Car_Model import CarConfig


@dataclass
class SolverConfig:
    """Parameter, die JEDER Solver hat. Eigene Configs erben hiervon."""

    max_paths: int = 5                 # höchstens so viele Wege pro Start/Ziel
    min_path_distance: float = 0.5     # m, Mindest-Hausdorff-Abstand zwischen zwei Wegen
    max_length_ratio: float = 1.5      # Wege länger als ratio * kürzester werden verworfen
    length_tau: float = 1.0            # m, Temperatur der Längengewichtung
    safety_margin: float = 0.05        # m, zusätzlicher Abstand zu Hindernissen über den Radius hinaus


@dataclass
class Solution:
    """Ein gefundener Weg."""

    waypoints: np.ndarray              # [K, 2] Polylinie im Weltframe, erster Punkt = Start, letzter = Ziel
    length: float                      # m
    weight: float                      # Anteil an allen Lösungen dieses Start/Ziel-Paars, Summe = 1


class BaseSolver(ABC):
    """Basisklasse für alle Solver."""

    Config = SolverConfig
    name: str = "base"

    def __init__(self, car: Optional[CarConfig] = None, cfg: Optional[SolverConfig] = None, **overrides):
        self.car = car if car is not None else CarConfig()
        self.cfg = cfg if cfg is not None else self.Config(**overrides)

    @property
    def inflation(self) -> float:
        """Mindestabstand des Fahrzeugmittelpunkts zu jedem Hindernis."""
        return self.car.radius + self.cfg.safety_margin

    # ── Von Unterklassen zu implementieren ──────────────────────────────────
    @abstractmethod
    def _candidates(self, scene, start: np.ndarray, goal: np.ndarray) -> Iterator[np.ndarray]:
        """Schlägt nacheinander kollisionsfreie Wege vor ([K, 2], erster Punkt = Start,
        letzter = Ziel), ungefähr nach Länge aufsteigend."""

    # ── Ablauf ──────────────────────────────────────────────────────────────
    def solve(self, scene, start, goal) -> list[Solution]:
        """Alle ausreichend verschiedenen Wege von start nach goal. Leere Liste = unlösbar."""
        cfg = self.cfg
        start = np.asarray(start, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)

        accepted: list[tuple[np.ndarray, float]] = []
        for path in self._candidates(scene, start, goal):
            length = polyline_length(path)
            if accepted and length > cfg.max_length_ratio * min(l for _, l in accepted):
                break
            if any(hausdorff(path, p) < cfg.min_path_distance for p, _ in accepted):
                continue
            accepted.append((path, length))
            if len(accepted) >= cfg.max_paths:
                break

        if not accepted:
            return []
        lengths = np.array([l for _, l in accepted])
        w = np.exp(-(lengths - lengths.min()) / cfg.length_tau)
        w /= w.sum()
        return [Solution(p, float(l), float(wi)) for (p, l), wi in zip(accepted, w)]

    def describe(self) -> dict:
        return {"class": type(self).__name__, "name": self.name, "params": asdict(self.cfg)}


# ─────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen für Polylinien
# ─────────────────────────────────────────────────────────────────────────────
def polyline_length(path: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def resample_polyline(path: np.ndarray, spacing: Optional[float] = None, n: Optional[int] = None) -> np.ndarray:
    """Polylinie gleichmäßig nach Bogenlänge abtasten, Endpunkte bleiben erhalten.
    Entweder mit festem Punktabstand (spacing) oder fester Punktanzahl (n)."""
    path = np.asarray(path, dtype=np.float64)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if n is None:
        if s[-1] < 1e-9:
            return path[:1].copy()
        n = max(int(np.ceil(s[-1] / spacing)) + 1, 2)
    s_new = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(s_new, s, path[:, 0]), np.interp(s_new, s, path[:, 1])], -1)


def hausdorff(a: np.ndarray, b: np.ndarray, spacing: float = 0.05) -> float:
    """Symmetrischer Hausdorff-Abstand zweier Polylinien."""
    pa, pb = resample_polyline(a, spacing), resample_polyline(b, spacing)
    d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=-1)
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))
