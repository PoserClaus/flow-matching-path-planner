"""
Car_Model.py — Fahrzeugparameter, Fahrzeugzustand und Koordinatentransformationen.

Bewusst einfach gehalten: Das Fahrzeug ist ein Kreis mit Radius cfg.radius.
Kinematik und Fahrbarkeitsprüfung (Unicycle) kommen später dazu.

Konventionen:
    - Einheiten SI (m, rad, s).
    - Weltframe: x, y, yaw wie in der Szene.
    - Fahrzeugframe: Ursprung im Fahrzeugmittelpunkt, x = Fahrtrichtung, y = links.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CarConfig:
    """Parameter des Roboters."""

    radius: float = 0.3          # m, Fahrzeug als Kreis


@dataclass
class VehicleState:
    """Zustand des Fahrzeugs zu einem Zeitpunkt (das, was Sensoren sehen dürfen)."""

    pose: np.ndarray             # [3] x, y, yaw      (Weltframe)
    velocity: np.ndarray         # [2] v, omega       (m/s, rad/s)
    acceleration: np.ndarray     # [2] v_dot, omega_dot (m/s^2, rad/s^2)
    t: float                     # Zeit in s

    @classmethod
    def standing(cls, pose) -> "VehicleState":
        """Stehendes Fahrzeug an der Pose (x, y, yaw)."""
        return cls(np.asarray(pose, dtype=np.float64), np.zeros(2), np.zeros(2), 0.0)


def world_to_vehicle(points, pose) -> np.ndarray:
    """Punkte [..., 2] vom Weltframe in den Fahrzeugframe der Pose (x, y, yaw)."""
    x, y, yaw = pose
    d = np.asarray(points, dtype=np.float64) - np.array([x, y])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)


def vehicle_to_world(points, pose) -> np.ndarray:
    """Punkte [..., 2] vom Fahrzeugframe der Pose (x, y, yaw) in den Weltframe."""
    x, y, yaw = pose
    p = np.asarray(points, dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * p[..., 0] - s * p[..., 1] + x, s * p[..., 0] + c * p[..., 1] + y], -1)


def wrap_angle(angle):
    """Winkel auf [-pi, pi). Skalar rein -> float raus, Array rein -> Array raus."""
    wrapped = (np.asarray(angle, dtype=np.float64) + np.pi) % (2 * np.pi) - np.pi
    return float(wrapped) if wrapped.ndim == 0 else wrapped
