"""
Base_Sensor.py — Basisklasse für alle Sensoren.

Ein Sensor bekommt pro Zeitschritt die Szene und den Fahrzeugzustand und
liefert eine Messung als numpy-Array mit fester Shape.

Messkette in observe() (entspricht der Physik eines echten Sensors):

    _measure()  ->  + Rauschen  ->  Sättigung  ->  Quantisierung  ->  Shape-Check
      ideal          analog         Messbereich       ADC

Konventionen:
    - Messungen erfolgen im SENSOR-/FAHRZEUGFRAME, nie im Weltframe.
    - Der Sensor fragt die Szene nur über ihre Abfragemethoden ab
      (z. B. scene.raycast_2d), nie direkt über die Boxen.
    - Zufall nur über den übergebenen rng, nie über np.random direkt.
    - Die Abtastrate ist global und wird im Car_Model festgelegt.
      Ein Sensor kennt dt nicht, er misst immer genau dann, wenn er aufgerufen wird.

Neuen Sensor schreiben:
    1. Config-Dataclass anlegen, die von SensorConfig erbt (eigene Parameter
       wie n_beams kommen dort hinein)
    2. Klasse von BaseSensor ableiten, Config und name setzen
    3. spec() und _measure() implementieren
    4. Optional reset() überschreiben, falls der Sensor einen inneren Zustand hat
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Optional, Union

import numpy as np

from Car_Model import VehicleState

# Ein Parameter darf eine Zahl (gilt für alle Komponenten) oder ein Tupel
# (eine Zahl pro Komponente, z. B. IMU: Beschleunigung und Drehrate) sein.
Param = Union[float, tuple]


# ─────────────────────────────────────────────────────────────────────────────
# Datenstrukturen
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SensorSpec:
    """Beschreibt die Form einer EINZELNEN Messung."""

    shape: tuple                 # z. B. (360,) für einen LiDAR mit 360 Strahlen
    dtype: str = "float32"
    unit: str = ""               # z. B. "m" oder "m/s^2 | rad/s"
    description: str = ""        # z. B. "Index 0 = vorne, gegen den Uhrzeigersinn"


@dataclass
class SensorConfig:
    """Parameter, die JEDER Sensor hat. Sensorspezifische Configs erben hiervon."""

    # Rauschen: Standardabweichung, 0 = aus
    noise_std: Param = 0.0
    # Sättigung: Messbereich, None = unbegrenzt
    min_value: Optional[Param] = None
    max_value: Optional[Param] = None
    # Quantisierung: kleinste unterscheidbare Stufe, 0 = aus
    resolution: Param = 0.0
    # Einbauposition relativ zum Fahrzeugmittelpunkt (m)
    mount_x: float = 0.0
    mount_y: float = 0.0
    mount_height: float = 0.2


# ─────────────────────────────────────────────────────────────────────────────
# Basisklasse
# ─────────────────────────────────────────────────────────────────────────────
class BaseSensor(ABC):
    """Basisklasse für alle Sensoren."""

    Config = SensorConfig
    name: str = "base"           # Schlüssel im obs-Dictionary des Rollouts

    def __init__(self, cfg: Optional[SensorConfig] = None, **overrides):
        """Entweder eine fertige Config übergeben oder einzelne Parameter:
            Lidar(noise_std=0.02, resolution=0.01)
        Nicht angegebene Parameter behalten ihren Default.
        """
        self.cfg = cfg if cfg is not None else self.Config(**overrides)

    # ── Von Unterklassen zu implementieren ──────────────────────────────────
    @abstractmethod
    def spec(self) -> SensorSpec:
        """Shape, dtype und Einheit einer einzelnen Messung."""

    @abstractmethod
    def _measure(self, scene, state: VehicleState) -> np.ndarray:
        """Ideale, rauschfreie Messung im Sensorframe."""

    def reset(self, rng: np.random.Generator) -> None:
        """Wird zu Beginn jeder Fahrt aufgerufen.
        Standardmäßig leer. Überschreiben für Sensoren mit innerem Zustand,
        z. B. einen IMU-Bias, der pro Fahrt neu gewürfelt wird."""

    # ── Messkette ───────────────────────────────────────────────────────────
    def observe(self, scene, state: VehicleState, rng: np.random.Generator) -> np.ndarray:
        """Vollständige Messung: ideal -> Rauschen -> Sättigung -> Quantisierung."""
        z = np.asarray(self._measure(scene, state), dtype=np.float64)

        spec = self.spec()
        if z.shape != tuple(spec.shape):
            raise ValueError(
                f"Sensor '{self.name}': _measure() liefert Shape {z.shape}, "
                f"spec() deklariert {tuple(spec.shape)}"
            )

        z = self._add_noise(z, rng)
        z = self._saturate(z)
        z = self._quantize(z)
        return z.astype(spec.dtype)

    def _add_noise(self, z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        std = self._param("noise_std", z.shape)
        if np.all(std == 0):
            return z
        return z + rng.normal(0.0, 1.0, z.shape) * std

    def _saturate(self, z: np.ndarray) -> np.ndarray:
        lo = -np.inf if self.cfg.min_value is None else self._param("min_value", z.shape)
        hi = np.inf if self.cfg.max_value is None else self._param("max_value", z.shape)
        return np.clip(z, lo, hi)

    def _quantize(self, z: np.ndarray) -> np.ndarray:
        res = self._param("resolution", z.shape)
        active = res > 0
        if not np.any(active):
            return z
        safe_res = np.where(active, res, 1.0)          # Division durch 0 vermeiden
        return np.where(active, safe_res * np.round(z / safe_res), z)

    # ── Hilfsfunktionen ─────────────────────────────────────────────────────
    def _param(self, field: str, shape: tuple) -> np.ndarray:
        """Config-Parameter auf die Shape der Messung bringen (Zahl oder Tupel)."""
        value = np.asarray(getattr(self.cfg, field), dtype=np.float64)
        try:
            return np.broadcast_to(value, shape)
        except ValueError:
            raise ValueError(
                f"Sensor '{self.name}': Parameter '{field}' mit Shape {value.shape} "
                f"passt nicht zur Messung mit Shape {shape}"
            ) from None

    def mount_pose(self, vehicle_pose: np.ndarray) -> np.ndarray:
        """Pose des Sensors im Weltframe aus der Fahrzeugpose (x, y, yaw)."""
        x, y, yaw = vehicle_pose
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([
            x + c * self.cfg.mount_x - s * self.cfg.mount_y,
            y + s * self.cfg.mount_x + c * self.cfg.mount_y,
            yaw,
        ])

    def describe(self) -> dict:
        """Vollständige Beschreibung für die Metadaten des Datensatzes."""
        return {
            "class": type(self).__name__,
            "name": self.name,
            "params": asdict(self.cfg),
            "spec": asdict(self.spec()),
        }