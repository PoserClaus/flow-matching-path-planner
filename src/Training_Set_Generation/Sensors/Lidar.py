"""
Lidar.py — 2D-LiDAR (horizontaler Scan in einer Ebene).

Messung: Abstand zum ersten Hindernis für jeden Strahl.
    Shape:   (n_beams,)
    Einheit: m
    Frame:   Sensorframe. Index 0 zeigt in Fahrtrichtung, die Indizes laufen
             gegen den Uhrzeigersinn (mathematisch positiv, wie yaw).
    Kein Treffer:       max_range  (über die Sättigung der Basisklasse)
    Zu nah (Blindzone): min_range  (über die Sättigung der Basisklasse)

Benötigt von der Szene:
    scene.raycast_2d(origin, angles, max_range, height) -> distances
        origin:  [2]  Sensorposition (x, y) im Weltframe
        angles:  [N]  Strahlrichtungen im Weltframe (rad)
        height:  Höhe der Scanebene; niedrigere Hindernisse werden nicht getroffen
        return:  [N]  Abstand zum ersten Treffer, max_range ohne Treffer
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from Car_Model import VehicleState

from .Base_Sensor import BaseSensor, SensorConfig, SensorSpec


@dataclass
class LidarConfig(SensorConfig):
    """Parameter des LiDARs. Werte möglichst aus dem Datenblatt des echten Sensors."""

    n_beams: int = 360                # Anzahl Strahlen
    fov: float = 2 * np.pi            # Öffnungswinkel in rad (2*pi = Rundumsicht)
    max_range: float = 8.0            # maximale Reichweite in m
    min_range: float = 0.15           # Blindbereich direkt am Sensor in m

    # Defaults der Basisklasse für einen typischen LiDAR überschreiben
    noise_std: float = 0.02           # m
    resolution: float = 0.01          # m

    def __post_init__(self):
        if self.n_beams < 1:
            raise ValueError("n_beams muss mindestens 1 sein")
        if not 0 < self.fov <= 2 * np.pi + 1e-9:
            raise ValueError("fov muss im Bereich (0, 2*pi] liegen")
        if not 0 <= self.min_range < self.max_range:
            raise ValueError("Es muss gelten: 0 <= min_range < max_range")

        # Messbereich der Sättigung an die Reichweite koppeln, damit niemand
        # die Werte doppelt eintragen muss.
        if self.min_value is None:
            self.min_value = self.min_range
        if self.max_value is None:
            self.max_value = self.max_range


class Lidar(BaseSensor):
    """2D-LiDAR mit gleichmäßig verteilten Strahlen."""

    Config = LidarConfig
    name = "lidar"

    def __init__(self, cfg: LidarConfig | None = None, **overrides):
        super().__init__(cfg, **overrides)
        self.beam_angles = self._compute_beam_angles()   # einmalig, im Sensorframe

    def _compute_beam_angles(self) -> np.ndarray:
        """Strahlrichtungen im Sensorframe.

        Rundumsicht (fov = 2*pi):  theta_i = i * fov / N,  i = 0 .. N-1
            -> Strahl 0 zeigt nach vorne, der letzte Strahl fällt NICHT auf den ersten.
        Teilsicht (fov < 2*pi):    gleichmäßig von -fov/2 bis +fov/2
            -> symmetrisch um die Fahrtrichtung.
        """
        n, fov = self.cfg.n_beams, self.cfg.fov
        if np.isclose(fov, 2 * np.pi):
            return np.arange(n) * (fov / n)
        if n == 1:
            return np.zeros(1)
        return np.linspace(-fov / 2, fov / 2, n)

    def spec(self) -> SensorSpec:
        if np.isclose(self.cfg.fov, 2 * np.pi):
            layout = "Index 0 = vorne, gegen den Uhrzeigersinn, Rundumsicht"
        else:
            layout = "Index 0 = rechter Rand (-fov/2), letzter Index = linker Rand (+fov/2)"
        return SensorSpec(
            shape=(self.cfg.n_beams,),
            unit="m",
            description=f"2D-LiDAR Abstände. {layout}. Kein Treffer = max_range.",
        )

    def _measure(self, scene, state: VehicleState) -> np.ndarray:
        # 1. Sensorpose aus der Fahrzeugpose (berücksichtigt den Einbauort)
        sensor_pose = self.mount_pose(state.pose)

        # 2. Strahlwinkel vom Sensorframe in den Weltframe drehen
        world_angles = sensor_pose[2] + self.beam_angles

        # 3. Szene abfragen
        return scene.raycast_2d(
            origin=sensor_pose[:2],
            angles=world_angles,
            max_range=self.cfg.max_range,
            height=self.cfg.mount_height,
        )

    def beam_endpoints(self, ranges: np.ndarray, vehicle_pose: np.ndarray) -> np.ndarray:
        """Hilfsfunktion zum Plotten: gemessene Abstände -> Trefferpunkte im Weltframe, [N, 2]."""
        sensor_pose = self.mount_pose(vehicle_pose)
        angles = sensor_pose[2] + self.beam_angles
        return sensor_pose[:2] + np.stack([np.cos(angles), np.sin(angles)], -1) * ranges[:, None]