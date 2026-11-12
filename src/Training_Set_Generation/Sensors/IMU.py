"""
IMU.py — Inertialsensor für ein Fahrzeug in der Ebene.

Messung: [a_x, a_y, omega_z]
    a_x      Längsbeschleunigung im Sensorframe   (m/s^2)
    a_y      Querbeschleunigung im Sensorframe    (m/s^2)
    omega_z  Gierrate                             (rad/s)

Physik (Fahrzeug in der Ebene, IMU um den Hebelarm (r_x, r_y) versetzt):
    a_x     = v_dot     - omega^2 * r_x - omega_dot * r_y
    a_y     = v * omega - omega^2 * r_y + omega_dot * r_x
    omega_z = omega

    v * omega       Zentripetalbeschleunigung der Kurvenfahrt
    omega^2 * r     Zentripetalanteil durch den versetzten Einbau
    omega_dot * r   Tangentialanteil durch Änderung der Drehrate

Fehlermodell:
    Bias          pro Fahrt konstant, in reset() gewürfelt
    Skalenfehler  pro Fahrt konstant, in reset() gewürfelt
    Rauschen, Sättigung, Quantisierung  über die Basisklasse

Hinweis zur Abtastrate:
    Die simulierte IMU liefert pro Zeitschritt einen Momentanwert. Auf dem
    echten Roboter die IMU-Daten auf den Planungstakt mitteln oder filtern,
    damit die Charakteristik zur Simulation passt.

Benötigt vom VehicleState: velocity (v, omega) und acceleration (v_dot, omega_dot).
Benötigt von der Szene: nichts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from Car_Model import VehicleState

from .Base_Sensor import BaseSensor, SensorConfig, SensorSpec

G = 9.81  # m/s^2


@dataclass
class IMUConfig(SensorConfig):
    """Parameter der IMU. Reihenfolge aller Tupel: (a_x, a_y, omega_z).

    Die Defaults sind Platzhalter in der Größenordnung typischer MEMS-IMUs.
    Möglichst durch Werte aus dem Datenblatt der echten IMU ersetzen.
    """

    # Weißes Rauschen pro Messung
    noise_std: tuple = (0.05, 0.05, 0.005)          # m/s^2, m/s^2, rad/s
    # Bias: pro Fahrt konstant, gewürfelt mit dieser Standardabweichung
    bias_std: tuple = (0.1, 0.1, 0.01)              # m/s^2, m/s^2, rad/s
    # Skalenfehler: pro Fahrt konstanter Faktor 1 + N(0, scale_error_std)
    scale_error_std: float = 0.01                   # 1 %
    # Messbereich: ±16 g und ±2000 °/s
    max_value: Optional[tuple] = (16 * G, 16 * G, np.deg2rad(2000))
    # Quantisierung: 16-Bit-Wandler über den gesamten Messbereich
    resolution: tuple = (32 * G / 2**16, 32 * G / 2**16, 2 * np.deg2rad(2000) / 2**16)

    def __post_init__(self):
        if self.scale_error_std < 0 or np.any(np.asarray(self.bias_std) < 0):
            raise ValueError("bias_std und scale_error_std dürfen nicht negativ sein")
        # Symmetrischer Messbereich, falls min_value nicht explizit gesetzt ist
        if self.min_value is None and self.max_value is not None:
            self.min_value = tuple(-np.asarray(self.max_value, dtype=float))


class IMU(BaseSensor):
    """IMU mit drei Komponenten: [a_x, a_y, omega_z]."""

    Config = IMUConfig
    name = "imu"

    def __init__(self, cfg: IMUConfig | None = None, **overrides):
        super().__init__(cfg, **overrides)
        # Ohne reset() arbeitet die IMU fehlerfrei (kein Bias, kein Skalenfehler)
        self.bias = np.zeros(3)
        self.scale = np.ones(3)

    def reset(self, rng: np.random.Generator) -> None:
        """Zu Beginn jeder Fahrt: Bias und Skalenfehler neu würfeln."""
        self.bias = rng.normal(0.0, 1.0, 3) * np.asarray(self.cfg.bias_std, dtype=float)
        self.scale = 1.0 + rng.normal(0.0, self.cfg.scale_error_std, 3)

    def spec(self) -> SensorSpec:
        return SensorSpec(
            shape=(3,),
            unit="m/s^2 | m/s^2 | rad/s",
            description="[a_x, a_y, omega_z] im Sensorframe. "
                        "x = Fahrtrichtung, y = links, z = oben.",
        )

    def _measure(self, scene, state: VehicleState) -> np.ndarray:
        v, omega = state.velocity
        v_dot, omega_dot = state.acceleration
        r_x, r_y = self.cfg.mount_x, self.cfg.mount_y

        # 1. Ideale Werte am Fahrzeugmittelpunkt + 2. Hebelarm-Terme
        a_x = v_dot - omega**2 * r_x - omega_dot * r_y
        a_y = v * omega - omega**2 * r_y + omega_dot * r_x
        ideal = np.array([a_x, a_y, omega])

        # 3. Skalenfehler und Bias (pro Fahrt konstant)
        return self.scale * ideal + self.bias