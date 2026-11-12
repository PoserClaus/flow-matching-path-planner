"""
Verzeichnis aller Sensoren.

Neuen Sensor hinzufügen:
    1. Datei in Sensors/ anlegen, Klasse erbt von BaseSensor
    2. Hier importieren und ins Dictionary eintragen
"""

from .Base_Sensor import BaseSensor, SensorConfig, SensorSpec
from .Lidar import Lidar, LidarConfig
from .IMU import IMU, IMUConfig

SENSORS: dict[str, type[BaseSensor]] = {
    "lidar": Lidar,
    "imu": IMU,
}