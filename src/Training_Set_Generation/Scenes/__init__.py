"""
Verzeichnis aller Szenen.

Neue Szene hinzufügen:
    1. Datei in Scenes/ anlegen, Klasse erbt von SceneGenerator
       (Dateiname darf NICHT mit einer Ziffer beginnen, sonst ist er nicht importierbar)
    2. Hier importieren und ins Dictionary eintragen
"""

from .Base_Scene import Scene, SceneConfig, SceneGenerator
from .Random_Boxes_Random_Start_Random_Target import RandomBoxesRandomStart, RandomBoxesRandomStartConfig

SCENES: dict[str, type[SceneGenerator]] = {
    "random_boxes_random_start": RandomBoxesRandomStart,
}
 