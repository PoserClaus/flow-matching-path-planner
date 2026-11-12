"""
Base_Model.py — Schnittstelle, die jedes Modell erfüllen muss.

    v = model(x_t, t, cond)

    x_t   [B, N, 2]  verrauschte Trajektorie zur Flow-Zeit t
    t     [B]        Flow-Zeit in [0, 1]  (0 = reines Rauschen, 1 = Trajektorie)
    cond  dict       cond["goal"] [B, 2]  und  cond[<sensor>], z. B. cond["lidar"] [B, 360]
    v     [B, N, 2]  vorhergesagte Geschwindigkeit

spec beschreibt die Datenformen (aus Dataset.py):
    spec["n_points"]   N, Punkte pro Trajektorie
    spec["goal_dim"]   2
    spec["obs"]        {sensor: Form}, z. B. {"lidar": [360]}

Neues Modell:
    1. Datei in Models/ anlegen, Klasse erbt von BaseModel, name setzen
    2. __init__(self, spec, <eigene Parameter>) und forward() schreiben
    3. In Models/__init__.py eintragen
    4. In der Config "model": {"type": "<name>", "params": {...}}
"""

from abc import ABC, abstractmethod

import torch.nn as nn


class BaseModel(nn.Module, ABC):
    name = "base"

    def __init__(self, spec: dict):
        super().__init__()
        self.spec = spec

    @abstractmethod
    def forward(self, x_t, t, cond):
        """Gibt die Geschwindigkeit [B, N, 2] zurück."""

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
