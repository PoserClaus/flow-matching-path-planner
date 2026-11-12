"""
MLP.py — Einfachstes Modell: alle Eingänge aneinanderhängen, durch ein MLP schicken.

    Eingang:  x_t (N*2) + t (1) + Ziel (2) + Sensoren roh (z. B. LiDAR 360)
    -> n_layers versteckte Schichten mit je hidden Neuronen (ReLU)
    -> Ausgang: N*2, umgeformt zu [B, N, 2]
"""

import torch
import torch.nn as nn

from .Base_Model import BaseModel


class MLP(BaseModel):
    name = "mlp"

    def __init__(self, spec: dict, hidden: int = 256, n_layers: int = 3):
        super().__init__(spec)
        n = spec["n_points"]
        n_obs = sum(torch.Size(shape).numel() for shape in spec["obs"].values())
        n_in = n * 2 + 1 + spec["goal_dim"] + n_obs

        layers = []
        d = n_in
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers.append(nn.Linear(d, n * 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t, t, cond):
        sensors = [cond[k].flatten(1) for k in sorted(self.spec["obs"])]
        x = torch.cat([x_t.flatten(1), t[:, None], cond["goal"], *sensors], dim=1)
        return self.net(x).view(-1, self.spec["n_points"], 2)
