"""
Flow_Matching.py — Loss und Sampling. Funktioniert mit jedem Modell aus Models/.

    x0 ~ N(0, I)                    Rauschen
    x1                              Experten-Trajektorie
    t  ~ U(0, 1)                    Flow-Zeit
    x_t = (1 - t) * x0 + t * x1     Punkt auf der Geraden zwischen Rauschen und Trajektorie

Training:  Das Modell soll die Richtung x1 - x0 vorhersagen.
           loss = mittlerer quadratischer Fehler
Sampling:  Bei Rauschen starten und n_steps kleine Schritte in die vorhergesagte
           Richtung gehen (Euler-Verfahren). Am Ende steht eine Trajektorie.
"""

import torch


def flow_matching_loss(model, x1, cond):
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], device=x1.device)
    x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
    v_pred = model(x_t, t, cond)
    return ((v_pred - (x1 - x0)) ** 2).mean()


@torch.no_grad()
def sample(model, cond, n_points, n_steps=32):
    batch_size = cond["goal"].shape[0]
    device = cond["goal"].device
    x = torch.randn(batch_size, n_points, 2, device=device)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((batch_size,), i * dt, device=device)
        x = x + dt * model(x, t, cond)
    return x
