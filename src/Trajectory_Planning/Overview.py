"""
Overview.py — Alle Läufe in runs/ nebeneinander: Configs und Metriken.

Pro Lauf eine Zeile mit:
    data.*      Config der Datengenerierung   (dataset_config.json, von Train.py)
    train.*     Config des Trainings          (config.json)
    metric.*    Ergebnisse der Validierung    (metrics.json, von Dashboard.py)

    runs/overview.csv   alle Spalten (z. B. in Excel öffnen)
    Terminal            nur Spalten, die sich zwischen den Läufen unterscheiden, plus Metriken

Aufruf (aus dem Ordner Trajectory_Planning):
    python Overview.py
"""

import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = ["collision", "length_ratio", "goal_error", "expert_dist", "final_val_loss"]


def flatten(d, prefix):
    """{"a": {"b": 1}} -> {"prefix.a.b": 1}. Listen werden zu Text."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = json.dumps(v) if isinstance(v, list) else v
    return out


def read_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def collect(runs_dir):
    rows = []
    for run in sorted(glob.glob(os.path.join(runs_dir, "*", "config.json"))):
        run_dir = os.path.dirname(run)
        row = {"run": os.path.basename(run_dir)}
        row.update(flatten(read_json(os.path.join(run_dir, "dataset_config.json")), "data"))
        row.update(flatten(read_json(run), "train"))
        row.update(flatten(read_json(os.path.join(run_dir, "metrics.json")), "metric"))
        rows.append(row)
    return rows


if __name__ == "__main__":
    runs_dir = os.path.join(HERE, "runs")
    rows = collect(runs_dir)
    if not rows:
        raise SystemExit("Keine Läufe in runs/ gefunden.")

    columns = list(dict.fromkeys(k for r in rows for k in r))
    out = os.path.join(runs_dir, "overview.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    # Terminal: nur Einstellungen, die sich unterscheiden, plus Metriken
    differing = [c for c in columns if not c.startswith(("metric.", "run"))
                 and len({str(r.get(c, "")) for r in rows}) > 1]
    shown = ["run"] + differing + [f"metric.{m}" for m in METRICS]

    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else ("–" if v in (None, "") else str(v))

    table = [[c.replace("metric.", "") for c in shown]] + [[fmt(r.get(c)) for c in shown] for r in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(shown))]
    for k, row in enumerate(table):
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)))
        if k == 0:
            print("  ".join("-" * w for w in widths))

    if not differing:
        print("\n(Alle Läufe haben dieselben Einstellungen.)")
    if any("metric.collision" not in r for r in rows):
        print("\n– = keine Metriken. Für diesen Lauf zuerst  python Dashboard.py runs/<lauf>  ausführen.")
    print(f"\nAlle Spalten: {out}")
