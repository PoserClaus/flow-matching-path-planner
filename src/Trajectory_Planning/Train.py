"""
Train.py — Ein Modell mit Flow Matching trainieren.

Aufruf (aus dem Ordner Trajectory_Planning, mit der venv):
    python Train.py                              # configs/default.json
    python Train.py --config configs/xyz.json

Ergebnis: runs/<name>_<zeit>/model.pt  (Gewichte + alles, um das Modell wieder aufzubauen)
          runs/<name>_<zeit>/config.json
          runs/<name>_<zeit>/log.csv   (step, train_loss, val_loss)
          runs/<name>_<zeit>/dataset_config.json   (Config, mit der der Datensatz erzeugt wurde)
"""

import argparse
import json
import os
import time

import torch

from Dataset import HERE, TrajectoryData
from Flow_Matching import flow_matching_loss
from Models import MODELS


def train(config, run_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(config["seed"])
    tcfg = config["training"]

    data = TrajectoryData(os.path.join(HERE, config["dataset"]), config["inputs"],
                          config["val_fraction"], config["normalize"], config["seed"], device)
    model = MODELS[config["model"]["type"]](data.spec, **config["model"]["params"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])

    print(f"Gerät:  {device}")
    print(f"Daten:  {len(data.train['x1'])} Train-Samples ({data.n_scenes[0]} Szenen), "
          f"{len(data.val['x1'])} Val-Samples ({data.n_scenes[1]} Szenen)")
    print(f"Modell: {config['model']['type']}, {model.n_parameters():,} Parameter\n")

    # Config, mit der der Datensatz erzeugt wurde, mit ablegen (für Overview.py)
    with open(os.path.join(run_dir, "dataset_config.json"), "w", encoding="utf-8") as f:
        json.dump(data.raw["meta"]["config"], f, indent=2)

    log = open(os.path.join(run_dir, "log.csv"), "w", encoding="utf-8")
    log.write("step,train_loss,val_loss\n")

    for step in range(1, tcfg["steps"] + 1):
        x1, cond = data.batch("train", tcfg["batch_size"])
        loss = flow_matching_loss(model, x1, cond)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % tcfg["eval_every"] == 0:
            with torch.no_grad():
                val_loss = flow_matching_loss(model, *data.batch("val", tcfg["batch_size"]))
            print(f"  step {step:6d}   train {loss.item():.4f}   val {val_loss.item():.4f}")
            log.write(f"{step},{loss.item():.6f},{val_loss.item():.6f}\n")

    log.close()
    torch.save({
        "model_type": config["model"]["type"],
        "model_params": config["model"]["params"],
        "spec": data.spec,
        "norm": data.norm,
        "state_dict": model.state_dict(),
    }, os.path.join(run_dir, "model.pt"))
    print(f"\nGespeichert: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(HERE, "configs", "default.json"))
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    run_dir = os.path.join(HERE, "runs", f"{config['name']}_{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    train(config, run_dir)
