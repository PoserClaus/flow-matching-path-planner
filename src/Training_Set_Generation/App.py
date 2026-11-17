"""
App.py — Oberfläche für die Datensatz-Generierung.

    Seitenleiste   Config einstellen + "Datensatz generieren" -> Test_Sets/<Datum_Uhrzeit>.npz
    Hauptbereich   Dashboard des neuen (oder eines älteren) Datensatzes:
                   Kennzahlen, Übersicht, Trajektorien-Explorer (mit A*-Raster), Config, Rohdaten

Start (aus dem Ordner Training_Set_Generation, mit der venv):
    streamlit run App.py
"""

from __future__ import annotations

import copy
import glob
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from Dataset_Format import load_dataset, save_dataset
from Generate_Testset import HERE, generate, new_dataset_path
from Plot_Testset import _lidar_from_meta, _solver_from_meta, plot_pair
from Scenes import SCENES
from Scenes.Plot_Scene import PATH_COLORS
from Solvers import SOLVERS

st.set_page_config(page_title="Datensatz-Generator", layout="wide")


# ─────────────────────────────────────────────────────────────────────────────
# Seitenleiste: Config einstellen
# ─────────────────────────────────────────────────────────────────────────────
def config_form(base: dict) -> dict:
    """Formular mit Startwerten aus base. Gibt die eingestellte Config zurück."""
    cfg = copy.deepcopy(base)
    scene_p = cfg["scene"].setdefault("params", {})
    solver_p = cfg["solver"].setdefault("params", {})
    lidar = next(s for s in cfg["sensors"] if s["type"] == "lidar")
    lidar_p = lidar.setdefault("params", {})

    with st.expander("Allgemein", expanded=True):
        c1, c2 = st.columns(2)
        cfg["n_scenes"] = c1.number_input(
            "Anzahl Szenen", 1, 10000, cfg["n_scenes"],
            help="Wie viele zufällige Räume erzeugt werden. Mehr Szenen = mehr Vielfalt, "
                 "aber längere Laufzeit (ca. 0,25 s pro Paar).")
        cfg["seed"] = c2.number_input(
            "Seed", 0, 10**9, cfg["seed"],
            help="Startwert des Zufallsgenerators. Gleicher Seed + gleiche Config = exakt gleicher Datensatz. "
                 "Anderer Seed = andere Szenen.")
        cfg["car"]["radius"] = st.number_input(
            "Roboterradius (m)", 0.05, 2.0, float(cfg["car"]["radius"]), 0.05,
            help="Der Roboter wird als Kreis modelliert. Der Solver hält den Robotermittelpunkt mindestens "
                 "so weit (plus Sicherheitsabstand) von Hindernissen fern.")

    with st.expander("Szene", expanded=True):
        cfg["scene"]["type"] = st.selectbox(
            "Szenengenerator", list(SCENES), list(SCENES).index(cfg["scene"]["type"]),
            help="Wie die Räume aufgebaut werden. Neue Generatoren in Scenes/ anlegen und in Scenes/__init__.py "
                 "eintragen, dann erscheinen sie hier.")
        x0, x1, y0, y1 = scene_p.get("bounds", [0.0, 8.0, -3.0, 3.0])
        c1, c2 = st.columns(2)
        length = c1.number_input("Raumlänge x (m)", 2.0, 50.0, float(x1 - x0), 0.5,
                                 help="Ausdehnung des Raums in x-Richtung. Der Raum ist von Wänden umgeben.")
        width = c2.number_input("Raumbreite y (m)", 2.0, 50.0, float(y1 - y0), 0.5,
                                help="Ausdehnung des Raums in y-Richtung. Der Raum ist von Wänden umgeben.")
        scene_p["bounds"] = [0.0, length, -width / 2, width / 2]
        scene_p["n_boxes"] = list(st.slider(
            "Boxen pro Szene", 0, 30, tuple(scene_p.get("n_boxes", [2, 6])),
            help="Pro Szene wird eine Anzahl zwischen Minimum und Maximum gewürfelt. "
                 "Passen nicht alle in den Raum, werden es weniger."))
        size = st.slider(
            "Boxgröße: Länge und Breite (m)", 0.1, 3.0,
            tuple(map(float, scene_p.get("box_length", [0.3, 1.0]))), 0.1,
            help="Länge und Breite jeder Box werden unabhängig voneinander in diesem Bereich gewürfelt. "
                 "Die Drehung der Box ist zufällig.")
        scene_p["box_length"] = scene_p["box_width"] = list(size)

    with st.expander("Start/Ziel-Paare"):
        p = cfg["pairs"]
        p["per_scene"] = st.number_input(
            "Paare pro Szene", 1, 100, p["per_scene"],
            help="Wie viele Fahraufgaben (Start → Ziel) pro Raum gewürfelt werden. "
                 "So entstehen aus einer Szene viele Trainingsbeispiele.")
        c1, c2 = st.columns(2)
        p["min_dist"] = c1.number_input(
            "Mindestabstand Start–Ziel (m)", 0.0, 50.0, float(p["min_dist"]), 0.5,
            help="Luftlinie, die Start und Ziel mindestens auseinanderliegen. "
                 "Größer = längere, interessantere Wege. Zu groß für den Raum = keine Paare.")
        p["min_clearance"] = c2.number_input(
            "Freiraum um Start/Ziel (m)", 0.0, 5.0, float(p["min_clearance"]), 0.05,
            help="So viel Platz muss um Start und Ziel frei sein. Sollte größer als "
                 "Roboterradius + Sicherheitsabstand sein, sonst findet der Solver oft keinen Weg.")

    with st.expander("Solver"):
        cfg["solver"]["type"] = st.selectbox(
            "Solver", list(SOLVERS), list(SOLVERS).index(cfg["solver"]["type"]),
            help="Verfahren, das die Wege sucht. astar = A* auf einem Raster, danach geglättet.")
        c1, c2 = st.columns(2)
        solver_p["resolution"] = c1.number_input(
            "Rasterzelle (m)", 0.02, 1.0, float(solver_p.get("resolution", 0.05)), 0.01,
            help="Kantenlänge einer A*-Rasterzelle. Kleiner = genauer, findet auch enge Lücken, "
                 "aber deutlich langsamer.")
        solver_p["safety_margin"] = c2.number_input(
            "Sicherheitsabstand (m)", 0.0, 1.0, float(solver_p.get("safety_margin", 0.05)), 0.01,
            help="Zusätzlicher Abstand zu Hindernissen über den Roboterradius hinaus. "
                 "Im A*-Raster rot als Sperrzone sichtbar.")
        solver_p["max_paths"] = c1.number_input(
            "Max. Wege pro Paar", 1, 20, solver_p.get("max_paths", 5),
            help="Höchstens so viele verschiedene Wege werden pro Paar gespeichert "
                 "(z. B. links oder rechts um eine Box).")
        solver_p["min_path_distance"] = c2.number_input(
            "Mindestabstand zweier Wege (m)", 0.0, 5.0, float(solver_p.get("min_path_distance", 0.5)), 0.1,
            help="Zwei Wege gelten erst als verschieden, wenn sie an ihrer weitesten Stelle "
                 "so weit auseinanderliegen (Hausdorff-Abstand). Sonst wird der zweite verworfen.")
        solver_p["max_length_ratio"] = c1.number_input(
            "Max. Länge × kürzester Weg", 1.0, 5.0, float(solver_p.get("max_length_ratio", 1.5)), 0.1,
            help="Umwege, die länger als dieses Vielfache des kürzesten Wegs sind, werden verworfen. "
                 "1,5 = höchstens 50 % länger.")
        solver_p["length_tau"] = c2.number_input(
            "Längengewichtung τ (m)", 0.05, 10.0, float(solver_p.get("length_tau", 1.0)), 0.1,
            help="Bestimmt das Gewicht jedes Wegs: w ∝ exp(−Umweg / τ). Kleiner τ = kurze Wege "
                 "stark bevorzugt. Großer τ = alle Wege fast gleich wichtig.")

    with st.expander("LiDAR"):
        c1, c2 = st.columns(2)
        lidar_p["n_beams"] = c1.number_input(
            "Strahlen", 1, 3600, lidar_p.get("n_beams", 360),
            help="Anzahl Messstrahlen, gleichmäßig über das Sichtfeld verteilt. "
                 "Das ist auch die Länge des LiDAR-Vektors, den das Netz bekommt.")
        lidar_p["max_range"] = c2.number_input(
            "Reichweite (m)", 0.5, 100.0, float(lidar_p.get("max_range", 8.0)), 0.5,
            help="Weiter entfernte Hindernisse sieht der LiDAR nicht. Strahlen ohne Treffer "
                 "liefern genau diesen Wert.")
        fov_deg = st.slider(
            "Sichtfeld (°)", 10, 360, int(round(np.rad2deg(lidar_p.get("fov", 2 * np.pi)))),
            help="Öffnungswinkel des Scans. 360° = Rundumsicht, kleiner = symmetrisch um die Fahrtrichtung.")
        lidar_p["fov"] = float(np.deg2rad(fov_deg))

    with st.expander("Trajektorie"):
        cfg["trajectory"]["n_points"] = st.number_input(
            "Punkte pro Weg", 2, 512, cfg["trajectory"]["n_points"],
            help="Jeder Weg wird auf so viele gleichmäßig verteilte Punkte gebracht "
                 "(erster = Start, letzter = Ziel). Das ist die Ausgabegröße des Netzes.")

    return cfg


with st.sidebar:
    st.header("Config")
    config_files = sorted(glob.glob(os.path.join(HERE, "configs", "*.json")))
    config_file = st.selectbox(
        "Startwerte aus", config_files, format_func=os.path.basename,
        help="Config-Datei aus configs/, deren Werte unten vorausgefüllt werden. Änderungen hier verändern "
             "die Datei nicht, sie werden nur im neuen Datensatz gespeichert.")
    with open(config_file, encoding="utf-8") as f:
        config = config_form(json.load(f))

    if st.button("Datensatz generieren", type="primary", width="stretch"):
        bar = st.progress(0.0, text="Generiere …")
        t0 = time.time()
        arrays, scenes, meta = generate(
            config, verbose=False,
            on_scene=lambda i, n: bar.progress(i / n, text=f"Szene {i}/{n}  ({time.time() - t0:.0f} s)"))
        meta["config_file"] = os.path.basename(config_file) + " (App)"
        meta["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
        out = new_dataset_path()
        os.makedirs(os.path.dirname(out), exist_ok=True)
        save_dataset(out, arrays, scenes, meta)
        bar.empty()
        st.success(f"Gespeichert: {os.path.basename(out)}  ({time.time() - t0:.0f} s)")
        st.session_state["dataset"] = out


# ─────────────────────────────────────────────────────────────────────────────
# Hauptbereich: Dashboard
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load(path: str) -> dict:
    return load_dataset(path)


def pair_table(data: dict) -> pd.DataFrame:
    """Eine Zeile pro Start/Ziel-Paar."""
    pairs, first, n_paths = np.unique(data["pair_id"], return_index=True, return_counts=True)
    shortest = [data["length"][data["pair_id"] == p].min() for p in pairs]
    return pd.DataFrame({
        "Paar": pairs,
        "Szene": data["scene_id"][first],
        "Wege": n_paths,
        "Luftlinie (m)": np.hypot(*data["goal_rel"][first, :2].T),
        "kürzester Weg (m)": shortest,
    })


def tab_overview(data: dict, pairs: pd.DataFrame):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Verschiedene Wege pro Paar**")
        st.bar_chart(pairs["Wege"].value_counts().sort_index(), x_label="Wege pro Paar", y_label="Anzahl Paare")
    with c2:
        st.markdown("**Länge aller Wege**")
        counts, edges = np.histogram(data["length"], bins=20)
        st.bar_chart(pd.Series(counts, index=np.round((edges[:-1] + edges[1:]) / 2, 1)),
                     x_label="Weglänge (m)", y_label="Anzahl Wege")
    st.markdown("**Alle Paare** (Tabelle sortierbar per Klick auf die Spalte)")
    st.dataframe(pairs, hide_index=True, width="stretch", height=300,
                 column_config={c: st.column_config.NumberColumn(format="%.2f")
                                for c in ("Luftlinie (m)", "kürzester Weg (m)")})


def tab_explorer(data: dict, pairs: pd.DataFrame):
    meta = data["meta"]
    c1, c2, c3 = st.columns([2, 3, 2])
    scene = c1.selectbox("Szene", sorted(pairs["Szene"].unique()))
    in_scene = pairs[pairs["Szene"] == scene]
    pair = c2.radio("Paar", in_scene["Paar"], horizontal=True,
                    format_func=lambda p: f"{p} ({int(in_scene.loc[in_scene['Paar'] == p, 'Wege'].iloc[0])} Wege)")
    show_grid = c3.toggle("A*-Raster", value=True, help="Rot = dort darf der Robotermittelpunkt nicht hin "
                                                         "(Hindernis + Radius + Sicherheitsabstand)")
    show_lidar = c3.toggle("LiDAR (rechts)", value=True)

    idx = np.flatnonzero(data["pair_id"] == pair)
    lengths = data["length"][idx]
    paths = pd.DataFrame({
        "Farbe": [PATH_COLORS[k % len(PATH_COLORS)] for k in range(len(idx))],
        "Weg": np.arange(1, len(idx) + 1),
        "Länge (m)": lengths,
        "× kürzester": lengths / lengths.min(),
        "Gewicht": data["weight"][idx],
    })
    show = st.multiselect("Wege anzeigen", list(range(len(idx))), default=list(range(len(idx))),
                          format_func=lambda k: f"Weg {k + 1}")
    if not show:
        st.info("Mindestens einen Weg auswählen.")
        return

    fig, (ax_w, ax_v) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.4, 1]})
    plot_pair(ax_w, ax_v, data, int(pair), _lidar_from_meta(meta) if show_lidar else None,
              _solver_from_meta(meta) if show_grid else None, show=show)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Links: Weltansicht. Rechts: Fahrzeugframe am Start = was das Netz sieht "
               "(x nach vorne, y nach links). Linienbreite = Gewicht.")

    st.dataframe(paths.style.map(lambda c: f"background-color: {c}; color: {c}", subset=["Farbe"]),
                 hide_index=True, width="stretch",
                 column_config={"Länge (m)": st.column_config.NumberColumn(format="%.2f"),
                                "× kürzester": st.column_config.NumberColumn(format="%.2f"),
                                "Gewicht": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0,
                                                                           format="%.2f")})


def tab_data(data: dict):
    rows = []
    for key, v in data.items():
        if isinstance(v, np.ndarray):
            first = np.round(v[0].reshape(-1)[:5].astype(float), 3).tolist()
            rows.append({"Array": key, "Form": str(v.shape), "Typ": str(v.dtype),
                         "erstes Sample": str(first) + (" …" if v[0].size > 5 else "")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("1 Sample = 1 Weg. Alle Wege eines Paars teilen sich dieselben Sensordaten. "
               "Format: siehe Dataset_Format.py")


st.title("Datensatz-Dashboard")
datasets = sorted(glob.glob(os.path.join(HERE, "Test_Sets", "*.npz")), reverse=True)  # neuester zuerst
if not datasets:
    st.info("Noch kein Datensatz vorhanden. Links einstellen und auf **Datensatz generieren** klicken.")
    st.stop()

current = st.session_state.get("dataset", datasets[0])
path = st.selectbox("Datensatz", datasets, datasets.index(current) if current in datasets else 0,
                    format_func=os.path.basename)
st.session_state["dataset"] = path
data = load(path)
meta = data["meta"]
pairs = pair_table(data)
st.caption(f"erzeugt {meta.get('created', '?')} · Startwerte aus {meta.get('config_file', '?')}")

m = st.columns(5)
m[0].metric("Szenen", len(data["scenes"]))
m[1].metric("Paare", len(pairs), help="Paar = ein Start- und ein Zielpunkt in einer Szene")
m[2].metric("ohne Lösung", meta["n_pairs_failed"], help="Paare, für die der Solver keinen Weg fand (verworfen)")
m[3].metric("Samples", len(data["weight"]), help="1 Sample = 1 Weg")
m[4].metric("Ø Wege pro Paar", f"{pairs['Wege'].mean():.1f}")

t1, t2, t3, t4 = st.tabs(["Trajektorien", "Übersicht", "Config", "Daten"])
with t1:
    tab_explorer(data, pairs)
with t2:
    tab_overview(data, pairs)
with t3:
    st.json(meta["config"])
with t4:
    tab_data(data)
