"""Central settings for the flow matching planner.

One `Settings` object holds everything that can differ between experiments,
grouped by who needs it:

  representation  how paths and worlds are encoded      (must match the model)
  model           network size                          (must match the model)
  source          source distribution of the flow       (jitter / candidates are
                                                         also needed at inference)
  augmentation    training crops and data augmentation  (training only)
  training        optimizer, epochs, validation         (training only)
  evaluation      held-out checks during training       (not part of the model)

Every run saves its settings as `settings.json` next to the weights. The viewer
and the benchmark load that file, so a model can never be used with a different
horizon or scale than it was trained with. Missing keys in an old file fall back
to the current defaults; unknown keys raise an error (usually a typo).

Typical use:
    s = Settings(name="gaussian_source")
    s.override(["source.kind=gaussian", "training.epochs=40"])
    s.save("runs/gaussian_source/settings.json")
    s = Settings.load("runs/gaussian_source/settings.json")
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

SOURCE_KINDS = ("bend", "gaussian", "line") 


# ------------------------------------------------------------------
# Groups
# ------------------------------------------------------------------
@dataclass
class Representation:
  """How paths and worlds are encoded. Must be identical in training and inference."""

  n_points: int = 40  # points per predicted path
  horizon_px: float = 240.0  # arc length covered by the n_points (~6 px spacing)
  coord_scale: float = 250.0  # all relative coordinates are divided by this
  max_obs: int = 6  # obstacle slots (nearest first); the dataset has 2-4

  @property
  def spacing_px(self) -> float:
    return self.horizon_px / (self.n_points - 1)


@dataclass
class Model:
  """Network size. Needed to rebuild the network when loading weights."""

  hidden_dim: int = 512
  obs_dim: int = 128  # output size of the per-obstacle encoder


@dataclass
class Source:
  """Source distribution x0 of the flow (what the path is generated from).

  kind="bend":     straight line to the goal + one random smooth sideways bend
                   + small per-point jitter. Candidates at inference use bends
                   spread over [-cand_bend_max_px, +cand_bend_max_px].
  kind="gaussian": plain Gaussian noise, x0 ~ N(0, gaussian_std^2) per
                   coordinate in normalized units. Candidates are independent
                   noise draws; the bend settings are ignored.
  """

  kind: str = "bend"
  bend_std_px: float = 70.0  # training: bend of the start line ~ N(0, std)
  jitter_std: float = 0.02  # normalized units (~5 px) of per-point noise
  cand_bend_max_px: float = 100.0  # inference: candidates span [-max, +max]
  gaussian_std: float = 1.0  # kind="gaussian": std in normalized units


@dataclass
class Augmentation:
  """Training crops and augmentation (training only)."""

  p_start: float = 0.15  # fraction of crops that start at the real episode start
  p_offpath: float = 0.5  # fraction of crops where the agent is shifted off path
  offpath_std_px: float = 8.0
  offpath_max_px: float = 20.0
  rejoin_px: float = 60.0  # the offset fades out along this arc length
  mirror: bool = True  # random mirroring of the relative world


@dataclass
class Training:
  """Optimizer, schedule and validation (training only)."""

  seed: int = 0
  epochs: int = 30
  batch_size: int = 256
  crops_per_episode: int = 16  # random crops per episode and epoch
  val_fraction: float = 0.05
  val_crops_per_episode: int = 4
  lr: float = 5e-4
  ema_decay: float = 0.999


@dataclass
class Evaluation:
  """Held-out checks during training."""

  eval_every: int = 10  # epochs between evaluations (and checkpoints)
  eval_margin_px: float = 14.0  # collision margin; the planner's narrowest clearance


GROUPS = {
    "representation": Representation,
    "model": Model,
    "source": Source,
    "augmentation": Augmentation,
    "training": Training,
    "evaluation": Evaluation,
}


# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
@dataclass
class Settings:
  name: str = "default"
  representation: Representation = field(default_factory=Representation)
  model: Model = field(default_factory=Model)
  source: Source = field(default_factory=Source)
  augmentation: Augmentation = field(default_factory=Augmentation)
  training: Training = field(default_factory=Training)
  evaluation: Evaluation = field(default_factory=Evaluation)

  # ---- checks ----
  def validate(self) -> "Settings":
    r, s, a, t, e = (
        self.representation, self.source, self.augmentation, self.training,
        self.evaluation,
    )
    problems = []
    if r.n_points < 2:
      problems.append("representation.n_points must be >= 2")
    if r.horizon_px <= 0:
      problems.append("representation.horizon_px must be > 0")
    if r.coord_scale <= 0:
      problems.append("representation.coord_scale must be > 0")
    if r.max_obs < 1:
      problems.append("representation.max_obs must be >= 1")
    if self.model.hidden_dim < 1 or self.model.obs_dim < 1:
      problems.append("model sizes must be >= 1")
    if s.kind not in SOURCE_KINDS:
      problems.append(f"source.kind must be one of {SOURCE_KINDS}, got {s.kind!r}")
    if min(s.bend_std_px, s.jitter_std, s.cand_bend_max_px, s.gaussian_std) < 0:
      problems.append("source stds and bend sizes must be >= 0")
    if s.kind == "gaussian" and s.gaussian_std <= 0:
      problems.append("source.gaussian_std must be > 0 for kind='gaussian'")
    for key in ("p_start", "p_offpath"):
      if not 0.0 <= getattr(a, key) <= 1.0:
        problems.append(f"augmentation.{key} must be in [0, 1]")
    if min(a.offpath_std_px, a.offpath_max_px, a.rejoin_px) < 0:
      problems.append("augmentation distances must be >= 0")
    if t.epochs < 1 or t.batch_size < 1 or t.crops_per_episode < 1:
      problems.append("training.epochs/batch_size/crops_per_episode must be >= 1")
    if not 0.0 < t.val_fraction < 1.0:
      problems.append("training.val_fraction must be in (0, 1)")
    if t.val_crops_per_episode < 1:
      problems.append("training.val_crops_per_episode must be >= 1")
    if t.lr <= 0:
      problems.append("training.lr must be > 0")
    if not 0.0 <= t.ema_decay < 1.0:
      problems.append("training.ema_decay must be in [0, 1)")
    if e.eval_every < 1:
      problems.append("evaluation.eval_every must be >= 1")
    if e.eval_margin_px < 0:
      problems.append("evaluation.eval_margin_px must be >= 0")
    if problems:
      raise ValueError("Invalid settings:\n  " + "\n  ".join(problems))
    return self

  def compatible_with(self, other: "Settings") -> bool:
    """True if `other`'s weights can be used with these representation/model settings."""
    return (
        self.representation == other.representation
        and self.model == other.model
    )

  # ---- (de)serialization ----
  def to_dict(self) -> dict:
    return dataclasses.asdict(self)

  @classmethod
  def from_dict(cls, data: dict) -> "Settings":
    """Strict on unknown keys, lenient on missing ones (defaults are used)."""
    data = dict(data)
    unknown = set(data) - {"name", *GROUPS}
    if unknown:
      raise KeyError(f"Unknown settings keys: {sorted(unknown)}")
    settings = cls(name=data.pop("name", "default"))
    for group_name, group_cls in GROUPS.items():
      values = dict(data.pop(group_name, {}))
      valid = {f.name for f in dataclasses.fields(group_cls)}
      bad = set(values) - valid
      if bad:
        raise KeyError(
            f"Unknown keys in '{group_name}': {sorted(bad)} "
            f"(valid: {sorted(valid)})"
        )
      setattr(settings, group_name, group_cls(**values))
    return settings.validate()

  def save(self, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(self.to_dict(), indent=2))
    return path

  @classmethod
  def load(cls, path) -> "Settings":
    return cls.from_dict(json.loads(Path(path).read_text()))

  # ---- command-line style overrides ----
  def override(self, assignments) -> "Settings":
    """Apply strings like 'source.kind=gaussian' or 'name=my_run' in place.

    Values are converted to the type of the current value (bool/int/float/str).
    """
    for item in assignments:
      if "=" not in item:
        raise ValueError(f"Override must look like 'group.key=value', got {item!r}")
      key, raw = item.split("=", 1)
      key, raw = key.strip(), raw.strip()

      if key == "name":
        self.name = raw
        continue
      if "." not in key:
        raise KeyError(
            f"Unknown setting {key!r}. Use 'name' or 'group.key' with group in "
            f"{sorted(GROUPS)}"
        )
      group_name, attr = key.split(".", 1)
      if group_name not in GROUPS:
        raise KeyError(f"Unknown group {group_name!r}; valid: {sorted(GROUPS)}")
      group = getattr(self, group_name)
      if not hasattr(group, attr) or attr.startswith("_"):
        valid = [f.name for f in dataclasses.fields(group)]
        raise KeyError(f"Unknown key {key!r}; valid in '{group_name}': {valid}")
      setattr(group, attr, _coerce(raw, getattr(group, attr), key))
    return self.validate()

  # ---- comparing experiments ----
  def changes(self, other: "Settings" | None = None) -> dict:
    """{'group.key': (other_value, self_value)} for everything that differs.

    With no argument, compares against the defaults.
    """
    other = other or Settings()
    diff = {}
    mine, theirs = self.to_dict(), other.to_dict()
    for group_name in GROUPS:
      for key, value in mine[group_name].items():
        if value != theirs[group_name][key]:
          diff[f"{group_name}.{key}"] = (theirs[group_name][key], value)
    return diff


def _coerce(raw: str, current, key: str):
  try:
    if isinstance(current, bool):  # bool before int: bool is a subclass of int
      low = raw.lower()
      if low in ("true", "1", "yes", "on"):
        return True
      if low in ("false", "0", "no", "off"):
        return False
      raise ValueError(raw)
    if isinstance(current, int):
      as_float = float(raw)
      if as_float != int(as_float):
        raise ValueError(raw)
      return int(as_float)  # accepts '80' and '1e2'
    if isinstance(current, float):
      return float(raw)
    return raw
  except ValueError:
    raise ValueError(
        f"Cannot convert {raw!r} for {key!r} (expected {type(current).__name__})"
    ) from None
