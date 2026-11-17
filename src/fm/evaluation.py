"""Open-loop evaluation on held-out worlds (used during training)."""

import numpy as np
import torch

from fm.data import crop_path
from fm.sampling import generate_candidates
from geometry import collision_count, path_length


@torch.no_grad()
def evaluate(model, episodes, settings, n_eps=200, n_cand=7, verbose=True):
  """Paths from the episode start; returns a dict of collision-free rates etc.

  single_ok:    the middle candidate (one sample) is collision-free
  best_ok:      at least one of the n_cand candidates is collision-free
  length_ratio: median length of the shortest collision-free candidate / expert
  end_error_px: median distance of that candidate's end from the expert's end
  """
  model.eval()
  rep = settings.representation
  margin = settings.evaluation.eval_margin_px
  single_ok, best_ok, ratios, end_errs = [], [], [], []
  for ep in episodes[:n_eps]:
    agent = ep["path"][0]
    cands = generate_candidates(model, agent, ep["goal"], ep["obs"], settings,
                                n_cand)
    expert = crop_path(ep, 0.0, rep)

    coll = np.array([collision_count(c, ep["obs"], margin) for c in cands])
    lens = np.array([path_length(c) for c in cands])
    ok = coll == 0
    single_ok.append(bool(ok[n_cand // 2]))
    best_ok.append(bool(ok.any()))
    if ok.any():
      j = int(np.argmin(np.where(ok, lens, np.inf)))
      ratios.append(lens[j] / max(path_length(expert), 1e-6))
      end_errs.append(float(np.linalg.norm(cands[j][-1] - expert[-1])))

  result = {
      "n_worlds": len(single_ok),
      "n_candidates": n_cand,
      "single_ok": float(np.mean(single_ok)),
      "best_ok": float(np.mean(best_ok)),
      "length_ratio": float(np.median(ratios)) if ratios else None,
      "end_error_px": float(np.median(end_errs)) if end_errs else None,
  }
  if verbose:
    ratio = result["length_ratio"]
    err = result["end_error_px"]
    print(
        f"    eval ({result['n_worlds']} held-out worlds): collision-free single "
        f"{result['single_ok']:.1%} | best of {n_cand} {result['best_ok']:.1%} | "
        f"length vs expert {'n/a' if ratio is None else f'{ratio:.2f}'} | "
        f"end error {'n/a' if err is None else f'{err:.1f}px'}"
    )
  return result
