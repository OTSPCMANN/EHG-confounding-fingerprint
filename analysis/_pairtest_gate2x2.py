# -*- coding: utf-8 -*-
"""Paired Wilcoxon tests for the 2x2 admission-gate factorial (same 10 seeds)."""
import sys
import glob, os, sys
import numpy as np
from scipy.stats import wilcoxon

# --- repository path bootstrap (added for the public release) -----------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "analysis"),
           os.path.join(_ROOT, "figures"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


sys.path.insert(0, os.path.join(_ROOT, "results"))
import _summarize_runs as S

EXPD = os.path.join(_ROOT, "experiments") + os.sep
SEEDS = [40, 41, 42, 456, 789, 2024, 2025, 2026, 2027, 2028]


def per_seed(cfg):
    out = {}
    for d in glob.glob(EXPD + cfg + "_seed*"):
        fps = glob.glob(os.path.join(d, "pipe_03_sfs_*.xlsx"))
        if not fps:
            continue
        p = S.parse(fps[0])
        if p is None:
            continue
        parts = os.path.basename(d).split("_seed")[1].split("_")
        seed = int(parts[0])
        stamp = "_".join(parts[1:])
        if seed not in out or stamp > out[seed][0]:
            out[seed] = (stamp, p)
    return {k: v[1] for k, v in out.items()}


CFG = {"CI/CI": "P1CI_P6CI_direct",
       "CI/EFF": "P1CI_P6eff050_direct",
       "EFF/CI": "P1eff_P6CI_direct",
       "EFF/EFF": "P1eff_P6eff050_direct"}

data = {k: per_seed(v) for k, v in CFG.items()}
for k in CFG:
    assert set(data[k]) == set(SEEDS), (k, sorted(data[k]))

pairs = [("CI/EFF", "EFF/CI"), ("CI/EFF", "EFF/EFF"),
         ("CI/CI", "EFF/CI"), ("CI/CI", "EFF/EFF")]
rows = []
for a, b in pairs:
    for metric in ["DB OOF", "DS mean"]:
        va = np.array([data[a][s][metric] for s in SEEDS])
        vb = np.array([data[b][s][metric] for s in SEEDS])
        diff = va - vb          # positive = first config higher
        stat, p = wilcoxon(va, vb)
        wins = int((diff > 0).sum())
        rows.append(f"{a:8s} vs {b:8s}  {metric:8s} "
                    f"mean diff {diff.mean():+.4f}  p={p:.4f}  first wins {wins}/10")
print("\n".join(rows))

import io
io.open(os.path.join(_ROOT, "results", "v3_pairtest_gate2x2_20260928.csv"), "w",
        encoding="utf-8").write(
        "pair,metric,mean_diff,p,first_wins\n" +
        "\n".join(f'"{a} vs {b}",{m},{(np.array([data[a][s][m] for s in SEEDS])-np.array([data[b][s][m] for s in SEEDS])).mean():.4f},{wilcoxon([data[a][s][m] for s in SEEDS],[data[b][s][m] for s in SEEDS])[1]:.4f},{int((np.array([data[a][s][m] for s in SEEDS])-np.array([data[b][s][m] for s in SEEDS])>0).sum())}/10'
                  for a, b in pairs for m in ["DB OOF", "DS mean"]) + "\n")
print("SAVED v3_pairtest_gate2x2_20260928.csv")
