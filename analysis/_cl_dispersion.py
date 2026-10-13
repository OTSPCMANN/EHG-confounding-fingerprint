# -*- coding: utf-8 -*-
"""集成 (RF+GB) vs 三个单家族: 逐种子离散度对比, 供 §4.3 引用.

动机: 配对 Wilcoxon 已显示集成在外部队列 10/10 胜出三个家族全部.
若集成的优势是方差抑制而非判别力, 则集成的逐种子 SD 应显著更小,
且 DB (开发期, n=300) 上差异小、DS (外部队列, n=26) 上差异大.
"""
import sys
import pandas as pd

# --- repository path bootstrap (added for the public release) -----------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "analysis"),
           os.path.join(_ROOT, "figures"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


B = os.path.join(_ROOT, "results") + os.sep
ENS = "P1CI_P6eff050_direct"
assert ENS.startswith("P1CI")
KIND = {"cllogreg": "logistic regression",
        "clsvm": "SVM (RBF)",
        "clmlp": "MLP (32 units)"}

ens = pd.read_csv(B + "v3_runs_summary.csv")
ens = ens[ens.config == ENS].sort_values("seed")
fam = pd.read_csv(B + "v3_clsummary_20260927.csv")

METRICS = ["DB mean", "DB OOF", "DS mean"]
rows = []
for m in METRICS:
    e = ens[m].values.astype(float)
    row = {"metric": m, "ens_mean": e.mean(), "ens_SD": e.std()}
    for k in KIND:
        row[k + "_SD"] = fam[fam.config.str.endswith(k)][m].std()
    row["worst_SD"] = max(row[k + "_SD"] for k in KIND)
    row["min_SD_reduction"] = row["worst_SD"] / row["ens_SD"]
    rows.append(row)
d = pd.DataFrame(rows)

print("=== 逐种子离散度 (集成 vs 三家族) ===")
print(d.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print()
print("=== 家族 SD / 集成 SD ===")
for m in METRICS:
    e = float(d[d.metric == m].ens_SD.iloc[0])
    parts = [f"{KIND[k]}={float(d[d.metric == m][k + '_SD'].iloc[0]) / e:.2f}x" for k in KIND]
    print(f"  {m:8s} ens_SD={e:.4f}   " + "  ".join(parts))

out = B + "v3_cl_dispersion_20260927.csv"
d.to_csv(out, index=False, encoding="utf-8-sig")
print("\n落盘:", out)
