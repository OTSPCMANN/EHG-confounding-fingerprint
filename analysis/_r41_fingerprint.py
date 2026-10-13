# -*- coding: utf-8 -*-
"""§4.1 Table 4 / Figure 3 的 v3 口径重算.

判定标准与手稿一致: bootstrap 95% CI 排除 0.5 记为显著.
纯计算, 不训练任何模型. 复用 run_experiment 的 build_matrix / pair_auc_ci.
输出到 R0/v3_table4_fingerprint_20260927.csv
"""
import os
import sys
import numpy as np
import pandas as pd

# --- repository path bootstrap (added for the public release) -----------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "analysis"),
           os.path.join(_ROOT, "figures"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


BASE = os.path.join(_ROOT)
sys.path.insert(0, BASE)
os.environ.setdefault("EHG_FT_CSV", os.path.join(
    BASE, "results", "checkpoints_lit_features_v3", "lit_features_v3.csv"))
import run_experiment as R  # noqa: E402

OUT = os.path.join(BASE, "results", "v3_table4_fingerprint_20260927.csv")


def main():
    db, ds, ALL_COLS = R.load_data()
    y = db.label.values.astype(int)
    lead = db.lead.values.astype(float)
    rect = db.rectime.values.astype(float)
    X = db[ALL_COLS].values.astype(np.float64)
    print(f"DB {len(db)} 条, 特征 {len(ALL_COLS)}")

    long, wide = R.build_matrix(X, y, lead, rect, ALL_COLS)

    # 显著判定: CI 排除 0.5
    long["sig"] = (long["LO"] > 0.5) | (long["HI"] < 0.5)
    long["dev"] = (long["AUC"] - 0.5).abs()

    rows = []
    for pn in ["P1", "P2", "P3", "P4", "P5", "P6"]:
        sub = long[long["Pair"] == pn]
        n_sig = int(sub["sig"].sum())
        rows.append({
            "Pairing": pn,
            "Type": {"P1": "Time-unmatched", "P2": "Time-matched",
                     "P3": "Time-unmatched", "P4": "Time-matched",
                     "P5": "Within-outcome (ref)", "P6": "Proxy probe"}[pn],
            "Significant (n/189)": n_sig,
            "Significant %": round(100.0 * n_sig / len(sub), 1),
            "Mean |AUC-0.5|": round(float(sub["dev"].mean()), 4),
            "Median |AUC-0.5|": round(float(sub["dev"].median()), 4),
            "Max |AUC-0.5|": round(float(sub["dev"].max()), 4),
        })
    tbl = pd.DataFrame(rows)
    print()
    print("=== §4.1 Table 4 (v3 池 189) ===")
    print(tbl.to_string(index=False))

    # 按通道分解 P6
    long["chan"] = long["Column Name"].str.extract(r"_s([1-6])$")[0].fillna("cross")
    p6 = long[long["Pair"] == "P6"]
    ch = (p6.groupby("chan")
            .agg(n=("sig", "size"), n_sig=("sig", "sum"),
                 mean_dev=("dev", "mean"))
            .reset_index())
    ch["sig %"] = (100 * ch["n_sig"] / ch["n"]).round(1)
    ch["mean_dev"] = ch["mean_dev"].round(4)
    print()
    print("=== P6 显著数按通道分解 ===")
    print(ch.to_string(index=False))

    long.to_csv(OUT.replace(".csv", "_long.csv"), index=False, encoding="utf-8-sig")
    tbl.to_csv(OUT, index=False, encoding="utf-8-sig")
    ch.to_csv(OUT.replace(".csv", "_bychan.csv"), index=False, encoding="utf-8-sig")
    print(f"\n落盘: {OUT}")


if __name__ == "__main__":
    main()
