# -*- coding: utf-8 -*-
"""§4.4 in-sample 上界 (v3 口径, 等价于手稿 T4).

定义: 全池 189 特征, 全量 DB(300) 训练, 同一批记录上评估 —— 无任何划分.
这是"没有留出任何验证信息"时的 AUC 上界, 用作量化评估膨胀的标尺.

输出: R0/v3_insample_20260927.csv  以及逐种子预测 npz (供 Figure 5 / bootstrap 复用)
"""
import os
import sys
import glob
import time
import numpy as np
import pandas as pd

# --- repository path bootstrap (added for the public release) -----------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "analysis"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


BASE = os.path.join(_ROOT)
sys.path.insert(0, BASE)
os.environ.setdefault("EHG_FT_CSV", os.path.join(
    BASE, "results", "checkpoints_lit_features_v3", "lit_features_v3.csv"))
import run_experiment as R  # noqa: E402

SEEDS = [40, 41, 42, 456, 789, 2024, 2025, 2026, 2027, 2028]
OUT = os.path.join(BASE, "results", "v3_insample_20260927.csv")


def main():
    db, ds, ALL_COLS = R.load_data()
    X = db[ALL_COLS].values.astype(np.float64)
    y = db.label.values.astype(int)
    print(f"全池 {len(ALL_COLS)} 特征, DB {len(y)} 条 (pos {int(y.sum())})")

    rows, preds = [], {}
    for s in SEEDS:
        t0 = time.time()
        # impute_median(X, X): 全量中位数插补 (无划分场景)
        Xi, _ = R.impute_median(X, X)
        p = R.rfgb_proba(Xi, y, Xi, s)          # 训练与评估同一批记录
        a = R.roc_auc_score(y, p)
        _, lo, hi = R.auc_ci(y, p)
        rows.append({"seed": s, "in-sample AUC": round(float(a), 4),
                     "boot_lo": round(float(lo), 4), "boot_hi": round(float(hi), 4),
                     "sec": round(time.time() - t0, 0)})
        preds[s] = p
        print(f"  seed {s}: {a:.4f}  [{lo:.3f},{hi:.3f}]  ({time.time()-t0:.0f}s)")

    d = pd.DataFrame(rows)
    v = d["in-sample AUC"].values
    print()
    print(f"in-sample: mean={v.mean():.4f}  SD={v.std(ddof=1):.4f}  "
          f"range=[{v.min():.4f}, {v.max():.4f}]")

    # 汇总落盘
    tot_n = len(v)
    grand = np.concatenate([preds[s] for s in SEEDS])
    ya = np.tile(y, tot_n)
    m, lo, hi = R.auc_ci(ya, grand, n_boot=2000, seed=44)
    print(f"池化(10种子×300条) bootstrap 95% CI: [{lo:.4f}, {hi:.4f}]")
    d.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"落盘: {OUT}")

    # 预测落盘供 Figure 5 / §4.4 直接用
    np.savez(os.path.join(BASE, "results", "v3_insample_preds_20260927.npz"),
             y_all=y, pred=grand)
    print("落盘: R0/v3_insample_preds_20260927.npz")


if __name__ == "__main__":
    main()
