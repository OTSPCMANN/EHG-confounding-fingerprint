# -*- coding: utf-8 -*-
"""§4.2 主性能的完整不确定性报告 (v3 口径, 兑现 §2.4 的 cluster bootstrap 承诺).

1) DB 开发期: 记录级 bootstrap 95% CI(池化 10 种子 OOF); 折间/种子间离散.
2) DS 外部集: 按女性 cluster bootstrap (重抽 18 名女性, 整簇保留) 95% CI.
3) ROC 曲线数据 (10 种子 OOF 的平均曲线 + 包络; DS 平均曲线).
输出: R0/v3_uncertainty_20260927.csv
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

# --- repository path bootstrap (added for the public release) -----------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "analysis"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


BASE = os.path.join(_ROOT)
sys.path.insert(0, BASE)
import run_experiment as R  # noqa: E402

TAG = "P1CI_P6eff050_direct"
EXPD = os.path.join(BASE, "experiments")
OUT = os.path.join(BASE, "results", "v3_uncertainty_20260927.csv")
OUT_ROC = os.path.join(BASE, "results", "v3_roc_20260927.csv")
B = 2000


def boot_ci(y, p, n_boot=B, seed=44, clusters=None):
    """记录级 bootstrap; 若给 clusters, 则做 cluster bootstrap(整簇重抽)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    if clusters is None:
        idx_pool = np.arange(n)
    else:
        idx_pool = [np.where(clusters == c)[0] for c in np.unique(clusters)]
    vals = []
    for _ in range(n_boot):
        if clusters is None:
            i = rng.integers(0, n, n)
        else:
            pick = rng.integers(0, len(idx_pool), len(idx_pool))
            i = np.concatenate([idx_pool[k] for k in pick])
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    files = sorted(glob.glob(os.path.join(EXPD, f"{TAG}_seed*", f"preds_{TAG}.npz")))
    if len(files) < 10:
        print(f"!! 预测只有 {len(files)} 份, 需 10"); return
    Y, P, YDS, PDS, MID = [], [], [], [], []
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        Y.append(d["y_all"]); P.append(d["oof_pred"])
        YDS.append(d["y_ds"]); PDS.append(d["ds_pred"]); MID.append(d["mother_id"])
    y = np.concatenate(Y); p = np.concatenate(P)
    yds = np.concatenate(YDS); pds = np.concatenate(PDS); mid = np.concatenate(MID)
    print(f"池化: DB {y.shape[0]} 条(×{len(files)}种子), DS {yds.shape[0]} 条(×{len(files)})")

    # --- DB: 记录级 bootstrap ---
    db_lo, db_hi = boot_ci(y, p, clusters=None)
    # --- DB: 种子间 OOF AUC 的分布 (逐种子记录级 AUC, 非池化) ---
    per_seed = [roc_auc_score(d["y_all"], d["oof_pred"]) for d in
                (np.load(f, allow_pickle=True) for f in files)]
    per_seed = np.array(per_seed)

    # --- DS: 女性 cluster bootstrap ---
    cin = np.array([np.where(mid == m)[0] for m in np.unique(mid)], dtype=object)
    ds_lo, ds_hi = boot_ci(yds, pds, clusters=mid)
    # 参照: 记录级 bootstrap(会低估)
    r_lo, r_hi = boot_ci(yds, pds, clusters=None)

    rows = [{
        "metric": "DB OOF AUC (record-level, pooled 10 seeds)",
        "point": round(float(roc_auc_score(y, p)), 4),
        "ci_low": round(db_lo, 4), "ci_high": round(db_hi, 4),
        "note": "record-level bootstrap B=2000",
    }, {
        "metric": "DB OOF AUC (per-seed mean)",
        "point": round(float(per_seed.mean()), 4),
        "ci_low": round(float(per_seed.min()), 4), "ci_high": round(float(per_seed.max()), 4),
        "note": f"SD={per_seed.std(ddof=1):.4f}; 区间是 10 颗种子的 min-max",
    }, {
        "metric": "DS AUC (woman-level cluster bootstrap)",
        "point": round(float(roc_auc_score(yds, pds)), 4),
        "ci_low": round(ds_lo, 4), "ci_high": round(ds_hi, 4),
        "note": "18 women, 整簇重抽 B=2000",
    }, {
        "metric": "DS AUC (record-level bootstrap, 参照)",
        "point": round(float(roc_auc_score(yds, pds)), 4),
        "ci_low": round(r_lo, 4), "ci_high": round(r_hi, 4),
        "note": "未做 cluster 校正, 作为低估参照",
    }]
    df = pd.DataFrame(rows)
    print()
    print(df.to_string(index=False))

    # --- ROC 数据 ---
    roc_rows = []
    fpr_db, tpr_db, _ = roc_curve(y, p)
    for i in range(0, len(fpr_db), max(1, len(fpr_db) // 60)):
        roc_rows.append({"set": "DB OOF (pooled)", "fpr": float(fpr_db[i]),
                         "tpr": float(tpr_db[i])})
    fpr_d, tpr_d, _ = roc_curve(yds, pds)
    for i in range(0, len(fpr_d), max(1, len(fpr_d) // 60)):
        roc_rows.append({"set": "DS cross-dataset", "fpr": float(fpr_d[i]),
                         "tpr": float(tpr_d[i])})
    rdf = pd.DataFrame(roc_rows)
    rdf.to_csv(OUT_ROC, index=False, encoding="utf-8-sig")

    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n落盘: {OUT}   ROC: {OUT_ROC}")
    print(f"DS 女性数 = {len(np.unique(mid))}, 每女性记录数 = "
          f"{ {m: int((mid == m).sum()) for m in np.unique(mid)} }")


if __name__ == "__main__":
    main()
