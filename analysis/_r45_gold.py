# -*- coding: utf-8 -*-
"""§4.5 D3 的文献口径复现 (Goldsztejn et al. 2023 包含规则) 在 v3 池下重算.

包含规则: 记录时孕周 >= 26 周 + 每名母亲取一条 (DS 的 26 条 -> 18 名母亲,
8 人各两条, 各取一条) + 合库 DB+DS.
在该子集上比较: (a) 记录孕周单变量 AUC; (b) v3 主配置全模型的 AUC
(DB 记录用 oof_pred, DS 记录用 ds_pred —— 两者都是"折外/预测集平均"口径).

输出: R0/v3_goldsztejn_repro_20260927.csv
"""
import os
import sys
import glob

# --- repository path bootstrap (added for the public release) -----------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "analysis"),
           os.path.join(_ROOT, "figures"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


import numpy as np
import pandas as pd

BASE = os.path.join(_ROOT)
sys.path.insert(0, BASE)
import run_experiment as R  # noqa: E402

TAG = "P1CI_P6eff050_direct"
EXPD = os.path.join(BASE, "experiments")
LIT = os.path.join(BASE, "results", "checkpoints_lit_features_v2", "lit_features_v2.csv")
MMAP = os.path.join(BASE, "results", "checkpoints_lit_features_v2", "tpehgt_ds_mother_map.csv")
OUT = os.path.join(BASE, "results", "v3_goldsztejn_repro_20260927.csv")
GOLD_WEEK = 26.0
SRC_DB, SRC_DS = "TPEHGDB", "TPEHGT"


def auc_ci_boot(y, p, n_boot=2000, seed=44):
    y = np.asarray(y, float); p = np.asarray(p, float)
    ok = ~np.isnan(p)
    y, p = y[ok], p[ok]
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(y)
    out = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        out.append(R.roc_auc_score(y[i], p[i]))
    lo = np.percentile(out, 2.5); hi = np.percentile(out, 97.5)
    return R.roc_auc_score(y, p), lo, hi


def main():
    lit = pd.read_csv(LIT)
    lit["src"] = lit["source"].astype(str).str.upper()
    db_lit = lit[lit.src == SRC_DB].reset_index(drop=True)
    ds_lit = lit[lit.src == SRC_DS].reset_index(drop=True)

    mm = pd.read_csv(MMAP)
    mm.columns = [c.strip().lstrip("\ufeff") for c in mm.columns]
    mother_of = mm.set_index("recname")["mother_id"].to_dict()

    files = sorted(glob.glob(os.path.join(EXPD, f"{TAG}_seed*", f"preds_{TAG}.npz")))
    if not files:
        print("!! 未找到预测文件"); return
    print(f"预测文件 {len(files)} 份")

    rows = []
    for fp in files:
        seed = os.path.basename(os.path.dirname(fp)).split("_seed")[1].split("_")[0]
        d = np.load(fp, allow_pickle=True)
        p_db = np.asarray(d["oof_pred"], float)
        p_ds = np.asarray(d["ds_pred"], float)
        assert len(p_db) == len(db_lit), "DB 预测行数与 lit 不符"
        assert len(p_ds) == len(ds_lit), "DS 预测行数与 lit 不符"

        def subset(frame, preds):
            # rectime = 记录时孕周(周); Goldsztejn 的 >=26 周是"记录时孕周"门槛,
            # 不是 lead (gest-rectime).
            f = frame[frame.rectime >= GOLD_WEEK].copy()
            f["y"] = f["label"].values.astype(int)
            f["p"] = preds[f.index.values].astype(float)
            f["s"] = f["rectime"].values.astype(float)
            return f

        sub = subset(db_lit, p_db)
        dsub = subset(ds_lit, p_ds)
        # 每名母亲取一条 (8 名母亲有两条, 各保留一条)
        dsub = dsub.groupby(dsub.recname.map(mother_of), sort=False).head(1)

        y = np.r_[sub.y.values, dsub.y.values]
        p = np.r_[sub.p.values, dsub.p.values]
        s = np.r_[sub.s.values, dsub.s.values]

        a_r, lo_r, hi_r = auc_ci_boot(y, s)
        a_m, lo_m, hi_m = auc_ci_boot(y, p)
        rows.append({
            "seed": seed, "n db": int(len(sub)), "n ds": int(len(dsub)),
            "rectime AUC": round(float(a_r), 4),
            "rectime CI": f"[{lo_r:.3f},{hi_r:.3f}]",
            "model AUC": round(float(a_m), 4),
            "model CI": f"[{lo_m:.3f},{hi_m:.3f}]",
            "delta (model-rectime)": round(float(a_m - a_r), 4),
        })

    df = pd.DataFrame(rows)
    print()
    print("=== 逐种子 (Goldsztejn 口径子集, v3 池) ===")
    print(df.to_string(index=False))
    print()
    for c in ["rectime AUC", "model AUC", "delta (model-rectime)"]:
        v = df[c].astype(float).values
        print(f"  {c:26s} mean={v.mean():.4f}  SD={v.std(ddof=1):.4f}  "
              f"min={v.min():.4f} max={v.max():.4f}")
    print(f"  子集规模 n_DB={df['n db'].iloc[0]} n_DS={df['n ds'].iloc[0]}")

    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n落盘: {OUT}")


if __name__ == "__main__":
    main()
