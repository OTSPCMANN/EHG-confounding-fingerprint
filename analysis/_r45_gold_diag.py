# -*- coding: utf-8 -*-
"""Goldsztejn 口径子集上"方向反转"的诊断: 是真信号还是小样本/病例构成伪影?

在 _r45_gold.py 报出 (子集 rectime AUC 0.3196 vs 全模型 0.5937, delta +0.274) 之后,
需要回答三件事:
  Q1 子集病例构成变了没有? (剔除 rectime<26 会机械改变患病率 -> AUC 口径不可直接横比)
  Q2 rectime 在子集上为什么会翻到随机以下? (截断效应 / 病例构成 / 低功效)
  Q3 delta 的抽样波动有多大? (逐种子范围 + bootstrap CI of delta)

同时给出"同人群内"的口径: 即在**全库**上算 rectime 与 model, 再与子集并排,
用来说明 AUC 只在同构成人群内可比.

输出: R0/v3_gold_diag_20260927.csv
"""
import os
import sys
import glob

# --- repository path bootstrap (added for the public release) -----------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "analysis"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


import numpy as np
import pandas as pd

BASE = os.path.join(_ROOT)
sys.path.insert(0, BASE)
import run_experiment as R  # noqa: E402

TAG = "P1CI_P6eff050_direct"
EXPD = os.path.join(BASE, "experiments")
LIT_V3 = os.path.join(BASE, "results", "checkpoints_lit_features_v3", "lit_features_v3.csv")
LIT_V2 = os.path.join(BASE, "results", "checkpoints_lit_features_v2", "lit_features_v2.csv")
MMAP = os.path.join(BASE, "results", "checkpoints_lit_features_v2", "tpehgt_ds_mother_map.csv")
OUT = os.path.join(BASE, "results", "v3_gold_diag_20260927.csv")
GOLD_WEEK = 26.0
SRC_DB, SRC_DS = "TPEHGDB", "TPEHGT"


def boot_delta(y, a, b, n_boot=2000, seed=45):
    """bootstrap CI of (AUC_b - AUC_a) — 两者共用同一批重抽样, 故区间是可比的."""
    y = np.asarray(y, float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    y, a, b = y[ok], np.asarray(a, float)[ok], np.asarray(b, float)[ok]
    rng = np.random.default_rng(seed)
    n = len(y)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        d.append(R.roc_auc_score(y[i], b[i]) - R.roc_auc_score(y[i], a[i]))
    lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
    return R.roc_auc_score(y, b) - R.roc_auc_score(y, a), lo, hi


def describe(tag, y, s, p):
    y = np.asarray(y, float)
    n_pos = int(y.sum())
    a_s = R.roc_auc_score(y, s) if 0 < n_pos < len(y) else np.nan
    a_p = R.roc_auc_score(y, p) if 0 < n_pos < len(y) else np.nan
    d, lo, hi = boot_delta(y, s, p)
    return {
        "scope": tag, "n": len(y), "n_pos": n_pos,
        "prev": round(float(y.mean()), 4),
        "mean_rectime_pos": round(float(np.asarray(s)[y == 1].mean()), 2),
        "mean_rectime_neg": round(float(np.asarray(s)[y == 0].mean()), 2),
        "rectime AUC": round(float(a_s), 4),
        "model AUC": round(float(a_p), 4),
        "delta": round(float(d), 4), "delta CI": f"[{lo:.3f},{hi:.3f}]",
        "delta 含0": "yes" if lo <= 0 <= hi else "no",
    }


def main():
    v3 = pd.read_csv(LIT_V3)
    v2 = pd.read_csv(LIT_V2)
    v3["src"] = v3["source"].astype(str).str.upper()
    v2["src"] = v2["source"].astype(str).str.upper()
    same = (set(v3.recname) == set(v2.recname)) and (len(v3) == len(v2) == 326)
    print(f"记录集一致性 (v3 vs v2 meta): {same}  n_v3={len(v3)} n_v2={len(v2)}")
    if not same:
        print("!! 记录集不同, 停止"); return
    db = v3[v3.src == SRC_DB].reset_index(drop=True)
    ds = v3[v3.src == SRC_DS].reset_index(drop=True)

    mm = pd.read_csv(MMAP)
    mm.columns = [c.strip().lstrip("\ufeff") for c in mm.columns]
    mother_of = mm.set_index("recname")["mother_id"].to_dict()

    files = sorted(glob.glob(os.path.join(EXPD, f"{TAG}_seed*", f"preds_{TAG}.npz")))
    print(f"预测文件 {len(files)} 份\n")

    rows = []
    for fp in files:
        seed = os.path.basename(os.path.dirname(fp)).split("_seed")[1].split("_")[0]
        d = np.load(fp, allow_pickle=True)
        p_db = np.asarray(d["oof_pred"], float)
        p_ds = np.asarray(d["ds_pred"], float)
        assert len(p_db) == len(db) and len(p_ds) == len(ds)

        def sub(frame, preds):
            f = frame[frame.rectime >= GOLD_WEEK].copy()
            f["y"] = f["label"].values.astype(int)
            f["p"] = preds[f.index.values].astype(float)
            f["s"] = f["rectime"].values.astype(float)
            return f

        sdb, sds = sub(db, p_db), sub(ds, p_ds)
        sds = sds.groupby(sds.recname.map(mother_of), sort=False).head(1)

        # (A) 全库同时点(未做任何子集化)
        rows.append(describe("全库 (DB oof + DS 预测集平均)",
                             np.r_[db.label.values.astype(int), ds.label.values.astype(int)],
                             np.r_[db.rectime.values, ds.rectime.values],
                             np.r_[p_db, p_ds]))
        # (B) Goldsztejn 子集
        rows.append(describe("Goldsztejn 子集 (rectime>=26, 每母一条)",
                             np.r_[sdb.y.values, sds.y.values],
                             np.r_[sdb.s.values, sds.s.values],
                             np.r_[sdb.p.values, sds.p.values]))
        # (C) 仅 DB 的子集 (排除 DS 病例构成差异)
        rows.append(describe("Goldsztejn 子集 · 仅 DB",
                             sdb.y.values, sdb.s.values, sdb.p.values))
        # (D) 仅 DS 的子集
        rows.append(describe("Goldsztejn 子集 · 仅 DS",
                             sds.y.values, sds.s.values, sds.p.values))

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    print("\n=== 逐种子: Goldsztejn 子集 delta ===")
    _, _, sub = [], None, None
    g = df[df.scope.str.startswith("Goldsztejn 子集 (")].reset_index(drop=True)
    for c in ["rectile", "model AUC", "delta"]:
        pass
    for c in ["rectime AUC", "model AUC", "delta"]:
        v = g[c].astype(float).values
        print(f"  {c:14s} mean={v.mean():.4f} SD={v.std(ddof=1):.4f} "
              f"[{v.min():.4f},{v.max():.4f}]")
    print(f"  子集 n_pos 逐种子: {g['n_pos'].unique().tolist()} / n={g['n'].unique().tolist()}")

    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n落盘: {OUT}")


if __name__ == "__main__":
    main()
