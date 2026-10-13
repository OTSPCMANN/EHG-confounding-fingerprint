# -*- coding: utf-8 -*-
"""§4.5 维度 D2 / D3 / D4 的 v3 口径重算 (纯计算, 复用已落盘的 OOF/DS 预测).

D2 队列来源: origin 二值指标的单独 AUC (记录级, 与特征池无关) + 队列基础率.
D3 记录时刻基线: rectime 单变量 AUC + 与全模型的配对 Delta AUC.
D4 临床分层: <37 / 早期 28-34 / 晚期 34-37 vs 足月 37-41 / >=41 的 AUC + CI.

输出: R0/v3_dims_D2D3D4_20260927.csv
"""
import os
import sys
import glob
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
import run_experiment as R  # noqa: E402

TAG = "P1CI_P6eff050_direct"
EXPD = os.path.join(BASE, "experiments")
OUT = os.path.join(BASE, "results", "v3_dims_D2D3D4_20260927.csv")


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
    files = sorted(glob.glob(os.path.join(EXPD, f"{TAG}_seed*", f"preds_{TAG}.npz")))
    if not files:
        print("!! 未找到预测文件, 请先跑 R4.2-1"); return
    print(f"预测文件 {len(files)} 份")

    rows = []
    for fp in files:
        seed = os.path.basename(os.path.dirname(fp)).split("_seed")[1].split("_")[0]
        d = np.load(fp, allow_pickle=True)
        y, p = d["y_all"], d["oof_pred"]
        yds = d["y_ds"]
        gest, rect = d["gest_all"], d["rect_all"]

        # --- D3: rectime 单变量 ---
        a_r, lo_r, hi_r = auc_ci_boot(y, rect)
        # --- 全模型 OOF ---
        a_m, lo_m, hi_m = auc_ci_boot(y, p)

        # --- D2: origin 的判别力 (记录级, 与特征池无关) ---
        # 用手稿的口径: 把"队列条件早产率"当作该记录的预测概率
        #   p_i = P(preterm | 该记录所属队列)
        # 则 origin 指标的 AUC = 随机取一对(DS记录, DB记录) 时前者 label 更高的概率.
        n_ds, n_db = len(yds), len(y)
        p_ds = float(yds.mean()); p_db = float(y.mean())
        score = np.r_[np.full(n_ds, p_ds), np.full(n_db, p_db)]
        yy = np.r_[yds, y]
        a_o = R.roc_auc_score(yy, score)

        # --- D4: 临床分层 ---
        def acc(mask_pos, mask_ctl):
            ii = np.where(mask_pos | mask_ctl)[0]
            if len(np.unique(y[ii])) < 2:
                return np.nan, np.nan, np.nan, 0, 0
            a_, l_, h_ = auc_ci_boot(y[ii], p[ii])
            return a_, l_, h_, int(mask_pos[ii].sum()), int(mask_ctl[ii].sum())

        all_pt = (gest < 37) & (y == 1)
        all_term = (gest >= 37) & (y == 0)
        early = (gest >= 28) & (gest < 34) & (y == 1)
        late = (gest >= 34) & (gest < 37) & (y == 1)
        lt35 = (gest < 35) & (y == 1)
        term37 = (gest >= 37) & (gest < 41) & (y == 0)
        post41 = (gest >= 41) & (y == 0)

        rows.append({
            "seed": seed,
            # D2
            "preterm rate DB": round(p_db, 4),
            "preterm rate DS": round(p_ds, 4),
            "origin indicator AUC": round(float(a_o), 4),
            # D3
            "rectime-only AUC": round(float(a_r), 4),
            "rectime CI": f"[{lo_r:.3f},{hi_r:.3f}]",
            "model OOF AUC": round(float(a_m), 4),
            "model CI": f"[{lo_m:.3f},{hi_m:.3f}]",
            "delta (model-rectime)": round(float(a_m - a_r), 4),
            # D4
            "all preterm <37": round(
                float(R.roc_auc_score(y[all_pt | all_term], p[all_pt | all_term])), 4),
            # 统一控制集: 所有临床分层对比均以 all-term (gest>=37, n=262) 为对照,
            # 与聚合行 "all preterm <37" 及 S4.4 strict 估计保持一致。
            "early 28-34 vs term": f"{R.roc_auc_score(y[early|all_term], p[early|all_term]):.4f}"
                                   if len(np.unique(y[early | all_term])) > 1 else "NA",
            "late 34-37 vs term": f"{R.roc_auc_score(y[late|all_term], p[late|all_term]):.4f}"
                                  if len(np.unique(y[late | all_term])) > 1 else "NA",
            "<35 weeks vs term (non-clinical)": (
                f"{R.roc_auc_score(y[lt35|all_term], p[lt35|all_term]):.4f}"
                if len(np.unique(y[lt35 | all_term])) > 1 else "NA"),
            "n early pos": int(early.sum()), "n late pos": int(late.sum()),
            "n term37": int(term37.sum()), "n post41": int(post41.sum()),
        })

    df = pd.DataFrame(rows)
    num = ["preterm rate DB", "origin indicator AUC", "rectime-only AUC",
           "model OOF AUC", "delta (model-rectime)"]
    print()
    print("=== 逐种子 ===")
    print(df.to_string(index=False))
    print()
    print("=== 10 种子均值 ===")
    for c in num:
        print(f"  {c:28s} {df[c].mean():.4f}  (SD {df[c].std(ddof=1):.4f})")

    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n落盘: {OUT}")


if __name__ == "__main__":
    main()
