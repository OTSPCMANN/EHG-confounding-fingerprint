# -*- coding: utf-8 -*-
"""D4 足月对照的 42 周拆分可行性分析 (只读计算, 复用已落盘 OOF 预测).

在开发集 (TPEHGDB) 上, 把 262 条足月对照 (gest>=37, y==0) 按 42 周拆成两组:
  T_37_42 : 37 <= gest < 42  (临床足月)
  T_42    : gest >= 42       (过期产, post-term)
再对每个早产分层 (排除 <28) 分别与这两组各自单独做 vs 的 AUC:
  all preterm <37 (38) / early 28-34 (10) / late 34-37 (27) / <35 (17)

输出: R0/v3_dims_term42_20261012.csv
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
OUT = os.path.join(BASE, "results", "v3_dims_term42_20261012.csv")


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
        print("!! 未找到预测文件"); return
    print(f"预测文件 {len(files)} 份\n")

    # --- 先看构成 (只与 gest/label 有关, 与 seed 无关) ---
    d0 = np.load(files[0], allow_pickle=True)
    y0, gest0 = d0["y_all"], d0["gest_all"]
    term = (gest0 >= 37) & (y0 == 0)
    t37_42 = (gest0 >= 37) & (gest0 < 42) & (y0 == 0)
    t42 = (gest0 >= 42) & (y0 == 0)
    print("=== 足月对照 262 的构成 ===")
    print(f"  全部足月 (>=37)      n={int(term.sum())}")
    print(f"  37-42               n={int(t37_42.sum())}")
    print(f"  >=42 (post-term)    n={int(t42.sum())}  gest 值: {sorted(set(np.round(gest0[t42],1)))}")
    print(f"  >=41                n={int(((gest0>=41)&(y0==0)).sum())}")
    print(f"  早产 <37            n={int(((gest0<37)&(y0==1)).sum())}")
    print(f"  早产 <28            n={int(((gest0<28)&(y0==1)).sum())}")

    strata = {
        "All preterm <37": (gest0 < 37) & (y0 == 1),
        "Early preterm 28-34": (gest0 >= 28) & (gest0 < 34) & (y0 == 1),
        "Late preterm 34-37": (gest0 >= 34) & (gest0 < 37) & (y0 == 1),
        "<35 weeks": (gest0 < 35) & (y0 == 1),
    }
    # 42 周拆分不可行 (>=42 仅 1 条); 补充临床可用的 41 周边界 (late term 41-42 + post-term >=42 = 29 条)
    t37_41 = (gest0 >= 37) & (gest0 < 41) & (y0 == 0)
    t41 = (gest0 >= 41) & (y0 == 0)
    print(f"  37-41               n={int(t37_41.sum())}")
    print(f"  >=41                n={int(t41.sum())}")
    ctrls = {"vs T(37-42)": t37_42, "vs T(>=42)": t42, "vs all term >=37": term,
             "vs T(37-41)": t37_41, "vs T(>=41)": t41}

    rows = []
    for fp in files:
        seed = os.path.basename(os.path.dirname(fp)).split("_seed")[1].split("_")[0]
        d = np.load(fp, allow_pickle=True)
        y, p = d["y_all"], d["oof_pred"]
        for sname, smask in strata.items():
            rec = {"seed": seed, "stratum": sname}
            for cname, cmask in ctrls.items():
                ii = np.where(smask | cmask)[0]
                if len(np.unique(y[ii])) < 2:
                    rec[cname] = np.nan
                else:
                    rec[cname] = R.roc_auc_score(y[ii], p[ii])
            rows.append(rec)

    df = pd.DataFrame(rows)
    print("\n=== 逐种子 AUC ===")
    print(df.to_string(index=False))
    print("\n=== 10 种子汇总 (mean [min, max]) ===")
    for sname in strata:
        g = df[df.stratum == sname]
        print(f"  {sname}")
        for cname in ctrls:
            v = g[cname].astype(float).values
            print(f"     {cname:18s} {v.mean():.4f}  [{v.min():.4f}, {v.max():.4f}]")

    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n落盘: {OUT}")


if __name__ == "__main__":
    main()