# -*- coding: utf-8 -*-
"""汇总分类器家族 (cllogreg / clsvm / clmlp) 的逐种子结果, 自动去重同 (config,seed) 的重复批次.

与 _summarize_runs.py 的区别: 同一 config+seed 可能有多批(任务中断重跑),
这里按目录名时间戳保留**最新**的一批, 避免把中间态混进均值.
"""
import os
import re
import sys
import glob
import datetime
import pandas as pd

# --- repository path bootstrap (added for the public release) -----------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "analysis"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


from _summarize_runs import parse  # noqa: E402

BASE = os.path.join(_ROOT)
EXPD = os.path.join(BASE, "experiments")


def summarize(cfgs, day):
    rows = {}
    for cfg in cfgs:
        pat = os.path.join(EXPD, f"{cfg}_seed*_{day}_*", "pipe_03_sfs_*.xlsx")
        for fp in sorted(glob.glob(pat)):
            sd = os.path.basename(os.path.dirname(fp))
            m = re.search(r"_seed(\d+)_(\d{8})_(\d{6})$", sd)
            if not m:
                continue
            seed, d8, t6 = int(m.group(1)), m.group(2), m.group(3)
            p = parse(fp)
            if p is None:
                continue
            key = (cfg, seed)
            # 同 seed 多批: 保留时间戳更大的
            if key in rows and rows[key][0] >= (d8, t6):
                continue
            rows[key] = ((d8, t6), {"config": cfg, "seed": seed, "dir": sd} | p)
    df = pd.DataFrame([v[1] for v in rows.values()])
    df = df.sort_values(["config", "seed"]).reset_index(drop=True)
    print(df.to_string(index=False))
    print()
    g = df.groupby("config").agg(
        n=("seed", "size"),
        DB_mean=("DB mean", "mean"), DB_sd=("DB mean", "std"),
        DB_OOF=("DB OOF", "mean"), DB_OOF_sd=("DB OOF", "std"),
        DS_mean=("DS mean", "mean"), DS_sd=("DS mean", "std"),
        unionFk=("union Fk", "mean"),
    ).round(4)
    print("=== 家族汇总 ===")
    print(g.to_string())
    return df, g


if __name__ == "__main__":
    day = sys.argv[1] if len(sys.argv) > 1 else datetime.date.today().strftime("%Y%m%d")
    cfgs = sys.argv[2:] or ["P1CI_P6eff050_direct_cllogreg",
                            "P1CI_P6eff050_direct_clsvm",
                            "P1CI_P6eff050_direct_clmlp"]
    df, g = summarize(cfgs, day)
    if not df.empty:
        out = os.path.join(BASE, "results", f"v3_clsummary_{day}.csv")
        df.to_csv(out, index=False, encoding="utf-8-sig")
        g.to_csv(out.replace(".csv", "_agg.csv"), encoding="utf-8-sig")
        print(f"\n落盘: {out}")
