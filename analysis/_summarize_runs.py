# -*- coding: utf-8 -*-
"""把若干配置的逐种子结果汇总成一张表 (§4.2 / §4.3 / §4.4 共用).

从 experiments/<cfg>[_<sel>]_seed<S>_*/pipe_03_sfs_*.xlsx 的 Overview sheet 提取:
  DB AUC (unbiased) [mean±SD] / DB OOF AUC / DS AUC (unbiased) [mean±SD]
输出: R0/v3_runs_summary_{日期}.csv
"""
import os
import re
import sys
import glob
import pandas as pd

# --- repository path bootstrap (added for the public release) -----------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "analysis"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


BASE = os.path.join(_ROOT)
EXPD = os.path.join(BASE, "experiments")


def parse(path):
    try:
        d = pd.read_excel(path, sheet_name="Overview")
    except Exception:
        return None
    if d.empty:
        return None
    r = d.iloc[0]
    def g(k):
        return r.get(k, None)
    m = re.match(r"^([\d.]+)±([\d.]+)$", str(g("DB AUC (unbiased)")))
    m2 = re.match(r"^([\d.]+)±([\d.]+)$", str(g("DS AUC (unbiased)")))
    # intersection Fk 存的是逗号分隔的特征名列表, 不是数字
    _iv = g("intersection Fk")
    if isinstance(_iv, str) and _iv.strip():
        n_iv = len([x for x in _iv.split(",") if x.strip()])
    else:
        n_iv = None
    return {
        "DB mean": float(m.group(1)) if m else None,
        "DB SD": float(m.group(2)) if m else None,
        "DB OOF": float(g("DB OOF AUC")) if pd.notna(g("DB OOF AUC")) else None,
        "DS mean": float(m2.group(1)) if m2 else None,
        "DS SD": float(m2.group(2)) if m2 else None,
        "DB per fold": str(g("DB per fold")),
        "union Fk": int(g("union Fk size")) if pd.notna(g("union Fk size")) else None,
        "intersection Fk_n": n_iv,
    }


def main(cfgs, out_name=None, day=None):
    # 只统计指定日期的目录, 避免把同名的更早批次(如 20260924)混进来
    import datetime
    day = day or datetime.date.today().strftime("%Y%m%d")
    # 同一配置可能有多批(不同口径/中途改名的家族), 用 EHG_SUM_SINCE 限定目录名子串
    SINCE = os.environ.get("EHG_SUM_SINCE", "")
    rows = []
    for cfg in cfgs:
        pat = os.path.join(EXPD, f"{cfg}_seed*_{day}_*", "pipe_03_sfs_*.xlsx")
        for fp in sorted(glob.glob(pat)):
            sd = os.path.basename(os.path.dirname(fp))
            if SINCE and SINCE not in sd:
                continue
            m = re.search(r"_seed(\d+)_", sd)
            rec = {"config": cfg, "seed": int(m.group(1)) if m else -1}
            rec.update(parse(fp) or {})
            rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        print("!! 无结果"); return None
    df = df.sort_values(["config", "seed"])
    print(df.to_string(index=False))
    print()
    g = df.groupby("config").agg(
        n=("seed", "size"),
        DB_mean=("DB mean", "mean"), DB_sd=("DB mean", "std"),
        DB_OOF=("DB OOF", "mean"), DB_OOF_sd=("DB OOF", "std"),
        DS_mean=("DS mean", "mean"), DS_sd=("DS mean", "std"),
        unionFk=("union Fk", "mean"),
    ).round(4)
    print("=== 汇总 ===")
    print(g.to_string())
    out = os.path.join(BASE, "results", out_name or "v3_runs_summary.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    g.to_csv(out.replace(".csv", "_agg.csv"), encoding="utf-8-sig")
    print(f"\n落盘: {out}")
    return df


if __name__ == "__main__":
    main(sys.argv[1:])
