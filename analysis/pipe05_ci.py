# -*- coding: utf-8 -*-
"""
pipe05_ci.py — 记录级 Bootstrap CI 后处理 (独立脚本, 不改动 pipe01-04, 不重跑 R1)
==============================================================================
原理:
  pipe03 的 pkl 保存了 all_Fk (每折特征集) + db_auc/ds_auc (每折 AUC);
  预测本身未保存, 但完全可精确重建:
    - 划分: StratifiedGroupKFold(5, shuffle=True, random_state=OUTER_SEED)  (seed 取自目录名)
    - 插补: impute_median(X_raw[tr], Xte)  (训练折中位数)
    - 模型: rfgb_proba(..., seed)  (RF/GB random_state=seed, 完全确定性)
  本脚本逐折重建 (y_true, p_hat), 先断言复现 AUC == 记录 AUC (|Δ| < 1e-9),
  通过后才做记录级 bootstrap (B=2000, percentile 95% CI)。

统计量 (与手稿报告口径一一对应):
  DB_oof       合并 OOF 预测 (n=300, 每条记录恰为测试一次) 的 AUC
  DB_foldmean  5 折 AUC 的均值 (手稿"无偏 DB"口径); fold-wise bootstrap:
               每折各自重采样测试记录, 重算折 AUC 后取均值
  DS_mean      5 折 DS AUC 的均值 (手稿"折内 DS"口径); 共享 DS 重采样索引:
               每次迭代重采样 26 条 DS 记录, 5 折模型在同一重采样上评估后取均值
               (保留折间相关性, 正确的配对重采样)
  DSbiased     全量 DB 训练 (union_Fk) → DS AUC (手稿"有偏 DS"口径), n=26 记录级 bootstrap
  DBbiased     全量 DB 训练 → DB AUC (有偏/参考), n=300 记录级 bootstrap

跨种子汇总 (>=2 个种子时):
  DB_oof / DS_mean / DSbiased 的重采样索引在种子间共享 (记录对齐),
  逐迭代对种子取均值 → "10 种子均值" 的记录级 CI (配对设计);
  DB_foldmean 因各种子测试集不同, 各种子独立重采样后逐迭代取均值。

用法:
  python pipe05_ci.py                    # 默认扫描主线 10-seed 目录 (P1CI_P6eff050_direct_seed*_*)
  python pipe05_ci.py <dir1> [dir2 ...]  # 显式指定含 pipe_03_sfs_*.pkl 的实验目录

输出:
  experiments/CI_<tag>_<n>runs_<ts>/bootstrap_ci_summary.xlsx  (PerSeed/Folds/Assert/Method)
  experiments/CI_<tag>_<n>runs_<ts>/boot_iters.npz             (逐迭代 AUC, 供后续配对 ΔAUC)
"""
import os
import re
import sys
import glob
import time
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

# --- repository path bootstrap (added for the public release) -----------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "analysis"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# 复用 run_experiment 的定义, 保证与原实验逐位等价 (load_data / impute_median / rfgb_proba)
from run_experiment import load_data, impute_median, rfgb_proba  # noqa: E402

BOOT_N = 2000
BOOT_SEED = 42          # OOF 共享索引流
BOOT_SEED_DS = 43       # DS 共享索引流 (26 条, 与 OOF 流独立)
FOLD_RNG_SEED = 42      # fold-wise bootstrap 每种子独立流
AUC_TOL = 1e-9

EXP_ROOT = os.path.join(_ROOT, "experiments")
DEFAULT_GLOB = os.path.join(EXP_ROOT, "P1CI_P6eff050_direct_seed*_*")


def ts():
    return time.strftime("%H:%M:%S")


def log(s):
    print(f"[{ts()}] {s}", flush=True)


def _boot_aucs(y, p, idx):
    """向量化 rank-AUC, 与 run_experiment.auc_ci 的实现逐行同源 (平秩/有效掩码一致)。
    y: (n,), p: (n,), idx: (B, n) -> (B,) 每次重采样的 AUC (无效迭代=nan)。"""
    yg, pg = y[idx], p[idx]
    n = idx.shape[1]
    order = np.argsort(pg, axis=1, kind="stable")
    sp = np.take_along_axis(pg, order, axis=1)
    pos = np.arange(n)
    new_run = np.empty(sp.shape, dtype=bool)
    new_run[:, 0] = True
    new_run[:, 1:] = sp[:, 1:] != sp[:, :-1]
    start_pos = np.maximum.accumulate(np.where(new_run, pos, 0), axis=1)
    is_last = np.empty(sp.shape, dtype=bool)
    is_last[:, -1] = True
    is_last[:, :-1] = sp[:, 1:] != sp[:, :-1]
    end_pos = np.minimum.accumulate(
        np.where(is_last, pos, n)[:, ::-1], axis=1)[:, ::-1]
    avg_sorted = (start_pos + end_pos) / 2.0 + 1.0
    ranks = np.empty_like(avg_sorted)
    np.put_along_axis(ranks, order, avg_sorted, axis=1)
    posm = yg == 1
    npos_b = posm.sum(axis=1)
    nneg_b = n - npos_b
    rsum = np.where(posm, ranks, 0.0).sum(axis=1)
    valid = (npos_b >= 2) & (nneg_b >= 2)
    aucs = np.full(idx.shape[0], np.nan)
    den = npos_b[valid] * nneg_b[valid]
    aucs[valid] = (rsum[valid] - npos_b[valid] * (npos_b[valid] + 1) / 2.0) / den
    return aucs


def _ci(iters):
    lo, hi = np.nanpercentile(iters, [2.5, 97.5])
    return float(lo), float(hi)


def parse_seed(d):
    m = re.search(r"_seed(\d+)_", os.path.basename(os.path.normpath(d)))
    if not m:
        raise ValueError(f"无法从目录名解析 seed: {d}")
    return int(m.group(1))


def find_runs(argv):
    if len(argv) > 1:
        dirs = [os.path.abspath(a) for a in argv[1:]]
    else:
        dirs = sorted(glob.glob(DEFAULT_GLOB))
    runs = []
    for d in dirs:
        if not os.path.isdir(d):
            log(f"[跳过] 目录不存在: {d}")
            continue
        if not glob.glob(os.path.join(d, "pipe_03_sfs_*.pkl")):
            log(f"[跳过] 无 pipe_03_sfs_*.pkl: {d}")
            continue
        runs.append(d)
    return runs


def rebuild_run(d, db, ds, ALL_COLS):
    """重建单个 run 的逐折预测 + 有偏评估, 并断言与记录 AUC 一致。"""
    seed = parse_seed(d)
    p03 = sorted(glob.glob(os.path.join(d, "pipe_03_sfs_*.pkl")))[0]
    with open(p03, "rb") as f:
        sfs = pickle.load(f)
    tag = sfs.get("tag", "?")
    all_Fk = sfs["all_Fk"]
    ref_db = sfs.get("db_auc", {})
    ref_ds = sfs.get("ds_auc", {})

    X_raw = db[ALL_COLS].values.astype(np.float64)
    y_all = db.label.values.astype(int)
    g_all = db.recname.values
    Xds = ds[ALL_COLS].values.astype(np.float64)
    yds = ds.label.values.astype(int)
    CIDX = {c: i for i, c in enumerate(ALL_COLS)}

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    splits = list(sgkf.split(X_raw, y_all, g_all))

    n_db = len(y_all)
    oof_p = np.full(n_db, np.nan)
    folds = []
    for k, (tr, te) in enumerate(splits, 1):
        F_k = all_Fk.get(f"fold{k}", [])
        sel = [CIDX[c] for c in F_k if c in CIDX]
        ytr, yte = y_all[tr], y_all[te]
        Xtr, Xte = impute_median(X_raw[tr], X_raw[te])
        _, Xds_imp = impute_median(X_raw[tr], Xds)
        if not sel:
            folds.append(dict(k=k, n_te=len(te), pos=int(yte.sum()),
                              yte=yte, p_db=None, p_ds=None,
                              auc_db=np.nan, auc_ds=np.nan))
            continue
        p_db = rfgb_proba(Xtr[:, sel], ytr, Xte[:, sel], seed)
        p_ds = rfgb_proba(Xtr[:, sel], ytr, Xds_imp[:, sel], seed)
        auc_db = float(roc_auc_score(yte, p_db))
        auc_ds = float(roc_auc_score(yds, p_ds))
        oof_p[te] = p_db
        folds.append(dict(k=k, n_te=len(te), pos=int(yte.sum()),
                          yte=yte, p_db=p_db, p_ds=p_ds,
                          auc_db=auc_db, auc_ds=auc_ds))

    # ---- 断言: 复现 AUC == pipe03 记录 AUC ----
    db_diff = 0.0
    ds_diff = 0.0
    for fd in folds:
        r_d = ref_db.get(f"fold{fd['k']}", np.nan)
        r_s = ref_ds.get(f"fold{fd['k']}", np.nan)
        if not (np.isnan(fd["auc_db"]) and np.isnan(r_d)):
            db_diff = max(db_diff, abs(fd["auc_db"] - r_d))
        if not (np.isnan(fd["auc_ds"]) and np.isnan(r_s)):
            ds_diff = max(ds_diff, abs(fd["auc_ds"] - r_s))

    # ---- 有偏评估复现 (pipe04) ----
    p04 = sorted(glob.glob(os.path.join(d, "pipe_04_eval_*.pkl")))
    feats, ref_bdb, ref_bds = None, np.nan, np.nan
    if p04:
        with open(p04[0], "rb") as f:
            ev = pickle.load(f)
        feats = ev.get("feats")
        ref_bdb = ev.get("db_auc_biased", np.nan)
        ref_bds = ev.get("ds_auc_biased", np.nan)
    if not feats:
        feats = [c for c in sfs.get("union_Fk", []) if c in CIDX]
    idx4 = [CIDX[c] for c in feats if c in CIDX]
    Xdb_imp, _ = impute_median(X_raw, X_raw)
    _, Xds_imp4 = impute_median(X_raw, Xds)
    p_bdb = rfgb_proba(Xdb_imp[:, idx4], y_all, Xdb_imp[:, idx4], seed)
    p_bds = rfgb_proba(Xdb_imp[:, idx4], y_all, Xds_imp4[:, idx4], seed)
    auc_bdb = float(roc_auc_score(y_all, p_bdb))
    auc_bds = float(roc_auc_score(yds, p_bds))
    bdb_diff = abs(auc_bdb - ref_bdb) if not np.isnan(ref_bdb) else np.nan
    bds_diff = abs(auc_bds - ref_bds) if not np.isnan(ref_bds) else np.nan

    passed = (db_diff < AUC_TOL) and (ds_diff < AUC_TOL) \
        and (np.isnan(bdb_diff) or bdb_diff < AUC_TOL) \
        and (np.isnan(bds_diff) or bds_diff < AUC_TOL)

    log(f"  seed{seed}: 复现断言 dbΔ={db_diff:.2e} dsΔ={ds_diff:.2e} "
        f"bdbΔ={bdb_diff:.2e} bdsΔ={bds_diff:.2e} -> {'PASS' if passed else 'FAIL'}")
    return dict(seed=seed, tag=tag, dir=d, folds=folds,
                y=y_all, yds=yds, oof_p=oof_p,
                p_bdb=p_bdb, p_bds=p_bds,
                auc_bdb=auc_bdb, auc_bds=auc_bds,
                db_diff=db_diff, ds_diff=ds_diff,
                bdb_diff=bdb_diff, bds_diff=bds_diff,
                passed=passed)


def bootstrap_run(res, oof_idx, ds_idx):
    """单个种子的逐迭代 bootstrap AUC。oof_idx/ds_idx 为种子间共享索引。"""
    y, oof_p = res["y"], res["oof_p"]
    ok = ~np.isnan(oof_p)
    if ok.all():
        oof_iters = _boot_aucs(y, oof_p, oof_idx)
    else:  # 理论上不发生 (R1 各折 F_k 非空); 保险: 仅在有效记录上重采样
        yv, pv = y[ok], oof_p[ok]
        idx = oof_idx[:, :len(yv)] % len(yv)
        oof_iters = _boot_aucs(yv, pv, idx)

    # fold-mean: 每折独立重采样 (rng 每种子独立)
    rng = np.random.RandomState(FOLD_RNG_SEED)
    fold_iter_cols = []
    for fd in res["folds"]:
        if fd["p_db"] is None:
            continue
        n_te = fd["n_te"]
        idx_k = rng.randint(0, n_te, size=(BOOT_N, n_te))
        fold_iter_cols.append(_boot_aucs(fd["yte"], fd["p_db"], idx_k))
    db_fm_iters = np.mean(np.vstack(fold_iter_cols), axis=0) if fold_iter_cols \
        else np.full(BOOT_N, np.nan)

    # DS_mean: 共享 DS 索引 (折间配对)
    ds_cols, ds_fold_iters = [], {}
    for fd in res["folds"]:
        if fd["p_ds"] is None:
            continue
        it = _boot_aucs(res["yds"], fd["p_ds"], ds_idx)
        ds_cols.append(it)
        ds_fold_iters[fd["k"]] = it
    ds_mean_iters = np.mean(np.vstack(ds_cols), axis=0) if ds_cols \
        else np.full(BOOT_N, np.nan)

    dsb_iters = _boot_aucs(res["yds"], res["p_bds"], ds_idx)
    bdb_iters = _boot_aucs(res["y"], res["p_bdb"], oof_idx)

    res.update(dict(
        db_oof_auc=float(roc_auc_score(res["y"], res["oof_p"])),
        db_fm_auc=float(np.nanmean([fd["auc_db"] for fd in res["folds"]])),
        ds_mean_auc=float(np.nanmean([fd["auc_ds"] for fd in res["folds"]])),
        db_oof_iters=oof_iters, db_fm_iters=db_fm_iters,
        ds_mean_iters=ds_mean_iters, dsb_iters=dsb_iters, bdb_iters=bdb_iters,
        ds_fold_iters=ds_fold_iters,
    ))
    return res


STAT_DEFS = [
    ("DB_oof", "db_oof_iters", "db_oof_auc"),
    ("DB_foldmean", "db_fm_iters", "db_fm_auc"),
    ("DS_mean", "ds_mean_iters", "ds_mean_auc"),
    ("DSbiased", "dsb_iters", "auc_bds"),
    ("DBbiased", "bdb_iters", "auc_bdb"),
]


def main():
    t0 = time.time()
    log("pipe05_ci 记录级 Bootstrap CI 后处理启动")
    runs = find_runs(sys.argv)
    if not runs:
        log("未找到任何实验目录, 退出")
        sys.exit(1)
    log(f"待处理 {len(runs)} 个 run: " + ", ".join(os.path.basename(r) for r in runs))

    db, ds, ALL_COLS = load_data()
    log(f"DB {len(db)} (pos {int(db.label.sum())}), DS {len(ds)} (pos {int(ds.label.sum())}), "
        f"特征 {len(ALL_COLS)}")

    # 种子间共享重采样索引 (记录对齐 → 跨种子均值为配对 bootstrap)
    rng_oof = np.random.RandomState(BOOT_SEED)
    oof_idx = rng_oof.randint(0, len(db), size=(BOOT_N, len(db)))
    rng_ds = np.random.RandomState(BOOT_SEED_DS)
    ds_idx = rng_ds.randint(0, len(ds), size=(BOOT_N, len(ds)))

    results = []
    n_fail = 0
    for d in runs:
        log(f"[重建] {os.path.basename(d)}")
        try:
            res = rebuild_run(d, db, ds, ALL_COLS)
        except Exception as e:
            log(f"  重建失败: {type(e).__name__}: {e}")
            n_fail += 1
            continue
        if not res["passed"]:
            n_fail += 1
            log("  断言未通过 (复现 AUC 与记录不一致), 该种子不输出 CI")
            results.append(res)
            continue
        res = bootstrap_run(res, oof_idx, ds_idx)
        results.append(res)

    done = [r for r in results if "db_oof_iters" in r]
    tag = results[0]["tag"] if results else "unknown"
    out_dir = os.path.join(EXP_ROOT, f"CI_{tag}_{len(results)}runs_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)

    # ---------- PerSeed ----------
    rows = []
    for r in results:
        row = {"seed": r["seed"], "dir": os.path.basename(r["dir"]),
               "assert": "PASS" if r["passed"] else "FAIL"}
        if "db_oof_iters" in r:
            for name, _key, _attr in STAT_DEFS:
                it = r[_key]
                lo, hi = _ci(it)
                row[f"{name}_AUC"] = round(float(r[_attr]), 4)
                row[f"{name}_LO"] = round(lo, 4)
                row[f"{name}_HI"] = round(hi, 4)
        rows.append(row)
    if len(done) >= 2:
        mean_row = {"seed": f"MEAN({len(done)} seeds)", "dir": "", "assert": ""}
        for name, key, attr in STAT_DEFS:
            stack = np.vstack([r[key] for r in done])          # (B, n_seeds)
            cross = np.nanmean(stack, axis=0)                  # 逐迭代种子均值
            lo, hi = _ci(cross)
            pt = float(np.mean([r[attr] for r in done]))
            mean_row[f"{name}_AUC"] = round(pt, 4)
            mean_row[f"{name}_LO"] = round(lo, 4)
            mean_row[f"{name}_HI"] = round(hi, 4)
        rows.append(mean_row)
    df_seed = pd.DataFrame(rows)

    # ---------- Folds ----------
    frows = []
    for r in results:
        for fd in r["folds"]:
            fr = {"seed": r["seed"], "fold": fd["k"], "n_test": fd["n_te"],
                  "pos_test": fd["pos"], "DB_AUC": round(fd["auc_db"], 4),
                  "DS_AUC": round(fd["auc_ds"], 4)}
            if "ds_fold_iters" in r and fd["k"] in r["ds_fold_iters"]:
                lo, hi = _ci(r["ds_fold_iters"][fd["k"]])
                fr["DS_LO"], fr["DS_HI"] = round(lo, 4), round(hi, 4)
            frows.append(fr)
    df_folds = pd.DataFrame(frows)

    # ---------- Assert ----------
    df_assert = pd.DataFrame([{
        "seed": r["seed"], "dir": os.path.basename(r["dir"]), "tag": r["tag"],
        "db_max_diff": r["db_diff"], "ds_max_diff": r["ds_diff"],
        "biased_db_diff": r["bdb_diff"], "biased_ds_diff": r["bds_diff"],
        "PASS": r["passed"]} for r in results])

    # ---------- Method ----------
    df_method = pd.DataFrame({"Note": [
        f"记录级 bootstrap CI, B={BOOT_N}, percentile 95% CI (2.5/97.5), 秩统计与 run_experiment.auc_ci 同源",
        f"共享重采样索引: OOF 流 RandomState({BOOT_SEED}) (n={len(db)}), DS 流 RandomState({BOOT_SEED_DS}) (n={len(ds)})",
        "DB_foldmean: fold-wise bootstrap, 每折独立重采样测试记录后取 5 折均值 (对应手稿'无偏 DB'口径)",
        "DS_mean: 共享 DS 索引, 5 折模型在同一 DS 重采样上评估后取均值 (对应'折内 DS'口径, 保留折间相关)",
        "DSbiased/DBbiased: 全量 DB 训练 (union_Fk/pipe04 feats) 的 AUC 记录级 bootstrap",
        f"复现断言: 逐折重建 AUC 与 pipe03/pipe04 记录值 |Δ| < {AUC_TOL:g}, 未通过者不输出 CI",
        f"跨种子 MEAN 行: 逐迭代对 {len(done)} 个种子取均值后的 percentile CI (记录级配对)",
        "boot_iters.npz 保存逐迭代 AUC (含共享索引), 供后续配对 ΔAUC bootstrap (R2/R4/R5 vs 基线)",
    ]})

    outx = os.path.join(out_dir, "bootstrap_ci_summary.xlsx")
    with pd.ExcelWriter(outx, engine="openpyxl") as xw:
        df_seed.to_excel(xw, sheet_name="PerSeed", index=False)
        df_folds.to_excel(xw, sheet_name="Folds", index=False)
        df_assert.to_excel(xw, sheet_name="Assert", index=False)
        df_method.to_excel(xw, sheet_name="Method", index=False)

    # ---------- npz (供配对 ΔAUC) ----------
    npz_path = os.path.join(out_dir, "boot_iters.npz")
    save = {"seeds": np.array([r["seed"] for r in results]),
            "oof_idx": oof_idx, "ds_idx": ds_idx}
    for r in done:
        s = r["seed"]
        save[f"db_oof_iters_seed{s}"] = r["db_oof_iters"]
        save[f"db_fm_iters_seed{s}"] = r["db_fm_iters"]
        save[f"ds_mean_iters_seed{s}"] = r["ds_mean_iters"]
        save[f"dsb_iters_seed{s}"] = r["dsb_iters"]
        save[f"bdb_iters_seed{s}"] = r["bdb_iters"]
        for k, it in r["ds_fold_iters"].items():
            save[f"ds_fold{k}_iters_seed{s}"] = it
    np.savez_compressed(npz_path, **save)

    # ---------- 控制台汇总 ----------
    log("\n===== 汇总 (AUC [95% CI]) =====")
    for r in done:
        lo, hi = _ci(r["db_fm_iters"])
        log(f"  seed{r['seed']}: DB_foldmean {r['db_fm_auc']:.3f} [{lo:.3f}, {hi:.3f}]")
        lo, hi = _ci(r["db_oof_iters"])
        log(f"           DB_oof      {r['db_oof_auc']:.3f} [{lo:.3f}, {hi:.3f}]")
        lo, hi = _ci(r["ds_mean_iters"])
        log(f"           DS_mean     {r['ds_mean_auc']:.3f} [{lo:.3f}, {hi:.3f}]")
        lo, hi = _ci(r["dsb_iters"])
        log(f"           DSbiased    {r['auc_bds']:.3f} [{lo:.3f}, {hi:.3f}]")
    if len(done) >= 2:
        for name, key, attr in STAT_DEFS:
            cross = np.nanmean(np.vstack([r[key] for r in done]), axis=0)
            lo, hi = _ci(cross)
            pt = float(np.mean([r[attr] for r in done]))
            log(f"  MEAN({len(done)}): {name} {pt:.3f} [{lo:.3f}, {hi:.3f}]")

    log(f"\n输出: {outx}")
    log(f"      {npz_path}")
    log(f"完成: {len(done)}/{len(results)} 个种子输出 CI, 断言失败 {n_fail} 个, "
        f"耗时 {(time.time()-t0)/60:.1f} min")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
