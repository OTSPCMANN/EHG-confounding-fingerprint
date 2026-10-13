# -*- coding: utf-8 -*-
"""
EHG 特征筛选管线 — 一体化脚本
================================
整合 pipe01-04 全部逻辑, 一键运行:
  段1: 配对 AUC 计算 (P1-P6, bootstrap CI)
  段2: Stage1 特征池形成 (判据筛选)
  段3: SFS 前向选择 + drop-one 去冗余
  段4: DB 无偏 + DS 跨域分类评估

用法:
  python run_experiment.py P1CI_P6eff050

输出:
  experiments/<tag>_<时间戳>/  含全部 xlsx + pkl + run.log
"""
import os
import sys
import pickle
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# --- repository path bootstrap (added for the public release) -----------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "analysis"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# 分类器家族 (§4.3 鲁棒性用): rfgb(默认, RF+GB 等权) / logreg / svm / mlp
# 非 rfgb 家族为单一模型 + class_weight balanced, 与 rfgb 的 balanced 处理保持可比
CLASSL = os.environ.get("EHG_CLASSL", "rfgb").lower()

warnings.filterwarnings("ignore")

# ======================== 全局配置 ========================
OUTER_SEED = 42
BOOT_N = 2000
BOOT_SEED = 42
LEAD_SPLIT = 6.0
RECT_EARLY = 26
RECT_LATE = 26
MIN_N = 20
DELTA = 0.05
SEEDS = [42, 123, 456, 789, 2024]

# 数据源: 可用环境变量 EHG_FT_CSV 覆盖 (v3 六通道池传
# D:\ECG_project\EHG\checkpoints_lit_features_v3\lit_features_v3.csv)
FT_CSV = os.environ.get(
    "EHG_FT_CSV",
    os.path.join(_ROOT, "results", "checkpoints_lit_features_v3",
                 "lit_features_v3.csv"))

# 特征选择方法 (对比实验用): direct(默认, 主线: 准入全集直接进分类器) / sfs(SFS+drop-one, 已证过严降级为辅助) / embed(C: L1 嵌入式选择)
# 2026-09-24 对比实验裁定: B direct 三指标全面显著优于 sfs (配对 Wilcoxon p<=0.004), 主线改 direct
SEL = os.environ.get("EHG_SELECT", "direct").lower()
# Feature ID 由 ehg_featmeta 提供 (编号固化, 不再依赖已弃用的 EHG论文0822/baseline_results_75col.xlsx)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from ehg_featmeta import load_fid_map  # noqa: E402

EXP_ROOT = os.path.join(_ROOT, "experiments")

PAIRS = [("P1", "G1vsG3"), ("P2", "G1vsG4"), ("P3", "G2vsG3"),
         ("P4", "G2vsG4"), ("P5", "G1vsG2"), ("P6", "G3vsG4")]

CONFIGS = {
    "P1CI_P6CI":      dict(rule14="ci",     rule6="ci",  eps=None),
    "P1CI_P6none":    dict(rule14="ci",     rule6="none", eps=None),
    "P1CI_P6eff020":  dict(rule14="ci",     rule6="eff", eps=0.02),
    "P1CI_P6eff030":  dict(rule14="ci",     rule6="eff", eps=0.03),
    "P1CI_P6eff040":  dict(rule14="ci",     rule6="eff", eps=0.04),
    "P1CI_P6eff050":  dict(rule14="ci",     rule6="eff", eps=0.05),
    "P1CI_P6eff060":  dict(rule14="ci",     rule6="eff", eps=0.06),
    "P1CI_P6eff070":  dict(rule14="ci",     rule6="eff", eps=0.07),
    "P1CI_P6eff080":  dict(rule14="ci",     rule6="eff", eps=0.08),
    "P1CI_P6eff090":  dict(rule14="ci",     rule6="eff", eps=0.09),
    "P1CI_P6eff100":  dict(rule14="ci",     rule6="eff", eps=0.10),
    "P1eff_P6CI":     dict(rule14="eff",    rule6="ci",  eps=None),
    "P1eff_P6eff050": dict(rule14="eff",    rule6="eff", eps=0.05),
    "P1eff_P6eff070": dict(rule14="eff",    rule6="eff", eps=0.07),
    "P1AND_P6eff070": dict(rule14="ci_eff", rule6="eff", eps=0.07),
}

# ======================== 工具函数 ========================
_logf = None


def ts():
    return time.strftime("%H:%M:%S")


def log(s):
    line = f"[{ts()}] {s}"
    print(line, flush=True)
    if _logf:
        _logf.write(line + "\n")
        _logf.flush()


def banner(title, params):
    log("=" * 60)
    log(f"[{title}]")
    for k, v in params.items():
        log(f"  {k:<16} = {v}")
    log("=" * 60)


def load_data():
    df = pd.read_csv(FT_CSV)
    db = df[df.source == "TPEHGDB"].reset_index(drop=True)
    ds = df[df.source == "TPEHGT"].reset_index(drop=True)
    DROP = ("recname", "source", "label", "gest", "rectime", "lead", "n_samples")
    ALL_COLS = [c for c in db.columns if c not in DROP]
    return db, ds, ALL_COLS


def impute_median(Xtr, Xte):
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    return (np.where(np.isnan(Xtr), med, Xtr),
            np.where(np.isnan(Xte), med, Xte))


def smote_before_split(X, y, g, gest, rect, lead, k_neighbors=5, random_state=0):
    """Minimal SMOTE applied to the FULL dataset BEFORE the CV split.

    Protocol-ladder rung T3 (resampling leakage): synthetic minority records are
    created by interpolating between a minority sample and one of its k nearest
    minority neighbours, then the augmented set is split into folds, so synthetic
    records leak into the test fold. Returns the augmented (X, y, g, gest,
    rect, lead). Synthetic group ids are unique sentinels so StratifiedGroupKFold
    distributes them across folds.
    """
    Xi, _ = impute_median(X, X)
    rng = np.random.RandomState(random_state)
    y = np.asarray(y)
    imin = np.where(y == 1)[0]
    imaj = np.where(y == 0)[0]
    n_min, n_maj = len(imin), len(imaj)
    if n_min == 0 or n_maj == 0:
        return X, y, g, gest, rect, lead
    need = n_maj - n_min
    Xm = Xi[imin]
    synth, meta = [], []
    for _ in range(need):
        i = rng.randint(n_min)
        d = np.linalg.norm(Xm - Xm[i], axis=1)
        nn = np.argsort(d)[1:k_neighbors + 1]
        j = nn[rng.randint(len(nn))]
        synth.append(Xm[i] + rng.rand() * (Xm[j] - Xm[i]))
        meta.append((gest[imin[i]], rect[imin[i]], lead[imin[i]], g[imin[i]]))
    Xs = np.vstack([Xi, np.asarray(synth, dtype=np.float64)])
    ys = np.r_[y, np.ones(need, dtype=int)]
    gs = np.r_[np.asarray(g, dtype=object),
               np.array([f"SM{i}" for i in range(need)], dtype=object)]
    ga = np.r_[gest, np.array([m[0] for m in meta])]
    ra = np.r_[rect, np.array([m[1] for m in meta])]
    la = np.r_[lead, np.array([m[2] for m in meta])]
    return Xs, ys, gs, ga, ra, la


# ======================== 共享: Bootstrap AUC ========================
def auc_ci(y, p, n_boot=BOOT_N, seed=BOOT_SEED):
    ok = ~np.isnan(p)
    y, p = y[ok], p[ok]
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos < 2 or nneg < 2:
        return np.nan, np.nan, np.nan
    auc = float(roc_auc_score(y, p))
    n = len(y)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(n_boot, n))
    pg, yg = p[idx], y[idx]
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
    npos_b, nneg_b = posm.sum(axis=1), n - posm.sum(axis=1)
    rsum = np.where(posm, ranks, 0.0).sum(axis=1)
    valid = (npos_b >= 2) & (nneg_b >= 2)
    aucs = np.full(n_boot, np.nan)
    den = npos_b[valid] * nneg_b[valid]
    aucs[valid] = (rsum[valid]
                   - npos_b[valid] * (npos_b[valid] + 1) / 2.0) / den
    aucs = aucs[~np.isnan(aucs)]
    if len(aucs) == 0:
        return auc, np.nan, np.nan
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(auc), float(lo), float(hi)


def group_masks(y, lead, rect):
    g1 = (y == 1) & (lead <= LEAD_SPLIT)
    g2 = (y == 1) & (lead > LEAD_SPLIT)
    g3 = (y == 0) & (rect < RECT_EARLY)
    g4 = (y == 0) & (rect >= RECT_LATE)
    return g1, g2, g3, g4


def pair_auc_ci(p_col, ga, gb, min_n):
    pp = np.r_[p_col[ga], p_col[gb]]
    yy = np.r_[np.ones(int(ga.sum())), np.zeros(int(gb.sum()))]
    ok = ~np.isnan(pp)
    if ok.sum() < min_n or len(np.unique(yy[ok])) < 2:
        return np.nan, np.nan, np.nan
    return auc_ci(yy[ok], pp[ok])


def build_matrix(X, y, lead, rect, cols):
    g1, g2, g3, g4 = group_masks(y, lead, rect)
    grp = {"G1": g1, "G2": g2, "G3": g3, "G4": g4}
    rows = []
    for j, c in enumerate(cols):
        p = X[:, j]
        for pn, tag in PAIRS:
            ga, gb = tag.split("vs")
            a, lo, hi = pair_auc_ci(p, grp[ga], grp[gb], MIN_N)
            rows.append({"Column Name": c, "Pair": pn, "Group": tag,
                         "AUC": a, "LO": lo, "HI": hi})
    wide = pd.DataFrame(rows).pivot_table(
        index="Column Name", columns="Pair", values="AUC").reindex(cols)
    return pd.DataFrame(rows), wide


# ======================== 共享: 判据函数 ========================
def _sig14_pre(long_df, col_name, rule14, delta):
    sub = long_df[long_df["Column Name"] == col_name]
    pair_map = {"P1": "G1vsG3", "P2": "G1vsG4", "P3": "G2vsG3", "P4": "G2vsG4"}
    sig, dirs_ = [], []
    for pn, grp in pair_map.items():
        row = sub[(sub["Pair"] == pn) & (sub["Group"] == grp)]
        if row.empty:
            continue
        a, lo, hi = float(row["AUC"].iloc[0]), float(row["LO"].iloc[0]), float(row["HI"].iloc[0])
        if np.isnan(a):
            continue
        if rule14 == "ci":
            d = (1 if lo > 0.5 else -1) if hi < 0.5 or lo > 0.5 else 0
            if d:
                sig.append(pn); dirs_.append(d)
        elif rule14 == "eff":
            d = (1 if a - 0.5 > delta else (-1 if 0.5 - a > delta else 0))
            if d:
                sig.append(pn); dirs_.append(d)
        elif rule14 == "ci_eff":
            d_ci = (1 if lo > 0.5 else -1) if hi < 0.5 or lo > 0.5 else 0
            d_ef = (1 if a - 0.5 > delta else (-1 if 0.5 - a > delta else 0))
            if d_ci and d_ci == d_ef:
                sig.append(pn); dirs_.append(d_ci)
    return (len(sig) >= 1) and (len(set(dirs_)) <= 1)


def _sig6_pre(long_df, col_name, rule6, eps):
    # rule6 == "none": 显式关闭反向门 (T0a 消融实验), 所有特征直接通过 P6 门
    if rule6 == "none":
        return True
    sub = long_df[(long_df["Column Name"] == col_name)
                  & (long_df["Pair"] == "P6") & (long_df["Group"] == "G3vsG4")]
    if sub.empty:
        return False
    a, lo, hi = float(sub["AUC"].iloc[0]), float(sub["LO"].iloc[0]), float(sub["HI"].iloc[0])
    if np.isnan(a):
        return False
    if rule6 == "ci":
        return lo <= 0.5 <= hi
    return abs(a - 0.5) < eps


# ======================== 共享: 分类器 ========================
def rfgb_proba(Xtr, ytr, Xte, seed):
    # 距离/梯度型模型(logreg/svm/mlp)必须缩放; 本池 189 特征的逐特征标准差
    # 跨 4 个数量级(7e-4 ~ 338), 不缩放时 SVC(gamma="scale") 的核会被最广特征的
    # 方差主导而退化, LogisticRegression 则陷入病态条件数.
    # scaler 只在训练折内拟合, 不泄漏测试折统计量.
    if CLASSL in ("logreg", "svm", "mlp"):
        from sklearn.preprocessing import StandardScaler
        Xtr = StandardScaler().fit_transform(Xtr)
        Xte = StandardScaler().fit_transform(Xte)
    if CLASSL == "logreg":
        m = LogisticRegression(max_iter=5000, class_weight="balanced",
                               random_state=seed).fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]
    if CLASSL == "svm":
        from sklearn.svm import SVC
        m = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                class_weight="balanced", random_state=seed).fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]
    if CLASSL == "mlp":
        from sklearn.neural_network import MLPClassifier
        m = MLPClassifier(hidden_layer_sizes=(32,), max_iter=600,
                          random_state=seed, early_stopping=True).fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]
    if CLASSL != "rfgb":
        raise ValueError(f"未知 EHG_CLASSL={CLASSL!r} (可用 rfgb/logreg/svm/mlp)")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=2,
        random_state=seed, class_weight="balanced").fit(Xtr, ytr)
    gb = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.03, max_depth=3, subsample=0.8,
        random_state=seed).fit(Xtr, ytr)
    return 0.5 * rf.predict_proba(Xte)[:, 1] + 0.5 * gb.predict_proba(Xte)[:, 1]


def cv_mean_auc(X, y, g, feats_idx):
    aucs = []
    for s in SEEDS:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=s)
        fa = []
        for tr, te in sgkf.split(X, y, g):
            rf = RandomForestClassifier(
                n_estimators=300, max_depth=6, min_samples_leaf=2,
                random_state=s, class_weight="balanced"
            ).fit(X[tr][:, feats_idx], y[tr])
            fa.append(roc_auc_score(y[te], rf.predict_proba(X[te][:, feats_idx])[:, 1]))
        aucs.append(np.mean(fa))
    return float(np.mean(aucs))


def sfs_select(pool_idx, X, y, g, tag):
    selected, cur, step = [], None, 0
    while pool_idx:
        step += 1
        best_i, best_m = None, None
        for cand in pool_idx:
            m = cv_mean_auc(X, y, g, selected + [cand])
            if best_m is None or m > best_m:
                best_i, best_m = cand, m
        if cur is not None and best_m <= cur + 1e-9:
            log(f"    {tag} SFS step{step} STOP (best {best_m:.4f} <= cur {cur:.4f})")
            break
        selected.append(best_i)
        pool_idx = [i for i in pool_idx if i != best_i]
        cur = best_m
        log(f"    {tag} SFS step{step}: +{best_i} -> {best_m:.4f}")
    return selected, cur


def drop_one(selected, X, y, g, tag):
    if not selected:
        return selected, np.nan
    cur = cv_mean_auc(X, y, g, selected)
    while len(selected) > 1:
        worst_i, worst_m = None, None
        for f in selected:
            m = cv_mean_auc(X, y, g, [x for x in selected if f != x])
            if worst_m is None or m > worst_m:
                worst_i, worst_m = f, m
        if worst_m > cur + 1e-9:
            selected.remove(worst_i)
            cur = worst_m
            log(f"    {tag} drop -{worst_i} -> {worst_m:.4f}")
        else:
            break
    return selected, cur


def embedded_select(pool_idx, X, y, g):
    """C 方案: L1 嵌入式选择. 在准入池上用 L1 Logistic 回归挑非零系数特征.
    最终分类器仍是 RF+GB(与 B/基线一致), 仅特征集不同, 以隔离"选择方法"这一变量."""
    if len(pool_idx) <= 1:
        return list(pool_idx)
    Xp = X[:, pool_idx].astype(np.float64)
    mu = np.nanmean(Xp, axis=0); sd = np.nanstd(Xp, axis=0)
    sd[sd == 0] = 1.0
    Xs = np.where(np.isnan(Xp), mu, Xp)
    Xs = (Xs - mu) / sd
    picked = []
    clf = None
    try:
        clf = LogisticRegression(penalty="l1", C=1.0, class_weight="balanced",
                                 solver="liblinear", max_iter=3000, random_state=0)
        clf.fit(Xs, y)
        coef = np.abs(clf.coef_[0])
        picked = [pool_idx[i] for i in range(len(pool_idx)) if coef[i] > 1e-5]
    except Exception:
        picked = []
    if not picked:  # 回退: |coef| 最大前 min(3, 池); 拟合彻底失败则退化为全集
        if clf is not None and hasattr(clf, "coef_"):
            coef = np.abs(clf.coef_[0])
            order = np.argsort(-coef)[:max(1, min(3, len(pool_idx)))]
            picked = [pool_idx[i] for i in order]
        else:
            picked = list(pool_idx)
    return picked


# ============================================================
#  段1: 配对 AUC 计算
# ============================================================
def stage1_auc(exp_dir, ALL_COLS):
    t0 = time.time()
    banner("PIPE01 配对 AUC 计算 (P1-P6)", {
        "OUTER_SEED": OUTER_SEED, "BOOT_N": BOOT_N, "BOOT_SEED": BOOT_SEED,
        "LEAD_SPLIT": LEAD_SPLIT, "RECT_EARLY": RECT_EARLY,
        "RECT_LATE": RECT_LATE, "MIN_N": MIN_N,
    })
    db, _, _ = load_data()
    y_all = db.label.values.astype(int)
    g_all = db.recname.values
    lead_all = db.lead.values.astype(float)
    rect_all = db.rectime.values.astype(float)
    X_raw = db[ALL_COLS].values.astype(np.float64)
    fid_for = load_fid_map()
    log(f"段1 | {len(y_all)} 条, {len(ALL_COLS)} 特征")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=OUTER_SEED)
    splits = list(sgkf.split(X_raw, y_all, g_all))

    long_full, wide_full = build_matrix(X_raw, y_all, lead_all, rect_all, ALL_COLS)
    log(f"全量300: 完成 {len(long_full)} 行")

    OUTX = os.path.join(exp_dir, "pipe_01_pair_auc.xlsx")
    ov_rows = []
    fold_aucs = {"Full300": long_full}
    with pd.ExcelWriter(OUTX, engine="openpyxl") as xw:
        wf = wide_full.copy()
        wf.insert(0, "Feature ID", wf.index.map(lambda c: fid_for.get(c, "")))
        wf.insert(0, "Column Name2", wf.index)
        wf = wf.rename(columns={"Column Name2": "Column Name"})
        wf.to_excel(xw, sheet_name="Full300", index=False)
        for k, (tr, te) in enumerate(splits, 1):
            tk = time.time()
            long_k, wide_k = build_matrix(X_raw[tr], y_all[tr],
                                          lead_all[tr], rect_all[tr], ALL_COLS)
            wk = wide_k.copy()
            wk.insert(0, "Feature ID", wk.index.map(lambda c: fid_for.get(c, "")))
            wk.insert(0, "Column Name2", wk.index)
            wk = wk.rename(columns={"Column Name2": "Column Name"})
            wk.to_excel(xw, sheet_name=f"Fold{k}", index=False)
            fold_aucs[k] = long_k
            ov_rows.append({"Fold": k, "N train": len(tr), "N test": len(te),
                            "Pos train": int(y_all[tr].sum()),
                            "Pos test": int(y_all[te].sum()),
                            "sec": round(time.time() - tk, 1)})
            log(f"折{k}: 完成 ({time.time()-tk:.0f}s)")
        pd.DataFrame(ov_rows).to_excel(xw, sheet_name="Overview", index=False)

    pkl = os.path.join(exp_dir, "pipe_01_fold_aucs.pkl")
    with open(pkl, "wb") as f:
        pickle.dump(fold_aucs, f, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"段1输出: xlsx + pkl({len(fold_aucs)} keys) 耗时 {(time.time()-t0)/60:.1f} min")
    return fold_aucs


# ============================================================
#  段2: Stage1 特征池形成
# ============================================================
def stage2_pool(exp_dir, tag, fold_aucs, ALL_COLS, base_tag=None):
    t0 = time.time()
    cfg = CONFIGS[base_tag if base_tag else tag]
    r14, r6, eps = cfg["rule14"], cfg["rule6"], cfg["eps"]
    eps6 = eps if eps is not None else ("none" if r6 == "none" else "CI")
    banner("PIPE02 Stage1 特征池", {
        "tag": tag, "rule14": r14, "rule6": r6, "eps": eps6,
        "DELTA": DELTA, "MIN_N": MIN_N,
    })
    fid_for = load_fid_map()

    long_full = fold_aucs["Full300"]
    full_ok = [c for c in ALL_COLS
               if _sig14_pre(long_full, c, r14, DELTA)
               and _sig6_pre(long_full, c, r6, eps)]

    # §4.4 T2 选择泄漏对照: EHG_LEAKGATE=1 时 gate 用全量(Full300)判据的固定结果,
    # 而不是在每个训练折内重算 —— 即"选择过程移出折外", 评估集信息反哺了选择.
    LEAKGATE = os.environ.get("EHG_LEAKGATE", "0") == "1"
    if LEAKGATE:
        log("  [LEAKGATE] 选择判据用全量 Full300 固定, 不在折内重算 (T2 泄漏对照)")

    fold_pass = {}
    for k in range(1, 6):
        tk = time.time()
        if LEAKGATE:
            passed_k = list(full_ok)
        else:
            long_k = fold_aucs[k]
            passed_k = [c for c in ALL_COLS
                        if _sig14_pre(long_k, c, r14, DELTA)
                        and _sig6_pre(long_k, c, r6, eps)]
        fold_pass[k] = set(passed_k)
        log(f"折{k}: pool={len(passed_k)} ({time.time()-tk:.1f}s)")

    mu = np.mean([len(fold_pass[k]) for k in range(1, 6)])
    rng_ = (min(len(fold_pass[k]) for k in range(1, 6)),
            max(len(fold_pass[k]) for k in range(1, 6)))
    uni = set().union(*fold_pass.values())

    OUTX = os.path.join(exp_dir, f"pipe_02_pool_{tag}.xlsx")
    det = pd.DataFrame({"Column Name": ALL_COLS})
    det.insert(0, "Feature ID", det["Column Name"].map(lambda c: fid_for.get(c, "")))
    for k in range(1, 6):
        det[f"fold{k}"] = det["Column Name"].isin(fold_pass[k]).astype(int)
    det["in_Full300pass"] = det["Column Name"].isin(full_ok).astype(int)

    ov = pd.DataFrame([{"Config": tag, "P1P4": r14, "P6": r6, "eps": eps6,
                        "fold sizes": str([len(fold_pass[k]) for k in range(1, 6)]),
                        "mean_pool": round(float(mu), 1),
                        "range": f"{rng_[0]}-{rng_[1]}", "union_size": len(uni),
                        "Full300_pass": len(full_ok)}])
    with pd.ExcelWriter(OUTX, engine="openpyxl") as xw:
        det.to_excel(xw, sheet_name="Pool", index=False)
        ov.to_excel(xw, sheet_name="Overview", index=False)

    pool_data = {"fold_pass": {k: sorted(v) for k, v in fold_pass.items()},
                 "full_ok": full_ok, "config": cfg, "tag": tag}
    pkl_out = os.path.join(exp_dir, f"pipe_02_pool_{tag}.pkl")
    with open(pkl_out, "wb") as f:
        pickle.dump(pool_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    log(f"段2输出: Full300={len(full_ok)}, folds={[len(fold_pass[k]) for k in range(1,6)]}, "
        f"mu={mu:.1f}, union={len(uni)}  耗时 {(time.time()-t0)/60:.1f} min")
    return pool_data


# ============================================================
#  段3: SFS + drop-one
# ============================================================
def stage3_sfs(exp_dir, tag, pool_data, ALL_COLS):
    t0 = time.time()
    banner("PIPE03 SFS + drop-one (无偏 DB+DS)", {
        "tag": tag, "SEEDS": SEEDS,
        "RF": "n=300, d6, leaf2, balanced",
        "GB": "n=150, lr0.03, d3, sub0.8",
    })
    db, ds, _ = load_data()
    SMOTEGATE = os.environ.get("EHG_SMOTE", "0") == "1"
    y_all = db.label.values.astype(int)
    g_all = db.recname.values
    gest_all = db.gest.values.astype(float)
    rect_all = db.rectime.values.astype(float)
    lead_all = db.lead.values.astype(float)
    X_raw = db[ALL_COLS].values.astype(np.float64)
    Xds = ds[ALL_COLS].values.astype(np.float64)
    yds = ds.label.values.astype(int)
    if SMOTEGATE:
        log("  [SMOTEGATE] SMOTE 应用于全量 DB 后再划分 (T3 重采样泄漏对照)")
        X_raw, y_all, g_all, gest_all, rect_all, lead_all = smote_before_split(
            X_raw, y_all, g_all, gest_all, rect_all, lead_all,
            k_neighbors=5, random_state=OUTER_SEED)
        log(f"  [SMOTEGATE] 增广后 DB {len(y_all)} 条 (pos {int(y_all.sum())})")
    CIDX = {c: i for i, c in enumerate(ALL_COLS)}

    fold_pass_lists = pool_data["fold_pass"]
    sgkf_out = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=OUTER_SEED)
    splits = list(sgkf_out.split(X_raw, y_all, g_all))

    fold_rows, all_Fk = [], {}
    oof_pred = np.full(len(y_all), np.nan)  # 记录级 OOF: 每折测试集预测回填
    ds_pred_folds = []  # 每折模型对全 DS 的预测, 供"预测集平均"口径使用
    for k, (tr, te) in enumerate(splits, 1):
        tk = time.time()
        ytr, yte = y_all[tr], y_all[te]
        Xtr, Xte = impute_median(X_raw[tr], X_raw[te])
        Xtr_for_ds, Xds_imp = impute_median(X_raw[tr], Xds)
        passed_k = fold_pass_lists[k]
        pool_idx = [CIDX[c] for c in passed_k if c in CIDX]
        log(f"---- {tag} 折{k}: train {len(tr)}, stage1 {len(passed_k)}")

        if pool_idx:
            if SEL == "direct":
                sel_k = list(pool_idx)
                log(f"    {tag}/f{k} DIRECT: 准入全集 {len(sel_k)} 直接进分类器")
            elif SEL == "embed":
                sel_k = embedded_select(list(pool_idx), Xtr, ytr, g_all[tr])
                log(f"    {tag}/f{k} EMBED: L1 选中 {len(sel_k)}/{len(pool_idx)}")
            else:  # sfs (默认): 前向选择 + drop-one 去冗余
                sel_k, m1 = sfs_select(list(pool_idx), Xtr, ytr, g_all[tr], f"{tag}/f{k}")
                sel_k, m2 = drop_one(sel_k, Xtr, ytr, g_all[tr], f"{tag}/f{k}")
        else:
            sel_k = []

        F_k = [ALL_COLS[i] for i in sel_k]
        all_Fk[f"fold{k}"] = F_k
        auc_db = np.nan
        auc_ds = np.nan
        if sel_k:
            p_db = rfgb_proba(Xtr[:, sel_k], ytr, Xte[:, sel_k], OUTER_SEED)
            auc_db = roc_auc_score(yte, p_db)
            oof_pred[te] = p_db  # 回填 OOF
            p_ds = rfgb_proba(Xtr[:, sel_k], ytr, Xds_imp[:, sel_k], OUTER_SEED)
            auc_ds = roc_auc_score(yds, p_ds)
            ds_pred_folds.append(p_ds)

        fold_rows.append({"Fold": k, "N train": len(tr), "N test": len(te),
                          "Pos test": int(yte.sum()), "Stage1 pass": len(passed_k),
                          "Stage1 list": ", ".join(passed_k),
                          "N selected": len(F_k), "F_k": ", ".join(F_k),
                          "DB AUC (unbiased)": auc_db,
                          "DS AUC (unbiased)": auc_ds,
                          "sec": round(time.time() - tk, 0)})
        log(f"    {tag} 折{k}: F_k({len(F_k)})={F_k} DB={auc_db:.3f} DS={auc_ds:.3f}")

    db_oof = roc_auc_score(y_all, oof_pred) if not np.any(np.isnan(oof_pred)) else np.nan
    log(f"  [OOF] DB AUC = {db_oof:.3f}  (记录级, {int(np.sum(~np.isnan(oof_pred)))}/{len(y_all)} 条)")
    dfr = pd.DataFrame(fold_rows)
    db_auc = np.array([r["DB AUC (unbiased)"] for r in fold_rows], dtype=float)
    ds_auc = np.array([r["DS AUC (unbiased)"] for r in fold_rows], dtype=float)
    u = set().union(*all_Fk.values())
    inter = set.intersection(*[set(v) for v in all_Fk.values()]) if all_Fk else set()

    OUTX = os.path.join(exp_dir, f"pipe_03_sfs_{tag}.xlsx")
    with pd.ExcelWriter(OUTX, engine="openpyxl") as xw:
        dfr.to_excel(xw, sheet_name="Folds", index=False)
        pd.DataFrame([{"Config": tag,
                       "DB AUC (unbiased)": f"{np.nanmean(db_auc):.3f}±{np.nanstd(db_auc):.3f}",
                       "DB OOF AUC": round(float(db_oof), 3) if db_oof is not None and not np.isnan(db_oof) else "NA",
                       "DS AUC (unbiased)": f"{np.nanmean(ds_auc):.3f}±{np.nanstd(ds_auc):.3f}",
                       "DB per fold": ", ".join(f"{v:.3f}" for v in db_auc),
                       "DS per fold": ", ".join(f"{v:.3f}" for v in ds_auc),
                       "union Fk size": len(u),
                       "intersection Fk": ", ".join(sorted(inter))}]).to_excel(
            xw, sheet_name="Overview", index=False)
        pd.DataFrame({"union_Fk": sorted(u)}).to_excel(xw, sheet_name="union_Fk", index=False)

    sfs_data = {"union_Fk": sorted(u), "all_Fk": all_Fk,
                "db_auc": {f"fold{k+1}": float(db_auc[k]) for k in range(len(db_auc))},
                "ds_auc": {f"fold{k+1}": float(ds_auc[k]) for k in range(len(ds_auc))},
                "db_oof": float(db_oof) if db_oof is not None and not np.isnan(db_oof) else None,
                "tag": tag}
    pkl_out = os.path.join(exp_dir, f"pipe_03_sfs_{tag}.pkl")
    with open(pkl_out, "wb") as f:
        pickle.dump(sfs_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ---- 落盘预测概率, 供 bootstrap CI / ROC / D2-D4 等下游分析复用, 避免重复训练 ----
    # 注意: DS 的女性(cluster)必须用 tpehgt_ds_mother_map.csv, recname 前缀 pNNN/tNNN 是 13+13, 不能用来推断
    try:
        _mm = pd.read_csv(os.path.join(
            _ROOT, "results", "checkpoints_lit_features_v2", "tpehgt_ds_mother_map.csv"))
        _mid = _mm.set_index("recname")["mother_id"].reindex(
            ds.recname.astype(str).values)
        if _mid.isna().any():
            raise ValueError(f"母亲映射缺 {int(_mid.isna().sum())} 条 DS 记录")
        # 手稿 §3.3 口径: 每个折内模型预测全 DS, 先平均 5 个预测集, 再算 AUC.
        # 直接堆叠 50 个预测(跨种子)会得到一个不同的点数, 不能与"预测集平均"混用.
        _dp = np.vstack(ds_pred_folds) if ds_pred_folds else \
            np.full((1, len(yds)), np.nan)
        ds_pred_avg = np.nanmean(_dp, axis=0)
        auc_ds_avg = roc_auc_score(yds, ds_pred_avg)
        log(f"  DS 预测集平均 AUC = {auc_ds_avg:.4f} (逐折均值 {np.nanmean(ds_auc):.4f})")
        np.savez(os.path.join(exp_dir, f"preds_{tag}.npz"),
                 y_all=y_all.astype(int), oof_pred=oof_pred.astype(float),
                 gest_all=gest_all, rect_all=rect_all, lead_all=lead_all,
                 y_ds=yds.astype(int), ds_pred=ds_pred_avg.astype(float),
                 ds_pred_all=_dp.astype(float),
                 mother_id=_mid.astype(str).values)
        log(f"  预测落盘 preds_{tag}.npz: DB {int(np.sum(~np.isnan(oof_pred)))} 条, "
            f"DS {len(ds_pred_avg)} 条, {int(_mid.nunique())} 名女性")
    except Exception as e:
        log(f"  !! 预测落盘失败(不影响主流程): {e}")

    log(f"==== {tag}: 无偏DB={np.nanmean(db_auc):.3f}±{np.nanstd(db_auc):.3f}, "
        f"OOF DB={db_oof:.3f}, "
        f"DS={np.nanmean(ds_auc):.3f}±{np.nanstd(ds_auc):.3f}  "
        f"耗时 {(time.time()-t0)/60:.1f} min")
    return sfs_data


# ============================================================
#  段4: 有偏评估 (全量 DB 训练 → DB + DS 预测)
# ============================================================
def stage4_eval(exp_dir, tag, sfs_data, ALL_COLS):
    t0 = time.time()
    banner("PIPE04 有偏评估 (全量 DB 训练)", {
        "tag": tag, "union_Fk size": len(sfs_data["union_Fk"]),
    })
    db, ds, _ = load_data()
    Xdb = db[ALL_COLS].values.astype(np.float64)
    ydb = db.label.values.astype(int)
    Xds = ds[ALL_COLS].values.astype(np.float64)
    yds = ds.label.values.astype(int)
    CIDX = {c: i for i, c in enumerate(ALL_COLS)}
    log(f"DB {len(ydb)} (pos {ydb.sum()}), DS {len(yds)} (pos {yds.sum()})")

    feats = [c for c in sfs_data["union_Fk"] if c in ALL_COLS]
    if not feats:
        raise RuntimeError("union_Fk 为空, 无法评估")
    log(f"特征集: {len(feats)} 个: {feats}")

    idx = [CIDX[c] for c in feats]
    Xdb_imp, _ = impute_median(Xdb, Xdb)
    _, Xds_imp = impute_median(Xdb, Xds)

    p_db = rfgb_proba(Xdb_imp[:, idx], ydb, Xdb_imp[:, idx], OUTER_SEED)
    auc_db = roc_auc_score(ydb, p_db)
    p_ds = rfgb_proba(Xdb_imp[:, idx], ydb, Xds_imp[:, idx], OUTER_SEED)
    auc_ds = roc_auc_score(yds, p_ds)

    log(f"  {tag}: 全量 DB 训练 → DB AUC={auc_db:.3f}, DS AUC={auc_ds:.3f}")

    OUTX = os.path.join(exp_dir, f"pipe_04_eval_{tag}.xlsx")
    with pd.ExcelWriter(OUTX, engine="openpyxl") as xw:
        pd.DataFrame([{"Config": tag, "Feat size": len(feats),
                       "Features": ", ".join(feats),
                       "DB AUC (biased)": round(auc_db, 3),
                       "DS AUC (biased)": round(auc_ds, 3)}]).to_excel(
            xw, sheet_name="Overview", index=False)

    eval_data = {"tag": tag, "feats": feats, "feat_size": len(feats),
                 "db_auc_biased": float(auc_db), "ds_auc_biased": float(auc_ds)}
    pkl_out = os.path.join(exp_dir, f"pipe_04_eval_{tag}.pkl")
    with open(pkl_out, "wb") as f:
        pickle.dump(eval_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    log(f"==== {tag}: 有偏 DB={auc_db:.3f}, DS={auc_ds:.3f}  耗时 {(time.time()-t0)/60:.1f} min")
    return eval_data


# ============================================================
#  主入口
# ============================================================
def main():
    global _logf, OUTER_SEED, LEAD_SPLIT

    if len(sys.argv) < 2 or sys.argv[1] not in CONFIGS:
        print("用法: python run_experiment.py <tag> [seed] [lead_split]")
        print(f"可用 tags: {list(CONFIGS.keys())}")
        print("示例: python run_experiment.py P1CI_P6eff050 123")
        print("      python run_experiment.py P1CI_P6eff050 42 4   # lead敏感性: lead_split=4")
        return
    tag = sys.argv[1]
    if len(sys.argv) >= 3:
        OUTER_SEED = int(sys.argv[2])
    if len(sys.argv) >= 4:
        LEAD_SPLIT = float(sys.argv[3])   # lead 分组阈值, 默认 6.0; 敏感性实验传 4/5/7/8
    stamp = time.strftime("%Y%m%d_%H%M%S")
    lead_tag = "" if LEAD_SPLIT == 6.0 else f"_lead{LEAD_SPLIT:g}"
    # 把非默认变量写进目录名, 避免不同配置的结果混在同一批目录里
    extra = ""
    if CLASSL != "rfgb":
        extra += f"_cl{CLASSL}"
    if os.environ.get("EHG_LEAKGATE", "0") == "1":
        extra += "_leak"
    if os.environ.get("EHG_SMOTE", "0") == "1":
        extra += "_smote"
    exp_dir = os.path.join(
        EXP_ROOT,
        f"{tag if SEL=='sfs' else f'{tag}_{SEL}'}{extra}{lead_tag}"
        f"_seed{OUTER_SEED}_{stamp}")
    os.makedirs(exp_dir, exist_ok=True)

    log_path = os.path.join(exp_dir, "run.log")
    _logf = open(log_path, "w", encoding="utf-8")

    try:
        log(f"实验: {tag}, SEL={SEL}, OUTER_SEED={OUTER_SEED}, LEAD_SPLIT={LEAD_SPLIT:g}")
        log(f"数据源 FT_CSV: {FT_CSV}")
        log(f"输出: {exp_dir}")
        t_total = time.time()

        db, _, ALL_COLS = load_data()
        log(f"数据: DB {len(db)} 条, {len(ALL_COLS)} 特征")

        fold_aucs = stage1_auc(exp_dir, ALL_COLS)
        naming_tag = tag if SEL == "sfs" else f"{tag}_{SEL}"
        pool_data = stage2_pool(exp_dir, naming_tag, fold_aucs, ALL_COLS, base_tag=tag)
        sfs_data = stage3_sfs(exp_dir, naming_tag, pool_data, ALL_COLS)
        eval_data = stage4_eval(exp_dir, naming_tag, sfs_data, ALL_COLS)

        log("=" * 60)
        log(f"全部完成! 耗时 {(time.time()-t_total)/60:.1f} min")
        db_auc_vals = list(sfs_data["db_auc"].values())
        ds_auc_vals = list(sfs_data["ds_auc"].values())
        log(f"  [无偏] DB AUC = {np.mean(db_auc_vals):.3f}±{np.std(db_auc_vals):.3f}")
        log(f"  [无偏] DS AUC = {np.mean(ds_auc_vals):.3f}±{np.std(ds_auc_vals):.3f}")
        db_oof_val = sfs_data.get("db_oof")
        log(f"  [OOF]  DB AUC = {db_oof_val:.3f}" if db_oof_val is not None else "  [OOF]  DB AUC = NA")
        log(f"  [有偏] DB AUC = {eval_data['db_auc_biased']:.3f}")
        log(f"  [有偏] DS AUC = {eval_data['ds_auc_biased']:.3f}")
        log(f"  结果: {exp_dir}")
        log("=" * 60)
    except Exception as e:
        log(f"!!! 运行异常: {e}")
        import traceback
        log(traceback.format_exc())
    finally:
        _logf.close()
        _logf = None

    for f in sorted(os.listdir(exp_dir)):
        sz = os.path.getsize(os.path.join(exp_dir, f))
        log(f"  {f}  ({sz/1024:.1f} KB)" if sz > 1024 else f"  {f}  ({sz} B)")


if __name__ == "__main__":
    main()
