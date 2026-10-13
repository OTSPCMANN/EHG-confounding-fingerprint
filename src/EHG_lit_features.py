# -*- coding: utf-8 -*-
"""
EHG 记录级手工特征提取 —— 合并版 (原 EHG_lit_features.py + EHG_lit_features_v2.py)
==================================================================================
本文件是特征提取环节的**唯一入口**, 产物直接供 pipe01_AUC.py 使用:

    checkpoints_union/rec_full.npy  (由 EHG_three_ds_union.py --full_records_only 生成, 与窗口缓存同目录)
            │
            │  python EHG_lit_features.py
            ▼
    checkpoints_lit_features_v2/lit_features_v2.csv   ← pipe01/pipe02/pipe03/pipe04
    checkpoints_lit_features_v2/feature_id_map.csv    ← 替代已弃用的 EHG论文0822/baseline_results_75col.xlsx
    checkpoints_lit_features_v2/pair_auc_matrix.csv
    checkpoints_lit_features_v2/honest_features.csv
    checkpoints_lit_features_v2/lit_features_report.txt

为什么要合并
------------
原 v1 只算 26 个单通道特征(+5 entx +3 跨通道 = 65 列), v2 再单独扫一遍信号补 5 个
特征(×mean/s3 = 10 列) 后按 recname 合并成 75 列。两次遍历整条记录(熵类最耗时)
且中间产物 lit_features_all.csv 只为合并而存在。合并后:
  · 每条记录只读一次信号, 一次算完 31 个单通道特征
  · 列顺序由 ehg_featmeta.COLUMN_ORDER 固定, 与历史产物逐列一致, 结果可复现
  · 不再依赖任何 xlsx 基线文件

特征体系 (75 列 = 36 个基础名 × mean/s3 + 3 个跨通道)
------------------------------------------------------
  A 时域(9): peak, rms, energy, area, waveform_len, variance, logdet, rms_env, p2p
  A burst(4): burst_dur, burst_rate, burst_interval, activation_dur
  B 频域(11): peak_freq_full, median_freq, centroid_freq, total_power,
              rel_fwl, rel_cont, rel_high, fwl_peak_freq, cv_env, burst_ratio, zcr
  C 传导(2): cv_vert(S1-S3), cv_horiz(S2-S4)      [跨通道, 无 mean/s3]
  D 非线性(6): apen, sampen, fuzzyen, mse, lzc, corr_dim
  E 传播(1): hv_ratio                              [跨通道, 无 mean/s3]
  F 其他(1): model_error
  G 派生(5): entx_{apen,sampen,fuzzyen,mse,lzc} = 原值 × (1 + rectime/37)

熵类计算: 分段法 — 30s 段(600点)逐段计算后取记录均值, 避免全长 O(n²);
          MSE 用 60s 段粗粒化到尺度 1-5。

诚实特征四条预注册标准
----------------------
  1. 时间配对检验: 激活类过 P2(短lead早产 vs 晚记录足月), 形态类过 P3(长lead早产 vs 早记录足月)
     folded AUC >= 0.60 且 bootstrap CI 不含 0.5
  2. 外部方向复现: TPEHGT DS 26 条 同向 且 folded AUC > 0.5
  3. 非孕周代理: P6(早记录足月 vs 晚记录足月) folded AUC <= 主配对 folded AUC + 0.05
  4. 双通道不矛盾: mean 版与 S3 版方向不相反

用法
----
    python EHG_lit_features.py                    # 全量 326 条 (DB 300 + TPEHGT 26)
    python EHG_lit_features.py --limit 8          # 冒烟测试
    python EHG_lit_features.py --workers 8        # 提高并行度
    python EHG_lit_features.py --skip-honest      # 只出特征表, 跳过配对矩阵/诚实判定
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.signal import welch, hilbert, butter, filtfilt
from scipy.spatial.distance import pdist
from sklearn.metrics import roc_auc_score

# --- repository path bootstrap (added for the public release) -----------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "analysis"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ehg_featmeta as FM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FULLREC_DIR = os.path.join(_ROOT, "checkpoints_union")  # 记录级全信号, 与窗口缓存同目录
FULLREC_PATH = os.path.join(FULLREC_DIR, "rec_full.npy")
FULLMETA_PATH = os.path.join(FULLREC_DIR, "rec_full_meta.csv")

FS = 20
CH = {"S1": 0, "S2": 1, "S3": 2, "S4": 3, "S5": 4, "S6": 5}
ELEC_DIST_M = 0.07          # 电极间距近似 (与图网络 σ=70mm 同源)
LEAD_SPLIT = 6.0            # 短/长 lead 分界 (周)
RECTIME_SPLIT_EARLY = 26.0  # 早记录足月: rectime < 26
RECTIME_SPLIT_LATE = 26.0   # 晚记录足月: rectime >= 26
BOOT_N = 2000
SEED = 42

ENT_M = 2
ENT_R = 0.15
SEG_ENT = FS * 30           # 熵类分段: 30s
SEG_MSE = FS * 60           # MSE 分段: 60s
MSE_SCALES = 5
MAX_SEG_ENT = 20            # 熵类最多 20 段 (10min, 控制 O(n²) 总量)
MAX_SEG_MSE = 10            # MSE 最多 10 段 (10min)
N_WORKERS = 4               # 多进程并行

AR_MAX_ORDER = 30           # model_error: AIC 搜索最大阶
CD_N_SUB = 2000             # corr_dim: 等距子采样点数 (确定性, 无随机性)
CD_M_MAX = 8                # corr_dim: 嵌入维上限
CD_M_REPORT = 6             # corr_dim: 报告 D2 的嵌入维

# numpy>=2.0 移除了 np.trapz, 这里做兼容
_TRAPZ = getattr(np, "trapezoid", None) or np.trapz


# ==================== 工具 ====================
def segments(x, seg_len, max_segs=None):
    """全长信号切为不重叠段 (去掉 NaN 尾), 返回段列表 (可封顶段数)"""
    x = x[~np.isnan(x)]
    n = (len(x) // seg_len) * seg_len
    if n < seg_len // 2:
        return []
    segs = [x[i:i + seg_len] for i in range(0, n, seg_len)]
    if max_segs is not None:
        segs = segs[:max_segs]
    return segs


def bootstrap_ci(y, p, n=BOOT_N):
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    a = roc_auc_score(y, p)
    rng = np.random.RandomState(SEED)
    boots = []
    for _ in range(n):
        idx = rng.choice(len(y), size=len(y), replace=True)
        yy = y[idx]
        if yy.sum() < 2 or (1 - yy).sum() < 2:
            continue
        try:
            boots.append(roc_auc_score(yy, p[idx]))
        except Exception:
            pass
    if not boots:
        return a, np.nan, np.nan
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return a, lo, hi


def ci_excludes_half(auc, lo, hi):
    """CI 是否不含 0.5 (任一方向)"""
    if any(np.isnan(v) for v in (auc, lo, hi)):
        return False
    if auc >= 0.5:
        return lo > 0.5
    return hi < 0.5


# ==================== A 时域 ====================
def time_domain_feats(x):
    x = x[~np.isnan(x)]
    if len(x) < FS * 60:
        return {}
    out = {}
    out["peak"] = float(np.max(np.abs(x)))
    out["rms"] = float(np.sqrt(np.mean(x ** 2)))
    out["energy"] = float(np.sum(x ** 2) / FS)
    out["area"] = float(_TRAPZ(np.abs(x)) / FS)
    out["waveform_len"] = float(np.sum(np.abs(np.diff(x))))
    v = float(np.var(x))
    out["variance"] = v
    out["logdet"] = float(np.log10(v)) if v > 0 else np.nan
    return out


def detect_bursts(x, fs=FS):
    """文献标准: 平滑包络 > 2×中位数基线 且 持续 > 10s, 合并 <2s 间隙"""
    x = x[~np.isnan(x)]
    if len(x) < fs * 60:
        return []
    env = np.abs(hilbert(x))
    k = fs * 2
    env_s = np.convolve(env, np.ones(k) / k, mode="same")
    baseline = np.median(env_s)
    if baseline <= 0:
        return []
    active = env_s > 2.0 * baseline
    segs = []
    in_seg = False
    start = 0
    for i, a in enumerate(active):
        if a and not in_seg:
            start, in_seg = i, True
        elif not a and in_seg:
            segs.append((start, i))
            in_seg = False
    if in_seg:
        segs.append((start, len(active)))
    merged = []
    for s, e in segs:
        if merged and s - merged[-1][1] < 2 * fs:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(s / fs, e / fs) for s, e in merged if (e - s) > 10 * fs]


def burst_feats(x, fs=FS):
    bursts = detect_bursts(x, fs)
    x = x[~np.isnan(x)]
    T = len(x) / fs
    if not bursts or T <= 0:
        return {"burst_dur": np.nan, "burst_rate": np.nan,
                "burst_interval": np.nan, "activation_dur": np.nan}
    durs = [e - s for s, e in bursts]
    intervals = [bursts[i][0] - bursts[i - 1][1]
                 for i in range(1, len(bursts))]
    return {
        "burst_dur": float(np.mean(durs)),
        "burst_rate": float(len(bursts) / (T / 60.0)),
        "burst_interval": float(np.mean(intervals)) if intervals else np.nan,
        "activation_dur": float(sum(durs) / T),
    }


# ==================== B 频域 ====================
def freq_domain_feats(x, fs=FS):
    x = x[~np.isnan(x)]
    if len(x) < fs * 60:
        return {}
    f, p = welch(x, fs=fs, nperseg=min(len(x), 3600))
    band = (f >= 0.08) & (f <= 5.0)
    fb, pb = f[band], p[band]
    ptot = pb.sum()
    if ptot <= 0:
        return {}
    out = {}
    out["peak_freq_full"] = float(fb[np.argmax(pb)])
    cum = np.cumsum(pb)
    out["median_freq"] = float(np.interp(ptot / 2, cum, fb))
    out["centroid_freq"] = float((fb * pb).sum() / ptot)
    out["total_power"] = float(ptot)
    out["rel_fwl"] = float(pb[(fb >= 0.125) & (fb <= 0.575)].sum() / ptot)
    out["rel_cont"] = float(pb[(fb >= 0.20) & (fb <= 0.45)].sum() / ptot)
    out["rel_high"] = float(pb[(fb >= 0.8) & (fb <= 3.0)].sum() / ptot)
    fwl = (fb >= 0.125) & (fb <= 0.575)
    if fwl.sum() > 0 and pb[fwl].sum() > 0:
        out["fwl_peak_freq"] = float(fb[fwl][np.argmax(pb[fwl])])
    env = np.abs(hilbert(x))
    k = fs * 2
    env_s = np.convolve(env, np.ones(k) / k, mode="same")
    mu, sd = np.mean(env_s), np.std(env_s)
    if sd > 0:
        out["cv_env"] = float(sd / mu)
    out["burst_ratio"] = float((env_s > mu + sd).mean())
    return out


# ==================== C 传导 (近似) ====================
def conduction_feats(sig6, fs=FS):
    out = {}
    b, a = butter(4, [0.34 / (fs / 2), 1.0 / (fs / 2)], btype="band")
    for name, (ia, ib) in {"vert": (CH["S1"], CH["S3"]),
                           "horiz": (CH["S2"], CH["S4"])}.items():
        xa, xb = sig6[:, ia], sig6[:, ib]
        ok = ~(np.isnan(xa) | np.isnan(xb))
        xa, xb = xa[ok], xb[ok]
        if len(xa) < fs * 120:
            out[f"cv_{name}"] = np.nan
            continue
        xa_f = filtfilt(b, a, xa)
        xb_f = filtfilt(b, a, xb)
        max_lag = int(2 * fs)
        n = len(xa_f)
        xa_c, xb_c = xa_f - xa_f.mean(), xb_f - xb_f.mean()
        denom = np.sqrt((xa_c ** 2).sum() * (xb_c ** 2).sum())
        if denom <= 0:
            out[f"cv_{name}"] = np.nan
            continue
        lags = np.arange(-max_lag, max_lag + 1)
        cc = np.empty(len(lags))
        for j, lg in enumerate(lags):
            if lg >= 0:
                cc[j] = (xa_c[:n - lg] * xb_c[lg:]).sum()
            else:
                cc[j] = (xa_c[-lg:] * xb_c[:n + lg]).sum()
        cc /= denom
        jbest = int(np.argmax(np.abs(cc)))
        lag, r = lags[jbest], cc[jbest]
        if abs(r) < 0.2 or lag == 0:
            out[f"cv_{name}"] = np.nan
        else:
            out[f"cv_{name}"] = ELEC_DIST_M / (abs(lag) / fs)
    return out


# ==================== D 非线性 (分段法) ====================
def _embed(x, m):
    n = len(x) - m + 1
    return np.lib.stride_tricks.sliding_window_view(x, m)[:n]


def _entropy_counts(x, m, r):
    """一次距离计算, 同时给出 ApEn/SampEn 所需的 per-向量计数 (含自身)."""
    emb = _embed(x, m)
    d = np.max(np.abs(emb[:, None, :] - emb[None, :, :]), axis=2)
    return (d <= r).sum(axis=1).astype(np.float64)


def entropy_pair(x, m=ENT_M, r_scale=ENT_R):
    """返回 (apen, sampen) 或 (nan, nan). x: 单段."""
    sd = np.std(x)
    if sd < 1e-12 or len(x) < 4 * m + 10:
        return np.nan, np.nan
    r = r_scale * sd
    cnt0 = _entropy_counts(x, m, r)          # m 维, 含自身
    cnt1 = _entropy_counts(x, m + 1, r)      # m+1 维, 含自身
    # ApEn (phi 差, 防 log0)
    phi0 = np.mean(np.log(np.maximum(cnt0, 1.0)))
    phi1 = np.mean(np.log(np.maximum(cnt1, 1.0)))
    apen = phi0 - phi1
    # SampEn (总匹配数, 排除自身)
    B = cnt0.sum() - len(cnt0)
    A = cnt1.sum() - len(cnt1)
    sampen = -np.log(A / B) if A > 0 and B > 0 else np.nan
    return float(apen), float(sampen)


def fuzzyen_seg(x, m=ENT_M, r_scale=ENT_R, n_order=2):
    sd = np.std(x)
    if sd < 1e-12 or len(x) < 4 * m + 10:
        return np.nan
    r = r_scale * sd

    def fuzzy_d(emb):
        d = np.max(np.abs(emb[:, None, :] - emb[None, :, :]), axis=2)
        mu = np.exp(-(d ** n_order) / r)
        np.fill_diagonal(mu, 0)
        return mu.sum(axis=1) / (mu.shape[0] - 1)

    e0 = _embed(x, m).copy()
    e0 = e0 - e0.mean(axis=1, keepdims=True)
    e1 = _embed(x, m + 1).copy()
    e1 = e1 - e1.mean(axis=1, keepdims=True)
    phi0 = fuzzy_d(e0).sum()
    phi1 = fuzzy_d(e1).sum()
    if phi0 <= 0 or phi1 <= 0:
        return np.nan
    return float(-np.log(phi1 / phi0))


def lzc_seg(x):
    s = (x > np.median(x)).astype(np.uint8)
    n = len(s)
    if n < 100:
        return np.nan
    i, k, l, c = 0, 1, 1, 1
    while l + k - 1 < n:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
        else:
            c += 1
            if l >= i + k:
                i, l, k = l, l + k, 1
            else:
                l, k = l + k, 1
    return float(c * np.log2(n) / n)


def mse_record(x):
    """多尺度熵: 60s 段粗粒化到尺度 1-5, SampEn 均值"""
    vals = []
    for seg in segments(x, SEG_MSE, MAX_SEG_MSE):
        for s in range(1, MSE_SCALES + 1):
            if s == 1:
                cg = seg
            else:
                cg = seg[:len(seg) // s * s].reshape(-1, s).mean(axis=1)
            _, sp = entropy_pair(cg)
            if not np.isnan(sp):
                vals.append(sp)
    return float(np.mean(vals)) if vals else np.nan


def nonlinear_feats(x):
    """x: 单通道全长. 熵类分段计算."""
    apens, sampens, fuzzies, lzcs = [], [], [], []
    for seg in segments(x, SEG_ENT, MAX_SEG_ENT):
        ap, sp = entropy_pair(seg)
        if not np.isnan(ap):
            apens.append(ap)
        if not np.isnan(sp):
            sampens.append(sp)
        fz = fuzzyen_seg(seg)
        if not np.isnan(fz):
            fuzzies.append(fz)
        lz = lzc_seg(seg)
        if not np.isnan(lz):
            lzcs.append(lz)
    return {
        "apen": float(np.mean(apens)) if apens else np.nan,
        "sampen": float(np.mean(sampens)) if sampens else np.nan,
        "fuzzyen": float(np.mean(fuzzies)) if fuzzies else np.nan,
        "lzc": float(np.mean(lzcs)) if lzcs else np.nan,
        "mse": mse_record(x),
    }


# ==================== v2 增补的 5 个特征 ====================
def model_error(x, max_order=AR_MAX_ORDER):
    """AR(Yule-Walker+AIC) 归一化预测残差 = 1-R². 越低 = 信号越规律可预测.

    Powar 2026 (s42452-026-08571-8): AR(Yule-Walker)+AIC 选阶, 归一化预测残差
    = 1-R², 度量宫缩规律性/可预测性。PDF 提取中公式损坏, 按其文字定义
    (原信号与 AR 预测的偏离) 实现为标准归一化残差方差。
    实现: 每阶直接解 Yule-Walker Toeplitz 方程 (x[n]=c·x[n-1..n-p]+e),
    避免手写 Levinson-Durbin 的符号约定错误。"""
    x = x[~np.isnan(x)]
    if len(x) < FS * 60:
        return np.nan
    xc = x - x.mean()
    n = len(xc)
    if np.sum(xc ** 2) <= 0:
        return np.nan
    # 自相关 (FFT, 有偏估计 — Yule-Walker 标准口径)
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    f = np.fft.rfft(xc, n=nfft)
    r = np.fft.irfft(f * np.conj(f), n=nfft)[:max_order + 1]
    if r[0] <= 0:
        return np.nan
    best_aic, best_c, best_p = np.inf, None, 0
    for p in range(1, max_order + 1):
        # Yule-Walker: R·c = b, R=toeplitz(r[0..p-1]), b=r[1..p]
        R = np.empty((p, p))
        for i in range(p):
            for j in range(p):
                R[i, j] = r[abs(i - j)]
        b = r[1:p + 1]
        try:
            c = np.linalg.solve(R, b)
        except np.linalg.LinAlgError:
            try:
                c = np.linalg.lstsq(R, b, rcond=None)[0]
            except Exception:
                break
        E = r[0] - float(c @ b)   # 残差方差 (Yule-Walker 解析式)
        aic = n * np.log(max(E, 1e-12)) + 2 * p
        if aic < best_aic:
            best_aic, best_c, best_p = aic, c, p
    if best_p < 1 or best_c is None:
        return np.nan
    # 样本内一步预测残差: e[n] = x[n] - Σ c[j]·x[n-j], n=p..N-1
    e = xc[best_p:n].copy()
    for j in range(1, best_p + 1):
        e -= best_c[j - 1] * xc[best_p - j:n - j]
    denom = np.sum(xc[best_p:] ** 2)
    if denom <= 0:
        return np.nan
    return float(np.sum(e ** 2) / denom)


def zcr(x):
    """过零率 (r3 原版口径: 符号变化数 / 采样点数)"""
    x = x[~np.isnan(x)]
    if len(x) < FS * 60:
        return np.nan
    signs = np.sign(x)
    signs[signs == 0] = 1
    return float(np.sum(np.abs(np.diff(signs)) > 0) / len(signs))


def rms_env(x):
    """Hilbert 包络 RMS (r3 原版口径)"""
    x = x[~np.isnan(x)]
    if len(x) < FS * 60:
        return np.nan
    env = np.abs(hilbert(x))
    return float(np.sqrt(np.mean(env ** 2)))


def p2p(x):
    """峰峰值 max - min"""
    x = x[~np.isnan(x)]
    if len(x) < FS * 60:
        return np.nan
    return float(np.max(x) - np.min(x))


def corr_dim(x, n_sub=CD_N_SUB, m_max=CD_M_MAX, m_report=CD_M_REPORT):
    """Grassberger-Procaccia 相关维数 D2.

    延迟 tau = 自相关函数第一极小值 (文献标准嵌入延迟选择);
    等距子采样 (无随机性), 嵌入维 1..m_max, 取 m_report 处的 log-log 斜率."""
    x = x[~np.isnan(x)]
    if len(x) < FS * 60:
        return np.nan
    xc = x - x.mean()
    n = len(xc)
    if np.sum(xc ** 2) <= 0:
        return np.nan
    # ACF (FFT)
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    f = np.fft.rfft(xc, n=nfft)
    acf = np.fft.irfft(f * np.conj(f), n=nfft)[:n // 2]
    if acf[0] <= 0:
        return np.nan
    acf /= acf[0]
    tau = None
    for i in range(1, min(len(acf) - 1, 400)):
        if acf[i] < acf[i - 1] and acf[i] <= acf[i + 1]:
            tau = i
            break
    if tau is None or tau < 1:
        tau = 20
    # 等距子采样 (确定性)
    hi = n - m_max * tau
    if hi <= n_sub:
        idx = np.arange(0, hi)
    else:
        idx = np.arange(0, hi, hi // n_sub)[:n_sub]
    xs = xc[idx]
    if len(idx) < 200:
        return np.nan
    # r 网格: m=1 距离分布的 5%-95% 分位 (对数等距)
    d1 = pdist(xs[:, None])
    if len(d1) == 0:
        return np.nan
    r_lo, r_hi = np.percentile(d1, 5), np.percentile(d1, 95)
    if r_hi <= r_lo * 1.01 or r_lo <= 0:
        return np.nan
    rs = np.exp(np.linspace(np.log(r_lo), np.log(r_hi), 25))
    slopes = []
    for m in range(1, m_max + 1):
        emb = np.empty((len(idx), m))
        for j in range(m):
            emb[:, j] = xc[idx + j * tau]
        d = pdist(emb)
        C = np.array([(d < r).mean() for r in rs])
        ok = (C > 0) & (C < 1)
        if ok.sum() < 5:
            slopes.append(np.nan)
            continue
        sl = np.polyfit(np.log(rs[ok]), np.log(C[ok]), 1)[0]
        slopes.append(sl)
    slopes = np.array(slopes)
    m_t = min(m_report, m_max)
    if np.isnan(slopes[:m_t]).all():
        valid = np.where(~np.isnan(slopes))[0]
        if len(valid) == 0:
            return np.nan
        m_t = valid[-1] + 1
    return float(slopes[m_t - 1])


# ==================== E 传播 ====================
def propagation_feats(sig6):
    e_h = (np.nansum(sig6[:, CH["S1"]] ** 2)
           + np.nansum(sig6[:, CH["S3"]] ** 2))
    e_v = (np.nansum(sig6[:, CH["S2"]] ** 2)
           + np.nansum(sig6[:, CH["S4"]] ** 2))
    return {"hv_ratio": float(e_v / e_h) if e_h > 0 else np.nan}


# ==================== 单记录提取 ====================
# v1 的 4 个"返回多键"的函数 + v2 的 5 个"返回单值"的函数
SINGLE_CH_FUNCS = [time_domain_feats, burst_feats, freq_domain_feats,
                   nonlinear_feats]
NEW_FEATS = {"model_error": model_error, "zcr": zcr,
             "rms_env": rms_env, "p2p": p2p, "corr_dim": corr_dim}
# 兼容别名: 历史 EHG_lit_features_v2 导出的是 NEW_FEATS
NEW_SINGLE_FUNCS = NEW_FEATS


def _nanmean_or_nan(vals):
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan


def extract_record(sig6, rectime, pool="v3"):
    """单条记录 -> 特征 dict.

    pool="v3" (2026-09-24 修订口径): 189 列
      31 基础特征 × S1..S6 独立通道 (后缀 _s1.._s6, 无通道平均) + 3 跨通道;
      不含 entx 重标定 (rectime 参数注入, 时钟 shortcut 风险)。
      列名与 ehg_featmeta.FEATURE_COLS_V3 一致。
    pool="v2" (历史口径): 75 列 (mean/s3 + entx), 保持逐位可复现。
    """
    # 1) 六个通道各算一遍 (S1..S6)
    per_ch = []
    for c in range(6):
        x = sig6[:, c]
        d = {}
        for fn in SINGLE_CH_FUNCS:
            d.update(fn(x))                       # v1 的 26 个
        for name, fn in NEW_FEATS.items():
            d[name] = fn(x)                       # v2 的 5 个
        per_ch.append(d)

    feats = {}
    if pool == "v3":
        # 2) 31 基础 × S1..S6 独立通道 (无通道平均, 无 entx)
        for b in FM.V1_SINGLE + FM.V2_SINGLE:
            for c in range(6):
                feats[f"{b}_{FM.CH_SUFFIXES[c]}"] = per_ch[c].get(b, np.nan)
        # 3) 跨通道特征 (单口径, 不展开)
        feats.update(conduction_feats(sig6))
        feats.update(propagation_feats(sig6))
        return feats

    # ---- v2 历史口径 ----
    for b in FM.V1_SINGLE + FM.V2_SINGLE:
        feats[f"{b}_mean"] = _nanmean_or_nan([d.get(b, np.nan) for d in per_ch])
        feats[f"{b}_s3"] = per_ch[CH["S3"]].get(b, np.nan)

    # 3) Ent_x 重标定 (Lou 2022): 原值 × (1 + rectime/37)
    for b in FM.ENTX_BASES:
        for suf in ("mean", "s3"):
            v = feats.get(f"{b}_{suf}", np.nan)
            feats[f"entx_{b}_{suf}"] = (v * (1 + rectime / 37.0)
                                        if v is not None and not np.isnan(v)
                                        else np.nan)

    # 4) 跨通道特征 (无 mean/s3 扩展)
    feats.update(conduction_feats(sig6))
    feats.update(propagation_feats(sig6))
    return feats


def _extract_one(task):
    """worker: 单条记录特征提取. task=(arr_row_idx, meta_dict, pool)"""
    idx, m, pool = task
    arr = np.load(FULLREC_PATH, mmap_mode="r")
    sig6 = np.array(arr[idx, :int(m["n_samples"]), :], dtype=np.float64)
    f = extract_record(sig6, m["rectime"], pool=pool)
    f.update(m)
    return f


# ==================== 配对 AUC 矩阵 + 诚实判定 ====================
ACT_KEYS = ["rel_fwl", "rel_cont", "rel_high", "burst", "rms", "energy",
            "peak", "area", "activation", "cv_env", "cv_", "total_power",
            "waveform_len", "variance", "logdet", "hv_ratio", "p2p"]


def build_pair_matrix(db, tg, feat_cols):
    G1 = db[(db.label == 1) & (db.lead <= LEAD_SPLIT)]
    G2 = db[(db.label == 1) & (db.lead > LEAD_SPLIT)]
    G3 = db[(db.label == 0) & (db.rectime < RECTIME_SPLIT_EARLY)]
    G4 = db[(db.label == 0) & (db.rectime >= RECTIME_SPLIT_LATE)]
    pairs = {
        "P1_G1vsG3": (G1, G3), "P2_G1vsG4": (G1, G4),
        "P3_G2vsG3": (G2, G3), "P4_G2vsG4": (G2, G4),
        "P5_G1vsG2": (G1, G2), "P6_G3vsG4": (G3, G4),
    }
    print(f"\n分组: G1短lead早产={len(G1)}, G2长lead早产={len(G2)}, "
          f"G3早记录足月={len(G3)}, G4晚记录足月={len(G4)}", flush=True)

    rows = []
    for fname in feat_cols:
        r = {"feature": fname}
        for pname, (ga, gb) in pairs.items():
            p = np.r_[ga[fname].values, gb[fname].values]
            y = np.r_[np.ones(len(ga)), np.zeros(len(gb))]
            ok = ~np.isnan(p)
            if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
                r[f"{pname}_auc"] = np.nan
                r[f"{pname}_lo"] = np.nan
                r[f"{pname}_hi"] = np.nan
                continue
            a, lo, hi = bootstrap_ci(y[ok], p[ok])
            r[f"{pname}_auc"], r[f"{pname}_lo"], r[f"{pname}_hi"] = a, lo, hi
        # TPEHGT DS 外部方向复现 (26 条, 13-13)
        p = np.r_[tg[tg.label == 1][fname].values,
                  tg[tg.label == 0][fname].values]
        y = np.r_[np.ones((tg.label == 1).sum()),
                  np.zeros((tg.label == 0).sum())]
        ok = ~np.isnan(p)
        r["TG_auc"] = (roc_auc_score(y[ok], p[ok])
                       if ok.sum() >= 15 and len(np.unique(y[ok])) == 2
                       else np.nan)
        rows.append(r)
    return pd.DataFrame(rows), pairs


def judge_honest(res, pool="v3"):
    rows = []
    for _, r in res.iterrows():
        f = r["feature"]
        is_act = any(k in f for k in ACT_KEYS)
        main_pair = "P2_G1vsG4" if is_act else "P3_G2vsG3"
        auc = r.get(f"{main_pair}_auc", np.nan)
        lo = r.get(f"{main_pair}_lo", np.nan)
        hi = r.get(f"{main_pair}_hi", np.nan)
        tg_auc = r.get("TG_auc", np.nan)
        p6 = r.get("P6_G3vsG4_auc", np.nan)
        folded = max(auc, 1 - auc) if not np.isnan(auc) else np.nan
        c1 = (not np.isnan(folded) and folded >= 0.60
              and ci_excludes_half(auc, lo, hi))
        c2 = (not np.isnan(tg_auc) and not np.isnan(auc)
              and np.sign(tg_auc - 0.5) == np.sign(auc - 0.5)
              and max(tg_auc, 1 - tg_auc) > 0.5)
        c3 = True
        if not np.isnan(p6) and not np.isnan(auc):
            c3 = max(p6, 1 - p6) <= folded + 0.05
        c4 = True
        if pool == "v2":
            # v2 历史口径: mean 版与 s3 版方向不相反
            if f.endswith("_mean"):
                f_s3 = f[:-5] + "_s3"
                if f_s3 in set(res["feature"]):
                    a_s3 = res.loc[res["feature"] == f_s3,
                                   f"{main_pair}_auc"].iloc[0]
                    if not np.isnan(a_s3) and not np.isnan(auc):
                        c4 = np.sign(a_s3 - 0.5) == np.sign(auc - 0.5)
        else:
            # v3 口径: 同基底 6 通道判别方向多数一致 (>=4/6 同号)
            if any(f.endswith(s) for s in FM.CHANNEL_SUFFIX_COLS):
                base = f.rsplit("_", 1)[0]
                aucs = []
                for s in FM.CH_SUFFIXES:
                    rr = res.loc[res["feature"] == f"{base}_{s}",
                                 f"{main_pair}_auc"]
                    if len(rr) and not np.isnan(rr.iloc[0]):
                        aucs.append(float(rr.iloc[0]))
                if aucs:
                    n_pos = sum(a - 0.5 > 0 for a in aucs)
                    c4 = max(n_pos, len(aucs) - n_pos) >= 4
        rows.append({
            "feature": f, "type": "activation" if is_act else "morphology",
            "main_pair": main_pair, "main_auc": auc,
            "main_lo": lo, "main_hi": hi, "folded_auc": folded,
            "tg_auc": tg_auc, "p6_auc": p6,
            "c1_time_matched": c1, "c2_tg_replicated": c2,
            "c3_not_gest_proxy": c3, "c4_dual_ch_consistent": c4,
            "honest": all([c1, c2, c3, c4]),
        })
    return pd.DataFrame(rows)


# ==================== 主流程 ====================
def main():
    ap = argparse.ArgumentParser(description="EHG 记录级手工特征提取 (合并版)")
    ap.add_argument("--workers", type=int, default=N_WORKERS,
                    help=f"并行进程数 (默认 {N_WORKERS})")
    ap.add_argument("--limit", type=int, default=0,
                    help="只处理前 N 条记录 (冒烟测试, 0=全部)")
    ap.add_argument("--pool", choices=("v2", "v3"), default="v3",
                    help="特征池口径: v3=六通道独立 189 列 (2026-09-24 修订, "
                         "默认); v2=历史 75 列 (mean/s3+entx)")
    ap.add_argument("--out-dir", default=None,
                    help="输出目录 (默认按 --pool: v3=checkpoints_lit_features_v3, "
                         "v2=checkpoints_lit_features_v2)")
    ap.add_argument("--skip-honest", action="store_true",
                    help="跳过配对 AUC 矩阵与诚实特征判定, 只出特征表")
    ap.add_argument("--no-partial", action="store_true",
                    help="关闭中途落盘 (默认每 25 条写一次 partial csv)")
    args = ap.parse_args()

    pool = args.pool
    _default_dir = FM.OUT_DIR_V3 if pool == "v3" else FM.OUT_DIR
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else _default_dir
    os.makedirs(out_dir, exist_ok=True)
    csv_name = "lit_features_v3.csv" if pool == "v3" else "lit_features_v2.csv"
    out_csv = os.path.join(out_dir, csv_name)
    partial_csv = os.path.join(out_dir, csv_name.replace(".csv", "_partial.csv"))
    col_order = FM.COLUMN_ORDER_V3 if pool == "v3" else FM.COLUMN_ORDER
    feat_cols_spec = FM.FEATURE_COLS_V3 if pool == "v3" else FM.FEATURE_COLS
    n_cols = len(feat_cols_spec)

    for p in (FULLREC_PATH, FULLMETA_PATH):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"缺少整条记录缓存: {p}\n"
                f"该产物由 EHG_three_ds_union.py 导出, 请先在项目根目录执行:\n"
                f"    python EHG_three_ds_union.py --full_records_only\n"
                f"注意: 不带 --full_records_only 时只生成 checkpoints_union/ 的\n"
                f"      滑窗缓存 (win_*.npy), 不会导出 rec_full.npy。")

    print("=" * 70)
    print(f"EHG 记录级手工特征提取 (pool={pool}, {n_cols} 列)")
    print(f"  数据: {FULLREC_DIR}")
    print(f"  输出: {out_dir}")
    print(f"  熵类: 分段法 ({SEG_ENT // FS}s×{MAX_SEG_ENT}段, "
          f"MSE {SEG_MSE // FS}s×{MAX_SEG_MSE}段)")
    print(f"  并行: {args.workers} workers")
    print("=" * 70, flush=True)

    meta = pd.read_csv(FULLMETA_PATH)
    # 防守: rec_full.* 必须来自 EHG_three_ds_union.py --full_records_only,
    # 否则列名/记录顺序与消费端不符会静默出错
    _exp_cols = {"recname", "source", "label_preterm", "gestation",
                 "rectime", "n_samples"}
    if _exp_cols - set(meta.columns):
        raise ValueError(
            "rec_full_meta.csv 缺少列 "
            f"{sorted(_exp_cols - set(meta.columns))}; 该文件必须由\n"
            "    python EHG_three_ds_union.py\n"
            "重新生成 (当前疑似非 union 导出的全信号缓存)")
    _arr = np.load(FULLREC_PATH, mmap_mode="r")
    if _arr.shape[0] != len(meta) or _arr.shape[2] != 6:
        raise ValueError(
            f"rec_full.npy 形状 {_arr.shape} 与 rec_full_meta.csv "
            f"({len(meta)} 行) 不一致, 请重新生成。")
    # 只取判别矩阵用到的两个库: ICEHG 全足月, 不参与
    keep = meta[meta["source"].isin(["TPEHGDB", "TPEHGT"])]
    if args.limit:
        keep = keep.head(args.limit)
    print(f"\n记录数: DB {int((keep.source == 'TPEHGDB').sum())} + "
          f"TPEHGT {int((keep.source == 'TPEHGT').sum())} = {len(keep)}",
          flush=True)

    # 保留原始行号 (映射 mmap 行), 不 reset_index
    tasks = [(idx, {"recname": r["recname"], "source": r["source"],
                    "label": int(r["label_preterm"]), "gest": r["gestation"],
                    "rectime": r["rectime"], "n_samples": int(r["n_samples"]),
                    "lead": r["gestation"] - r["rectime"]}, pool)
             for idx, r in keep.iterrows()]

    t0 = time.time()
    rows = []
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_extract_one, t): t[0] for t in tasks}
        done = 0
        for fut in as_completed(futs):
            rows.append(fut.result())
            done += 1
            if done % 25 == 0 or done == len(tasks):
                el = time.time() - t0
                eta = el / done * (len(tasks) - done)
                print(f"  {done}/{len(tasks)} 条 ({el:.0f}s, "
                      f"ETA {eta / 60:.0f}min)", flush=True)
                if not args.no_partial and done < len(tasks):
                    pd.DataFrame(rows).to_csv(partial_csv, index=False,
                                              encoding="utf-8-sig")

    # 稳定顺序 + 固定列顺序 (与历史产物逐列一致)
    rows.sort(key=lambda d: d["recname"])
    df = pd.DataFrame(rows)
    missing = [c for c in col_order if c not in df.columns]
    if missing:
        print(f"  [警告] 缺失列补 NaN: {missing}")
        for c in missing:
            df[c] = np.nan
    df = df[list(col_order)]
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[CSV] {out_csv}  形状 {df.shape}")

    # Feature ID 映射 (替代已弃用的 EHG论文0822/baseline_results_75col.xlsx)
    fid = pd.DataFrame({
        "Column Name": feat_cols_spec,
        "Feature ID": [FM.feature_id(c) for c in feat_cols_spec],
    })
    fid_path = os.path.join(out_dir, "feature_id_map.csv")
    fid.to_csv(fid_path, index=False, encoding="utf-8-sig")
    print(f"[CSV] {fid_path}  ({len(fid)} 行)")

    if args.skip_honest:
        print("\n[跳过] 配对矩阵 / 诚实判定 (--skip-honest)")
        print("[完成]")
        return

    db = df[df.source == "TPEHGDB"].copy()
    tg = df[df.source == "TPEHGT"].copy()
    feat_cols = [c for c in df.columns if c not in FM.META_COLS]
    res, _ = build_pair_matrix(db, tg, feat_cols)
    res.to_csv(os.path.join(out_dir, "pair_auc_matrix.csv"),
               index=False, encoding="utf-8-sig")
    print(f"[CSV] pair_auc_matrix.csv: {len(res)} 特征 × 6 配对 + TG")

    hon = judge_honest(res, pool=pool)
    hon.to_csv(os.path.join(out_dir, "honest_features.csv"),
               index=False, encoding="utf-8-sig")
    n_honest = int(hon["honest"].sum())
    print(f"\n通过全部四标准: {n_honest}/{len(hon)}")
    if n_honest:
        print("\n诚实特征清单:")
        for _, r in hon[hon["honest"]].sort_values(
                "folded_auc", ascending=False).iterrows():
            print(f"  {r['feature']:<24s} {r['type']:<11s} "
                  f"AUC={r['main_auc']:.3f} "
                  f"[{r['main_lo']:.3f},{r['main_hi']:.3f}] "
                  f"TG={r['tg_auc']:.3f} P6={r['p6_auc']:.3f}")

    rep = ["=" * 78,
           "EHG 手工特征 · 配对判别力与诚实特征判定",
           f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
           "=" * 78,
           f"数据: TPEHG DB {len(db)} + TPEHGT DS {len(tg)}",
           f"特征列: {len(feat_cols)}",
           f"分组阈值: LEAD_SPLIT={LEAD_SPLIT}, "
           f"RECT_EARLY={RECTIME_SPLIT_EARLY}, RECT_LATE={RECTIME_SPLIT_LATE}",
           f"通过诚实四标准: {n_honest}/{len(hon)}", ""]
    if n_honest:
        rep.append("诚实特征 (folded AUC 降序):")
        for _, r in hon[hon["honest"]].sort_values(
                "folded_auc", ascending=False).iterrows():
            rep.append(f"  {r['feature']:<24s} {r['type']:<11s} "
                       f"AUC={r['main_auc']:.3f} "
                       f"[{r['main_lo']:.3f},{r['main_hi']:.3f}] "
                       f"TG={r['tg_auc']:.3f} P6={r['p6_auc']:.3f}")
    rep.append("\nTOP-25 主配对 folded AUC (探索性, 未过诚实标准也列出):")
    hs = hon.dropna(subset=["folded_auc"]).sort_values(
        "folded_auc", ascending=False).head(25)
    for _, r in hs.iterrows():
        rep.append(f"  {r['feature']:<24s} {r['type']:<11s} "
                   f"AUC={r['main_auc']:.3f} TG={r['tg_auc']:.3f} "
                   f"P6={r['p6_auc']:.3f} honest={r['honest']} "
                   f"c1={r['c1_time_matched']} c2={r['c2_tg_replicated']} "
                   f"c3={r['c3_not_gest_proxy']} c4={r['c4_dual_ch_consistent']}")
    with open(os.path.join(out_dir, "lit_features_report.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(rep) + "\n")
    print(f"\n[TXT] 报告: lit_features_report.txt")
    print("[完成]")


if __name__ == "__main__":
    main()
