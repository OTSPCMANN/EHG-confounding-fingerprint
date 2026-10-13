# -*- coding: utf-8 -*-
"""
EHG 特征元数据 —— 单一真源 (single source of truth)
====================================================
为 `EHG_lit_features.py`(特征提取) 与各 `pipe*.py`(实验流水线) 提供:

  1. FEATURE_COLS : 75 个特征列的规范顺序 (与历史 lit_features_v2.csv 逐列一致)
  2. BASE_FEATURE_IDS : 基础特征名 -> Feature ID (F1..F39) 的固化映射
  3. load_fid_map()   : 供 pipe01/02、run_experiment、stage1_pool_2x2 调用

背景
----
Feature ID 原先从 `EHG论文0822/baseline_results_75col.xlsx` 的
`PairAUC_Raw_CI` sheet 读取 (Column Name <-> Feature ID)。
该 xlsx 是"全量 300 条上跑的 75 列基线", 现已决定弃用, 但 F1..F39 这套编号
已被论文表格与 stage1/pool/sfs 产物引用, 故在此固化为常量:
  · 任何机器、任何时间生成的编号都一致, 不再依赖外部 xlsx
  · 编号完全沿用历史产物 (见 experiments/P1CI_P6eff050_seed42_20260824_210938/
    pipe_01_pair_auc.xlsx 的 Full300 sheet), 保证与已有实验结果可对照

列顺序约定 (必须严格保持, 否则 pipe02/03 的池与历史结果不可比)
--------------------------------------------------------------
  [v1 的 26 个单通道特征 × _mean]          1-26
  [v1 的 26 个单通道特征 × _s3]           27-52
  [5 个 entx_ 重标定特征 × (mean,s3)]     53-62
  [3 个跨通道特征 cv_vert/cv_horiz/hv]    63-65
  [v2 增补 5 特征 × (mean,s3)]            66-75
"""
import sys
import os

# --- repository path bootstrap (added for the public release) -----------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "analysis"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================== 路径 ====================
OUT_DIR = os.path.join(_ROOT, "results", "checkpoints_lit_features_v2")
FT_CSV = os.path.join(OUT_DIR, "lit_features_v2.csv")
FID_CSV = os.path.join(OUT_DIR, "feature_id_map.csv")

# ==================== 基础特征 -> Feature ID (F1..F39, 无缺号) ====================
BASE_FEATURE_IDS = {
    # A 时域 (F1-F9)
    "peak": 1, "rms": 2, "energy": 3, "area": 4, "waveform_len": 5,
    "variance": 6, "logdet": 7, "rms_env": 8, "p2p": 9,
    # A burst (F10-F13)
    "burst_dur": 10, "burst_rate": 11, "burst_interval": 12,
    "activation_dur": 13,
    # B 频域 (F14-F24)
    "peak_freq_full": 14, "median_freq": 15, "centroid_freq": 16,
    "total_power": 17, "rel_fwl": 18, "rel_cont": 19, "rel_high": 20,
    "fwl_peak_freq": 21, "cv_env": 22, "burst_ratio": 23, "zcr": 24,
    # C 传导 / E 传播 (F25-F27, 跨通道, 无 mean/s3 后缀)
    "cv_vert": 25, "cv_horiz": 26, "hv_ratio": 27,
    # D 非线性 (F28-F32)
    "apen": 28, "sampen": 29, "fuzzyen": 30, "mse": 31, "lzc": 32,
    # D 非线性 · entx 重标定版 (F33-F37)
    "entx_apen": 33, "entx_sampen": 34, "entx_fuzzyen": 35,
    "entx_mse": 36, "entx_lzc": 37,
    # v2 增补 (F38-F39)
    "corr_dim": 38, "model_error": 39,
}

# 跨通道特征: 无通道扩展, ID 不带 -mean/-s3 后缀
CROSS_CHANNEL = ("cv_vert", "cv_horiz", "hv_ratio")

# v1 的 26 个单通道特征 (顺序 = extract_record 内 all_keys 的自然顺序)
V1_SINGLE = (
    "peak", "rms", "energy", "area", "waveform_len", "variance", "logdet",
    "burst_dur", "burst_rate", "burst_interval", "activation_dur",
    "peak_freq_full", "median_freq", "centroid_freq", "total_power",
    "rel_fwl", "rel_cont", "rel_high", "fwl_peak_freq", "cv_env",
    "burst_ratio",
    "apen", "sampen", "fuzzyen", "lzc", "mse",
)

# v2 增补的 5 个单通道特征 (顺序 = EHG_lit_features_v2.NEW_FEATS 的字典顺序)
V2_SINGLE = ("model_error", "zcr", "rms_env", "p2p", "corr_dim")

# 熵特征做 entx 重标定的 5 个基底 (顺序 = v1 extract_record 中的循环顺序)
ENTX_BASES = ("apen", "sampen", "fuzzyen", "mse", "lzc")

# 非特征列 (记录元数据)
META_COLS = ("recname", "source", "label", "gest", "rectime",
             "n_samples", "lead")


def _build_feature_cols():
    cols = []
    cols += [f"{b}_mean" for b in V1_SINGLE]
    cols += [f"{b}_s3" for b in V1_SINGLE]
    for b in ENTX_BASES:
        cols += [f"entx_{b}_mean", f"entx_{b}_s3"]
    cols += list(CROSS_CHANNEL)
    for b in V2_SINGLE:
        cols += [f"{b}_mean", f"{b}_s3"]
    return cols


FEATURE_COLS = _build_feature_cols()          # 75 列, 顺序即历史产物顺序
COLUMN_ORDER = FEATURE_COLS + list(META_COLS)  # 写入 CSV 的完整列顺序


# ==================== v3: 六通道独立口径 (2026-09-24 修订) ====================
# 修订动机:
#   1) 通道平均 (_mean) 会让反相信号互相抵消 -> 弃用 mean 视图, 31 基础特征按
#      S1..S6 独立通道输出 (后缀 _s1.._s6, 与旧 _s3 命名风格一致);
#   2) entx 重标定 = 原值 × (1 + rectime/37), 把记录孕周参数注入特征
#      (时钟 shortcut 风险) -> 整组移除。
# 新池: 31 基础 × 6 通道 = 186 + 跨通道 3 (cv_vert/cv_horiz/hv_ratio, 单口径) = 189。
CH_SUFFIXES = ("s1", "s2", "s3", "s4", "s5", "s6")
CHANNEL_SUFFIX_COLS = tuple("_" + s for s in CH_SUFFIXES)   # ("_s1",..,"_s6")


def _build_feature_cols_v3():
    cols = []
    for b in V1_SINGLE:
        cols += [f"{b}_{s}" for s in CH_SUFFIXES]
    for b in V2_SINGLE:
        cols += [f"{b}_{s}" for s in CH_SUFFIXES]
    cols += list(CROSS_CHANNEL)
    return cols


FEATURE_COLS_V3 = _build_feature_cols_v3()                     # 189 列
COLUMN_ORDER_V3 = FEATURE_COLS_V3 + list(META_COLS)            # CSV 完整列顺序
OUT_DIR_V3 = os.path.join(_ROOT, "results", "checkpoints_lit_features_v3")
FT_CSV_V3 = os.path.join(OUT_DIR_V3, "lit_features_v3.csv")


def feature_id(col):
    """列名 -> Feature ID.

      peak_mean        -> F1-mean
      peak_s3          -> F1-s3
      entx_mse_s3      -> F36-s3
      cv_vert          -> F25
      未知列           -> "" (空串, 与历史 load_fid_map 的 .get 缺省行为一致)
    """
    name = str(col)
    if name in CROSS_CHANNEL:
        return f"F{BASE_FEATURE_IDS[name]}"
    for suf in ("_mean", "_s3") + CHANNEL_SUFFIX_COLS:
        if name.endswith(suf):
            base = name[: -len(suf)]
            if base in BASE_FEATURE_IDS:
                return "F{}-{}".format(BASE_FEATURE_IDS[base], suf[1:])
    return ""


def build_fid_map(cols=None):
    """{列名: Feature ID}. cols=None 时用规范的 75 列顺序."""
    cols = list(FEATURE_COLS) if cols is None else list(cols)
    return {c: feature_id(c) for c in cols}


def _read_fid_csv(path):
    import pandas as pd
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    if not {"Column Name", "Feature ID"}.issubset(d.columns):
        return None
    d = d.dropna(subset=["Column Name"])
    return dict(zip(d["Column Name"].astype(str), d["Feature ID"].astype(str)))


def load_fid_map(ft_csv=None):
    """供 pipe01/pipe02/run_experiment/stage1_pool_2x2 使用.

    优先级:
      1. feature_id_map.csv (由 EHG_lit_features.py 生成, 与 ft_csv 同目录)
      2. 直接读 ft_csv 的表头, 按固化规则推导
      3. 兜底: 按出现顺序编号 F01..Fnn (不应发生, 仅防御)
    """
    ft_csv = FT_CSV if ft_csv is None else ft_csv
    candidates = [os.path.join(os.path.dirname(ft_csv), "feature_id_map.csv"),
                  FID_CSV]
    for p in candidates:
        m = _read_fid_csv(p)
        if m:
            return m
    try:
        import pandas as pd
        cols = [c for c in pd.read_csv(ft_csv, nrows=0).columns
                if c not in META_COLS]
        m = {c: feature_id(c) for c in cols}
        if any(v for v in m.values()):
            return m
        return {c: f"F{i:02d}" for i, c in enumerate(cols, 1)}
    except Exception:
        return {}


if __name__ == "__main__":
    print(f"特征列: {len(FEATURE_COLS)}")
    print(f"基础特征 ID: {len(BASE_FEATURE_IDS)} (F1..F39)")
    miss = [c for c in FEATURE_COLS if not feature_id(c)]
    print(f"无 ID 的列: {miss if miss else '无'}")
    dup = len(set(FEATURE_COLS)) != len(FEATURE_COLS)
    print(f"列重复: {dup}")
    for c in FEATURE_COLS[:3] + FEATURE_COLS[62:66] + FEATURE_COLS[-2:]:
        print(f"  {c:<24s} {feature_id(c)}")
