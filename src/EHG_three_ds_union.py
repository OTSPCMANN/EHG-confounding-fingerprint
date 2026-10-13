# -*- coding: utf-8 -*-
"""
EHG 三数据集合并脚本
====================
将 TPEHG DB / TPEHGT DS / ICEHG DS 三个数据集统一预处理并合并，
输出记录级全信号 rec_full.npy + rec_full_meta.csv，供后续特征提取与训练使用。

统一方案：
  - 取各数据集的 3 个 EHG 原始通道 (S1/S2/S3)
  - 派生 6 通道: S1, S2, S3, S4=S1-S2+S3, S5=S1-S2, S6=S3-S2
  - 统一 0.08-5Hz 四阶 Butterworth 双向滤波 (scipy.signal.filtfilt)
  - 标签: Gestation < 37 → 早产(1), >= 37 → 足月(0)

职责边界 (关键):
  本脚本只负责"记录级全信号"的整理与导出 (rec_full.npy / rec_full_meta.csv)。
  这两个文件是 EHG_lit_features.py(手工特征) 与 pipe_fusion.py(DL 融合) 的唯一数据来源。

  5 折划分 / 多 seed 划分 / 滑窗 / 单数据集缓存 等, 统一交给下游:
    - pipe01_AUC.py / run_experiment.py  (负责 5 折 StratifiedGroupKFold + 多 seed)
    - EHG_lit_features.py / pipe_fusion.py (按 recname 在 TPEHGDB+TPEHGT 子集上取数)
  因此本脚本不再生成 win_*.npy / record_split.csv / preprocess_config.txt /
  checkpoints_DB_base|extra|term 等历史缓存, 也不再依赖它们。

依赖: numpy, scipy, pandas, wfdb
运行: python EHG_three_ds_union.py
"""

import sys
import os
import glob
import time
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt

# --- repository path bootstrap (added for the public release) -----------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "analysis"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ==================== 配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_ROOT, "data")
TPEHGDB_DIR = os.path.join(DATA_DIR, "TPEHG_DB", "tpehgdb")
TPEHGT_DIR = os.path.join(DATA_DIR, "TPEHGT_DS")
ICEHG_DIR = os.path.join(DATA_DIR, "ICEHG_DS")

OUTPUT_DIR = os.path.join(_ROOT, "checkpoints_union")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FS = 20                  # 采样率 Hz
THRESHOLD_WEEKS = 37     # 早产阈值: Gestation < 37 → 早产 (临床标准: <37周, 不含37.0)

# 各数据集原始 EHG 通道索引
TPEHGDB_RAW_IDX = [0, 4, 8]   # S1原始(0), S2原始(4), S3原始(8)
TPEHGT_RAW_IDX = [0, 2, 4]    # EHG1(0), EHG2(2), EHG3(4)
ICEHG_RAW_IDX = [0, 2, 4]     # S1(0), S2(2), S3(4)

ICEHG_SUBDIRS = [
    "early_induced", "later_induced",
    "early_cesarean", "later_cesarean",
    "early_induced-cesarean", "later_induced-cesarean"
]


# ==================== ICEHG Table 真值解析 (.hea 部分字段有误, 以 Table 为准) ====================
def _parse_icehg_tables():
    """解析 ICEHG_DS/Table_1/2/3.txt, 返回 {recname: {临床字段}} 真值字典"""
    clinical = {}
    for tbl_name in ["Table_1.txt", "Table_2.txt", "Table_3.txt"]:
        path = os.path.join(ICEHG_DIR, tbl_name)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()
        for ln in lines:
            ln = ln.strip()
            if not ln.startswith("|"):
                continue
            if any(kw in ln for kw in ["Pregnancy No.", "Age", "deliveries",
                                         "Mean", "Std.", "Recording"]):
                continue
            parts = [p.strip() for p in ln.split("|")]
            if len(parts) < 14:
                continue
            age = parts[2].strip()
            height = parts[9].strip()
            placent = parts[10].strip()
            gestation = parts[11].strip()
            newborn = parts[12].strip()
            early_id = parts[3].strip()
            early_rectime = parts[4].strip()
            early_weight = parts[5].strip()
            later_id = parts[6].strip()
            later_rectime = parts[7].strip()
            later_weight = parts[8].strip()
            for rec_id, rt, wt in [
                (early_id, early_rectime, early_weight),
                (later_id, later_rectime, later_weight),
            ]:
                if rec_id and rec_id != "~" and rec_id.isdigit():
                    key = "icehg" + rec_id
                    d = {}
                    if age and age != "~":
                        d["Age"] = float(age)
                    if height and height != "~":
                        d["Height"] = float(height)
                    if placent and placent != "~":
                        d["Placental_position"] = placent
                    if gestation and gestation != "~":
                        d["gestation"] = float(gestation)
                    if newborn and newborn != "~":
                        d["Newborn_weight"] = float(newborn)
                    if rt and rt != "~":
                        d["rectime"] = float(rt)
                    if wt and wt != "~":
                        d["Weight"] = float(wt)
                    clinical[key] = d
    return clinical


# ==================== 工具函数 ====================
def parse_hea_comments(hea_path):
    """解析 .hea 头文件的注释段，返回字段字典"""
    fields = {}
    with open(hea_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            # 去掉前导 # 和空格
            content = line.lstrip("#").strip()
            if "\t" in content:
                parts = content.split("\t", 1)
            else:
                # 用空格分隔的也兼容
                parts = content.split(None, 1)
            if len(parts) == 2:
                key, val = parts[0].strip(), parts[1].strip()
                fields[key] = val
    return fields


def to_float(val, default=None):
    """安全转 float"""
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


# ==================== 临床协变量 (记录级元信息) ====================
# 仅作记录元信息: scan 函数会把它们写进 df, 便于追踪/审计。
# 注意: 本脚本导出的 rec_full_meta.csv 只保留
#   recname/source/label_preterm/gestation/rectime/n_samples
# 这 6 列 (EHG_lit_features.py / pipe_fusion.py 的契约列);
# 临床协变量列不在 rec_full_meta 中, 需要时可从原始 .hea 重新解析。
CLINICAL_NUM = ["Age", "Parity", "Abortions", "Weight",
                "Height", "Newborn_weight"]
CLINICAL_CAT = ["Placental_position", "Hypertension", "Diabetes", "Smoker",
                "Bleeding_first_trimester", "Bleeding_second_trimester",
                "Funneling"]
CLINICAL_COLS = CLINICAL_NUM + CLINICAL_CAT


def parse_clinical(fields):
    """从 .hea 注释解析临床协变量, 缺失字段回退 None (DataFrame 中即 NaN)"""
    out = {}
    for k in CLINICAL_NUM:
        out[k] = to_float(fields.get(k))
    for k in CLINICAL_CAT:
        v = fields.get(k)
        out[k] = v if (v is not None and str(v).strip() != "") else None
    return out


def butter_bandpass_filter(data, lowcut=0.08, highcut=5.0, fs=FS, order=4):
    """4阶 Butterworth 双向带通滤波 (零相位)"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data, axis=0)


def derive_6channels(s1, s2, s3):
    """从 3 个原始通道派生 6 通道"""
    s4 = s1 - s2 + s3
    s5 = s1 - s2
    s6 = s3 - s2
    return np.stack([s1, s2, s3, s4, s5, s6], axis=1)


# ==================== 文件路径解析 / 通道索引 ====================
def build_filepath_for_rec(row):
    """根据 source + recname 构建 wfdb 读取路径, 返回不含 .hea/.dat 的 path"""
    src = row["source"]
    name = row["recname"]
    if src == "TPEHGDB":
        return os.path.join(TPEHGDB_DIR, name)
    elif src == "TPEHGT":
        return os.path.join(TPEHGT_DIR, name)
    else:
        tp = row.get("timepoint", "later")
        rt = row.get("rectype", "induced")
        if pd.isna(tp):
            tp = "later"
        if pd.isna(rt):
            rt = "induced"
        candidates = [f"{tp}_{rt}"]
        for d in ["early_induced", "later_induced",
                  "early_cesarean", "later_cesarean",
                  "early_induced-cesarean", "later_induced-cesarean"]:
            if d not in candidates:
                candidates.append(d)
        for d in candidates:
            p = os.path.join(ICEHG_DIR, d, name)
            if os.path.exists(p + ".hea"):
                return p
        return os.path.join(ICEHG_DIR, candidates[0], name)


def get_raw_idx_for_source(src):
    if src == "TPEHGDB":
        return TPEHGDB_RAW_IDX
    elif src == "TPEHGT":
        return TPEHGT_RAW_IDX
    else:
        return ICEHG_RAW_IDX


# ==================== 数据集扫描 ====================
def scan_tpehgdb():
    """扫描 TPEHG DB，返回记录信息列表"""
    records = []
    hea_files = sorted(glob.glob(os.path.join(TPEHGDB_DIR, "*.hea")))
    for hea_path in hea_files:
        recname = os.path.splitext(os.path.basename(hea_path))[0]
        dat_path = hea_path.replace(".hea", ".dat")
        if not os.path.exists(dat_path):
            continue
        fields = parse_hea_comments(hea_path)
        gestation = to_float(fields.get("Gestation"))
        rectime = to_float(fields.get("Rectime"))
        recid = fields.get("RecID", recname)
        if gestation is None:
            continue
        label = 1 if gestation < THRESHOLD_WEEKS else 0
        # TPEHG DB 1记录/孕妇, 孕妇ID = recname
        records.append({
            "source": "TPEHGDB",
            "recname": recname,
            "filepath": hea_path[:-4],   # wfdb 不带后缀
            "recid": recid,
            "subject_id": recname,        # 孕妇ID (1记录/孕妇)
            "pair_recid": recname,        # 无配对, 自身
            "gestation": gestation,
            "rectime": rectime if rectime else 0,
            "rectype": "natural",         # TPEHG DB 全是自然发动
            "timepoint": "early" if rectime and rectime < 26 else "later",
            "raw_idx": TPEHGDB_RAW_IDX,
            "label_preterm": label,
        })
        records[-1].update(parse_clinical(fields))
    return records


def scan_tpehgt():
    """扫描 TPEHGT DS，返回记录信息列表"""
    records = []
    hea_files = sorted(glob.glob(os.path.join(TPEHGT_DIR, "*.hea")))
    for hea_path in hea_files:
        recname = os.path.splitext(os.path.basename(hea_path))[0]
        dat_path = hea_path.replace(".hea", ".dat")
        if not os.path.exists(dat_path):
            continue
        fields = parse_hea_comments(hea_path)
        # TPEHGT DS 文件名首字母: p=早产, t=足月, n=非孕妇
        first_letter = recname.split("_")[-1][0] if "_" in recname else recname[-1]
        rectype_raw = fields.get("RecType", "")
        gestation = to_float(fields.get("Gestation"))
        rectime = to_float(fields.get("Rectime"))
        recid = fields.get("RecID", recname)
        # 跳过非孕妇
        if first_letter == "n" or "non" in rectype_raw.lower():
            continue
        if gestation is None:
            continue
        label = 1 if gestation < THRESHOLD_WEEKS else 0
        # TPEHGT DS 同一孕妇可能多条记录, 用 recid 作为孕妇ID
        records.append({
            "source": "TPEHGT",
            "recname": recname,
            "filepath": hea_path[:-4],
            "recid": recid,
            "subject_id": recid,           # 同一孕妇多记录绑定
            "pair_recid": recname,         # 单时点, 无配对
            "gestation": gestation,
            "rectime": rectime if rectime else 31,
            "rectype": "natural",
            "timepoint": "later",          # TPEHGT DS 只有 ~31w
            "raw_idx": TPEHGT_RAW_IDX,
            "label_preterm": label,
        })
        records[-1].update(parse_clinical(fields))
    return records


def scan_icehg():
    """扫描 ICEHG DS, 返回记录信息列表"""
    records = []
    # 加载 Table 真值用于修正 .hea 错误
    table_gt = _parse_icehg_tables()
    for subdir in ICEHG_SUBDIRS:
        subdir_path = os.path.join(ICEHG_DIR, subdir)
        if not os.path.isdir(subdir_path):
            continue
        hea_files = sorted(glob.glob(os.path.join(subdir_path, "*.hea")))
        for hea_path in hea_files:
            recname = os.path.splitext(os.path.basename(hea_path))[0]
            dat_path = hea_path.replace(".hea", ".dat")
            if not os.path.exists(dat_path):
                continue
            fields = parse_hea_comments(hea_path)
            # 用 Table 真值覆盖 .hea 中可能错误的字段 (如 icehg1235 Age=3→34)
            if recname in table_gt:
                for k, v in table_gt[recname].items():
                    fields[k] = str(v) if not isinstance(v, str) else v
            gestation = to_float(fields.get("Gestation"))
            rectime = to_float(fields.get("Rectime"))
            recid = fields.get("RecID", recname)
            pair_recid = fields.get("Pair_RecID", recname)
            rectype = fields.get("RecType", "").lower()
            if gestation is None:
                continue
            # ICEHG DS 全部足月, label_preterm 恒为 0
            label = 1 if gestation < THRESHOLD_WEEKS else 0
            # 分娩方式 3 分类标签
            if "induced-cesarean" in subdir or "induced-ceasarean" in subdir:
                dtype = "induced-cesarean"
            elif "induced" in subdir:
                dtype = "induced"
            elif "cesarean" in subdir:
                dtype = "cesarean"
            else:
                dtype = "unknown"
            records.append({
                "source": "ICEHG",
                "recname": recname,
                "filepath": hea_path[:-4],
                "recid": recid,
                "subject_id": recid,            # ICEHG 1记录/孕妇(但early/later配对)
                "pair_recid": pair_recid,       # early/later 配对绑定
                "gestation": gestation,
                "rectime": rectime if rectime else 0,
                "rectype": dtype,
                "timepoint": "early" if subdir.startswith("early") else "later",
                "raw_idx": ICEHG_RAW_IDX,
                "label_preterm": label,         # ICEHG 全为 0
                "label_delivery": {"induced": 0, "cesarean": 1, "induced-cesarean": 2}.get(dtype, -1),
            })
            records[-1].update(parse_clinical(fields))
    return records


# ==================== 记录级全信号导出 ====================
def export_full_records(df, out_dir):
    """导出整条滤波后记录 (不做滑窗), 供记录级手工特征提取使用.

    输出:
      rec_full.npy       (n_rec, L_max, 6) float32, 短记录 NaN 填充
      rec_full_meta.csv  recname/source/label_preterm/gestation/rectime/n_samples
    与既有 win_*.npy 缓存完全独立, 互不影响.

    该 rec_full.* 是 EHG_lit_features.py(手工特征) 与 pipe_fusion.py(DL 融合)
    的唯一数据来源; 二者均按 recname 在 "TPEHGDB + TPEHGT" 子集上取数,
    ICEHG(126 条全足月) 虽一并导出但会被消费端忽略。若 rec_full.* 不是由
    本函数生成(列名/记录顺序不符), 两个消费端会在启动时报错。
    """
    os.makedirs(out_dir, exist_ok=True)
    recs = []
    meta_rows = []
    t0 = time.time()
    for i, row in df.iterrows():
        try:
            record = wfdb.rdrecord(build_filepath_for_rec(row))
            sig = record.p_signal
            raw_idx = get_raw_idx_for_source(row["source"])
            s1 = sig[:, raw_idx[0]]
            s2 = sig[:, raw_idx[1]]
            s3 = sig[:, raw_idx[2]]
            sig6 = derive_6channels(s1, s2, s3)
            sig6_filt = butter_bandpass_filter(sig6)
            recs.append(sig6_filt.astype(np.float32))
            meta_rows.append({
                "recname": row["recname"],
                "source": row["source"],
                "label_preterm": int(row["label_preterm"]),
                "gestation": row["gestation"],
                "rectime": row["rectime"],
                "n_samples": int(sig6_filt.shape[0]),
            })
        except Exception as e:
            print(f"  [跳过] {row['recname']}: {e}")
        if (i + 1) % 50 == 0 or i == len(df) - 1:
            print(f"    {i+1}/{len(df)} 条, 累计 {len(recs)} 有效 "
                  f"({time.time()-t0:.0f}s)", flush=True)
    if not recs:
        print("  [错误] 无有效记录")
        return
    lmax = max(r.shape[0] for r in recs)
    buf = np.full((len(recs), lmax, 6), np.nan, dtype=np.float32)
    for j, r in enumerate(recs):
        buf[j, :r.shape[0], :] = r
    np.save(os.path.join(out_dir, "rec_full.npy"), buf)
    pd.DataFrame(meta_rows).to_csv(
        os.path.join(out_dir, "rec_full_meta.csv"),
        index=False, encoding="utf-8-sig")
    n_MB = buf.nbytes / (1024 ** 2)
    print(f"  [NPY] rec_full.npy: shape={buf.shape}, {n_MB:.1f} MB")
    print(f"  [CSV] rec_full_meta.csv: {len(meta_rows)} 行")


# ==================== 主流程 ====================
def main():
    print("=" * 70)
    print("EHG 三数据集合并脚本 (记录级全信号导出)")
    print(f"  TPEHG DB  : {TPEHGDB_DIR}")
    print(f"  TPEHGT DS : {TPEHGT_DIR}")
    print(f"  ICEHG DS  : {ICEHG_DIR}")
    print(f"  输出目录   : {OUTPUT_DIR}")
    print(f"  早产阈值   : Gestation < {THRESHOLD_WEEKS} 周")
    print("=" * 70)

    # ---- 1. 扫描三数据集 ----
    print("\n[1/2] 扫描三数据集...")
    recs_tpehgdb = scan_tpehgdb()
    recs_tpehgt = scan_tpehgt()
    recs_icehg = scan_icehg()
    all_records = recs_tpehgdb + recs_tpehgt + recs_icehg

    print(f"  TPEHG DB  : {len(recs_tpehgdb)} 条")
    print(f"  TPEHGT DS : {len(recs_tpehgt)} 条 (已排除非孕妇)")
    print(f"  ICEHG DS  : {len(recs_icehg)} 条")
    print(f"  合计       : {len(all_records)} 条")

    if len(all_records) == 0:
        print("[错误] 未扫描到任何记录, 请检查目录路径")
        return

    df = pd.DataFrame(all_records)

    # ---- 2. 导出记录级全信号 ----
    print("\n[2/2] 导出记录级全信号 (rec_full.npy + rec_full_meta.csv)...")
    print("      (5 折划分 / 多 seed / 滑窗 由下游 pipe01_AUC.py / run_experiment.py 负责)")
    export_full_records(df, OUTPUT_DIR)

    print(f"\n[完成] 记录级全信号已保存: {OUTPUT_DIR}")
    print(f"  - rec_full.npy      (n_rec, L_max, 6) float32, 短记录 NaN 填充")
    print(f"  - rec_full_meta.csv  recname/source/label_preterm/gestation/rectime/n_samples")
    print(f"\n  下游消费端:")
    print(f"    a) 手工特征:   EHG_lit_features.py   (在 TPEHGDB + TPEHGT 子集取数)")
    print(f"    b) DL 融合:     pipe_fusion.py       (在 TPEHGDB + TPEHGT 子集取数)")


if __name__ == "__main__":
    main()
