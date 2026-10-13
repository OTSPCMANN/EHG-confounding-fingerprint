# -*- coding: utf-8 -*-
"""Figure 3 (ESWA) — 189 x 6 confounding fingerprint heatmap + table export.

Input : checkpoints_lit_features_v3/pair_auc_matrix.csv  (189 features x P1..P6 AUC/lo/hi + TG)
Output: fig3_confounding_fingerprint_189x6.png / .pdf      (publication figure)
        Fig3_fingerprint_wide.csv                          (feature x P1..P6 AUC, for manual redraw)
        Fig3_fingerprint_long.csv                          (long: feature, channel, pairing, auc, ci, sig)
        Fig3_fingerprint_README.txt                        (reading notes)

Style: Times New Roman, full-width 185 mm, diverging RdBu_r centred at 0.5,
black outline on cells whose bootstrap 95% CI excludes 0.5.
"""
import sys
import os
import re
import numpy as np
import pandas as pd
import matplotlib

# --- repository path bootstrap (added for the public release) -----------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "analysis"),
           os.path.join(_ROOT, "figures"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

BASE = os.path.join(_ROOT)
CSV = os.path.join(BASE, "results", "checkpoints_lit_features_v3", "pair_auc_matrix.csv")
OUTDIR = os.path.join(BASE, "figures", "output")
PNG = os.path.join(OUTDIR, "fig3_confounding_fingerprint_189x6.png")
PDF = os.path.join(OUTDIR, "fig3_confounding_fingerprint_189x6.pdf")
WIDE = os.path.join(OUTDIR, "Fig3_fingerprint_wide.csv")
LONG = os.path.join(OUTDIR, "Fig3_fingerprint_long.csv")
README = os.path.join(OUTDIR, "Fig3_fingerprint_README.txt")

PAIRS = ["P1", "P2", "P3", "P4", "P5", "P6"]
STEM = {"P1": "P1_G1vsG3", "P2": "P2_G1vsG4", "P3": "P3_G2vsG3",
        "P4": "P4_G2vsG4", "P5": "P5_G1vsG2", "P6": "P6_G3vsG4"}
PLABEL = {"P1": "P1\nG1-G3", "P2": "P2\nG1-G4", "P3": "P3\nG2-G3",
          "P4": "P4\nG2-G4", "P5": "P5\nG1-G2", "P6": "P6\nG3-G4"}
# time-matched pairs (confounder-controlled) vs time-unmatched
TIMEMATCH = {"P2", "P4"}
CROSS = ("cv_vert", "cv_horiz", "hv_ratio")

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.serif"] = ["Times New Roman"]


def channel_of(name):
    m = re.search(r"_s([1-6])$", name)
    return ("S" + m.group(1)) if m else "cross"


def main():
    df = pd.read_csv(CSV)
    assert len(df) == 189, "expected 189 features, got %d" % len(df)
    assert list(df.columns)[0] == "feature"

    df["channel"] = df["feature"].map(channel_of)
    groups = {("S%d" % i): df[df["channel"] == "S%d" % i] for i in range(1, 7)}
    groups["cross"] = df[df["channel"] == "cross"]
    assert len(groups["cross"]) == 3
    assert sum(len(g) for g in groups.values()) == 189

    # ---------- build matrices + significance ----------
    def matrix(g):
        M = np.full((len(g), 6), np.nan)
        S = np.zeros((len(g), 6), dtype=bool)
        for i, (_, r) in enumerate(g.iterrows()):
            for j, p in enumerate(PAIRS):
                a = r["%s_auc" % STEM[p]]
                lo = r["%s_lo" % STEM[p]]
                hi = r["%s_hi" % STEM[p]]
                M[i, j] = a
                S[i, j] = np.isfinite(lo) and np.isfinite(hi) and ((lo > 0.5) or (hi < 0.5))
        return M, S

    # ---------- table exports ----------
    wide = df[["feature", "channel"]].copy()
    for p in PAIRS:
        wide["%s_auc" % p] = df["%s_auc" % STEM[p]]
    wide["TG_auc"] = df["TG_auc"]
    wide["n_sig_pairings"] = 0
    long_rows = []
    for _, r in df.iterrows():
        nsig = 0
        for p in PAIRS:
            a = r["%s_auc" % STEM[p]]
            lo = r["%s_lo" % STEM[p]]
            hi = r["%s_hi" % STEM[p]]
            sig = bool(np.isfinite(lo) and np.isfinite(hi) and ((lo > 0.5) or (hi < 0.5)))
            nsig += int(sig)
            long_rows.append({
                "feature": r["feature"], "channel": r["channel"], "pairing": p,
                "contrast": STEM[p].split("_", 1)[1], "auc": a,
                "ci_lo": lo, "ci_hi": hi, "significant": int(sig),
            })
        wide.loc[wide["feature"] == r["feature"], "n_sig_pairings"] = nsig
    long = pd.DataFrame(long_rows)
    wide = wide.sort_values(["channel", "feature"]).reset_index(drop=True)
    wide.to_csv(WIDE, index=False, encoding="utf-8-sig")
    long.to_csv(LONG, index=False, encoding="utf-8-sig")

    sig_counts = long.groupby("pairing")["significant"].sum().reindex(PAIRS)
    print("significant counts per pairing:", dict(sig_counts))

    # ---------- figure ----------
    cmap = plt.get_cmap("RdBu_r")
    norm = TwoSlopeNorm(vmin=0.30, vcenter=0.5, vmax=0.70)

    fig = plt.figure(figsize=(7.28, 9.8))  # 185 mm wide
    gs = fig.add_gridspec(4, 2, hspace=0.30, wspace=0.34,
                          left=0.085, right=0.955, top=0.888, bottom=0.115)

    order = [("S1", 0, 0), ("S2", 0, 1), ("S3", 1, 0), ("S4", 1, 1),
             ("S5", 2, 0), ("S6", 2, 1), ("cross", 3, 0)]
    last_im = None
    for key, rr, cc in order:
        ax = fig.add_subplot(gs[rr, cc])
        g = groups[key]
        M, S = matrix(g)
        im = ax.imshow(M, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        last_im = im
        for i in range(M.shape[0]):
            for j in range(6):
                if S[i, j]:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="black", linewidth=0.9, zorder=3))
        # P6 (gestational-age probe) column highlight
        ax.add_patch(Rectangle((5 - 0.5, -0.5), 1, M.shape[0], fill=False,
                               edgecolor="#7a5c10", linewidth=1.6, zorder=4))
        ax.set_xticks(range(6))
        ax.set_xticklabels([PLABEL[p] for p in PAIRS], fontsize=6)
        ylab = [f.rsplit("_s", 1)[0] if key != "cross" else f for f in g["feature"]]
        ax.set_yticks(range(len(ylab)))
        ax.set_yticklabels(ylab, fontsize=4.6)
        ax.tick_params(length=0)
        for lbl, p in zip(ax.get_xticklabels(), PAIRS):
            if p in TIMEMATCH:
                lbl.set_color("#1f4e79")
            if p == "P6":
                lbl.set_color("#7a5c10")
                lbl.set_fontweight("bold")
        ax.set_title("Channel %s  (n=%d)" % (key, len(g)), fontsize=9, fontweight="bold", pad=4)
        for j in range(1, 6):
            ax.axvline(j - 0.5, color="white", lw=0.6)
        for i in range(1, M.shape[0]):
            ax.axhline(i - 0.5, color="white", lw=0.6)
        ax.invert_yaxis()

    # ---------- colorbar (row 3, col 1) ----------
    cax = fig.add_subplot(gs[3, 1])
    cb = fig.colorbar(last_im, cax=cax, orientation="vertical", fraction=0.6)
    cb.set_label("Univariate AUC (0.5 = chance)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cax.text(1.9, 0.02, "AUC > 0.5\n(discriminates)", transform=cax.transAxes,
             ha="left", va="bottom", fontsize=6.5, color="#b2182b")
    cax.text(1.9, 0.98, "AUC < 0.5\n(reversed)", transform=cax.transAxes,
             ha="left", va="top", fontsize=6.5, color="#2166ac")

    fig.suptitle("Figure 3. Confounding fingerprint of the 189-feature candidate pool (D1)",
                 fontsize=11, fontweight="bold", y=0.985)
    fig.text(0.5, 0.958,
             "Each cell = univariate AUC of one candidate feature on one of the six paired-discrimination "
             "contrasts. Time-matched pairs (P2, P4) carry far fewer significant cells than time-unmatched\n"
             "pairs (P1, P3), and 111/189 features discriminate on the P6 gestational-age probe.",
             ha="center", va="top", fontsize=6.8, linespacing=1.4)
    fig.text(0.085, 0.088,
             "Black outline: bootstrap 95% CI excludes 0.5 (significant).   "
             "Amber column (P6): within-term gestational-age probe.   "
             "Blue labels (P2, P4): time-matched (confounder-controlled).",
             ha="left", va="top", fontsize=6.6)

    fig.savefig(PNG, dpi=300)
    fig.savefig(PDF)
    plt.close(fig)
    print("wrote:", PNG)
    print("wrote:", PDF)
    print("wrote:", WIDE)
    print("wrote:", LONG)

    with open(README, "w", encoding="utf-8") as fh:
        fh.write(
            "Figure 3 data export — 189 x 6 confounding fingerprint\n"
            "=====================================================\n\n"
            "Source : checkpoints_lit_features_v3/pair_auc_matrix.csv\n"
            "Figure : fig3_confounding_fingerprint_189x6.png / .pdf\n\n"
            "Files\n"
            "-----\n"
            "Fig3_fingerprint_wide.csv  : 189 rows x (feature, channel, P1..P6 auc, TG_auc, n_sig_pairings)\n"
            "Fig3_fingerprint_long.csv  : 1134 rows x (feature, channel, pairing, contrast, auc, ci_lo, ci_hi, significant)\n\n"
            "Pairings\n"
            "--------\n"
            "P1 G1-G3 (time-unmatched)   P2 G1-G4 (time-matched)\n"
            "P3 G2-G3 (time-unmatched)   P4 G2-G4 (time-matched)\n"
            "P5 G1-G2 (within-outcome ref)   P6 G3-G4 (within-term gestational-age probe)\n\n"
            "Significance\n"
            "------------\n"
            "significant = 1 when the bootstrap 95% CI (B=2000, record-level) excludes 0.5.\n"
            "Significant counts per pairing: " + ", ".join("%s=%d" % (p, int(sig_counts[p])) for p in PAIRS) + "\n\n"
            "Redraw notes\n"
            "------------\n"
            "- Diverging colour scale RdBu_r, centred at 0.5, range [0.30, 0.70].\n"
            "- Draw a black outline on cells with significant = 1.\n"
            "- Facet by channel: S1-S6 (31 features each) + 3 cross-channel features.\n"
            "- Highlight the P6 column; mark P2/P4 as time-matched.\n"
        )
    print("wrote:", README)


if __name__ == "__main__":
    main()