# EHG Confounding Fingerprint

Reference implementation and derived results for **CAFA-C**, a four-dimension,
confounding-aware evaluation framework for electrohysterogram (EHG) based
preterm-birth prediction.

This repository accompanies the manuscript:

> **A Four-Dimension Evaluation Framework with a Reporting Checklist for
> Trustworthy Electrohysterogram-Based Preterm Birth Prediction: From Feature
> Admission to Cohort Origin**

The framework audits a predictor along four dimensions:

| Dimension | Question it addresses |
| --- | --- |
| **D1** Feature admission | Which of the 189 candidate features pass the paired-discrimination gates? |
| **D2** Cohort origin | Does performance hold on an external cohort? |
| **D3** Recording-time baseline | How much of the reported AUC is already explained by gestational age at the time of recording? |
| **D4** Clinical stratification | Does performance hold within clinically meaningful strata? |

The candidate feature pool has **189 features** = 31 base descriptors x 6 derived
channels (S1-S3 raw, S4 = S1-S2+S3, S5 = S1-S2, S6 = S3-S2) + 3 cross-channel
features (`cv_vert`, `cv_horiz`, `hv_ratio`).

## Repository layout

```text
src/                 core pipeline (data union -> feature extraction -> staged runner)
figures/             figure generators (Figures 3-9) and shared plotting helpers
analysis/            post-hoc analyses behind the tables and robustness panels
results/             derived feature tables and the aggregated CSV results consumed by figures
assets/              Figure 1 source (hand-drawn SVG)
data/                placeholder - the raw public datasets are downloaded, not shipped
docs/                reproducibility notes
manuscript/          placeholder for the HTML file referenced by the figure scripts
experiments/         output root of src/run_experiment.py (empty in the repository)
checkpoints_union/   intermediate cache root (empty in the repository; recreated on the first run)
```

### Core pipeline (`src/`)

| File | Role |
| --- | --- |
| `EHG_three_ds_union.py` | Preprocesses and merges the datasets into record-level arrays (`checkpoints_union/rec_full.npy`, `rec_full_meta.csv`). |
| `EHG_lit_features.py` | Extracts the 189-feature pool and the paired-AUC fingerprint into `results/checkpoints_lit_features_v3/`. |
| `ehg_featmeta.py` | Single source of truth for feature names, column order and Feature-ID mapping. |
| `run_experiment.py` | Staged runner: paired AUC (P1-P6) -> admission gates -> classifier evaluation. |

Run the pipeline in order:

```bash
python src/EHG_three_ds_union.py --full_records_only   # build record-level arrays
python src/EHG_lit_features.py                         # build the 189-feature pool (v3)
python src/run_experiment.py P1CI_P6eff050 42          # one outer seed of the main configuration
```

`run_experiment.py` is configured entirely through optional environment variables
(the defaults reproduce the main configuration of the paper):

| Variable | Default | Meaning |
| --- | --- | --- |
| `EHG_FT_CSV` | `results/checkpoints_lit_features_v3/lit_features_v3.csv` | Feature-pool CSV. |
| `EHG_SELECT` | `direct` | Feature selection: `direct` (main line), `sfs`, `embed`. |
| `EHG_CLASSL` | `rfgb` | Classifier family: `rfgb` (RF+GB ensemble), `logreg`, `svm`, `mlp`. |
| `EHG_SMOTE` | `0` | Apply SMOTE before the split (protocol-ladder rung, `1` = on). |
| `EHG_LEAKGATE` | `0` | Enable the leakage gate (protocol-ladder rung, `1` = on). |

Positional arguments are `<config-tag> [outer-seed] [lead-split]`;
`python src/run_experiment.py` with no arguments prints the 15 available config tags.

### Figure and table generators

| Paper item | Script | Consumes |
| --- | --- | --- |
| Table 4 / Figure 3 | `figures/make_fig3_final.py` | `results/checkpoints_lit_features_v3/pair_auc_matrix.csv` |
| Figure 4 | `figures/make_figs_4to9.py` (`fig4_roc_build` then `fig4_roc`) | `experiments/` + `results/checkpoints_lit_features_v2/tpehgt_ds_mother_map.csv` |
| Figure 5 | `figures/make_figs_4to9.py` (`fig5`) | `results/v3_eps_full_summary_20260924.csv` |
| Figure 6 | `figures/make_figs_4to9.py` (`fig6`) | `experiments/` (per-seed `pipe_03_sfs_*.xlsx`) |
| Figure 7 | `figures/make_figs_4to9.py` (`fig7`) | `results/v3_runs_summary.csv`, `results/v3_clsummary_20260927.csv` |
| Figure 8 | `figures/make_figs_4to9.py` (`fig8`) | `experiments/` (per-seed `pipe_03_sfs_*.xlsx`) |
| Figure 9 | `figures/make_figs_4to9.py` (`fig9`) | `results/v3_dims_D2D3D4_20260927.csv` |
| Figure 1 | `assets/fig1_framework.svg` | hand-drawn SVG (no generator) |

Figures are written to `figures/output/`. Figure 3 also exports
`Fig3_fingerprint_wide.csv` and `Fig3_fingerprint_long.csv` for manual redrawing.

### Analyses (`analysis/`)

| Script | Output |
| --- | --- |
| `_pairtest_gate2x2.py` | `results/v3_pairtest_gate2x2_20260928.csv` |
| `_r41_fingerprint.py` | `results/v3_table4_fingerprint_20260927.csv` (+ `_long`, `_bychan` variants) |
| `_r42_uncertainty.py` | `results/v3_uncertainty_20260927.csv`, `results/v3_roc_20260927.csv` |
| `_r44_insample.py` | `results/v3_insample_20260927.csv` |
| `_r45_dims.py` | `results/v3_dims_D2D3D4_20260927.csv` |
| `_r45_term42.py` | `results/v3_dims_term42_20261012.csv` |
| `_r45_gold.py`, `_r45_gold_diag.py` | Goldsztejn-style sub-cohort re-analysis |
| `_cl_dispersion.py` | Ensemble-vs-family dispersion comparison |
| `_summarize_cl.py` | `results/v3_clsummary_*.csv` |
| `figures/_summarize_runs.py` | `results/v3_runs_summary.csv` |

## Data

The raw datasets are public but are **not** redistributed here. See
[data/README.md](data/README.md) for download links and the directory layout the
scripts expect. The derived feature tables under `results/` are shipped so that
Figures 3, 5, 7 and 9 and the accompanying tables can be reproduced without the
raw signals.

## Environment

Python 3.10.9 was used for the manuscript. Install dependencies with:

```bash
pip install -r requirements.txt
```

## Reproducing the paper

1. Download the datasets and place them under `data/` (see [data/README.md](data/README.md)).
2. Build the record-level arrays and the feature pool:
   ```bash
   python src/EHG_three_ds_union.py --full_records_only
   python src/EHG_lit_features.py
   ```
3. Run the main configuration for the ten outer seeds:
   ```bash
   bash scripts/run_10seeds.sh
   ```
   (any Python with the dependencies works; on Windows PowerShell see
   [docs/reproducibility.md](docs/reproducibility.md)).
4. Regenerate the figures and tables:
   ```bash
   python figures/make_fig3_final.py
   python figures/make_figs_4to9.py
   ```

See [docs/reproducibility.md](docs/reproducibility.md) for the exact seeds,
the fixed bootstrap streams and the list of artefacts that are not regenerated
by the shipped scripts.

## Known gaps

- **Figure 2** (dataset overview) has no generator script in this repository.
- **Figure 1** is a hand-drawn SVG (`assets/fig1_framework.svg`), not generated from data.
- **Figures 4, 6 and 8** read per-seed outputs under `experiments/`, which are
  produced by `src/run_experiment.py` and are not shipped (they are large).
- `results/v3_eps_full_summary_20260924.csv` (Figure 5) has no generator script in
  this repository; the pre-computed summary is shipped as-is.
- The manuscript HTML consumed by the figure scripts is not included.

## License

Released under the MIT License - see [LICENSE](LICENSE).
