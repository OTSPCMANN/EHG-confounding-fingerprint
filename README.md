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
analysis/            post-hoc analyses and shared summary helpers behind the tables
results/             derived feature tables and the aggregated CSV result exports
assets/              Figure 1 source (hand-drawn SVG)
data/                placeholder - the raw public datasets are downloaded, not shipped
docs/                reproducibility notes
manuscript/          placeholder (empty in the repository)
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

### Figure and table data exports

The figures in the manuscript were drawn manually from the exported tables below.
This release ships the computation that produces those tables, not the drawing code.

| Paper item | Exported data | Produced by |
| --- | --- | --- |
| Table 4 / Figure 3 | `results/checkpoints_lit_features_v3/pair_auc_matrix.csv` | `src/EHG_lit_features.py` |
| Figure 4 (ROC) | per-seed outputs under `experiments/` + `results/checkpoints_lit_features_v2/tpehgt_ds_mother_map.csv` | `src/run_experiment.py` |
| Figure 5 (epsilon sweep) | `results/v3_eps_full_summary_20260924.csv` | shipped as-is (see Known gaps) |
| Figure 6 (admission criteria) | per-seed `experiments/**/pipe_03_sfs_*.xlsx` | `src/run_experiment.py` |
| Figure 7 (classifier family) | `results/v3_runs_summary.csv`, `results/v3_clsummary_20260927.csv` | `analysis/_summarize_runs.py`, `analysis/_summarize_cl.py` |
| Figure 8 (reference scale) | per-seed `experiments/**/pipe_03_sfs_*.xlsx`, `results/v3_insample_20260927.csv` | `src/run_experiment.py`, `analysis/_r44_insample.py` |
| Figure 9 (clinical strata) | `results/v3_dims_D2D3D4_20260927.csv` | `analysis/_r45_dims.py` |
| Figure 1 | `assets/fig1_framework.svg` | hand-drawn SVG (no generator) |

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
| `_summarize_runs.py` | `results/v3_runs_summary.csv` |
| `pipe05_ci.py` | `experiments/CI_<tag>_<n>runs_<timestamp>/bootstrap_ci_summary.xlsx` (record-level bootstrap intervals) |

## Data

The raw datasets are public but are **not** redistributed here. See
[data/README.md](data/README.md) for download links and the directory layout the
scripts expect. The derived feature tables under `results/` are shipped so that the reported
tables and the figure data can be checked without the raw signals.

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
4. Regenerate the exported tables behind the figures and manuscript tables
   (the summary scripts read the per-seed outputs written in step 3):

   ```bash
   python analysis/pipe05_ci.py            # record-level bootstrap intervals
   python analysis/_summarize_runs.py      # per-configuration run summary
   python analysis/_summarize_cl.py        # classifier-family summary
   python analysis/_r41_fingerprint.py     # fingerprint table
   python analysis/_r42_uncertainty.py     # ROC and uncertainty exports
   python analysis/_r44_insample.py        # reference-scale levels
   python analysis/_r45_dims.py            # D2-D4 exports
   ```

See [docs/reproducibility.md](docs/reproducibility.md) for the exact seeds,
the fixed bootstrap streams and the list of artefacts that are not regenerated
by the shipped scripts.

## Known gaps

- **Figure drawing code is not included.** Every manuscript figure was drawn
  manually from the exported CSV tables listed above.
- **Figure 2** (dataset overview) has no generator script in this repository.
- **Figure 1** is a hand-drawn SVG (`assets/fig1_framework.svg`), not generated from data.
- **Figures 4, 6 and 8** read per-seed outputs under `experiments/`, which are
  produced by `src/run_experiment.py` and are not shipped (they are large).
- `results/v3_eps_full_summary_20260924.csv` (Figure 5) has no generator script in
  this repository; the pre-computed summary is shipped as-is.
- `results/checkpoints_lit_features_v2/lit_features_v2.csv` (the 75-feature v2 pool)
  is not shipped, so `analysis/_r45_gold.py` and `analysis/_r45_gold_diag.py` cannot
  run unmodified.
- The manuscript HTML is not included.

## License

Released under the MIT License - see [LICENSE](LICENSE).
