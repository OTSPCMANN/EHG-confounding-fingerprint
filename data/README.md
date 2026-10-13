# Data

The raw datasets are public but are **not** redistributed in this repository.
Download them from PhysioNet and place them under this `data/` directory using
the layout below.

| Cohort | Source | Used for |
| --- | --- | --- |
| TPEHG DB | Term-Preterm EHG Database - https://physionet.org/content/tpehgdb/ | Development cohort (300 records: 38 preterm / 262 term). Feature extraction and model development. |
| TPEHGT DS | Term-Preterm EHG Database with Tocogram - https://physionet.org/content/tpehgt/ | External validation cohort (26 records from 18 women). |
| ICEHG DS | Icelandic EHG database (optional) | **Not used** by the analyses in the paper: all records are term. It is only scanned by `src/EHG_three_ds_union.py`. |

If a directory below is absent the union step simply skips it, so the two
PhysioNet cohorts are sufficient to reproduce every result in the paper.

## Expected layout

```text
data/
  TPEHG_DB/
    tpehgdb/        # *.hea + *.dat (TPEHG DB records)
  TPEHGT_DS/        # *.hea + *.dat (TPEHGT DS records)
  ICEHG_DS/         # optional; ignored by the paper analyses
    early_induced/ later_induced/ early_cesarean/ later_cesarean/
    early_induced-cesarean/ later_induced-cesarean/
    Table_1.txt Table_2.txt Table_3.txt
```

THESE directories are consumed by:

- `src/EHG_three_ds_union.py` - reads the raw signals and writes the
  record-level arrays to `checkpoints_union/`.
- `src/EHG_lit_features.py` - reads `checkpoints_union/rec_full.npy` (and its
  metadata) to build the 189-feature pool.

The derived feature tables that these steps produce are shipped under
`results/checkpoints_lit_features_v3/`, so the downstream figure and table
scripts can be run without re-downloading the raw signals.
