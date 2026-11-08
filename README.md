# The resolution limit of a model comparison

Code and results for the paper *The resolution limit of a model comparison:
rankings that a dataset cannot support*.

The repository contains the analysis code for the five components named in the
paper's availability statement (protocol ladder, variance decomposition,
resolution analysis, cross-domain run, sample-size sweep), the expected schema
of the proprietary plant record, and the per-configuration results behind every
number quoted in the text.

## What reproduces what

| Paper item | Artefact in `outputs/` | Regenerate with |
|---|---|---|
| Table 6, plant, 22 configurations × 4 tiers (88 rows) | `g4_plant_ladder.csv` | `scripts/run_ladder_plant.py`, needs the plant record, about 10 min |
| Table 6, wide-range database (88 rows) | `g4_gpvisc_ladder.csv` | `scripts/run_full_grid_gpvisc.py`, public data only, about 10 min |
| One configuration end to end with per-fold predictions | `metrics_rebuilt.csv`, `oof_*.npy` | `scripts/run_ladder_gpvisc.py`, about 2 min |
| §4.6 sample-size sweep, 14 cells | `sweep_paired.csv` | shipped, produced by `scripts/make_sweep_figure.py` |
| Table 6, literature compilation (88 rows) | `g4_compilation_ladder.csv` | shipped; the assembly script is not, see Data below |

`python verify.py` checks 24 numbers against these files and exits non-zero if
any of them moved.

Two of the commands above were re-run against the shipped data while this
release was being prepared: `run_ladder_gpvisc.py` and `verify.py` (24/24). The
plant ladder and the full grid were executed in the authors' environment and
produced the tables shipped here; the copies in `scripts/` are those programs
with their paths redirected, and `run_full_grid_gpvisc.py` was smoke-tested on
the first protocol tier before release.

## Quick start

```bash
pip install -r requirements.txt

# 1. public data (downloads the melt-viscosity database and checks its checksum)
python scripts/download_public.py --check

# 2. the ladder on public data, end to end, about two minutes on a laptop CPU
python scripts/run_ladder_gpvisc.py

# 3. verify the shipped results against the published numbers
python verify.py
```

## Data

Three datasets are involved, and they are not equally available.

- **Wide-range database** (10,739 records, 1,367 groups). Public. Fetched by
  `scripts/download_public.py` from its cited location, with a checksum on the
  raw file. Nothing is repackaged.
- **Literature compilation** (242 records, 241 compositions). Redistributed as a
  machine-readable table in which every row carries its source label. The
  per-configuration results are shipped here; the assembly script is not,
  because the underlying workbooks belong to their original authors.
- **Plant record** (967 records, 110 slag samples, 1450–1530 °C). Proprietary,
  held under a data-use agreement. It is **not** redistributed. What is shipped
  instead is its schema (`data/schema/plant_record.json`) and the per-fold
  outputs, so every table and figure can be regenerated from the plant owner's
  own copy without modification.

Set `DSTAR_DATA`, `DSTAR_OUT`, `DSTAR_PLANT_CSV` or `DSTAR_GPVISC_CSV` to
relocate any of these.

## The resolution limit

For a paired comparison of two candidates over `k` folds, with `σ_d` the
standard deviation of the paired fold-level differences,

```
Δ* = T · σ_d / √k,     T = t_{0.975, k−1} + t_{0.80, k−1}
```

`T = 3.7174` at `k = 5`, against `2.8016` for the normal approximation. Two
models whose gap falls below `Δ*` cannot be separated by this design at 5 %
significance with 80 % power. `scripts/res_common.py` implements the estimator.

## Numbers pinned by `verify.py`

- Plant ladder, median R² by tier: +0.9773 / +0.7274 / +0.0534 / −1.5302
- Plant ladder, Extra-Trees on F3: RMSE 0.0000 / 0.2163 / 0.4804 / 0.5575
- Wide-range ladder, Extra-Trees on F3: RMSE 0.4312 / 0.5381 / 0.6262 / 0.9255
- Compilation ladder, Extra-Trees on F1: pooled R² at T2 and T3, 0.631 and 0.148
- Sample-size sweep: 14 cells

## Layout

```
paths.py                central path resolution, overridable by environment variable
verify.py               checks the shipped outputs against the published numbers
scripts/
  download_public.py    fetch + checksum the public databases
  run_ladder_plant.py   22 configurations × 4 tiers on the plant record
  run_ladder_gpvisc.py  the same ladder on the wide-range database
  run_crossdomain.py    seven public datasets
  run_resolution.py     variance decomposition and the resolution analysis
  sim_delta.py          calibration of Δ* (20,000 replicates per cell)
  sim_fold_corr.py      cross-fold correlation
  res_common.py         shared loading and the Δ* estimator
  make_figures_resolution.py, make_sweep_figure.py
outputs/                per-configuration results, per-fold predictions
data/schema/            expected schema of the plant record
```

## Licence

MIT, see `LICENSE`. The plant record is not covered by this licence and is not
included.

## Contact

Jinyu Wang, Hongwei Guo (corresponding), Helan Liang (corresponding),
Soochow University.
