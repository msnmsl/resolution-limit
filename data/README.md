# Data

Three datasets are involved and they are not equally available.

## Public: the wide-range database

10,739 records over 1,367 groups. Fetched by `scripts/download_public.py` from
the gpvisc release (DOI 10.5281/zenodo.13843250, version 0.3.5) and checked
against the recorded SHA-256. Nothing is repackaged.

```bash
python scripts/download_public.py --check
```

## Redistributable: the literature compilation

242 records over 241 compositions, one record per composition. Redistributed as
a machine-readable table in which every row carries its source label. The
per-configuration results are shipped under `outputs/`; the assembly script is
not, because the underlying workbooks belong to their original authors.

## Proprietary: the plant record

967 records over 110 slag samples, 1450-1530 degrees Celsius, five furnaces.
Held under a data-use agreement and **not** redistributed here.

What is shipped instead:

- `schema/plant_record.json` documents every expected column, the grain of a
  record, the grouping key and the target, so the plant owner can regenerate
  every table and figure from their own copy without editing any code;
- `outputs/g4_plant_ladder.csv` holds the per-configuration results for the
  four protocol tiers;
- `outputs/oof_*.npy` hold the per-record out-of-fold predictions.

To run the ladder against your own copy:

```bash
export DSTAR_PLANT_CSV=/path/to/interval_data.csv
python scripts/run_ladder_plant.py
```
