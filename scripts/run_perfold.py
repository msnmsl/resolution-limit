# -*- coding: utf-8 -*-
"""Phase 1: run the grid and record EVERY fit per fold.

Three grids, all on identical folds within each grid:

  A  full matrix     11 models x 4 schemes x 2 repeats x 5 folds, one training seed
                     per repeat (this reproduces the original pipeline but keeps the
                     per-fold rows instead of averaging them away)
  B  variance split  the strongest combos, split seed FIXED, several TRAINING seeds
                     -> separates "how the data was divided" from "how the algorithm
                        rolled"
  C  mechanistic     B0/B1/B2/B3 on the same folds as A -> enables the paired
                     ML-vs-mechanism test

Outputs (out/):
  perfold_A.csv  perfold_B.csv  perfold_base.csv  run_log.txt

CPU only.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback

import numpy as np
import pandas as pd

from res_common import (BASELINE_NAMES, MODEL_NAMES, N_REPEAT, N_SPLITS, OUT_DIR,
                        SCHEMES, SEED, load_data, make_models, mechanistic_predictions,
                        metrics, scheme_frame, split_folds)

LOG: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


def grid_a(df: pd.DataFrame, models_filter: list[str], schemes_filter: list[str]
           ) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, oof = [], []
    for scheme in schemes_filter:
        d = scheme_frame(df, scheme).reset_index(drop=True)
        feats = SCHEMES[scheme]
        y = d["ln_eta"].values
        for rep in range(N_REPEAT):
            split_seed = SEED + rep          # matches the original pipeline
            train_seed = SEED + rep          # ...including the seed-per-repeat coupling
            folds = split_folds(d["group"].values, N_SPLITS, split_seed)
            for fold, (tr, te) in enumerate(folds):
                Xtr, Xte = d.iloc[tr][feats].values, d.iloc[te][feats].values
                for mname in models_filter:
                    try:
                        m = make_models(train_seed)[mname]
                        m.fit(Xtr, y[tr])
                        p = np.asarray(m.predict(Xte)).ravel()
                        mt = metrics(y[te], p)
                    except Exception as exc:                     # noqa: BLE001
                        log(f"  !! {scheme}/{mname}/rep{rep}/fold{fold}: {exc}")
                        continue
                    rows.append({"scheme": scheme, "model": mname, "rep": rep, "fold": fold,
                                 "split_seed": split_seed, "train_seed": train_seed,
                                 "n_test": len(te), "n_group_test": d.iloc[te]["group"].nunique(),
                                 **mt})
                    # point-level out-of-fold predictions: these are what make the
                    # paired comparisons powerful (n = test points, not n = folds)
                    oof.append(pd.DataFrame({
                        "scheme": scheme, "model": mname, "rep": rep, "fold": fold,
                        "group": d.iloc[te]["group"].values,
                        "furnace": d.iloc[te]["furnace"].values,
                        "T_C": d.iloc[te]["T_C"].values,
                        "liquid": (d.iloc[te]["eta_Pas"].values < 0.8),
                        "y_true": y[te], "y_pred": p,
                    }))
            log(f"  [A] {scheme} rep{rep} done")
    return pd.DataFrame(rows), pd.concat(oof, ignore_index=True)


def grid_b(df: pd.DataFrame, combos: list[tuple[str, str]], n_train_seeds: int,
           split_seed: int = SEED) -> pd.DataFrame:
    """Fixed split, varying training seed -> isolates training-side randomness."""
    rows = []
    for scheme, mname in combos:
        d = scheme_frame(df, scheme).reset_index(drop=True)
        feats = SCHEMES[scheme]
        y = d["ln_eta"].values
        folds = split_folds(d["group"].values, N_SPLITS, split_seed)
        for fold, (tr, te) in enumerate(folds):
            Xtr, Xte = d.iloc[tr][feats].values, d.iloc[te][feats].values
            for k in range(n_train_seeds):
                tseed = SEED + 100 + k          # deliberately disjoint from grid A seeds
                try:
                    m = make_models(tseed)[mname]
                    m.fit(Xtr, y[tr])
                    p = np.asarray(m.predict(Xte)).ravel()
                    mt = metrics(y[te], p)
                except Exception as exc:                         # noqa: BLE001
                    log(f"  !! B {scheme}/{mname}/fold{fold}/ts{tseed}: {exc}")
                    continue
                rows.append({"scheme": scheme, "model": mname, "fold": fold,
                             "split_seed": split_seed, "train_seed": tseed,
                             "n_test": len(te), **mt})
        log(f"  [B] {scheme}/{mname} done ({n_train_seeds} training seeds x {N_SPLITS} folds)")
    return pd.DataFrame(rows)


def grid_c(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mechanistic baselines on the grid-A folds, so pairing is exact."""
    rows, oof = [], []
    for scheme in SCHEMES:
        d = scheme_frame(df, scheme).reset_index(drop=True)
        y = d["ln_eta"].values
        for rep in range(N_REPEAT):
            split_seed = SEED + rep
            folds = split_folds(d["group"].values, N_SPLITS, split_seed)
            for fold, (tr, te) in enumerate(folds):
                trd, ted = d.iloc[tr], d.iloc[te]
                preds = mechanistic_predictions(trd, ted, y[tr])
                for bname, p in preds.items():
                    mt = metrics(y[te], p)
                    rows.append({"scheme": scheme, "model": bname, "rep": rep, "fold": fold,
                                 "split_seed": split_seed, "n_test": len(te), **mt})
                    oof.append(pd.DataFrame({
                        "scheme": scheme, "model": bname, "rep": rep, "fold": fold,
                        "group": d.iloc[te]["group"].values,
                        "furnace": d.iloc[te]["furnace"].values,
                        "T_C": d.iloc[te]["T_C"].values,
                        "liquid": (d.iloc[te]["eta_Pas"].values < 0.8),
                        "y_true": y[te], "y_pred": p,
                    }))
            log(f"  [C] {scheme} rep{rep} done")
    return pd.DataFrame(rows), pd.concat(oof, ignore_index=True)


def grid_t(df: pd.DataFrame, models: list[str], schemes: list[str],
           min_test: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-TEMPERATURE-out.

    Every sampling is measured at a shared grid of temperatures, so in the main
    (leave-one-sample-out) protocol the model never has to predict a temperature it has
    not seen. This grid holds out an entire temperature, so it measures how much of the
    reported accuracy is temperature interpolation -- the easy part -- and how much is
    genuine composition extrapolation.
    """
    rows, oof = [], []
    for scheme in schemes:
        d = scheme_frame(df, scheme).reset_index(drop=True)
        feats = SCHEMES[scheme]
        y = d["ln_eta"].values
        counts = d["T_C"].value_counts()
        temps = sorted(t for t, c in counts.items() if c >= min_test)
        for T in temps:
            te = np.where(d["T_C"].values == T)[0]
            tr = np.where(d["T_C"].values != T)[0]
            if len(tr) < 50:
                continue
            Xtr, Xte = d.iloc[tr][feats].values, d.iloc[te][feats].values
            for mname in models:
                try:
                    m = make_models(SEED + 300)[mname]
                    m.fit(Xtr, y[tr])
                    p = np.asarray(m.predict(Xte)).ravel()
                    mt = metrics(y[te], p)
                except Exception as exc:                         # noqa: BLE001
                    log(f"  !! T {scheme}/{mname}/T={T}: {exc}")
                    continue
                rows.append({"scheme": scheme, "model": mname, "holdout": float(T),
                             "regime": "leave_temp", "n_test": len(te), **mt})
                oof.append(pd.DataFrame({
                    "scheme": scheme, "model": mname, "rep": 0, "fold": int(T),
                    "group": d.iloc[te]["group"].values,
                    "furnace": d.iloc[te]["furnace"].values,
                    "T_C": d.iloc[te]["T_C"].values,
                    "liquid": (d.iloc[te]["eta_Pas"].values < 0.8),
                    "y_true": y[te], "y_pred": p,
                }))
        log(f"  [T] {scheme} done ({len(temps)} temperatures held out)")
    return pd.DataFrame(rows), pd.concat(oof, ignore_index=True)


def grid_f(df: pd.DataFrame, models: list[str], schemes: list[str]
           ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-FURNACE-out: does the model transfer to a furnace it has never seen?"""
    rows, oof = [], []
    for scheme in schemes:
        d = scheme_frame(df, scheme).reset_index(drop=True)
        feats = SCHEMES[scheme]
        y = d["ln_eta"].values
        for fu in sorted(d["furnace"].unique()):
            te = np.where(d["furnace"].values == fu)[0]
            tr = np.where(d["furnace"].values != fu)[0]
            Xtr, Xte = d.iloc[tr][feats].values, d.iloc[te][feats].values
            for mname in models:
                try:
                    m = make_models(SEED + 400)[mname]
                    m.fit(Xtr, y[tr])
                    p = np.asarray(m.predict(Xte)).ravel()
                    mt = metrics(y[te], p)
                except Exception as exc:                         # noqa: BLE001
                    log(f"  !! F {scheme}/{mname}/{fu}: {exc}")
                    continue
                rows.append({"scheme": scheme, "model": mname, "holdout": fu,
                             "regime": "leave_furnace", "n_test": len(te), **mt})
                oof.append(pd.DataFrame({
                    "scheme": scheme, "model": mname, "rep": 0, "fold": -1,
                    "group": d.iloc[te]["group"].values,
                    "furnace": d.iloc[te]["furnace"].values,
                    "T_C": d.iloc[te]["T_C"].values,
                    "liquid": (d.iloc[te]["eta_Pas"].values < 0.8),
                    "y_true": y[te], "y_pred": p,
                }))
        log(f"  [F] {scheme} done ({d['furnace'].nunique()} furnaces held out)")
    return pd.DataFrame(rows), pd.concat(oof, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="tiny subset, for checking the plumbing in ~1 minute")
    ap.add_argument("--train-seeds", type=int, default=5,
                    help="how many training seeds for the variance decomposition (grid B)")
    ap.add_argument("--top-combos", type=int, default=6,
                    help="how many of the strongest model x scheme combos go into grid B")
    ap.add_argument("--skip-b", action="store_true")
    ap.add_argument("--skip-regimes", action="store_true",
                    help="skip the leave-one-temperature and leave-one-furnace grids")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = load_data()
    log(f"data: {df['group'].nunique()} groups / {len(df)} points / "
        f"liquid-segment {int((df['eta_Pas'] < 0.8).sum())} points")

    models = ["Ridge", "ExtraTrees"] if args.quick else MODEL_NAMES
    schemes = ["F1"] if args.quick else list(SCHEMES)

    log(f"\n=== grid A: {len(models)} models x {len(schemes)} schemes x "
        f"{N_REPEAT} repeats x {N_SPLITS} folds ===")
    try:
        a, oof_a = grid_a(df, models, schemes)
    except Exception:                                            # noqa: BLE001
        traceback.print_exc()
        return 1
    a.to_csv(OUT_DIR / "perfold_A.csv", index=False, encoding="utf-8-sig")
    oof_a.to_csv(OUT_DIR / "oof_A.csv", index=False, encoding="utf-8-sig")
    log(f"grid A: {len(a)} fits -> perfold_A.csv   ({len(oof_a)} point predictions -> oof_A.csv)")

    log("\n=== grid C: mechanistic baselines on the same folds ===")
    c, oof_c = grid_c(df)
    c.to_csv(OUT_DIR / "perfold_base.csv", index=False, encoding="utf-8-sig")
    oof_c.to_csv(OUT_DIR / "oof_base.csv", index=False, encoding="utf-8-sig")
    log(f"grid C: {len(c)} fits -> perfold_base.csv   ({len(oof_c)} point predictions)")

    if not args.skip_b:
        top = (a.groupby(["scheme", "model"])["rmse_ln"].mean()
               .sort_values().head(args.top_combos).index.tolist())
        log(f"\n=== grid B: variance decomposition on the {len(top)} strongest combos ===")
        for s, m in top:
            log(f"    {s}/{m}")
        b = grid_b(df, top, args.train_seeds)
        b.to_csv(OUT_DIR / "perfold_B.csv", index=False, encoding="utf-8-sig")
        log(f"grid B: {len(b)} fits -> perfold_B.csv")

    if not args.skip_regimes:
        log("\n=== grid T: leave-one-temperature-out ===")
        t, oof_t = grid_t(df, models, schemes)
        t.to_csv(OUT_DIR / "perfold_T.csv", index=False, encoding="utf-8-sig")
        oof_t.to_csv(OUT_DIR / "oof_T.csv", index=False, encoding="utf-8-sig")
        log(f"grid T: {len(t)} fits -> perfold_T.csv")

        log("\n=== grid F: leave-one-furnace-out ===")
        f, oof_f = grid_f(df, models, schemes)
        f.to_csv(OUT_DIR / "perfold_F.csv", index=False, encoding="utf-8-sig")
        oof_f.to_csv(OUT_DIR / "oof_F.csv", index=False, encoding="utf-8-sig")
        log(f"grid F: {len(f)} fits -> perfold_F.csv")

    log(f"\ntotal wall clock: {time.time() - t0:.0f}s")
    (OUT_DIR / "run_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
