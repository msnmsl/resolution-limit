# -*- coding: utf-8 -*-
"""Figures for the sample-size sweep: what actually drives resolvability.

The sweep shrinks a fixed dataset, so within each curve only n changes. That isolates
the effect of n from the effect of the domain.

    python make_sweep_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_crossdomain import N_FOLDS
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import OUT, PLANT_CSV, GPVISC_CSV  # noqa: E402


HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

# Sum of the two-sided 5% and 80%-power quantiles; t with df = k-1 is used
# instead of the normal approximation (2026-10-24, paper-wide switch).
Z = 2.7764451051977987 + 0.9409644160109035
# sigma_fold is a FOLD-level statistic (sd of a model's error across the split x fold
# cells), so the resolution limit divides by sqrt(k folds), not by sqrt(n rows).
# Dividing a fold-level sd by sqrt(rows) mixes two different scales.
# Cross-check: the viscosity dataset has sigma_fold = 0.0475 over k = 5 folds, giving
# Delta* = 0.079 in ln(eta) units -- which matches its independent power table, where
# resolving 0.05 would need 286 slag samples against the 110 available.
DS_DENOM = np.sqrt(N_FOLDS)
COL = {"concrete": "#2B6CB0", "airfoil": "#2F855A", "diabetes": "#C05621"}
DOM = {"concrete": "concrete (civil)", "airfoil": "airfoil (aerospace)",
       "diabetes": "diabetes (medical)"}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 9,
    "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.constrained_layout.use": True,
})


def main() -> None:
    s = pd.read_csv(OUT / "sweep.csv", encoding="utf-8-sig")
    s = s.drop_duplicates(subset=["dataset", "n"])          # the full row appeared twice
    s["delta_star"] = Z * s["fold_sd"] / DS_DENOM           # superseded below
    # 2026-10-24: the paper moved to the paired-test definition, so Delta* and
    # the in-band count come from sweep_paired.csv rather than from fold_sd.
    sp = pd.read_csv(OUT / "sweep_paired.csv")
    s = s.merge(sp[["dataset", "n", "band", "delta"]], on=["dataset", "n"], how="left")
    s["delta_star"] = s["delta"]
    s["in_band"] = s["band"]

    tab = pd.read_csv(OUT / "cross_dataset_table.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(2, 2, figsize=(11.2, 8.0))

    # (a) tau vs n -- within each dataset only n changes
    A = ax[0, 0]
    for d, g in s.groupby("dataset"):
        g = g.sort_values("n")
        A.plot(g["n"], g["tau"], "o-", color=COL[d], lw=2, ms=6, label=DOM[d])
        A.annotate(f"n={int(g['n'].iloc[-1])}", (g["n"].iloc[-1], g["tau"].iloc[-1]),
                   fontsize=7, xytext=(5, -2), textcoords="offset points", color=COL[d])
    A.axhline(0.8, color="#2F855A", ls=":", lw=1)
    A.axhline(0.6, color="#B22222", ls=":", lw=1)
    A.text(85, 0.815, "stable", fontsize=7.5, color="#2F855A")
    A.text(85, 0.40, "not resolvable", fontsize=7.5, color="#B22222")
    A.set_xscale("log")
    A.set_xlabel("training samples")
    A.set_ylabel("Kendall $\\tau$ between fold rankings")
    A.set_ylim(0, 1.0)
    A.set_title("(a) Ranking stability against sample size\nmonotone in all three datasets")
    A.legend(loc="lower right", fontsize=7.5)

    # (b) tau vs R^2, sweep + the seven datasets
    B = ax[0, 1]
    for d, g in s.groupby("dataset"):
        g = g.sort_values("best_r2")
        B.plot(g["best_r2"], g["tau"], "o-", color=COL[d], lw=1.6, ms=6,
               label=DOM[d], alpha=.9)
    sw = tab.dropna(subset=["tau"])
    for fam, mk, c, lb in (("public", "s", "#4A5568", "public table"),
                           ("industrial", "^", "#B22222", "industrial")):
        gg = sw[sw.family == fam]
        B.scatter(gg["best_r2"], gg["tau"], marker=mk, s=55, color=c, zorder=3,
                  label=lb, edgecolor="white", linewidth=.7)
    # plain text: with the arrow, the head landed on top of the word "tau"
    B.text(0.625, 0.66, "diabetes: $R^2$ barely moves\nwhile $\\tau$ nearly doubles",
           fontsize=7, color=COL["diabetes"], ha="center", va="center")
    B.set_xlabel("best model $R^2$")
    B.set_ylabel("Kendall $\\tau$")
    B.set_ylim(0, 1.0)
    B.set_title("(b) Ranking stability against $R^2$\n"
                "positive but disproportionate")
    B.legend(loc="upper left", fontsize=7)

    # (c) tau vs the resolution limit Delta*
    C = ax[1, 0]
    for d, g in s.groupby("dataset"):
        g = g.sort_values("delta_star")
        C.plot(g["delta_star"], g["tau"], "o-", color=COL[d], lw=1.6, ms=6,
               label=DOM[d])
    C.set_xscale("log")
    C.set_xlabel("resolution limit $\\Delta^* = 3.72\\,\\sigma_{d}/\\sqrt{k}$"
                 f"  (k = {N_FOLDS} folds)  [units of target sd]")
    C.set_ylabel("Kendall $\\tau$")
    C.set_ylim(0, 1.0)
    # where the industrial slag-viscosity dataset sits: sigma_fold = 0.0475 over k = 5
    D_VISC = Z * 0.04755 / np.sqrt(5)
    C.axvline(D_VISC, color="#B22222", ls="--", lw=1.1)
    C.text(D_VISC * 1.12, 0.92, f"slag viscosity\n$\\Delta^*$ = {D_VISC:.3f}",
           fontsize=7, color="#B22222", va="top")
    C.set_title("(c) Ranking stability against the resolution limit\n"
                "ordered within datasets, not between them")
    C.legend(loc="lower right", fontsize=7)

    # (d) the protocol's own output: in-band fraction vs tau
    D = ax[1, 1]
    for d, g in s.groupby("dataset"):
        g = g.sort_values("in_band")
        D.plot(g["in_band"], g["tau"], "o-", color=COL[d], lw=1.6, ms=6,
               label=DOM[d])
    D.set_xlabel("models the paired test cannot separate  "
                 "(the protocol's own output)")
    D.set_ylabel("Kendall $\\tau$")
    D.set_ylim(0, 1.0)
    from scipy import stats
    allp = s.dropna(subset=["tau"])
    r = stats.spearmanr(allp["in_band"], allp["tau"])
    D.set_title(f"(d) Ranking stability against the in-band count\n"
                f"Spearman $\\rho$ = {r.statistic:.2f} over the {len(allp)} cells")
    D.legend(loc="upper right", fontsize=7)
    D.text(0.97, 0.04, "more models inside the band: the ranking is not resolvable",
           transform=D.transAxes, ha="right", fontsize=7, color="#4A5568")

    # No suptitle: IEEE figures carry their title in the caption, and duplicating it
    # inside the image costs vertical space that an 8-page budget cannot spare.
    for ext in ("png", "svg", "pdf"):
        fig.savefig(FIG / f"fig_sweep.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  figures/fig_sweep.png / .svg")

    s.to_csv(OUT / "sweep_with_delta.csv", index=False, encoding="utf-8-sig")
    print(s[["dataset", "n", "best_r2", "fold_sd", "delta_star", "tau"]]
          .to_string(index=False, float_format=lambda v: f"{v:.5f}"))


if __name__ == "__main__":
    main()
