# -*- coding: utf-8 -*-
"""Figures for the resolution study.

Labels are English on purpose: the target venue (IEEE TIM/TII) requires it, and it
sidesteps the CJK-font problem in matplotlib entirely.

Draws five panels, each usable standalone or assembled into one overview:
  rank_instability   are the rankings reproducible at all?
  band               how wide is the "indistinguishable" group?
  variance           split randomness vs training randomness
  power              how many slag samples to resolve a given gap
  regimes            which kind of generalisation is measured

CPU only, seconds.  Writes PNG (200 dpi) and SVG (vector, for the paper).
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from res_common import OUT_DIR

FIG = OUT_DIR / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.constrained_layout.use": True,
})
C_BAD, C_OK, C_NEU = "#C05621", "#2F855A", "#2B6CB0"


# ---------------------------------------------------------------------------------
def load():
    a = pd.read_csv(OUT_DIR / "perfold_A.csv", encoding="utf-8-sig")
    t = pd.read_csv(OUT_DIR / "perfold_T.csv", encoding="utf-8-sig")
    f = pd.read_csv(OUT_DIR / "perfold_F.csv", encoding="utf-8-sig")
    b = pd.read_csv(OUT_DIR / "perfold_B.csv", encoding="utf-8-sig")
    rep = json.loads((OUT_DIR / "analysis.json").read_text(encoding="utf-8"))
    return a, t, f, b, rep


def _foldings(a: pd.DataFrame) -> dict:
    a = a.copy()
    a["combo"] = a["scheme"] + "/" + a["model"]
    out = {}
    for (rr, ff), g in a.groupby(["rep", "fold"]):
        out[(int(rr), int(ff))] = g.sort_values("rmse_ln")["combo"].tolist()
    return out


def _taus(foldings: dict) -> list[float]:
    from scipy import stats
    ks = sorted(foldings)
    vals = []
    for i, j in combinations(range(len(ks)), 2):
        ri = {c: n for n, c in enumerate(foldings[ks[i]])}
        rj = {c: n for n, c in enumerate(foldings[ks[j]])}
        common = [c for c in ri if c in rj]
        vals.append(float(stats.kendalltau([ri[c] for c in common],
                                           [rj[c] for c in common]).statistic))
    return vals


def _jaccards(foldings: dict, k: int = 5) -> list[float]:
    ks = sorted(foldings)
    sets = [set(foldings[x][:k]) for x in ks]
    return [len(sets[i] & sets[j]) / len(sets[i] | sets[j])
            for i, j in combinations(range(len(sets)), 2)]


# ---------------------------------------------------------------------------------
def panel_instability(ax1, ax2, a: pd.DataFrame) -> None:
    foldings = _foldings(a)
    taus, jacs = _taus(foldings), _jaccards(foldings)

    ax1.hist(taus, bins=np.linspace(-0.2, 1.0, 25), color=C_NEU, alpha=.8)
    ax1.axvline(1.0, color="k", lw=1, ls="--")
    ax1.axvline(float(np.mean(taus)), color=C_BAD, lw=1.5)
    ax1.set_xlabel("Kendall $\\tau$ between fold rankings")
    ax1.set_ylabel("fold pairs")
    ax1.set_title("(a) Do two folds order the models the same way?\n"
                  f"mean $\\tau$ = {np.mean(taus):.2f}   (1.0 = identical)")
    ax1.set_xlim(-0.25, 1.05)

    ax2.hist(jacs, bins=np.linspace(0, 1, 21), color=C_BAD, alpha=.8)
    ax2.axvline(1.0, color="k", lw=1, ls="--")
    ax2.axvline(float(np.mean(jacs)), color="k", lw=1.5)
    ax2.set_xlabel("Jaccard overlap of the top-5 set")
    ax2.set_ylabel("fold pairs")
    ax2.set_title("(b) Do two folds agree on the top-5?\n"
                  f"mean overlap = {np.mean(jacs):.2f}   (1.0 = identical)")
    ax2.set_xlim(-0.03, 1.03)


def panel_flipmatrix(ax, a: pd.DataFrame) -> None:
    a = a.copy()
    a["combo"] = a["scheme"] + "/" + a["model"]
    order = a.groupby("combo")["rmse_ln"].mean().sort_values().index.tolist()
    foldings = _foldings(a)
    ks = sorted(foldings)
    n = len(order)

    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i, j] = np.nan
                continue
            want = i < j
            agree = sum(1 for k in ks
                        if (foldings[k].index(order[i]) < foldings[k].index(order[j])) == want)
            M[i, j] = 1 - agree / len(ks)

    im = ax.imshow(M, cmap="RdYlGn_r", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(c) Pairwise rank-flip rate across folds\n"
                 "(rows/cols ordered by mean RMSE; darker red = order flips more often)")
    ax.set_xlabel("combo, best → worst"); ax.set_ylabel("combo, best → worst")
    cb = ax.figure.colorbar(im, ax=ax, fraction=.046, pad=.02)
    cb.set_label("flip rate")
    # mark the indistinguishability frontier
    sd = a.groupby("combo")["rmse_ln"].std().median()
    best = a.groupby("combo")["rmse_ln"].mean().min()
    mean = a.groupby("combo")["rmse_ln"].mean()
    nb = int((mean - best <= sd).sum())
    ax.axhline(nb - .5, color="k", lw=1.2, ls="--")
    ax.axvline(nb - .5, color="k", lw=1.2, ls="--")
    ax.text(.02, nb + .6, f"{nb} combos within one fold-sd",
            transform=ax.transData, fontsize=7, va="top")


def _set_title(ax, tag, text: str) -> None:
    """Panel letters belong only to the overview figure, and a standalone figure needs
    no title at all -- the IEEE caption already carries it, and a second copy inside
    the image both duplicates the caption and costs vertical space.

    tag="(a) " -> titled and lettered (overview)
    tag=""     -> titled, no letter       (standalone, keeps the message in-frame)
    tag=None   -> no title                (standalone, caption carries it)
    """
    if tag is None:
        return
    ax.set_title(f"{tag}{text}" if tag else text)


def panel_band(ax, a: pd.DataFrame, rep: dict, tag=None) -> None:
    a = a.copy()
    a["combo"] = a["scheme"] + "/" + a["model"]
    g = a.groupby("combo")["rmse_ln"].agg(["mean", "std"]).sort_values("mean")
    x = np.arange(len(g))
    best, sd = g["mean"].iloc[0], g["std"].median()

    ax.errorbar(x, g["mean"], yerr=g["std"], fmt="o", ms=3, lw=0, elinewidth=1,
                capsize=1.5, color=C_NEU, alpha=.9)
    ax.axhline(best, color="k", lw=1)
    ax.axhspan(best, best + sd, color=C_BAD, alpha=.13)
    ax.axhline(best + sd, color=C_BAD, lw=1, ls="--")
    # label the dashed line from above. Placing it at the line's own height, as before,
    # drew the rule straight through the text.
    ax.annotate(f"best + 1 fold-sd  ({sd:.4f})", xy=(0.985, best + sd),
                xycoords=("axes fraction", "data"), xytext=(0, 5),
                textcoords="offset points", fontsize=7.5, color=C_BAD,
                ha="right", va="bottom")
    n_in = int((g["mean"] - best <= sd).sum())
    ax.axvline(n_in - 0.5, color=C_BAD, lw=1, ls=":")
    ax.set_xlabel("model × feature scheme, ordered by mean RMSE")
    ax.set_ylabel("RMSE(ln $\\eta$)  ±  fold-to-fold sd")
    ax.set_ylim(g["mean"].min() - sd, g["mean"].max() + sd * 1.15)
    _set_title(ax, tag, f"{n_in} of {len(g)} combinations lie within one "
                         "fold-sd of the best;\ntheir ordering is not resolvable "
                         "from this dataset")


def panel_variance(ax, b: pd.DataFrame, rep: dict, tag=None) -> None:
    """Grouped bars rather than a scatter: with six combos the labels overlap and the
    only message that matters is the RATIO."""
    b = b.copy()
    b["combo"] = b["scheme"] + "/" + b["model"]
    rows = []
    for c, g in b.groupby("combo"):
        sp = float(g.groupby("train_seed")["rmse_ln"].std(ddof=1).median())
        per_seed = g.groupby("train_seed")["rmse_ln"].mean()
        tr = float(per_seed.std(ddof=1)) if len(per_seed) > 1 else 0.0
        rows.append((c, sp, tr))
    rows.sort(key=lambda r: -r[1])
    names = [r[0] for r in rows]
    sp = np.array([r[1] for r in rows])
    tr = np.array([r[2] for r in rows])
    eps = 1e-5
    y = np.arange(len(names))
    ax.barh(y + .19, sp, height=.36, color=C_NEU, label="data split (training seed fixed)")
    ax.barh(y - .19, np.maximum(tr, eps), height=.36, color=C_BAD,
            label="training seed (split fixed)")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7.5)
    ax.invert_yaxis()
    for yy, s_, t_ in zip(y, sp, tr):
        ax.text(s_ * 1.15, yy + .19, f"{s_:.4f}", va="center", fontsize=6.5)
        ax.text(max(t_, eps) * 1.15, yy - .19,
                "0" if t_ < eps else f"{t_:.4f}", va="center", fontsize=6.5, color=C_BAD)
    ax.set_xscale("log")
    ax.set_xlabel("sd of RMSE(ln $\\eta$)  [log scale]")
    # Legend below the axes. Inside the frame it sat on top of the shortest bar row
    # and hid that row's value label.
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2,
              frameon=False)
    r = rep.get("variance_split", {}).get("median_train_over_split_ratio", float("nan"))
    _set_title(ax, tag, "Where the uncertainty comes from: median training-seed / "
                         f"split variance = {r:.2f}")



def panel_power(ax, a: pd.DataFrame, rep: dict, tag=None) -> None:
    per_fold = a.groupby(["scheme", "model", "rep", "fold"])["rmse_ln"].mean()
    sd = float(per_fold.groupby(level=["scheme", "model"]).std(ddof=1).median())
    z = 1.959963985 + 0.841621234

    d = np.geomspace(0.004, 0.2, 200)
    samples_per_fold = 110 / 5
    n = np.ceil(z ** 2 * sd ** 2 / d ** 2) * samples_per_fold

    ax.plot(d, n, color=C_NEU, lw=2)
    ax.axhline(110, color=C_BAD, lw=1.5)
    ax.text(0.0045, 128, "this dataset: 110 slag samples", color=C_BAD, fontsize=7.5)
    for dd in (0.01, 0.02, 0.05):
        nn = float(np.ceil(z ** 2 * sd ** 2 / dd ** 2) * samples_per_fold)
        ax.plot([dd], [nn], "ko", ms=4)
        ax.annotate(f"{nn:,.0f}", (dd, nn), fontsize=7, xytext=(4, 3),
                    textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("gap in RMSE(ln $\\eta$) to be confirmed")
    ax.set_ylabel("slag samples required")
    _set_title(ax, tag, "Slag samples required to confirm a given difference\n"
                         f"median fold-sd = {sd:.4f}, 22 samples per fold; "
                         "differences below ~0.06 are out of reach")


def panel_regimes(ax, rep: dict) -> None:
    rg = rep.get("regimes", {})
    s = rg.get("summary", {})
    names = [("leave_sample", "leave-one-sample\n(composition\nextrapolation)"),
             ("leave_temp", "leave-one-temperature\n(pure temperature\ninterpolation)"),
             ("leave_furnace", "leave-one-furnace\n(transfer to a\nnew furnace)")]
    vals = [s.get(k, {}).get("best_mean_rmse", np.nan) for k, _ in names]
    cols = [C_NEU, C_OK, C_NEU]
    bars = ax.bar([n for _, n in names], vals, color=cols, alpha=.85, width=.6)
    for bar, v in zip(bars, vals):
        if np.isfinite(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + .008, f"{v:.4f}",
                    ha="center", fontsize=8)
    ie = rg.get("interior_vs_edge")
    if ie:
        ax.axhline(ie["interior_rmse"], color=C_OK, lw=1.2, ls=":")
        ax.text(2.42, ie["interior_rmse"] + .008,
                f"interior-only {ie['interior_rmse']:.4f}", fontsize=7,
                color=C_OK, ha="right")
        ax.axhline(ie["edge_rmse"], color=C_BAD, lw=1.2, ls=":")
        ax.text(2.42, ie["edge_rmse"] + .008,
                f"edges-only {ie['edge_rmse']:.4f}", fontsize=7, color=C_BAD, ha="right")
    ax.set_ylabel("best RMSE(ln $\\eta$)")
    ax.set_title("(d) Which generalisation is being measured?\n"
                 "temperature interpolation is 3.7× easier than composition extrapolation")
    ax.tick_params(axis="x", labelsize=7)


# ---------------------------------------------------------------------------------
def save(fig, name: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.png / .svg")


def main() -> None:
    a, t, f, b, rep = load()

    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.4))
    panel_instability(ax[0], ax[1], a)
    save(fig, "fig_rank_instability")

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    panel_flipmatrix(ax, a)
    save(fig, "fig_flip_matrix")

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    panel_band(ax, a, rep, tag=None)
    save(fig, "fig_band")

    fig, ax = plt.subplots(figsize=(4.6, 4.1))
    panel_variance(ax, b, rep, tag=None)
    save(fig, "fig_variance")

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    panel_power(ax, a, rep, tag=None)
    save(fig, "fig_power")

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    panel_regimes(ax, rep)
    save(fig, "fig_regimes")

    # overview: 2 x 3
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.0))
    panel_instability(axes[0, 0], axes[0, 1], a)
    panel_flipmatrix(axes[0, 2], a)
    panel_band(axes[1, 0], a, rep, tag="(a) ")
    panel_variance(axes[1, 1], b, rep, tag="(b) ")
    panel_power(axes[1, 2], a, rep, tag="(c) ")
    fig.suptitle("Model comparison on small industrial data: what is actually resolvable?",
                 fontsize=13)
    save(fig, "fig_overview")

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    panel_regimes(ax, rep)
    save(fig, "fig_overview_regimes")


if __name__ == "__main__":
    main()
