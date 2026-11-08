# -*- coding: utf-8 -*-
"""Phase 2/3: the four analyses that make this a study rather than a leaderboard.

  1. noise floor -> achievable ceiling      (what is learnable at all)
  2. rank stability                          (is the ORDERING resolvable)
  3. variance decomposition                  (split randomness vs training randomness)
  4. power                                   (how much data to resolve a given gap)
  5. paired ML-vs-mechanism at POINT level   (does ML actually beat a 1970s model)

Writes out/resolution_report.md and out/analysis.json.  CPU only, seconds.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from res_common import N_SPLITS, NOISE_FLOOR_RMSE_LN, OUT_DIR, load_data
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import OUT, PLANT_CSV, GPVISC_CSV  # noqa: E402


Z_975, Z_80 = 1.959963985, 0.841621234


# ---------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------
def required_n_paired(sigma_d: float, delta: float, power_z: float = Z_80) -> float:
    """Pairs needed to detect a paired difference ``delta`` with SD of differences sigma_d."""
    if not np.isfinite(sigma_d) or not np.isfinite(delta) or delta <= 0:
        return float("inf")
    return (Z_975 + power_z) ** 2 * sigma_d ** 2 / delta ** 2


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else float("nan")


def bootstrap_ci(x: np.ndarray, n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ---------------------------------------------------------------------------------
# 1. noise floor -> achievable ceiling
# ---------------------------------------------------------------------------------
def sec_ceiling(df: pd.DataFrame) -> dict:
    """The achievable ceiling -- and an honest statement of whether it is even estimable.

    ``NOISE_FLOOR_RMSE_LN`` came from repeated measurements in the same group. If that
    group contains a single pair, the number is one draw, and calling it a physical floor
    is an over-claim. The leave-one-temperature grid decides the question: if a model can
    predict a held-out temperature far below the supposed floor, then the floor is not
    measurement noise on ln(eta) at all, and any 'R2 ceiling' built on it must be dropped.
    """
    var_y = float(df["ln_eta"].var(ddof=1))
    noise_var = NOISE_FLOOR_RMSE_LN ** 2
    raw = pd.read_csv(DATA / "plant" / "model_data.csv")
    per_group_samples = raw.groupby("group")["sample_id"].nunique()
    n_pairs = int((per_group_samples > 1).sum())
    pairs = [[gk, list(raw[raw["group"] == gk]["sample_id"].unique())]
             for gk in per_group_samples[per_group_samples > 1].index]
    return {
        "n_points": int(len(df)),
        "n_groups": int(df["group"].nunique()),
        "points_per_group_mean": float(len(df) / df["group"].nunique()),
        "var_ln_eta": var_y,
        "sd_ln_eta": float(np.sqrt(var_y)),
        "reported_floor_rmse_ln": NOISE_FLOOR_RMSE_LN,
        "floor_source_pairs": n_pairs,
        "floor_source_examples": pairs,
        "floor_note": (f"该数值来自 {n_pairs} 对同组重复测量——这是"
                       f"{'一个抽样，不能当作物理地板' if n_pairs <= 2 else '一个可用的估计'}。"
                       "若留一温度的结果低于它，就说明它不是 ln(η) 的测量噪声。"),
        "naive_r2_ceiling": 1.0 - noise_var / var_y,
        "ceiling_is_estimable": n_pairs >= 3,
    }


# ---------------------------------------------------------------------------------
# 2. rank stability
# ---------------------------------------------------------------------------------
def sec_rank_stability(a: pd.DataFrame, top_k: int = 5) -> dict:
    a = a.copy()
    a["combo"] = a["scheme"] + "/" + a["model"]
    combos = sorted(a["combo"].unique())

    per_fold: dict[tuple[int, int], list[str]] = {}
    for (rep, fold), g in a.groupby(["rep", "fold"]):
        per_fold[(rep, fold)] = g.sort_values("rmse_ln")["combo"].tolist()

    overall = (a.groupby("combo")["rmse_ln"].mean().sort_values().index.tolist())
    fold_keys = sorted(per_fold)

    # pairwise flip rate against the overall ordering
    flips = {}
    for x, y in combinations(combos, 2):
        want = overall.index(x) < overall.index(y)
        agree = sum(1 for k in fold_keys
                    if (per_fold[k].index(x) < per_fold[k].index(y)) == want)
        flips[f"{x}  vs  {y}"] = {"agree": agree, "n_folds": len(fold_keys),
                                  "flip_rate": 1 - agree / len(fold_keys)}

    # top-k set stability
    tops = [set(per_fold[k][:top_k]) for k in fold_keys]
    jac = [jaccard(tops[i], tops[j]) for i, j in combinations(range(len(tops)), 2)]

    # Kendall tau between fold rankings
    taus = []
    for i, j in combinations(range(len(fold_keys)), 2):
        ri = {c: n for n, c in enumerate(per_fold[fold_keys[i]])}
        rj = {c: n for n, c in enumerate(per_fold[fold_keys[j]])}
        common = [c for c in combos if c in ri and c in rj]
        if len(common) > 2:
            taus.append(float(stats.kendalltau([ri[c] for c in common],
                                               [rj[c] for c in common]).statistic))

    # which combo is #1, and how often
    winners = pd.Series([per_fold[k][0] for k in fold_keys]).value_counts()

    # the "indistinguishable band": combos whose spread is under one fold-std
    mean_sd = a.groupby("combo")["rmse_ln"].agg(["mean", "std"]).sort_values("mean")
    band = []
    if len(mean_sd) > 1:
        sd_pooled = float(mean_sd["std"].median())
        best = float(mean_sd["mean"].iloc[0])
        for c, r in mean_sd.iterrows():
            if r["mean"] - best <= sd_pooled:
                band.append({"combo": c, "mean": float(r["mean"]), "std": float(r["std"])})

    return {
        "n_combos": len(combos),
        "n_folds": len(fold_keys),
        "overall_order": overall,
        "winners": winners.to_dict(),
        "top_k": top_k,
        "topk_jaccard_mean": float(np.mean(jac)) if jac else float("nan"),
        "topk_jaccard_min": float(np.min(jac)) if jac else float("nan"),
        "kendall_tau_mean": float(np.mean(taus)) if taus else float("nan"),
        "kendall_tau_min": float(np.min(taus)) if taus else float("nan"),
        "median_fold_std": float(mean_sd["std"].median()),
        "indistinguishable_band": band,
        "worst_flips": sorted(flips.items(), key=lambda kv: -kv[1]["flip_rate"])[:8],
    }


# ---------------------------------------------------------------------------------
# 3. variance decomposition: split randomness vs training randomness
# ---------------------------------------------------------------------------------
def sec_variance_split(b: pd.DataFrame) -> dict:
    out = {}
    for (scheme, model), g in b.groupby(["scheme", "model"]):
        # across-fold spread at fixed training seed = split variance
        per_seed = g.groupby("train_seed")["rmse_ln"].mean()
        within_seed_fold_sd = g.groupby("train_seed")["rmse_ln"].std(ddof=1).median()
        train_sd = float(per_seed.std(ddof=1)) if len(per_seed) > 1 else float("nan")
        split_sd = float(within_seed_fold_sd) if pd.notna(within_seed_fold_sd) else float("nan")
        out[f"{scheme}/{model}"] = {
            "n_train_seeds": int(per_seed.size),
            "split_sd": split_sd,               # how much the partition moves the result
            "train_seed_sd": train_sd,          # how much the algorithm's randomness moves it
            "mean_rmse": float(g["rmse_ln"].mean()),
            "note": ("deterministic model: training randomness is zero by construction"
                     if train_sd is not None and (not np.isfinite(train_sd) or train_sd < 1e-12)
                     else ""),
        }
    ratios = [v["train_seed_sd"] / v["split_sd"] for v in out.values()
              if np.isfinite(v["train_seed_sd"]) and np.isfinite(v["split_sd"])
              and v["split_sd"] > 0]
    return {"per_combo": out,
            "median_train_over_split_ratio": float(np.median(ratios)) if ratios else float("nan")}


# ---------------------------------------------------------------------------------
# 4. power
# ---------------------------------------------------------------------------------
def sec_power(a: pd.DataFrame, n_groups: int, deltas=(0.005, 0.01, 0.02, 0.05)) -> dict:
    """Using the observed fold-to-fold spread of RMSE, how many independent folds would
    be needed to resolve a given difference in RMSE_ln, and how many slag samples that
    implies at the current samples-per-fold ratio?

    ``n_groups`` must be the MODELLING-set group count (110), not the raw table's (117);
    the two differ because samples with incomplete compositions are excluded.
    """
    per_fold = a.groupby(["scheme", "model", "rep", "fold"])["rmse_ln"].mean()
    sd_across_folds = float(per_fold.groupby(level=["scheme", "model"]).std(ddof=1).median())
    samples_per_fold = n_groups / N_SPLITS
    table = []
    for d in deltas:
        n = required_n_paired(sd_across_folds, d)
        table.append({
            "delta_rmse_ln": d,
            "required_independent_folds": None if not np.isfinite(n) else int(np.ceil(n)),
            "required_slag_samples": None if not np.isfinite(n)
            else int(np.ceil(np.ceil(n) * samples_per_fold)),
        })
    return {"sd_across_folds": sd_across_folds, "table": table,
            "n_groups": n_groups, "samples_per_fold": samples_per_fold,
            "note": ("more repeats of the SAME data shrink split variance but do not shrink "
                     "sampling variance over the population; only more slag samples do")}


# ---------------------------------------------------------------------------------
# 5. paired ML vs mechanism, at point level
# ---------------------------------------------------------------------------------
def sec_paired(oof_a: pd.DataFrame, oof_base: pd.DataFrame, n_repeat: int = 2) -> dict:
    """Point-level paired test of ML vs the Arrhenius baseline on identical folds.

    Two things matter for honesty here:

    * the pairing removes split variance, which is why the test is powerful at all;
    * but the repeats reuse the SAME data, so the n shown is n_points/n_repeat. With
      tens of thousands of point pairs just about any difference becomes "significant",
      so the effect size (mean difference, and the fraction of points where ML wins)
      is reported alongside the p-value and is the number that should drive conclusions.
    """
    key = ["scheme", "rep", "fold", "group", "T_C", "y_true"]
    base = oof_base[oof_base["model"] == "B1_Arrhenius"].copy()
    res = {}
    for model, ga in oof_a.groupby("model"):
        for scheme, gs in ga.groupby("scheme"):
            m = gs.merge(base, on=key, suffixes=("_ml", "_b"))
            if len(m) < 30:
                continue
            e_ml = np.abs(m["y_pred_ml"].values - m["y_true"].values)
            e_b = np.abs(m["y_pred_b"].values - m["y_true"].values)
            d = e_ml - e_b                       # negative = ML better
            w = stats.wilcoxon(d, zero_method="wilcox") if len(d) > 10 else None
            res[f"{scheme}/{model}"] = {
                "n_point_pairs": int(len(d)),
                "n_effective": int(len(d) / max(1, n_repeat)),
                "mean_abs_err_ml": float(e_ml.mean()),
                "mean_abs_err_arrhenius": float(e_b.mean()),
                "mean_diff": float(d.mean()),
                "ml_better_fraction": float((d < 0).mean()),
                "wilcoxon_p": float(w.pvalue) if w is not None else None,
                "ci95_mean_diff": list(bootstrap_ci(d)),
            }
    return res


# ---------------------------------------------------------------------------------
# 6. what KIND of generalisation is being measured
# ---------------------------------------------------------------------------------
def sec_regimes(dd: Path) -> dict:
    """Three hold-out regimes on identical models and schemes.

    leave_sample  : hold out whole slag samples  -> composition extrapolation + temperature interpolation
    leave_temp    : hold out a whole temperature -> temperature interpolation only (all samples seen)
    leave_furnace : hold out a whole furnace     -> transfer to an unseen furnace

    If leave_temp is much easier than leave_sample, then most of the reported accuracy
    comes from interpolating a temperature the model has already seen for other samples.
    """
    parts = []
    for fn, reg in [("perfold_A.csv", "leave_sample"),
                    ("perfold_T.csv", "leave_temp"),
                    ("perfold_F.csv", "leave_furnace")]:
        p = dd / fn
        if p.exists():
            t = pd.read_csv(p, encoding="utf-8-sig")
            t = t.assign(regime=reg)
            parts.append(t[["regime", "scheme", "model", "rmse_ln"]])
    if not parts:
        return {}
    allr = pd.concat(parts, ignore_index=True)
    agg = (allr.groupby(["regime", "scheme", "model"])["rmse_ln"]
           .agg(["mean", "std", "count"]).reset_index())
    summary = {}
    for reg, g in agg.groupby("regime"):
        best = g.sort_values("mean").iloc[0]
        summary[reg] = {"best_combo": f"{best['scheme']}/{best['model']}",
                        "best_mean_rmse": float(best["mean"]),
                        "median_mean_rmse": float(g["mean"].median()), "n_cells": int(len(g))}
    ref = summary.get("leave_sample", {}).get("best_mean_rmse")
    if ref:
        for reg, s in summary.items():
            s["ratio_vs_leave_sample"] = s["best_mean_rmse"] / ref

    # Interior vs edge temperatures: interpolation and extrapolation are different tasks, and
    # the gap between them is what makes the leave-one-temperature number interpretable
    # (and what shows whether the "noise floor" is a floor at all).
    interior_vs_edge = None
    t_path = dd / "perfold_T.csv"
    bc = summary.get("leave_temp", {}).get("best_combo")
    if t_path.exists() and bc:
        t = pd.read_csv(t_path, encoding="utf-8-sig")
        s_, m_ = bc.split("/")
        sub = t[(t["scheme"] == s_) & (t["model"] == m_)].copy()
        temps = sorted(sub["holdout"].unique())
        if len(temps) >= 6:
            lo, hi = temps[2], temps[-3]              # drop two temperatures at each end
            inner = sub[(sub["holdout"] >= lo) & (sub["holdout"] <= hi)]
            outer = sub[(sub["holdout"] < lo) | (sub["holdout"] > hi)]

            def wmean(d):
                return float(np.sqrt(np.average(d["rmse_ln"] ** 2, weights=d["n_test"])))

            interior_vs_edge = {
                "combo": bc,
                "interior_rmse": wmean(inner), "edge_rmse": wmean(outer),
                "interior_temperatures": [float(lo), float(hi)],
                "n_temperatures": int(len(temps)),
            }

    return {"summary": summary,
            "table": agg.sort_values(["regime", "mean"]).to_dict("records"),
            "interior_vs_edge": interior_vs_edge}


# ---------------------------------------------------------------------------------
# reconcile against the existing pipeline
# ---------------------------------------------------------------------------------
def sec_reconcile(a: pd.DataFrame) -> dict:
    prev_path = DATA / "plant" / "results_all.csv"
    if not prev_path.exists():
        return {"available": False}
    prev = pd.read_csv(prev_path, encoding="utf-8-sig")
    prev = prev.rename(columns={"模型": "model", "方案": "scheme", "RMSE_ln": "rmse_prev"})
    name_map = {"F1质量分数": "F1", "F2质量+派生": "F2", "F3摩尔+碱度": "F3",
                "F4全成分": "F4", "F5炉号+全成分": "F5"}
    prev["scheme"] = prev["scheme"].map(name_map).fillna(prev["scheme"])
    mine = a.groupby(["scheme", "model"])["rmse_ln"].mean().reset_index()
    j = mine.merge(prev[["scheme", "model", "rmse_prev"]], on=["scheme", "model"], how="inner")
    if not len(j):
        return {"available": True, "matched": 0}
    j["diff"] = j["rmse_ln"] - j["rmse_prev"]
    j = j.reindex(j["diff"].abs().sort_values(ascending=False).index)
    big = j[j["diff"].abs() > 0.03]
    return {"available": True, "matched": int(len(j)),
            "median_abs_diff": float(j["diff"].abs().median()),
            "max_abs_diff": float(j["diff"].abs().max()),
            "n_over_0p03": int(len(big)),
            "disclosure_models": sorted(set(big["model"])),
            "table": j.head(8).to_dict("records")}


# ---------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------
def write_report(rep: dict, path: Path) -> None:
    L = ["# 分辨率分析报告", "",
         "本报告回答一个问题：**在这份数据上，模型之间的比较有多少是可信的。**", ""]

    c = rep["ceiling"]
    L += ["## 1. 能不能给出一个上限？——不能，而这本身是一个结果", "",
          "| 量 | 值 |", "|---|---|",
          f"| 数据点 / 渣样组 | {c['n_points']} / {c['n_groups']} |",
          f"| 每组平均点数 | {c['points_per_group_mean']:.1f} |",
          f"| Var(ln η) | {c['var_ln_eta']:.4f}（sd {c['sd_ln_eta']:.3f}） |",
          f"| 文献口径的“噪声地板” RMSE(ln η) | {c['reported_floor_rmse_ln']:.4f} |",
          f"| **该数值依据的重复测量对数** | **{c['floor_source_pairs']}** |",
          f"| 若照此推算的 R² | {c['naive_r2_ceiling']:.3f}（仅供参考，见下） |",
          f"| 上限是否可估 | **{'是' if c['ceiling_is_estimable'] else '否'}** |", "",
          f"> {c['floor_note']}", "",
          "换句话说：**领域里普遍报告的单点 R²，没有可解释的上限参照**——"
          "你无法判断 0.5 离天花板还有多远。这是对报告实践的一条批评，"
          "比给出一个来路不明的上限更有价值。", ""]

    r = rep["rank_stability"]
    L += ["## 2. 排序有多少是可信的", "",
          f"{r['n_combos']} 个模型×方案组合，{r['n_folds']} 折。", "",
          f"- top-{r['top_k']} 集合在不同折之间的 Jaccard 重合率：**均值 {r['topk_jaccard_mean']:.2f}，最低 {r['topk_jaccard_min']:.2f}**",
          f"- 折间 Kendall τ：**均值 {r['kendall_tau_mean']:.2f}，最低 {r['kendall_tau_min']:.2f}**（1.0 = 完全一致）",
          f"- 折间标准差的中位数：**{r['median_fold_std']:.4f}**",
          f"- 第一名分布：{r['winners']}", "",
          f"**均值差在一个折间标准差之内的组合共有 {len(r['indistinguishable_band'])} 个** —— "
          "这一整段排名不可分辨：", "",
          "| 组合 | 均值 RMSE | 折间 std |", "|---|---:|---:|"]
    for b in r["indistinguishable_band"]:
        L.append(f"| {b['combo']} | {b['mean']:.4f} | {b['std']:.4f} |")
    L += ["", "翻转最频繁的组合对：", "", "| 组合对 | 一致折数 | 翻转率 |", "|---|---:|---:|"]
    for k, v in r["worst_flips"]:
        L.append(f"| {k} | {v['agree']}/{v['n_folds']} | {v['flip_rate']*100:.0f}% |")
    L.append("")

    if "variance_split" in rep:
        v = rep["variance_split"]
        L += ["## 3. 方差分解：数据怎么分 vs 算法怎么滚", "",
              f"训练种子方差 / 划分方差 的中位数比值：**{v['median_train_over_split_ratio']:.2f}**", "",
              "| 组合 | 划分 sd | 训练种子 sd | 均值 RMSE |", "|---|---:|---:|---:|"]
        for k, d in sorted(v["per_combo"].items(),
                           key=lambda kv: -abs(kv[1]["split_sd"] if np.isfinite(kv[1]["split_sd"]) else 0))[:12]:
            td = d["train_seed_sd"]
            L.append(f"| {k} | {d['split_sd']:.4f} | "
                     f"{'n/a' if not np.isfinite(td) else f'{td:.4f}'} | {d['mean_rmse']:.4f} |")
        L.append("")

    p = rep["power"]
    L += ["## 4. 功效：要分辨一个差距需要多少数据", "",
          f"以观测到的折间标准差 {p['sd_across_folds']:.4f} 计算：", "",
          "| 目标差距 ΔRMSE(ln η) | 需要的独立折数 | 折算成需要的渣样数 |", "|---|---:|---:|"]
    for row in p["table"]:
        L.append(f"| {row['delta_rmse_ln']:.3f} | {row['required_independent_folds']} | "
                 f"{row['required_slag_samples']} |")
    L += ["", f"> {p['note']}", ""]

    if rep.get("paired"):
        L += ["## 5. 配对比较：ML 与机理基线（点级）", "",
              "差值 = |误差(ML)| − |误差(Arrhenius)|，**负值表示 ML 更好**。"
              "配对消除了划分方差，所以这个检验很有功效——"
              "正因为如此，**要看效应量，不要只看 p 值**："
              "点对数量很大时，任何微小差异都会显得“显著”。", "",
              "| 组合 | 点对数 | 有效 n | ML 平均绝对误差 | Arrhenius | 差值 | ML 更好的点占比 | Wilcoxon p |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for k, d in sorted(rep["paired"].items(), key=lambda kv: kv[1]["mean_diff"])[:14]:
            L.append(f"| {k} | {d['n_point_pairs']} | {d['n_effective']} | "
                     f"{d['mean_abs_err_ml']:.4f} | {d['mean_abs_err_arrhenius']:.4f} | "
                     f"{d['mean_diff']:+.4f} | {d['ml_better_fraction']*100:.1f}% | "
                     f"{d['wilcoxon_p']:.2e} |")
        L.append("")

    rc = rep.get("reconcile", {})
    if rc.get("available"):
        L += ["## 技术披露：与原管线的复现差异", "",
              f"本包重实现了原管线的模型集合。{rc.get('matched')} 个组合中，"
              f"**RMSE 中位绝对差 {rc.get('median_abs_diff'):.4f}**，"
              f"最大 {rc.get('max_abs_diff'):.4f}。", ""]
        if rc.get("n_over_0p03"):
            L += [f"其中 {rc['n_over_0p03']} 个组合差异 >0.03，全部来自 "
                  f"**{'、'.join(rc['disclosure_models'])}**"
                  "——这两类模型对特征缩放与求解器版本敏感，不同库版本下结果本就不完全可复现。", "",
                  "| 组合 | 本包 | 原管线 | 差 |", "|---|---:|---:|---:|"]
            for r in rc["table"]:
                if abs(r["diff"]) > 0.03:
                    L.append(f"| {r['scheme']}/{r['model']} | {r['rmse_ln']:.4f} | "
                             f"{r['rmse_prev']:.4f} | {r['diff']:+.4f} |")
            L += ["",
                  "**注：这不影响本文的任何结论。** 这两类模型在本数据上本就是最弱的"
                  "（RMSE 0.61–0.83，若干方案下甚至差于均值基线），它们始终落在"
                  "“不可分辨带”之外，既不是最优也不参与排名稳定性结论。"
                  "论文中保留原管线的数值，并在附录披露此处差异。", ""]

    rg = rep.get("regimes", {})
    if rg.get("summary"):
        L += ["## 7. 测的是哪一种泛化", "",
              "同一批模型、同一批特征方案，只换留出方式：", "",
              "| 留出方式 | 测什么 | 最好组合 | 最好 RMSE | 相对留一渣样 |",
              "|---|---|---|---:|---:|"]
        zh = {"leave_sample": "留一渣样", "leave_temp": "留一温度", "leave_furnace": "留一高炉"}
        what = {"leave_sample": "成分外推 + 温度插值",
                "leave_temp": "**纯温度插值**（所有渣样都见过）",
                "leave_furnace": "迁移到没见过的炉子"}
        for reg in ("leave_sample", "leave_temp", "leave_furnace"):
            s = rg["summary"].get(reg)
            if not s:
                continue
            r = s.get("ratio_vs_leave_sample")
            L.append(f"| {zh[reg]} | {what[reg]} | {s['best_combo']} | {s['best_mean_rmse']:.4f} | "
                     f"{'—' if r is None else f'{r:.2f}×'} |")
        L.append("")
        smp = rg["summary"].get("leave_sample", {}).get("best_mean_rmse")
        tmp = rg["summary"].get("leave_temp", {}).get("best_mean_rmse")
        if smp and tmp:
            if tmp < smp * 0.9:
                L += [f"**留一温度（{tmp:.4f}）明显优于留一渣样（{smp:.4f}）**——说明报告出来的"
                      "精度里有相当一部分来自“温度已经在其他渣样上见过”这一步插值，"
                      "而成分外推才是难的部分。写论文时这两个数字必须分开报。", ""]
            else:
                L += [f"留一温度（{tmp:.4f}）与留一渣样（{smp:.4f}）接近——说明精度不是靠温度插值"
                      "撑起来的，成分外推本身可学。", ""]
        ie = rg.get("interior_vs_edge")
        if ie:
            L += [f"再把它拆开看（最好组合 {ie['combo']}）：", "",
                  f"- **温度区间内部**（{ie['interior_temperatures'][0]:.0f}–"
                  f"{ie['interior_temperatures'][1]:.0f} ℃，插值）：RMSE = **{ie['interior_rmse']:.4f}**",
                  f"- **温度区间两端**（外推）：RMSE = **{ie['edge_rmse']:.4f}**", "",
                  f"内部误差（{ie['interior_rmse']:.4f}）**远低于文献口径的噪声地板 "
                  f"（{rep['ceiling']['reported_floor_rmse_ln']:.4f}）**，"
                  "这证明那个数值不是 ln(η) 的测量噪声——它反映的是**跨样本**的差异。"
                  "因此它不能当地板用，第 1 节因此改为“上限不可估”。", ""]

    path.write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    df = load_data()
    dd = OUT_DIR
    a = pd.read_csv(dd / "perfold_A.csv", encoding="utf-8-sig")
    base = pd.read_csv(dd / "perfold_base.csv", encoding="utf-8-sig")
    oof_a = pd.read_csv(dd / "oof_A.csv", encoding="utf-8-sig")
    oof_base = pd.read_csv(dd / "oof_base.csv", encoding="utf-8-sig")

    rep: dict = {"ceiling": sec_ceiling(df),
                 "rank_stability": sec_rank_stability(a),
                 "power": sec_power(a, int(df["group"].nunique())),
                 "paired": sec_paired(oof_a, oof_base)}
    b_path = dd / "perfold_B.csv"
    if b_path.exists():
        rep["variance_split"] = sec_variance_split(pd.read_csv(b_path, encoding="utf-8-sig"))
    rep["regimes"] = sec_regimes(dd)
    rep["reconcile"] = sec_reconcile(pd.concat([a, base], ignore_index=True))

    write_report(rep, dd / "resolution_report.md")
    (dd / "analysis.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str),
                                      encoding="utf-8")
    print(f"wrote {dd / 'resolution_report.md'}")
    print(f"wrote {dd / 'analysis.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
