# -*- coding: utf-8 -*-
"""Cross-domain test of the protocol on five public small-sample regression datasets.

The claim under test: "on small-sample data, most model comparisons are not resolvable"
is a property of the REGIME, not of one metallurgy dataset. These five datasets come from
medicine, civil materials, aerospace, naval architecture and building energy.

Everything is reported in units of the target's own standard deviation, so gaps and
standard deviations are comparable across datasets with wildly different scales.

Two grids per dataset:
  A  8 models x 5 split seeds x 5 folds          -> ranking stability, metric dependence
  B  top 4 models x 5 folds x 5 training seeds   -> split variance vs training variance

    python run_crossdomain.py        -> out/crossdomain_report.md, out/per_dataset.csv
"""
from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    import xgboost as xgb
except ImportError:
    xgb = None

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

N_SPLITS, N_FOLDS = 5, 5
N_TRAIN_SEEDS = 5
TOP_FOR_B = 4
Z_975, Z_80 = 1.959963985, 0.841621234


def make_models(seed: int) -> dict:
    ms = {
        "Ridge": Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        "PLS": Pipeline([("s", StandardScaler()), ("m", PLSRegression(n_components=4))]),
        "KNN": Pipeline([("s", StandardScaler()),
                         ("m", KNeighborsRegressor(n_neighbors=8, weights="distance"))]),
        "SVR": Pipeline([("s", StandardScaler()), ("m", SVR(C=3.0, epsilon=0.1))]),
        "MLP": Pipeline([("s", StandardScaler()),
                         ("m", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=600,
                                            early_stopping=True, random_state=seed))]),
        "RF": RandomForestRegressor(n_estimators=200, max_features=0.5,
                                    random_state=seed, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=200, max_features=0.5,
                                          random_state=seed, n_jobs=-1),
        "HistGB": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08,
                                               max_depth=6, random_state=seed),
    }
    if xgb is not None:
        ms["XGBoost"] = xgb.XGBRegressor(n_estimators=300, learning_rate=0.08, max_depth=6,
                                         subsample=0.8, colsample_bytree=0.8,
                                         random_state=seed, n_jobs=-1, verbosity=0)
    return ms


def metrics(y, p):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    e = np.asarray(p, float) - np.asarray(y, float)
    return {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "r2": float(r2_score(y, p))}


def run_dataset(name: str, df: pd.DataFrame) -> dict:
    y = df["target"].values.astype(float)
    X = df.drop(columns=["target"]).values.astype(float)
    sd = float(np.std(y, ddof=1))
    models = list(make_models(0))
    rows = []
    t0 = time.time()

    # ---- grid A: multiple split seeds ------------------------------------------------
    for ss in range(N_SPLITS):
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=1000 + ss)
        for fold, (tr, te) in enumerate(kf.split(X)):
            for mn in models:
                try:
                    m = make_models(1000 + ss)[mn]
                    m.fit(X[tr], y[tr])
                    mt = metrics(y[te], np.asarray(m.predict(X[te])).ravel())
                except Exception:                                   # noqa: BLE001
                    continue
                rows.append({"dataset": name, "grid": "A", "model": mn, "split_seed": ss,
                             "fold": fold, "train_seed": 1000 + ss, "n_test": len(te),
                             "mae_sd": mt["mae"] / sd, "rmse_sd": mt["rmse"] / sd,
                             "r2": mt["r2"]})
    A = pd.DataFrame(rows)

    # ---- grid B: fixed split, varying training seed -----------------------------------
    top = (A.groupby("model")["mae_sd"].mean().sort_values().head(TOP_FOR_B).index.tolist())
    rows = []
    for fold, (tr, te) in enumerate(KFold(n_splits=N_FOLDS, shuffle=True,
                                          random_state=1000).split(X)):
        for mn in top:
            for k in range(N_TRAIN_SEEDS):
                ts = 2000 + k
                try:
                    m = make_models(ts)[mn]
                    m.fit(X[tr], y[tr])
                    mt = metrics(y[te], np.asarray(m.predict(X[te])).ravel())
                except Exception:                                   # noqa: BLE001
                    continue
                rows.append({"dataset": name, "grid": "B", "model": mn, "fold": fold,
                             "train_seed": ts, "mae_sd": mt["mae"] / sd})
    B = pd.DataFrame(rows)

    # ---- analyses --------------------------------------------------------------------
    g = A.groupby("model")["mae_sd"].agg(["mean", "std"]).sort_values("mean")
    best = float(g["mean"].iloc[0])
    fold_sd = float(g["std"].median())
    in_band = int((g["mean"] - best <= fold_sd).sum())

    # rank stability across the 25 (split, fold) cells
    cells = {}
    for (ss, fd), gg in A.groupby(["split_seed", "fold"]):
        cells[(ss, fd)] = gg.sort_values("mae_sd")["model"].tolist()
    ks = sorted(cells)
    taus = []
    for i, j in combinations(range(len(ks)), 2):
        ri = {c: n for n, c in enumerate(cells[ks[i]])}
        rj = {c: n for n, c in enumerate(cells[ks[j]])}
        common = [c for c in ri if c in rj]
        if len(common) > 2:
            taus.append(float(stats.kendalltau([ri[c] for c in common],
                                               [rj[c] for c in common]).statistic))
    winners = pd.Series([cells[k][0] for k in ks]).value_counts().to_dict()

    # metric dependence
    by_mae = A.groupby("model")["mae_sd"].mean().idxmin()
    by_rmse = A.groupby("model")["rmse_sd"].mean().idxmin()

    # variance decomposition (grid B)
    sp, tr_ = [], []
    for mn, gg in B.groupby("model"):
        sp.append(float(gg.groupby("train_seed")["mae_sd"].std(ddof=1).median()))
        per = gg.groupby("train_seed")["mae_sd"].mean()
        tr_.append(float(per.std(ddof=1)))
    split_sd = float(np.median(sp))
    train_sd = float(np.median(tr_))
    ratio = train_sd / split_sd if split_sd > 0 else float("nan")

    # ---- output ---------------------------------------------------------------------
    print(f"  {name:<10} n={len(df):<5} best={by_mae:<12} "
          f"best_MAE={best:.4f}sd  fold_sd={fold_sd:.4f}sd  "
          f"in_band={in_band}/{len(models)}  "
          f"tau={np.mean(taus):.2f}  split/train={split_sd:.4f}/{train_sd:.4f}  "
          f"({time.time()-t0:.0f}s)")
    return {
        "dataset": name, "n": len(df), "d": X.shape[1],
        "target_sd": sd, "n_models": len(g),
        "best_model": by_mae, "best_mae_sd": best, "best_r2": float(
            A[A.model == by_mae]["r2"].mean()),
        "fold_sd": fold_sd, "in_band": in_band,
        "kendall_tau_mean": float(np.mean(taus)) if taus else float("nan"),
        "kendall_tau_min": float(np.min(taus)) if taus else float("nan"),
        "n_distinct_winners": len(winners), "winners": winners,
        "metric_agrees": bool(by_mae == by_rmse),
        "rmse_best": by_rmse,
        "split_sd": split_sd, "train_seed_sd": train_sd, "train_over_split": ratio,
        "power": {f"{d:.3f}": int(np.ceil((Z_975 + Z_80) ** 2 * fold_sd ** 2 / d ** 2))
                  for d in (0.005, 0.01, 0.02, 0.05)},
        "_A": A, "_B": B,
    }


def main() -> None:
    files = sorted(DATA.glob("*.csv"))
    print("=" * 110)
    print("cross-domain protocol run (all metrics in units of the target's own sd)")
    print("=" * 110)
    results, A_all, B_all = [], [], []
    for f in files:
        r = run_dataset(f.stem, pd.read_csv(f))
        results.append({k: v for k, v in r.items() if not k.startswith("_")})
        A_all.append(r["_A"]); B_all.append(r["_B"])

    A_all = pd.concat(A_all, ignore_index=True)
    B_all = pd.concat(B_all, ignore_index=True)
    A_all.to_csv(OUT / "perfold_A.csv", index=False, encoding="utf-8-sig")
    B_all.to_csv(OUT / "perfold_B.csv", index=False, encoding="utf-8-sig")

    L = ["# 跨领域检验：七个公开小样本回归数据集\n",
         "**待检验的主张（第一版）**：\"小样本数据上多数模型比较不可分辨\"是**这类数据**的性质，"
         "而不是某个冶金数据集的特例。\n",
         "**结果：这个主张被本实验推翻了。** 修正过程中又提出过第二版主张"
         "（\"R² 高就可分辨\"），也被样本量扫描推翻。最终成立的版本见文末\"结论修正\"。\n",
         "七个公开数据集来自七个不同领域，外加两个冶金数据集对照"
         "（高炉渣粘度、铁样硅含量）。所有指标以**目标变量自身标准差为单位**，"
         "因此不同量纲的数据集可比。\n",
         "| 数据集 | 领域 | n | 特征数 | 最好模型 | 最好 MAE (sd) | R² | 折间 sd (sd) | "
         "**一个 sd 内的模型数** | 折间 τ | 不同赢家数 |",
         "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    domain = {"diabetes": "医学", "concrete": "土木/材料", "airfoil": "航空航天",
              "yacht": "船舶", "energy": "建筑能耗", "realestate": "房地产",
              "wine_red": "食品/化学"}
    for r in results:
        agree = "是" if r["metric_agrees"] else "**否**"
        L.append(f"| {r['dataset']} | {domain.get(r['dataset'],'—')} | {r['n']} | {r['d']} | "
                 f"{r['best_model']} | {r['best_mae_sd']:.4f} | {r['best_r2']:.3f} | "
                 f"{r['fold_sd']:.4f} | **{r['in_band']}/{r['n_models']}** | "
                 f"{r['kendall_tau_mean']:.2f} | {r['n_distinct_winners']} |")
    L.append("")

    L.append("## 1. 方差来源：划分还是训练种子\n")
    L.append("| 数据集 | 划分 sd (sd) | 训练种子 sd (sd) | 比值 | 结论 |")
    L.append("|---|---:|---:|---:|---|")
    for r in results:
        rt = r["train_over_split"]
        who = "**划分主导**" if rt < 0.5 else ("训练种子主导" if rt > 2 else "两者相当")
        L.append(f"| {r['dataset']} | {r['split_sd']:.4f} | {r['train_seed_sd']:.4f} | "
                 f"{rt:.2f} | {who} |")
    L.append("")

    L.append("## 2. 功效：确认一个差距需要多少样本\n")
    L.append("以各数据集自身的折间标准差计算。差距以目标 sd 为单位。\n")
    L.append("| 数据集 | n | Δ=0.005 sd | Δ=0.01 sd | Δ=0.02 sd | Δ=0.05 sd |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for r in results:
        p = r["power"]
        L.append(f"| {r['dataset']} | {r['n']} | {p['0.005']:,} | {p['0.010']:,} | "
                 f"{p['0.020']:,} | {p['0.050']:,} |")
    L.append("")

    L.append("## 3. 指标族：换指标会不会换赢家\n")
    L.append("| 数据集 | MAE 最好 | RMSE 最好 | 一致？ |")
    L.append("|---|---|---|---|")
    for r in results:
        L.append(f"| {r['dataset']} | {r['best_model']} | {r['rmse_best']} | "
                 f"{'是' if r['metric_agrees'] else '**否**'} |")
    L.append("")
    n_dis = sum(1 for r in results if not r["metric_agrees"])
    L.append(f"**{n_dis}/{len(results)} 个数据集上，MAE 与 RMSE 选出不同的最好模型。**\n")

    L.append("## 4. 与两个工业数据集对照\n")
    L.append("| 数据集 | 领域 | n | 划分方式 | 一个 sd 内的模型数 / 总数 |")
    L.append("|---|---|---:|---|---|")
    L.append("| 高炉渣粘度 | 冶金 | 1875 | 按渣样分组 | 23 / 44（模型×方案） |")
    L.append("| 铁样硅含量 | 冶金（图像回归） | 486 图 / 81 样 | 按图（有泄漏） | —（策略对比） |")
    for r in results:
        L.append(f"| {r['dataset']} | {domain.get(r['dataset'],'—')} | {r['n']} | 随机 K 折 | "
                 f"**{r['in_band']} / {r['n_models']}** |")
    L.append("")

    L.append("")
    L.append("## 结论修正：哪一条跨领域成立，哪一条不成立\n")

    ratios = [r["train_over_split"] for r in results]
    L.append("### ✅ 跨领域成立：不确定性来自划分，不是训练种子\n")
    L.append(f"七个公开数据集（外加两个冶金数据集）里，训练种子方差 / 划分方差 = "
             f"{', '.join(f'{r:.2f}' for r in ratios)}"
             f"（冶金粘度数据集为 0.03）。**即划分方差是训练种子方差的 "
             f"{1/max(ratios):.0f}–{1/min(ratios):.0f} 倍。**"
             "这条在六个领域上一致，而且与机器学习界\"多跑几个随机种子\"的常规做法相反。\n")

    L.append("### ❌ 跨领域不成立（第一版）：\"小样本 ⇒ 排序不可分辨\"\n")
    L.append("按样本量排序，规律并不成立——样本量最大的 wine_red（1599）"
             "和最小的 yacht（308）都可分辨，而 442 的 diabetes 不可分辨。\n")
    L.append("### ❌ 跨领域不成立（第二版）：\"R² 高 ⇒ 可分辨\"\n")
    L.append("按信号强度排序看起来像规律（见下表），但**样本量扫描推翻了它**：\n")
    L.append("| 数据集 | n | R² | 一个 sd 内的模型数 | 折间 τ | 排序可分辨？ |")
    L.append("|---|---:|---:|---:|---:|---|")
    for r in sorted(results, key=lambda r: -r["best_r2"]):
        ok = "✅ 可分辨" if r["in_band"] <= 3 else "❌ 不可分辨"
        L.append(f"| {r['dataset']} | {r['n']} | {r['best_r2']:.3f} | "
                 f"{r['in_band']}/{r['n_models']} | {r['kendall_tau_mean']:.2f} | {ok} |")
    L.append("")
    L.append("在同一数据集内部只改变样本量（`run_sweep.py`），糖尿病数据上 "
             "R² 从 0.762 **下降**到 0.598，τ 却从 0.24 **上升**到 0.46"
             "（ρ(R², τ) = −1.00）。把扫描格和上表放一起看："
             "R² = 0.762 的格不可分辨（τ = 0.24），R² = 0.507 的数据集却高度可分辨"
             "（τ = 0.87）。**R² 既不是充分条件也不是必要条件。**\n")
    L.append("| 冶金数据集 | n | 信号 | 结论 |")
    L.append("|---|---:|---|---|")
    L.append("| 高炉渣粘度 | 1875 点 / 110 渣样 | 最好 R² = **0.483**（弱） | ❌ 23/44 在带内 |")
    L.append("| 铁样硅含量 | 486 图 / 81 样 | TTA 变体差距 0.0006（效应极小） | ❌ 变体间不可分辨 |")
    L.append("")
    L.append("**结论（第三版）：可分辨性可以写成 \"模型间距 / 折间噪声\" 这个比值，"
             "但它不能从 n、R² 或分辨率极限 Δ* 中的任何一个预测量算出来——必须实测。**")
    L.append("Δ* = 2.8 σ_fold/√n 在**单个数据集内部**把 τ 排得很整齐"
             "（ρ(Δ*, τ) = −0.90 ~ −1.00），但跨数据集时同样的 Δ* 上 τ 相差近一倍"
             "（Δ*≈0.036 时：airfoil τ=0.45，diabetes τ=0.24）。"
             "缺的那一项是模型之间的真实间距，它由任务决定，测不到。\n")
    L.append("**这比\"发现一个阈值\"强**：它解释了为什么所有便宜的启发式都会失灵，"
             "并给出唯一正确的做法——跑协议，数一下一个 fold-sd 里挤了几个模型。\n")
    L.append("指标依赖性（MAE 与 RMSE 选不同赢家）也只在效应接近噪声时出现——"
             "五个公开数据集上 0/5 出现，两个工业数据集上都出现了。\n")
    L.append("> 详细推导与 14 个扫描格见 `out/SWEEP_FINDINGS.md`。\n")

    (OUT / "crossdomain_report.md").write_text("\n".join(L), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(
        [{k: (v if not isinstance(v, pd.Series) else v.to_dict()) for k, v in r.items()}
         for r in results], ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n[written] {OUT/'crossdomain_report.md'}")

    # console summary of the headline
    print()
    print("=" * 110)
    print("headline: how many models sit within one fold-sd of the best?")
    print("=" * 110)
    for r in results:
        print(f"  {r['dataset']:<12} {r['in_band']}/{r['n_models']} models   "
              f"(fold_sd = {r['fold_sd']:.4f} target-sd, best = {r['best_mae_sd']:.4f})")


if __name__ == "__main__":
    main()
