# -*- coding: utf-8 -*-
"""gpvisc 数据库上的 22 模型 x 4 协议消融实验（第三个独立数据集）。
协议定义与武钢对齐，R4 因无时间维度改为「文献来源外推」。
输出 _render/_gpvisc_ladder.csv 与 _gpvisc_ladder.txt
"""
import io
import math
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import OUT, GPVISC_CSV  # noqa: E402


ROOT = str(Path(__file__).resolve().parents[1])
REN = str(OUT)
SEED, NSPLIT, NREP = 42, 5, 2
K0 = 273.15
T_MIN = 1000.0
MIN_GROUP = 5
LN_MAX = 12.0

L = []
def log(s=""):
    L.append(str(s))
    print(s.encode("ascii", "replace").decode(), flush=True)

# ---------------- 数据 ----------------
src = str(GPVISC_CSV)
raw = pd.read_csv(src)
log("raw: %d rows" % len(raw))

OX = ["sio2", "tio2", "al2o3", "feo", "fe2o3", "mno", "na2o", "k2o", "mgo", "cao", "p2o5", "h2o"]
d = raw.dropna(subset=["viscosity", "T", "Name"]).copy()
d = d[np.isfinite(d["viscosity"]) & np.isfinite(d["T"])]
d = d[d["T"] >= T_MIN]
g = d.groupby("Name").size()
d = d[d["Name"].isin(g[g >= MIN_GROUP].index)].copy()
d["lnEta"] = d["viscosity"] * math.log(10.0)
d = d[d["lnEta"] <= LN_MAX].copy()
d = d.reset_index(drop=True)          # 过滤后索引有空洞，后续按位置取数的折生成器依赖连续索引
d["TK"] = d["T"]
d["invT"] = 1000.0 / d["TK"]
d["R"] = d["cao"] / d["sio2"].replace(0, np.nan)
d["R"] = d["R"].fillna(d["R"].median())
d["MgO_Al2O3"] = d["mgo"] / d["al2o3"].replace(0, np.nan)
d["MgO_Al2O3"] = d["MgO_Al2O3"].fillna(0.0)
d["R4"] = d["R"] + 0.8 * (d["R"] - 1.0) * (d["al2o3"] / 20.0)
d["ref_f"] = d["ref"].fillna("UNKNOWN").astype(str)

log("filtered: %d points / %d groups (Name)" % (len(d), d["Name"].nunique()))
log("lnEta range: %.2f .. %.2f   T: %.0f..%.0f K" % (d["lnEta"].min(), d["lnEta"].max(), d["TK"].min(), d["TK"].max()))
log("refs: %d" % d["ref_f"].nunique())

y = d["lnEta"].values
groups = d["Name"].values
refs = d["ref_f"].values

# ---------------- 特征方案 ----------------
SCHEMES = {
    "F1-mol": OX,
    "F2-mol+der": OX + ["R", "R4", "MgO_Al2O3"],
    "F3-mol+R+T": OX + ["R", "MgO_Al2O3", "invT"],
    "F4-all+T": OX + ["R", "R4", "MgO_Al2O3", "invT"],
}
for k, cols in SCHEMES.items():
    X = d[cols].astype(float).values
    d["__X_" + k] = list(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))


def getX(k):
    return np.array([list(v) for v in d["__X_" + k].values], dtype=float)


# ---------------- 四档协议 ----------------
def folds_R1():
    """按点随机切分（同点重复会分散 -> 泄露）"""
    for rep in range(NREP):
        kf = KFold(NSPLIT, shuffle=True, random_state=SEED + rep)
        for tr, te in kf.split(np.zeros(len(y))):
            yield rep, tr, te


def folds_R2():
    """局部泄漏：同一条成分曲线内按温度排序后奇偶交错，同曲线相邻点跨集"""
    dd = d.sort_values(["Name", "T"])
    rank = dd.groupby("Name", sort=False).cumcount().values
    for rep in range(NREP):
        lab_sorted = (rank + rep) % 2
        lab = np.empty(len(d), dtype=int)
        lab[dd.index.values] = lab_sorted
        for k in range(NSPLIT):
            te = np.where(lab == (k % 2))[0]
            if len(te) == 0:
                continue
            yield rep, np.where(lab != (k % 2))[0], te


def folds_R3():
    """整条曲线留出（与武钢口径一致）"""
    for rep in range(NREP):
        rng = np.random.RandomState(SEED + rep)
        perm = rng.permutation(pd.unique(groups))
        remap = {g_: i for i, g_ in enumerate(perm)}
        gidx = np.array([remap[g_] for g_ in groups])
        gk = GroupKFold(n_splits=NSPLIT)
        for tr, te in gk.split(np.zeros(len(y)), groups=gidx):
            yield rep, tr, te


def folds_R4():
    """文献来源外推（gpvisc 无时间维度，用来源代替时间）"""
    gk = GroupKFold(n_splits=NSPLIT)
    for tr, te in gk.split(np.zeros(len(y)), groups=refs):
        yield 0, tr, te


PROTOS = [("R1", folds_R1, "按点随机切分（泄露）"),
          ("R2", folds_R2, "局部泄漏（同曲线相邻点跨集）"),
          ("R3", folds_R3, "整条曲线留出（GroupKFold，与武钢一致）"),
          ("R4", folds_R4, "文献来源外推（gpvisc 无时间维度）")]

# ---------------- 模型 ----------------
def mk_models(seed):
    return {
        "ExtraTrees": ExtraTreesRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
        "RF": RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
        "HistGB": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                                max_depth=6, random_state=seed),
        "XGBoost": xgb.XGBRegressor(n_estimators=300, learning_rate=0.06, max_depth=6,
                                    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                    random_state=seed, n_jobs=-1, verbosity=0),
        "LightGBM": lgb.LGBMRegressor(n_estimators=300, learning_rate=0.06, num_leaves=31,
                                      subsample=0.8, colsample_bytree=0.8, random_state=seed,
                                      n_jobs=-1, verbosity=-1),
        "CatBoost": cb.CatBoostRegressor(iterations=300, learning_rate=0.06, depth=6,
                                         random_seed=seed, verbose=0, allow_writing_files=False),
        "Ridge": Pipeline([("s", StandardScaler()), ("m", Ridge())]),
        "PLS": Pipeline([("s", StandardScaler()),
                         ("m", PLSRegression(n_components=3))]),
        "KNN": Pipeline([("s", StandardScaler()), ("m", KNeighborsRegressor(n_neighbors=8))]),
    }


def fit_arrhenius(tr, te, sc):
    """ln(eta) = a + b*invT  （机理基线，与项目 B1 一致）"""
    Xb = np.column_stack([np.ones(len(tr)), sc[tr, -1]])
    c, *_ = np.linalg.lstsq(Xb, y[tr], rcond=None)
    return c[0] + c[1] * sc[te, -1]


def fit_H2(tr, te, sc):
    """成分 -> (活化能 k, 截距 a) 映射，再用 Arrhenius 外推（H2 型混合）"""
    Xb = np.column_stack([np.ones(len(tr)), sc[tr, -1]])
    cb_, *_ = np.linalg.lstsq(Xb, y[tr], rcond=None)
    resid = y[tr] - (cb_[0] + cb_[1] * sc[tr, -1])
    mu, sd = sc[tr, :-1].mean(0), sc[tr, :-1].std(0)
    sd[sd == 0] = 1.0
    Z = (sc[tr, :-1] - mu) / sd
    W = (sc[te, :-1] - mu) / sd
    B = np.hstack([np.ones((len(Z), 1)), Z])
    c, *_ = np.linalg.lstsq(B, resid, rcond=None)
    return (cb_[0] + cb_[1] * sc[te, -1]) + (np.hstack([np.ones((len(W), 1)), W]) @ c)


COMBOS = [
    ("ExtraTrees", "F3-mol+R+T"), ("RF", "F3-mol+R+T"), ("Ridge", "F3-mol+R+T"),
    ("PLS", "F3-mol+R+T"), ("XGBoost", "F3-mol+R+T"), ("CatBoost", "F3-mol+R+T"),
    ("HistGB", "F3-mol+R+T"), ("KNN", "F3-mol+R+T"), ("LightGBM", "F3-mol+R+T"),
    ("ExtraTrees", "F1-mol"), ("ExtraTrees", "F2-mol+der"), ("ExtraTrees", "F4-all+T"),
    ("Ridge", "F1-mol"), ("PLS", "F1-mol"), ("RF", "F2-mol+der"),
    ("Ridge", "F2-mol+der"), ("PLS", "F2-mol+der"), ("KNN", "F2-mol+der"),
    ("B1-Arrhenius", "机理基线"), ("H2-活化能映射", "创新混合"),
    ("LHW-4成分+T", "外部方法"), ("LHW-6氧+T", "外部方法"),
]

LHW4 = ["sio2", "al2o3", "cao", "mgo", "invT"]
LHW6 = ["sio2", "al2o3", "cao", "mgo", "feo", "fe2o3", "invT"]

rows = []
t0 = time.time()
MOD_CACHE = {}


def get_model(name, rep):
    """按 (模型, rep) 缓存实例，避免每折重建（CatBoost/LightGBM 构造开销大）"""
    key = (name, rep)
    if key not in MOD_CACHE:
        if name.startswith("LHW-"):
            MOD_CACHE[key] = Pipeline([("s", StandardScaler()), ("m", Ridge())])
        else:
            MOD_CACHE[key] = mk_models(SEED + rep)[name]
    return MOD_CACHE[key]


log("")
log("running %d combos x %d protocols ..." % (len(COMBOS), len(PROTOS)))
for pname, gen, pdesc in PROTOS:
    fl = list(gen())
    log("")
    log("--- %s : %s (%d folds) ---" % (pname, pdesc, len(fl)))
    for cname, scheme in COMBOS:
        if cname == "B1-Arrhenius":
            sc = np.hstack([np.ones((len(d), 1)), d["invT"].values.reshape(-1, 1)])
        elif cname == "H2-活化能映射":
            sc = np.hstack([d[["R", "MgO_Al2O3", "sio2", "al2o3", "cao", "mgo", "feo"]].values.astype(float),
                            d["invT"].values.reshape(-1, 1)])
        elif cname == "LHW-4成分+T":
            sc = getX("F1-mol")[:, :0]
            rawcols = d[LHW4].values.astype(float)
            sc = np.hstack([rawcols, d["invT"].values.reshape(-1, 1)])
        elif cname == "LHW-6氧+T":
            sc = np.hstack([d[LHW6].values.astype(float), d["invT"].values.reshape(-1, 1)])
        else:
            sc = getX(scheme)

        rms, r2s, preds = [], [], np.full(len(y), np.nan)
        for rep, tr, te in fl:
            if cname == "B1-Arrhenius":
                p = fit_arrhenius(tr, te, sc)
            elif cname == "H2-活化能映射":
                p = fit_H2(tr, te, sc)
            else:
                mdl = get_model(cname, rep)
                mdl.fit(sc[tr], y[tr])
                p = mdl.predict(sc[te])
                p = np.asarray(p).ravel()
            preds[te] = p
            rms.append(float(np.sqrt(np.mean((y[te] - p) ** 2))))
            mu_t = y[te].mean()
            sst = float(np.sum((y[te] - mu_t) ** 2))
            r2s.append(1 - float(np.sum((y[te] - p) ** 2)) / sst if sst > 0 else np.nan)
        ok = ~np.isnan(preds)
        pe = np.exp(preds[ok])
        ae = np.abs(pe - np.exp(y[ok]))
        rows.append({
            "模型": cname, "方案": scheme, "协议": pname, "来源": "gpvisc",
            "折数": len(fl), "测试点": len(y),
            "rmse_ln": float(np.mean(rms)), "R2": float(np.nanmean(r2s)),
            "rmse_Pa_s": float(np.sqrt(np.mean((np.exp(preds[ok]) - np.exp(y[ok])) ** 2))),
            "mae_Pa_s": float(ae.mean()),
            "rel_med": float(np.median(ae / np.exp(y[ok])) * 100),
            "fold_std": float(np.std(rms)),
        })
        log("   %-18s %-14s RMSE=%.4f  R2=%+.4f" % (cname, scheme, rows[-1]["rmse_ln"], rows[-1]["R2"]))
    log("   [elapsed %.1fs]" % (time.time() - t0))

out = pd.DataFrame(rows)
out.to_csv(os.path.join(REN, "_gpvisc_ladder.csv"), index=False, encoding="utf-8-sig")

log("")
log("=" * 100)
log("gpvisc 数据集：22 模型 x 4 协议 汇总")
log("数据源 %s" % src)
log("筛选：T>=%.0fK, 组点数>=%d, lnEta<=%.0f  ->  %d 点 / %d 成分组 / %d 文献来源"
    % (T_MIN, MIN_GROUP, LN_MAX, len(d), d["Name"].nunique(), d["ref_f"].nunique()))
log("=" * 100)
log("")
log("| 模型 | 方案 | R1 R² | R2 R² | R3 R² | R4 R² | R3 RMSE(ln η) |")
log("|---|---|---|---|---|---|---|")
for cname, scheme in COMBOS:
    s = out[(out["模型"] == cname) & (out["方案"] == scheme)].set_index("协议")
    if len(s) == 0:
        continue
    def g(p, col, f="%.4f"):
        return (f % s.loc[p, col]) if p in s.index else "—"
    log("| %s | %s | %s | %s | **%s** | %s | %s |" % (
        cname, scheme, g("R1", "R2"), g("R2", "R2"), g("R3", "R2"), g("R4", "R2"),
        g("R3", "rmse_ln")))

log("")
log("--- 各协议中位 ---")
log("| 协议 | 中位 RMSE(ln η) | 中位 R² | R²>0.5 的方法数 |")
log("|---|---|---|---|")
for pname, _, desc in PROTOS:
    s = out[out["协议"] == pname]
    log("| %s %s | %.4f | %+.4f | %d |" % (
        pname, desc, s["rmse_ln"].median(), s["R2"].median(), (s["R2"] > 0.5).sum()))

log("")
log("--- R1 -> R3 塌陷幅度（按模型） ---")
piv = out.pivot_table(index=["模型", "方案"], columns="协议", values="R2")
piv = piv.dropna(subset=["R1", "R3"])
piv["drop"] = piv["R1"] - piv["R3"]
log("| 模型 | 方案 | R1 R² | R3 R² | 落差 |")
log("|---|---|---|---|---|")
for (m, s_), r in piv.sort_values("drop", ascending=False).iterrows():
    log("| %s | %s | %.4f | %.4f | %.4f |" % (m, s_, r["R1"], r["R3"], r["drop"]))
log("")
log("塌陷幅度：中位 %.4f，均值 %.4f，最大 %.4f（%s）" % (
    piv["drop"].median(), piv["drop"].mean(), piv["drop"].max(), piv["drop"].idxmax()))

io.open(os.path.join(REN, "_gpvisc_ladder.txt"), "w", encoding="utf-8").write("\n".join(L))
log("")
log("saved -> %s  (%.1fs)" % (os.path.join(REN, "_gpvisc_ladder.csv"), time.time() - t0))
