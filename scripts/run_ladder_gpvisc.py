# -*- coding: utf-8 -*-
"""复刻 _gpvisc_ladder.py 的 ET-F3-mol+R+T，核对四档指标，并导出逐点 OOF

原脚本在 2_归档候选/_code_archive/_gpvisc_ladder.py，本次逐字沿用其
数据清洗、特征、折生成与指标口径（折间平均，不是 pooled）。
"""
import math
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold, KFold
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import OUT, PLANT_CSV, GPVISC_CSV  # noqa: E402


ROOT = str(Path(__file__).resolve().parents[1])
OUT = str(OUT)
os.makedirs(OUT, exist_ok=True)

SEED, NSPLIT, NREP = 42, 5, 2
T_MIN, MIN_GROUP, LN_MAX = 1000.0, 5, 12.0
OX = ["sio2", "tio2", "al2o3", "feo", "fe2o3", "mno", "na2o", "k2o", "mgo", "cao", "p2o5", "h2o"]

src = str(GPVISC_CSV)
raw = pd.read_csv(src)
d = raw.dropna(subset=["viscosity", "T", "Name"]).copy()
d = d[np.isfinite(d["viscosity"]) & np.isfinite(d["T"])]
d = d[d["T"] >= T_MIN]
g = d.groupby("Name").size()
d = d[d["Name"].isin(g[g >= MIN_GROUP].index)].copy()
d["lnEta"] = d["viscosity"] * math.log(10.0)
d = d[d["lnEta"] <= LN_MAX].copy().reset_index(drop=True)
d["invT"] = 1000.0 / d["T"]
d["R"] = d["cao"] / d["sio2"].replace(0, np.nan)
d["R"] = d["R"].fillna(d["R"].median())
d["MgO_Al2O3"] = (d["mgo"] / d["al2o3"].replace(0, np.nan)).fillna(0.0)

y = d["lnEta"].values
groups = d["Name"].values
refs = d["ref"].fillna("UNKNOWN").astype(str).values
COLS = OX + ["R", "MgO_Al2O3", "invT"]
X = np.nan_to_num(d[COLS].astype(float).values, nan=0.0, posinf=0.0, neginf=0.0)
print("filtered: %d points / %d groups (论文 10,739 / 1,367)" % (len(y), d["Name"].nunique()))


def folds_R1():
    for rep in range(NREP):
        for tr, te in KFold(NSPLIT, shuffle=True, random_state=SEED + rep).split(np.zeros(len(y))):
            yield rep, tr, te


def folds_R2():
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
    for rep in range(NREP):
        rng = np.random.RandomState(SEED + rep)
        perm = rng.permutation(pd.unique(groups))
        remap = {g_: i for i, g_ in enumerate(perm)}
        gidx = np.array([remap[g_] for g_ in groups])
        for tr, te in GroupKFold(n_splits=NSPLIT).split(np.zeros(len(y)), groups=gidx):
            yield rep, tr, te


def folds_R4():
    for tr, te in GroupKFold(n_splits=NSPLIT).split(np.zeros(len(y)), groups=refs):
        yield 0, tr, te


PROTOS = [("R1", folds_R1), ("R2", folds_R2), ("R3", folds_R3), ("R4", folds_R4)]
rows = []
for pname, gen in PROTOS:
    fl = list(gen())
    rms, r2s, preds = [], [], np.full(len(y), np.nan)
    for rep, tr, te in fl:
        m = ExtraTreesRegressor(n_estimators=300, random_state=SEED + rep, n_jobs=-1)
        p = m.fit(X[tr], y[tr]).predict(X[te]).ravel()
        preds[te] = p
        rms.append(float(np.sqrt(np.mean((y[te] - p) ** 2))))
        sst = float(np.sum((y[te] - y[te].mean()) ** 2))
        r2s.append(1 - float(np.sum((y[te] - p) ** 2)) / sst if sst > 0 else np.nan)
    rows.append(dict(协议=pname, 折数=len(fl), rmse_ln=float(np.mean(rms)), R2=float(np.nanmean(r2s))))
    np.save(os.path.join(OUT, f"oof_{pname}.npy"), preds)
    print("%s  folds=%2d  RMSE(折间均)=%.4f  R2(折间均)=%+.4f" % (pname, len(fl), rows[-1]["rmse_ln"], rows[-1]["R2"]))

pd.DataFrame(rows).to_csv(os.path.join(OUT, "metrics_rebuilt.csv"), index=False, encoding="utf-8-sig")
np.save(os.path.join(OUT, "y.npy"), y)
print()
print("reference values from the published grid (ET / F3-mol+R+T):")
print("  R1 0.4312 / 0.9880   R2 0.5381 / 0.9812   R3 0.6262 / 0.9742   R4 0.9255 / 0.9073")
