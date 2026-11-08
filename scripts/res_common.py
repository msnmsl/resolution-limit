# -*- coding: utf-8 -*-
"""Shared config for the resolution-limits study.

Everything here mirrors analysis/model_pipeline.py so that the numbers produced by
this package can be reconciled against the existing results_all.csv. The differences
are deliberate and are the whole point of this package:

  1. every fit is recorded PER FOLD, not aggregated, so rank stability is computable;
  2. the SPLIT seed and the TRAINING seed are separate knobs, so the total variation
     can be decomposed into "how the data was divided" vs "how the algorithm rolled";
  3. the mechanistic baselines are evaluated on the SAME folds, so ML-vs-mechanism can
     be tested PAIRED (which is far more powerful than comparing two means).

CPU only. The whole grid is minutes, not GPU-hours.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import sys
from pathlib import Path as _P

sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from paths import DATA, OUT  # noqa: E402


# The three boosted-tree libraries are optional: the study works on whatever is
# installed, and the model set is reported so a run can never be silently compared
# against a different set. xgboost/lightgbm/catboost are what the original pipeline
# used; without them the grid simply has fewer members.
try:
    import lightgbm as lgb
except ImportError:                                             # pragma: no cover
    lgb = None
try:
    import xgboost as xgb
except ImportError:                                             # pragma: no cover
    xgb = None
try:
    from catboost import CatBoostRegressor
except ImportError:                                             # pragma: no cover
    CatBoostRegressor = None

# ---------------------------------------------------------------------------------
# paths and constants (match analysis/model_pipeline.py)
# ---------------------------------------------------------------------------------
ANALYSIS_DIR = DATA / "plant"
OUT_DIR = OUT

SEED = 42
N_SPLITS = 5
N_REPEAT = 2
LIQ_ETA = 0.8          # liquid-segment threshold, Pa.s
NOISE_FLOOR_RMSE_LN = 0.3486   # from analysis/results_all.csv (repeat-measurement reproducibility)

SCHEMES = {
    "F1": ["SiO2", "Al2O3", "CaO", "MgO", "invT"],
    "F2": ["SiO2", "Al2O3", "CaO", "MgO", "R", "R4", "MgO_Al2O3", "invT"],
    "F3": ["x_SiO2", "x_Al2O3", "x_CaO", "x_MgO", "Lambda", "invT"],
    "F4": ["SiO2", "Al2O3", "CaO", "MgO", "R", "R4", "MgO_Al2O3", "FeO", "S", "invT"],
}
# F4 (and F5) need FeO/S present; keep that subset separate, exactly as the original does
SCHEMES_NEED_FEO_S = {"F4", "F5"}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(ANALYSIS_DIR / "model_data.csv")
    df = df.dropna(subset=["SiO2", "Al2O3", "CaO", "MgO", "ln_eta"]).copy()
    return df


def scheme_frame(df: pd.DataFrame, scheme: str) -> pd.DataFrame:
    """Rows and columns usable for a given feature scheme."""
    if scheme in SCHEMES_NEED_FEO_S:
        d = df.dropna(subset=["FeO", "S"]).copy()
    else:
        d = df
    return d


class SafePLS(PLSRegression):
    """PLSRegression that clamps ``n_components`` to the number of features.

    The original pipeline asked for 6 components on every scheme, but F1 has only 5
    features. Older scikit-learn silently truncated the request; current versions raise
    ``n_components upper bound is 5``. Clamping reproduces the intended behaviour without
    depending on the installed version, so the same fit runs everywhere.
    """

    def fit(self, X, y, **kwargs):                              # noqa: N803
        n_feat = np.asarray(X).shape[1]
        if self.n_components is not None and self.n_components > n_feat:
            self.n_components = n_feat
        return super().fit(X, y, **kwargs)


def make_models(seed: int) -> dict:
    """Same model family and hyperparameters as analysis/model_pipeline.py.

    Boosted-tree members are added only if their library is importable, so the package
    degrades gracefully; ``MODEL_NAMES`` therefore reflects what actually ran.
    """
    models = {
        "Ridge": Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        "PLS": Pipeline([("sc", StandardScaler()), ("m", SafePLS(n_components=6))]),
        "KNN": Pipeline([("sc", StandardScaler()),
                         ("m", KNeighborsRegressor(n_neighbors=8, weights="distance"))]),
        "SVR": Pipeline([("sc", StandardScaler()),
                         ("m", SVR(C=3.0, epsilon=0.1, gamma="scale"))]),
        "MLP": Pipeline([("sc", StandardScaler()),
                         ("m", MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                            max_iter=800, early_stopping=True,
                                            random_state=seed))]),
        "RF": RandomForestRegressor(n_estimators=400, max_features=0.5,
                                    random_state=seed, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=400, max_features=0.5,
                                          random_state=seed, n_jobs=-1),
        "HistGB": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                                max_depth=6, random_state=seed),
    }
    if xgb is not None:
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=400, learning_rate=0.06, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbosity=0)
    if lgb is not None:
        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.06, num_leaves=31, subsample=0.8,
            colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbosity=-1)
    if CatBoostRegressor is not None:
        models["CatBoost"] = CatBoostRegressor(
            iterations=400, learning_rate=0.06, depth=6, random_seed=seed,
            verbose=0, allow_writing_files=False)
    return models


MODEL_NAMES = list(make_models(0).keys())

# ---------------------------------------------------------------------------------
# folds: GroupKFold over SLAG SAMPLES (the `group` column is date+furnace)
# ---------------------------------------------------------------------------------
def split_folds(groups: np.ndarray, n_splits: int = N_SPLITS,
                split_seed: int = SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    """One permutation of the group order -> one 5-fold partition.

    Changing `split_seed` gives a genuinely different partition of the same data.
    Both the ML grid and the mechanistic baselines are evaluated on the identical
    folds, which is what makes the paired comparison possible.
    """
    uniq = pd.unique(groups)
    rng = np.random.RandomState(split_seed)
    perm = rng.permutation(uniq)
    remap = {g: i for i, g in enumerate(perm)}
    gidx = np.array([remap[g] for g in groups])
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(np.zeros(len(groups)), groups=gidx))


# ---------------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------------
def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "rmse_ln": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae_ln": float(mean_absolute_error(y_true, y_pred)),
        "r2_ln": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }


# ---------------------------------------------------------------------------------
# mechanistic baselines (same formulas as analysis/model_pipeline.py)
# ---------------------------------------------------------------------------------
def mechanistic_predictions(trd: pd.DataFrame, ted: pd.DataFrame,
                            y_tr: np.ndarray) -> dict[str, np.ndarray]:
    """B0 mean / B1 Arrhenius / B2 Urbain / B3 corrected-Urbain.

    Parameters are fitted on the LIQUID segment of the training fold and applied to the
    whole test fold, matching the original pipeline. Standardisation statistics come
    from the training fold only (no leakage).
    """
    liq_tr = trd["eta_Pas"].values < LIQ_ETA
    trl = trd[liq_tr]
    yl = y_tr[liq_tr]

    out: dict[str, np.ndarray] = {}
    out["B0_mean"] = np.full(len(ted), float(y_tr.mean()))

    # B1: ln(eta) = a + b/T  on the liquid segment
    Xb = np.column_stack([np.ones(len(trl)), trl["invT"].values])
    cb, *_ = np.linalg.lstsq(Xb, yl, rcond=None)
    out["B1_Arrhenius"] = cb[0] + cb[1] * ted["invT"].values

    # B2/B3: Urbain-style reduced viscosity, quadratic in optical basicity (+R4 terms)
    lam_m, lam_s = trl["Lambda"].mean(), trl["Lambda"].std() + 1e-12
    r4_m, r4_s = trl["R4"].mean(), trl["R4"].std() + 1e-12

    def design(df: pd.DataFrame):
        z = (df["Lambda"].values - lam_m) / lam_s
        u = (df["R4"].values - r4_m) / r4_s
        X2 = np.column_stack([np.ones(len(df)), z, z ** 2])
        X3 = np.column_stack([X2, u, z * u])
        return X2, X3

    lnT_tr = np.log(trl["T_C"].values + 273.15)
    w_tr = trl["invT"].values - 0.2683
    yp = (yl - lnT_tr + 11.6725) / w_tr
    X2, X3 = design(trl)
    rid2 = Ridge(alpha=1.0).fit(X2, yp)
    rid3 = Ridge(alpha=1.0).fit(X3, yp)

    def urb(df: pd.DataFrame, model, ext: bool) -> np.ndarray:
        _, Xq = design(df)
        B = model.predict(Xq if ext else Xq[:, :3])
        lnT = np.log(df["T_C"].values + 273.15)
        return lnT - 11.6725 + B * (df["invT"].values - 0.2683)

    out["B2_Urbain"] = urb(ted, rid2, ext=False)
    out["B3_corrUrbain"] = urb(ted, rid3, ext=True)
    return out


BASELINE_NAMES = ["B0_mean", "B1_Arrhenius", "B2_Urbain", "B3_corrUrbain"]
