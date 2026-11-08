# -*- coding: utf-8 -*-
"""Sensitivity of the resolution limit to cross-fold correlation.

Eq. (1) divides the paired dispersion by sqrt(k), which is the standard error of
a mean of k *independent* fold-level differences. Cross-validation folds share
training data, so the fold errors are correlated, and Bates, Hastie and
Tibshirani (2024) show that this makes the naive interval optimistic. This
script quantifies how much.

Model
-----
Fold-level differences are equicorrelated with coefficient rho:

    d_j = delta + sigma_d * ( sqrt(rho) * z0 + sqrt(1 - rho) * z_j )

The marginal SD of d_j is sigma_d for every rho, so a sample SD still estimates
sigma_d; what changes is that the variance of the mean is

    Var(mean d) = sigma_d^2 * (1 + (k-1) rho) / k

while the analyst divides by sqrt(k) regardless. The standard error is therefore
understated by sqrt(1 + (k-1) rho), and the paired test is anti-conservative by
exactly that factor.

Reported: variance inflation, realised type-I rate at delta = 0, and realised
power at the limit the analyst would compute, Delta* = T * sd(d) / sqrt(k).

Run:  python sim_fold_corr.py
Out:  sim_fold_corr.json
"""

import json
import math
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
K = 5
T_SUM = stats.t.ppf(0.975, K - 1) + stats.t.ppf(0.80, K - 1)
T975 = stats.t.ppf(0.975, K - 1)
N_REP = 20000
RHOS = [0.0, 0.1, 0.3, 0.6, 0.9]
SEED = 20261025


def sim(rho, delta_scale, n_rep=N_REP, seed=SEED):
    """Rejection rate of the paired test; delta_scale multiplies T*sd/sqrt(k)."""
    rng = np.random.default_rng(seed)
    rej = 0
    for _ in range(n_rep):
        z0, z = rng.normal(), rng.normal(size=K)
        d = math.sqrt(rho) * z0 + math.sqrt(1.0 - rho) * z          # SD = 1
        sd = d.std(ddof=1)
        delta = delta_scale * T_SUM * sd / math.sqrt(K)             # analyst's limit
        dd = d + delta
        t = dd.mean() / (dd.std(ddof=1) / math.sqrt(K))
        if abs(t) > T975:
            rej += 1
    return rej / n_rep


def main():
    rows = []
    print(f'{"rho":>5s} {"inflation":>10s} {"type-I":>8s} {"power@D*":>9s}')
    for rho in RHOS:
        infl = math.sqrt(1.0 + (K - 1) * rho)
        t1 = sim(rho, 0.0)
        pw = sim(rho, 1.0)
        rows.append({"rho": rho, "inflation": infl, "type_I": t1, "power_at_limit": pw})
        print(f"{rho:5.2f} {infl:10.2f} {t1:8.3f} {pw:9.3f}")

    out = {"k": K, "n_rep": N_REP, "seed": SEED, "T_sum": T_SUM, "rows": rows}
    with open(os.path.join(HERE, "sim_fold_corr.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nwrote sim_fold_corr.json")


if __name__ == "__main__":
    main()
