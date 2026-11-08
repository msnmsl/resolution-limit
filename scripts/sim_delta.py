# -*- coding: utf-8 -*-
"""Calibration of the resolution limit Delta*: a coverage simulation.

Question the paper has to answer
--------------------------------
Delta* is quoted as "the smallest difference detectable at 5% significance with
80% power". Two things can be wrong with that label:

1. the normal quantile is used where the fold count gives 4 degrees of freedom;
2. sigma_fold is the spread of ONE candidate's fold errors, while the quantity
   being tested is a difference between two candidates on the SAME folds. If
   the two error vectors are correlated with coefficient rho, the standard
   error of the difference is sigma_fold * sqrt(2(1-rho)) / sqrt(k), which is
   smaller than sigma_fold/sqrt(k) when rho > 0 -- so using sigma_fold makes
   Delta* conservative in that direction. The paper's own data give a paired
   dispersion 1.54x the single-candidate one, i.e. rho < 0, the opposite
   direction, which is why the text calls Delta* a lower bound.

This script simulates the fold-level errors directly, so the truth is known:

    e_Aj ~ N(0, sigma)                  fold error of candidate A
    d_j  = delta + N(0, sigma * r)      A-versus-B difference on fold j
    e_Bj = e_Aj + d_j

with r = sigma_d / sigma_fold. r = sqrt(2) is independent candidates, r = 1.54
is what the slag data show. Four decision rules are compared:

    R1  |mean(d)| > Z   * sd(e_A)/sqrt(k)        normal, single candidate
    R2  |mean(d)| > T   * sd(e_A)/sqrt(k)        t(4),  single candidate
    R3  paired t-test on d, p < 0.05             the conventional test
    R4  |mean(d)| > T   * sd(d)/sqrt(k)          t(4), paired difference

Reported: type-I rate at delta = 0, and power at delta = the Delta* the paper
quotes (computed from the true sigma, not the estimate).

Run:  python sim_delta.py
Out:  sim_delta_results.json, and a printed table.
"""

import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

K = 5
Z_SUM = 1.959963984540054 + 0.8416212335729143        # z_.975 + z_.80
# t quantiles with df = k - 1 = 4
T975_4 = 2.7764451051977987
T80_4 = 0.9409644160109035
T_SUM = T975_4 + T80_4
SIGMA = 1.0            # scale-free: only ratios matter
N_REP = 20000
RATIOS = [1.0, 1.20, 1.38]   # sigma_d / sigma_fold, 1.38 measured on the slag data
SEED = 20261024


def run(r_ratio, delta, n_rep=N_REP, seed=SEED):
    """Return rejection rates for the four rules."""
    rng = np.random.default_rng(seed)
    k = K
    rej = np.zeros(4, dtype=float)

    for _ in range(n_rep):
        e_a = rng.normal(0.0, SIGMA, k)                 # candidate A fold errors
        d = delta + rng.normal(0.0, SIGMA * r_ratio, k)  # paired differences
        sd_a = e_a.std(ddof=1)
        sd_d = d.std(ddof=1)
        m = d.mean()

        # R1 normal, single candidate
        if abs(m) > Z_SUM * sd_a / math.sqrt(k):
            rej[0] += 1
        # R2 t, single candidate
        if abs(m) > T_SUM * sd_a / math.sqrt(k):
            rej[1] += 1
        # R3 paired t-test
        if sd_d > 0:
            tstat = abs(m) / (sd_d / math.sqrt(k))
            if tstat > T975_4:
                rej[2] += 1
        # R4 t, paired difference
        if abs(m) > T_SUM * sd_d / math.sqrt(k):
            rej[3] += 1

    return (rej / n_rep).tolist()


def main():
    rows = []
    for r in RATIOS:
        # Equation (1) builds Delta* from the paired dispersion, so the limit is
        # r-dependent: Delta* = T * (sigma * r) / sqrt(k). Evaluating power at a
        # fixed SIGMA/SQRT(K) instead would compare the rules against a limit
        # that Eq.(1) does not define.
        dstar_t = T_SUM * (SIGMA * r) / math.sqrt(K)
        dstar_z = Z_SUM * (SIGMA * r) / math.sqrt(K)

        null_z = run(r, 0.0)          # type-I at delta = 0
        pow_z = run(r, dstar_z)       # power at the sigma_d-based Delta* (normal version unused in paper)
        pow_t = run(r, dstar_t)       # power at the sigma_d-based Delta*
        rows.append({
            "sigma_d_over_sigma_fold": r,
            "typeI_at_delta_0": dict(zip(["R1_normal", "R2_t", "R3_paired_t", "R4_paired_t"], null_z)),
            "power_at_delta_star_normal": dict(zip(["R1_normal", "R2_t", "R3_paired_t", "R4_paired_t"], pow_z)),
            "power_at_delta_star_t": dict(zip(["R1_normal", "R2_t", "R3_paired_t", "R4_paired_t"], pow_t)),
        })

    out = {
        "k": K, "n_rep": N_REP, "seed": SEED,
        "Z_sum": Z_SUM, "T_sum_df4": T_SUM,
        "delta_star_normal_in_sigma": dstar_z,
        "delta_star_t_in_sigma": dstar_t,
        "rows": rows,
    }
    with open(os.path.join(HERE, "sim_delta_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"k={K}  Z={Z_SUM:.4f}  T(df4)={T_SUM:.4f}  reps={N_REP}")
    print(f"Delta*/sigma:  normal {dstar_z:.4f}   t {dstar_t:.4f}   ratio {T_SUM/Z_SUM:.4f}")
    print()
    hdr = f'{"sd_d/sd_f":>9s} | {"type-I at delta=0":^44s} | {"power at Delta*_normal":^44s}'
    print(hdr)
    print(f'{"":>9s} | {"R1":>10s}{"R2":>10s}{"R3":>12s}{"R4":>12s} | {"R1":>10s}{"R2":>10s}{"R3":>12s}{"R4":>12s}')
    print("-" * len(hdr))
    keys = ["R1_normal", "R2_t", "R3_paired_t", "R4_paired_t"]
    for row in rows:
        a = row["typeI_at_delta_0"]
        b = row["power_at_delta_star_normal"]
        print(f'{row["sigma_d_over_sigma_fold"]:9.3f} | '
              f'{"".join(f"{a[k]:10.3f}" for k in keys[:2])}'
              f'{"".join(f"{a[k]:12.3f}" for k in keys[2:])} | '
              f'{"".join(f"{b[k]:10.3f}" for k in keys[:2])}'
              f'{"".join(f"{b[k]:12.3f}" for k in keys[2:])}')
    print()
    print("R1 normal, single candidate | R2 t(4), single candidate | "
          "R3 paired t-test | R4 t(4), paired difference")


if __name__ == "__main__":
    main()
