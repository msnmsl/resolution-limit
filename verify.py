# -*- coding: utf-8 -*-
"""Check every number the paper quotes against the shipped CSVs.

Exit code is non-zero if any check fails, so this can gate a release.

    python verify.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "outputs"

fails: list[str] = []
checked = 0


def check(label: str, got, want, tol=5e-4) -> None:
    global checked
    checked += 1
    ok = (abs(float(got) - float(want)) <= tol) if isinstance(want, (int, float)) else got == want
    flag = "ok  " if ok else "FAIL"
    print(f"  {flag} {label:56s} {got!r:>10}  (expected {want})")
    if not ok:
        fails.append(label)


def tier_table(path: Path, key_model: str, key_scheme: str):
    d = pd.read_csv(path, encoding="utf-8-sig")
    piv = d.pivot_table(index=["模型", "方案"], columns="协议", values="R2")
    return d, piv.loc[(key_model, key_scheme)]


print("=== plant ladder (22 configurations x 4 tiers) ===")
p = OUT / "g4_plant_ladder.csv"
d = pd.read_csv(p, encoding="utf-8-sig")
check("rows", len(d), 88)
check("configurations", d.groupby(["模型", "方案"]).ngroups, 22)
med = d.pivot_table(index=["模型", "方案"], columns="协议", values="R2").median()
for t, want in [("R1", 0.9773), ("R2", 0.7274), ("R3", 0.0534), ("R4", -1.5302)]:
    check(f"median R2 {t}", round(float(med[t]), 4), want)

et = d[(d["模型"] == "ExtraTrees") & (d["方案"] == "F3摩尔+碱度")].set_index("协议")
for t, want in [("R1", 0.0000), ("R2", 0.2163), ("R3", 0.4804), ("R4", 0.5575)]:
    check(f"ET-F3 RMSE {t}", round(float(et.loc[t, "rmse_ln"]), 4), want)

print("\n=== wide-range ladder ===")
g = pd.read_csv(OUT / "g4_gpvisc_ladder.csv", encoding="utf-8-sig")
check("rows", len(g), 88)
ge = g[(g["模型"] == "ExtraTrees") & (g["方案"] == "F3-mol+R+T")].set_index("协议")
for t, want in [("R1", 0.4312), ("R2", 0.5381), ("R3", 0.6262), ("R4", 0.9255)]:
    check(f"ET-F3 RMSE {t}", round(float(ge.loc[t, "rmse_ln"]), 4), want)

print("\n=== literature compilation ===")
c = pd.read_csv(OUT / "g4_compilation_ladder.csv", encoding="utf-8-sig")
check("rows", len(c), 88)
ce = c[(c["模型"] == "ExtraTrees") & (c["方案"] == "F1-4氧化物质分数")].set_index("协议")
check("ET-F1 R2 T2", round(float(ce.loc["R2", "R2"]), 3), 0.631)
check("ET-F1 R2 T3", round(float(ce.loc["R3", "R2"]), 3), 0.148)

print("\n=== sample-size sweep ===")
s = pd.read_csv(OUT / "sweep_paired.csv")
check("cells", len(s), 14)
check("columns", sorted(s.columns), sorted(["dataset", "n", "band", "n_models", "med_sd_d", "delta"]))

print("\n=== per-fold predictions ===")
for t in ["R1", "R2", "R3", "R4"]:
    a = np.load(OUT / f"oof_{t}.npy")
    check(f"oof_{t} length", len(a), 10739)

print(f"\n{checked - len(fails)}/{checked} checks passed")
if fails:
    print("FAILED:", *fails, sep="\n  ")
    sys.exit(1)
print("every quoted number traces to a shipped file")
