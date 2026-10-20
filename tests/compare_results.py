"""Compare results CSVs between two runs (reproducibility check, W4).

Usage: python tests/compare_results.py <results_dir_A> <results_dir_B> [file ...]
Ignores the `commit` column. Reports, per file: identical, or the largest absolute numeric
difference per column plus any non-numeric mismatches.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT = ["metrics_w3.csv", "primary_A.csv", "primary_B.csv", "probe.csv", "crosstype.csv",
           "nmf_fit_log.csv", "nmf_convergence_check.csv"]


def compare(a, b):
    A = pd.read_csv(a).drop(columns="commit", errors="ignore")
    B = pd.read_csv(b).drop(columns="commit", errors="ignore")
    if A.shape != B.shape or list(A.columns) != list(B.columns):
        return f"SHAPE/COLUMNS differ: {A.shape} vs {B.shape}"
    if A.equals(B):
        return "identical"
    out = []
    for c in A.columns:
        if pd.api.types.is_numeric_dtype(A[c]) and pd.api.types.is_numeric_dtype(B[c]):
            d = np.nanmax(np.abs(A[c].to_numpy(float) - B[c].to_numpy(float))) if len(A) else 0
            if d > 0:
                out.append(f"{c}: max |diff| {d:.4g}")
        elif not A[c].astype(str).equals(B[c].astype(str)):
            out.append(f"{c}: {int((A[c].astype(str) != B[c].astype(str)).sum())} non-numeric mismatches")
    return "differs -> " + "; ".join(out)


if __name__ == "__main__":
    da, db = Path(sys.argv[1]), Path(sys.argv[2])
    for f in sys.argv[3:] or DEFAULT:
        if (da / f).exists() and (db / f).exists():
            print(f"{f:32s} {compare(da / f, db / f)}")
        else:
            print(f"{f:32s} missing in {'A' if not (da / f).exists() else 'B'}")
