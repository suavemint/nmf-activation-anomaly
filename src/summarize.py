"""Readable tables from a metrics CSV (metrics.py output). Prints markdown and writes
<stem>_tables.md next to the input.

Usage: python summarize.py results/metrics_w1.csv
"""
import sys
from pathlib import Path

import pandas as pd

ORDER_SETS = ["A_test", "A_ctrl", "A_base", "NC_near", "NC_near_lenstrat", "A_vs_NCnear", "A_vs_NCnear_lenstrat", "A_dev",
              "B_test", "NC_far", "B_vs_NCfar", "B_dev"]


def ci(r, m):
    return f"{r[m]:.3f} [{r[m + '_lo']:.3f}, {r[m + '_hi']:.3f}]"


def table(res, metric, pooling, k=32):
    r = res[(res.pooling == pooling) & res.detector.isin(["length", "mahalanobis", f"pca_k{k}", f"nmf_k{k}", f"rnmf_k{k}"])].copy()
    r = r[r[metric].notna()]  # stratified sets report AUROC only
    r["cell"] = r.apply(lambda x: ci(x, metric), axis=1)
    sets = [s for s in ORDER_SETS if s in set(r.eval_set)]
    t = r.pivot_table(index=["layer", "features", "detector"], columns="eval_set", values="cell", aggfunc="first")
    return t[sets]


def main(path):
    res = pd.read_csv(path, keep_default_na=False)
    out = [f"# Tables from `{Path(path).name}` (commit {res.commit.iloc[0]})\n",
           "Rows: layer / nonnegativity variant / detector. Cells: point [95% bootstrap CI]. "
           "`A_base` uses base-model features; everything else uses the backdoored model (D9).\n"]
    for pooling in ["mean", "last"]:
        for metric in ["auroc", "tpr@1%fpr"]:
            out.append(f"\n## {metric}, {pooling} pooling, k=32\n")
            out.append(table(res, metric, pooling).to_markdown())
    for pooling in ["mean", "last"]:
        sweep = res[res.detector.str.startswith(("pca_k", "nmf_k", "rnmf_k")) & (res.pooling == pooling)
                    & res.eval_set.isin(["A_test", "A_ctrl", "B_test"])]
        if len(sweep):
            cols = [f"{f}_k{k}" for f in ["pca", "nmf", "rnmf"] for k in (8, 16, 32, 64) if f"{f}_k{k}" in set(sweep.detector)]
            out.append(f"\n## Rank sweep, AUROC, {pooling} pooling\n")
            out.append(sweep.pivot_table(index=["eval_set", "layer", "features"], columns="detector", values="auroc")[cols]
                       .round(3).to_markdown())
    txt = "\n".join(out)
    Path(path).with_name(Path(path).stem + "_tables.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main(sys.argv[1])
