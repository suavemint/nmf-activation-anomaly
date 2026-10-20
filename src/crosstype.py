"""Cross-type selection for unsupervised detectors (PROTOCOL §6, D25).

Per detector family: choose (layer, pooling, variant, k) by AUROC on one type's dev set, then
report that configuration on A_test and B_test. Reads results/metrics_w3.csv (all from LoRA-model
features). Writes results/crosstype.csv.
"""
import pandas as pd

from config import RESULTS

FAMILIES = {"nmf": "nmf_k", "rnmf": "rnmf_k", "pca": "pca_k", "mahalanobis": "mahalanobis", "sae": "sae"}


def main():
    m = pd.read_csv(RESULTS / "metrics_w3.csv")
    m = m[m.model == "lora"]
    key = ["layer", "features", "pooling", "detector"]
    rows = []
    for fam, prefix in FAMILIES.items():
        f = m[m.detector.str.startswith(prefix) if prefix.endswith("_k") else m.detector.eq(prefix)]
        if fam == "sae":
            f = f[f.features == "mlp_relu"]  # variant-independent; listed once (D26)
        for sel_set, sel_type in [("A_dev", "A"), ("B_dev", "B")]:
            dev = f[f.eval_set == sel_set].sort_values(key).sort_values("auroc", ascending=False, kind="stable")
            best = dev.iloc[0]
            cfg = {k: best[k] for k in key}
            r = dict(family=fam, selected_on=sel_type, **cfg, dev_auroc=best.auroc)
            for ts in ["A_test", "B_test"]:
                t = f[(f.eval_set == ts)]
                for k in key:
                    t = t[t[k] == cfg[k]]
                t = t.iloc[0]
                r[f"{ts}_auroc"], r[f"{ts}_lo"], r[f"{ts}_hi"] = t.auroc, t.auroc_lo, t.auroc_hi
                r[f"{ts}_tpr1"] = t["tpr@1%fpr"]
            rows.append(r)
    res = pd.DataFrame(rows)
    res["commit"] = m.commit.iloc[0]
    res.to_csv(RESULTS / "crosstype.csv", index=False, float_format="%.4f")
    print(res.drop(columns="commit").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
