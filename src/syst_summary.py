"""Combine systematic variations (PROTOCOL A4) -> results/systematics/systematics.csv + SYSTEMATICS.md

Per primary cell: sigma_s = sample std of Delta over source s's variations including the nominal;
sigma_syst = sqrt(sum sigma_s^2). Robust (A4 rule): the A1 99.375% interval widened by 2*sigma_syst on
both sides still excludes 0. Organisms failing D16 acceptance are excluded from S4 (reported).
Sanity: every source file's recomputed nominal Delta must equal results/primary_{A,B}.csv.
"""
import json

import numpy as np
import pandas as pd

from config import RESULTS

SYS = RESULTS / "systematics"
KEY = ["type", "layer", "features", "pooling"]
SOURCES = {"S1": "init", "S2": "train", "S3": "scale"}


def main():
    prim = pd.concat([pd.read_csv(RESULTS / f"primary_{t}.csv") for t in ["A", "B"]], ignore_index=True)
    prim = prim.set_index(KEY)
    out, notes = [], []

    frames = {}
    for s, name in SOURCES.items():
        f = SYS / f"{name}.csv"
        if f.exists():
            frames[s] = pd.read_csv(f)
            nom = frames[s][frames[s].variation == "nominal"].set_index(KEY).delta
            dmax = float((nom - prim.delta.reindex(nom.index)).abs().max())
            notes.append(f"{s} ({name}): nominal recomputation vs published Δ, max |diff| = {dmax:.2e}")
            assert dmax < 1e-6, f"{s} nominal does not reproduce primary_*.csv"
        else:
            notes.append(f"{s} ({name}): not run yet")

    org = []
    for f in sorted(SYS.glob("organism_s*.csv")):
        seed = int(f.stem.split("_s")[1])
        chk = json.load(open(SYS / f"backdoor_check_s{seed}.json"))
        acc = chk["accepted"]
        notes.append(f"S4 organism seed {seed}: val triggered {chk['val_triggered']:.1%}, untriggered "
                     f"{chk['val_untriggered']:.1%}, control {chk['val_control_phrase']:.1%} -> "
                     f"{'accepted, included' if acc else 'FAILED acceptance, excluded'}")
        if acc:
            org.append(pd.read_csv(f))
    if org:
        frames["S4"] = pd.concat(org, ignore_index=True)

    for key, p in prim.iterrows():
        r = dict(zip(KEY, key), delta=p.delta, stat_lo=p.amended_lo, stat_hi=p.amended_hi,
                 verdict_A1=p.amended_verdict)
        var2 = 0.0
        for s in ["S1", "S2", "S3", "S4"]:
            if s not in frames:
                r[f"sigma_{s}"] = np.nan
                continue
            g = frames[s].set_index(KEY).loc[[key]]
            vals = g.delta.to_numpy()
            if s == "S4":
                vals = np.r_[p.delta, vals]  # nominal organism + accepted seeds
            sig = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            r[f"sigma_{s}"], r[f"n_{s}"] = sig, len(vals)
            var2 += 0.0 if np.isnan(sig) else sig ** 2
        r["sigma_syst"] = np.sqrt(var2)
        lo, hi = p.amended_lo - 2 * r["sigma_syst"], p.amended_hi + 2 * r["sigma_syst"]
        if p.amended_verdict == "inconclusive":
            r["A4"] = "inconclusive (stat)"
        elif lo > 0 or hi < 0:
            r["A4"] = f"{p.amended_verdict}: robust to systematics"
        else:
            r["A4"] = "consistent with zero within systematics"
        out.append(r)

    res = pd.DataFrame(out)
    res.to_csv(SYS / "systematics.csv", index=False, float_format="%.5f")

    md = ["# Systematic uncertainties (PROTOCOL A4)", "",
          "Δ = AUROC(NMF k=32) − max(PCA k=32, Mahalanobis). Stat: A1 99.375% paired-bootstrap interval. "
          "σ per source = std of Δ over its variations incl. nominal; σ_syst = quadrature sum. "
          "Robust = interval ± 2σ_syst still excludes 0.", "", "Checks and organisms:", ""]
    md += [f"- {n}" for n in notes] + [""]
    md += ["| type | layer | features | pooling | Δ | stat interval | σ_S1 init | σ_S2 train | σ_S3 scale | σ_S4 organism | σ_syst | A1 verdict | A4 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x: "—" if pd.isna(x) else f"{x:.4f}"
    for r in res.itertuples():
        md.append(f"| {r.type} | {r.layer} | {r.features} | {r.pooling} | {r.delta:+.4f} | [{r.stat_lo:+.4f}, {r.stat_hi:+.4f}] | "
                  f"{f(r.sigma_S1)} | {f(r.sigma_S2)} | {f(r.sigma_S3)} | {f(r.sigma_S4)} | {r.sigma_syst:.4f} | {r.verdict_A1} | {r.A4} |")
    md += ["", "Counts: " + ", ".join(f"{k}: {v}" for k, v in res.A4.value_counts().items())]
    (SYS / "SYSTEMATICS.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
