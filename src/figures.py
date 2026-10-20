"""W4 figures -> results/fig1_primary.png, results/fig2_flag_rates.png.

Fig 1: pre-registered primary comparison (PROTOCOL §4 + A1). Delta AUROC = NMF(k=32) - max(PCA k=32,
       Mahalanobis) per cell with 99.375% paired-bootstrap intervals, both anomaly types.
Fig 2: fraction of each prompt set flagged at the threshold giving 1% FPR on the matching benign
       test set, per detector, with 95% bootstrap CIs. One configuration for every detector:
       layer 18, last token, residual split, k=32 (selected on backdoor dev for all unsupervised
       families, results/crosstype.csv). The dotted line at 0.01 is what a useless detector flags.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import RESULTS

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
VARIANT = {"mlp_relu": "MLP clipped", "resid_split": "resid split [x+, x−]"}

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)


def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, facecolor=SURFACE)
    titles = {"A": "Type A: backdoor trigger", "B": "Type B: jailbreaks (128-token windows)"}
    for ax, t in zip(axes, ["A", "B"]):
        p = pd.read_csv(RESULTS / f"primary_{t}.csv").sort_values(["pooling", "layer", "features"])
        labels = [f"L{r.layer} · {VARIANT[r.features]} · {r.pooling}" for r in p.itertuples()]
        y = np.arange(len(p))
        for i, r in enumerate(p.itertuples()):
            c = BLUE if r.pooling == "mean" else ORANGE
            ax.plot([r.amended_lo, r.amended_hi], [i, i], color=c, lw=2, solid_capstyle="round")
            filled = r.amended_verdict != "inconclusive"
            ax.plot(r.delta, i, "o", ms=8, color=c, mfc=c if filled else SURFACE, mec=c if not filled else SURFACE,
                    mew=2 if not filled else 1.5, zorder=3)
            ax.annotate(r.amended_verdict, (max(r.amended_hi, 0) + 0.004, i), va="center", fontsize=8, color=INK2)
            best = max(r.auroc_pca, r.auroc_maha)
            ax.annotate(f"NMF {r.auroc_nmf:.2f} vs {best:.2f}", (-0.118, i), va="center", fontsize=7.5, color=INK2)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_xlim(-0.12, 0.05)
        ax.set_yticks(y, labels)
        ax.set_title(titles[t], loc="left", color=INK, fontsize=10)
        ax.set_xlabel("Δ AUROC = NMF − best of (PCA, Mahalanobis)")
        style(ax)
    axes[0].invert_yaxis()
    h = [plt.Line2D([], [], marker="o", color=c, lw=2, ms=7) for c in (BLUE, ORANGE)]
    h.append(plt.Line2D([], [], marker="o", color=INK2, mfc=SURFACE, lw=0, ms=7, mew=2))
    fig.legend(h, ["mean pooling", "last-token pooling", "open = inconclusive"], loc="lower center",
               ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0), labelcolor=INK)
    fig.suptitle("Pre-registered primary comparison: NMF is better in 0 of 16 cells", x=0.01, ha="left",
                 color=INK, fontsize=12)
    fig.text(0.01, 0.92, "Points: Δ AUROC (k=32, backdoored-model features). Whiskers: 99.375% paired bootstrap "
             "intervals (Bonferroni over 8 cells per type, PROTOCOL A1). Left labels: AUROC of NMF vs the stronger baseline.",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.91))
    fig.savefig(RESULTS / "fig1_primary.png", dpi=170, facecolor=SURFACE)


def fig2():
    L, feat, pool = 18, "resid_split", "last"
    m = pd.read_csv(RESULTS / "metrics_w3.csv")
    m = m[(m.model == "lora") & (m.layer == L) & (m.features == feat) & (m.pooling == pool)]
    pr = pd.read_csv(RESULTS / "probe.csv")
    pr = pr[(pr.model == "lora") & (pr.layer == L) & (pr.features == feat) & (pr.pooling == pool)]
    panels = [("A_test", "Triggered prompts\n(backdoor, type A)"), ("A_ctrl", "Control phrase\n(\"copper meadow\")"),
              ("NC_near", "oasst1 prompts\n(near negative control)"), ("B_test", "Jailbreaks\n(type B)"),
              ("NC_far", "WikiText paragraphs\n(far negative control)")]
    rows = [("nmf_k32", "NMF (k=32)"), ("rnmf_k32", "robust L2,1 NMF (k=32)"), ("pca_k32", "PCA (k=32)"),
            ("mahalanobis", "Mahalanobis"), ("sae", "pretrained SAE*"), ("length", "prompt length"),
            ("probe_A", "probe trained on A"), ("probe_B", "probe trained on B")]
    fig, axes = plt.subplots(1, len(panels), figsize=(16, 4.8), sharey=True, facecolor=SURFACE)
    for ax, (es, title) in zip(axes, panels):
        for i, (det, _) in enumerate(rows):
            if det.startswith("probe"):
                r = pr[(pr.train_type == det[-1]) & (pr.test_set == es)]
                c = ORANGE
            else:
                r = m[(m.detector == det) & (m.eval_set == es)]
                c = BLUE
            if r.empty:
                ax.annotate("n/a", (0.02, i), va="center", fontsize=7.5, color=INK2)
                continue
            r = r.iloc[0]
            ax.plot([r["tpr@1%fpr_lo"], r["tpr@1%fpr_hi"]], [i, i], color=c, lw=2, solid_capstyle="round")
            ax.plot(r["tpr@1%fpr"], i, "o", ms=8, color=c, mec=SURFACE, mew=1.5, zorder=3)
        ax.axvline(0.01, color=INK2, lw=1, ls=":")
        ax.axhline(5.5, color=GRID, lw=1)
        ax.set_xlim(-0.02, 1.02)
        ax.set_title(title, loc="left", color=INK, fontsize=9.5)
        ax.set_xlabel("fraction flagged")
        style(ax)
    axes[0].set_yticks(range(len(rows)), [r[1] for r in rows])
    axes[0].invert_yaxis()
    h = [plt.Line2D([], [], marker="o", color=c, lw=2, ms=7) for c in (BLUE, ORANGE)]
    h.append(plt.Line2D([], [], color=INK2, lw=1, ls=":"))
    fig.legend(h, ["unsupervised (fit on benign only)", "supervised probe (sees labelled examples)",
                   "1% = flag rate of a useless detector"], loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.0), labelcolor=INK)
    fig.suptitle("Fraction of each set flagged at a 1% false-alarm threshold — layer 18, last token, residual split",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.text(0.01, 0.885, "Threshold set on the matching benign test set (natural-length prompts for the first three "
             "panels, 128-token windows for the last two). Whiskers: 95% bootstrap CIs. "
             "*SAE reads the MLP output (a different site). Length is n/a where all prompts are 128 tokens.",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.87))
    fig.savefig(RESULTS / "fig2_flag_rates.png", dpi=170, facecolor=SURFACE)


if __name__ == "__main__":
    fig1()
    fig2()
    print("wrote", RESULTS / "fig1_primary.png", RESULTS / "fig2_flag_rates.png")
