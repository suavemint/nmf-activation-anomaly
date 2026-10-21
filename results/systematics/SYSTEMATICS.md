# Systematic uncertainties (PROTOCOL A4)

Δ = AUROC(NMF k=32) − max(PCA k=32, Mahalanobis). Stat: A1 99.375% paired-bootstrap interval. σ per source = std of Δ over its variations incl. nominal; σ_syst = quadrature sum. Robust = interval ± 2σ_syst still excludes 0.

Checks and organisms:

- S1 (init): nominal recomputation vs committed full-precision scores, max |ΔAUROC| = 4.9e-07
- S2 (train): nominal recomputation vs committed full-precision scores, max |ΔAUROC| = 4.9e-07
- S3 (scale): nominal recomputation vs committed full-precision scores, max |ΔAUROC| = 4.9e-07

| type | layer | features | pooling | Δ | stat interval | σ_S1 init | σ_S2 train | σ_S3 scale | σ_S4 organism | σ_syst | A1 verdict | A4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 12 | mlp_relu | mean | -0.0614 | [-0.0712, -0.0521] | 0.0004 | 0.0019 | 0.0130 | — | 0.0132 | worse | worse: robust to systematics |
| A | 12 | mlp_relu | last | -0.0092 | [-0.0187, -0.0005] | 0.0003 | 0.0004 | 0.0009 | — | 0.0010 | worse | consistent with zero within systematics |
| A | 12 | resid_split | mean | -0.0549 | [-0.0629, -0.0460] | 0.0015 | 0.0036 | 0.0033 | — | 0.0051 | worse | worse: robust to systematics |
| A | 12 | resid_split | last | -0.0098 | [-0.0218, +0.0009] | 0.0002 | 0.0002 | 0.0022 | — | 0.0022 | inconclusive | inconclusive (stat) |
| A | 18 | mlp_relu | mean | -0.0601 | [-0.0702, -0.0492] | 0.0005 | 0.0017 | 0.0095 | — | 0.0097 | worse | worse: robust to systematics |
| A | 18 | mlp_relu | last | -0.0110 | [-0.0146, -0.0068] | 0.0002 | 0.0016 | 0.0079 | — | 0.0081 | worse | consistent with zero within systematics |
| A | 18 | resid_split | mean | -0.0495 | [-0.0588, -0.0389] | 0.0004 | 0.0013 | 0.0034 | — | 0.0037 | worse | worse: robust to systematics |
| A | 18 | resid_split | last | -0.0043 | [-0.0058, -0.0027] | 0.0008 | 0.0004 | 0.0021 | — | 0.0022 | worse | consistent with zero within systematics |
| B | 12 | mlp_relu | mean | -0.0289 | [-0.0524, -0.0043] | 0.0059 | 0.0073 | 0.0073 | — | 0.0119 | worse | consistent with zero within systematics |
| B | 12 | mlp_relu | last | -0.0126 | [-0.0252, +0.0002] | 0.0026 | 0.0034 | 0.0071 | — | 0.0083 | inconclusive | inconclusive (stat) |
| B | 12 | resid_split | mean | -0.0273 | [-0.0527, -0.0002] | 0.0033 | 0.0058 | 0.0053 | — | 0.0085 | worse | consistent with zero within systematics |
| B | 12 | resid_split | last | -0.0081 | [-0.0265, +0.0084] | 0.0033 | 0.0037 | 0.0046 | — | 0.0068 | inconclusive | inconclusive (stat) |
| B | 18 | mlp_relu | mean | -0.0539 | [-0.0805, -0.0259] | 0.0024 | 0.0012 | 0.0285 | — | 0.0286 | worse | consistent with zero within systematics |
| B | 18 | mlp_relu | last | -0.0272 | [-0.0395, -0.0155] | 0.0006 | 0.0037 | 0.0111 | — | 0.0117 | worse | consistent with zero within systematics |
| B | 18 | resid_split | mean | -0.0566 | [-0.0778, -0.0344] | 0.0033 | 0.0058 | 0.0040 | — | 0.0078 | worse | worse: robust to systematics |
| B | 18 | resid_split | last | +0.0042 | [-0.0075, +0.0144] | 0.0023 | 0.0049 | 0.0055 | — | 0.0078 | inconclusive | inconclusive (stat) |

Counts: consistent with zero within systematics: 7, worse: robust to systematics: 5, inconclusive (stat): 4
