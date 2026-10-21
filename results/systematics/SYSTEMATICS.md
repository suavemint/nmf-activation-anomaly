# Systematic uncertainties (PROTOCOL A4)

Δ = AUROC(NMF k=32) − max(PCA k=32, Mahalanobis). Stat: A1 99.375% paired-bootstrap interval. σ per source = std of Δ over its variations incl. nominal; σ_syst = quadrature sum. Robust = interval ± 2σ_syst still excludes 0.

Checks and organisms:

- S1 (init): nominal recomputation vs committed full-precision scores, max |ΔAUROC| = 4.9e-07
- S2 (train): nominal recomputation vs committed full-precision scores, max |ΔAUROC| = 4.9e-07
- S3 (scale): nominal recomputation vs committed full-precision scores, max |ΔAUROC| = 4.9e-07
- S4 organism seed 2: val triggered 89.7%, untriggered 0.7%, control 1.3% -> accepted, included
- S4 organism seed 3: val triggered 87.0%, untriggered 0.3%, control 0.3% -> accepted, included
- S4 organism seed 4: val triggered 90.0%, untriggered 0.0%, control 0.0% -> accepted, included

| type | layer | features | pooling | Δ | stat interval | σ_S1 init | σ_S2 train | σ_S3 scale | σ_S4 organism | σ_syst | A1 verdict | A4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 12 | mlp_relu | mean | -0.0614 | [-0.0712, -0.0521] | 0.0004 | 0.0019 | 0.0130 | 0.0053 | 0.0142 | worse | worse: robust to systematics |
| A | 12 | mlp_relu | last | -0.0092 | [-0.0187, -0.0005] | 0.0003 | 0.0004 | 0.0009 | 0.0017 | 0.0020 | worse | consistent with zero within systematics |
| A | 12 | resid_split | mean | -0.0549 | [-0.0629, -0.0460] | 0.0015 | 0.0036 | 0.0033 | 0.0019 | 0.0055 | worse | worse: robust to systematics |
| A | 12 | resid_split | last | -0.0098 | [-0.0218, +0.0009] | 0.0002 | 0.0002 | 0.0022 | 0.0006 | 0.0023 | inconclusive | inconclusive (stat) |
| A | 18 | mlp_relu | mean | -0.0601 | [-0.0702, -0.0492] | 0.0005 | 0.0017 | 0.0095 | 0.0045 | 0.0106 | worse | worse: robust to systematics |
| A | 18 | mlp_relu | last | -0.0110 | [-0.0146, -0.0068] | 0.0002 | 0.0016 | 0.0079 | 0.0042 | 0.0091 | worse | consistent with zero within systematics |
| A | 18 | resid_split | mean | -0.0495 | [-0.0588, -0.0389] | 0.0004 | 0.0013 | 0.0034 | 0.0099 | 0.0106 | worse | worse: robust to systematics |
| A | 18 | resid_split | last | -0.0043 | [-0.0058, -0.0027] | 0.0008 | 0.0004 | 0.0021 | 0.0081 | 0.0084 | worse | consistent with zero within systematics |
| B | 12 | mlp_relu | mean | -0.0289 | [-0.0524, -0.0043] | 0.0059 | 0.0073 | 0.0073 | 0.0082 | 0.0144 | worse | consistent with zero within systematics |
| B | 12 | mlp_relu | last | -0.0126 | [-0.0252, +0.0002] | 0.0026 | 0.0034 | 0.0071 | 0.0150 | 0.0171 | inconclusive | inconclusive (stat) |
| B | 12 | resid_split | mean | -0.0273 | [-0.0527, -0.0002] | 0.0033 | 0.0058 | 0.0053 | 0.0031 | 0.0091 | worse | consistent with zero within systematics |
| B | 12 | resid_split | last | -0.0081 | [-0.0265, +0.0084] | 0.0033 | 0.0037 | 0.0046 | 0.0175 | 0.0188 | inconclusive | inconclusive (stat) |
| B | 18 | mlp_relu | mean | -0.0539 | [-0.0805, -0.0259] | 0.0024 | 0.0012 | 0.0285 | 0.0033 | 0.0288 | worse | consistent with zero within systematics |
| B | 18 | mlp_relu | last | -0.0272 | [-0.0395, -0.0155] | 0.0006 | 0.0037 | 0.0111 | 0.0079 | 0.0141 | worse | consistent with zero within systematics |
| B | 18 | resid_split | mean | -0.0566 | [-0.0778, -0.0344] | 0.0033 | 0.0058 | 0.0040 | 0.0018 | 0.0080 | worse | worse: robust to systematics |
| B | 18 | resid_split | last | +0.0042 | [-0.0075, +0.0144] | 0.0023 | 0.0049 | 0.0055 | 0.0063 | 0.0100 | inconclusive | inconclusive (stat) |

Counts: consistent with zero within systematics: 7, worse: robust to systematics: 5, inconclusive (stat): 4

## Detection level across organisms (descriptive; added after S4 results, D31)

AUROC of the best baseline (max of PCA k=32, Mahalanobis) / NMF k=32, per accepted organism. Not part of the A4 decision rule.

### Type A

|                             |   ('auroc_nmf', 'seed 1 (published)') |   ('auroc_nmf', 'seed 2') |   ('auroc_nmf', 'seed 3') |   ('auroc_nmf', 'seed 4') |   ('best_baseline', 'seed 1 (published)') |   ('best_baseline', 'seed 2') |   ('best_baseline', 'seed 3') |   ('best_baseline', 'seed 4') |
|:----------------------------|--------------------------------------:|--------------------------:|--------------------------:|--------------------------:|------------------------------------------:|------------------------------:|------------------------------:|------------------------------:|
| (12, 'mlp_relu', 'last')    |                                 0.476 |                     0.478 |                     0.488 |                     0.492 |                                     0.485 |                         0.486 |                         0.496 |                         0.497 |
| (12, 'mlp_relu', 'mean')    |                                 0.473 |                     0.502 |                     0.502 |                     0.502 |                                     0.534 |                         0.551 |                         0.554 |                         0.554 |
| (12, 'resid_split', 'last') |                                 0.482 |                     0.486 |                     0.493 |                     0.497 |                                     0.492 |                         0.496 |                         0.502 |                         0.506 |
| (12, 'resid_split', 'mean') |                                 0.48  |                     0.511 |                     0.51  |                     0.505 |                                     0.535 |                         0.564 |                         0.566 |                         0.556 |
| (18, 'mlp_relu', 'last')    |                                 0.852 |                     0.575 |                     0.56  |                     0.602 |                                     0.863 |                         0.585 |                         0.578 |                         0.621 |
| (18, 'mlp_relu', 'mean')    |                                 0.444 |                     0.439 |                     0.453 |                     0.436 |                                     0.504 |                         0.492 |                         0.515 |                         0.491 |
| (18, 'resid_split', 'last') |                                 0.942 |                     0.654 |                     0.638 |                     0.684 |                                     0.947 |                         0.671 |                         0.662 |                         0.698 |
| (18, 'resid_split', 'mean') |                                 0.496 |                     0.451 |                     0.44  |                     0.443 |                                     0.545 |                         0.521 |                         0.511 |                         0.505 |

### Type B

|                             |   ('auroc_nmf', 'seed 1 (published)') |   ('auroc_nmf', 'seed 2') |   ('auroc_nmf', 'seed 3') |   ('auroc_nmf', 'seed 4') |   ('best_baseline', 'seed 1 (published)') |   ('best_baseline', 'seed 2') |   ('best_baseline', 'seed 3') |   ('best_baseline', 'seed 4') |
|:----------------------------|--------------------------------------:|--------------------------:|--------------------------:|--------------------------:|------------------------------------------:|------------------------------:|------------------------------:|------------------------------:|
| (12, 'mlp_relu', 'last')    |                                 0.579 |                     0.543 |                     0.506 |                     0.569 |                                     0.591 |                         0.56  |                         0.55  |                         0.582 |
| (12, 'mlp_relu', 'mean')    |                                 0.631 |                     0.628 |                     0.64  |                     0.643 |                                     0.66  |                         0.676 |                         0.673 |                         0.677 |
| (12, 'resid_split', 'last') |                                 0.548 |                     0.557 |                     0.506 |                     0.535 |                                     0.556 |                         0.565 |                         0.535 |                         0.58  |
| (12, 'resid_split', 'mean') |                                 0.584 |                     0.577 |                     0.579 |                     0.588 |                                     0.611 |                         0.612 |                         0.61  |                         0.618 |
| (18, 'mlp_relu', 'last')    |                                 0.478 |                     0.461 |                     0.404 |                     0.428 |                                     0.505 |                         0.506 |                         0.436 |                         0.469 |
| (18, 'mlp_relu', 'mean')    |                                 0.578 |                     0.576 |                     0.566 |                     0.562 |                                     0.632 |                         0.634 |                         0.628 |                         0.62  |
| (18, 'resid_split', 'last') |                                 0.631 |                     0.583 |                     0.522 |                     0.576 |                                     0.627 |                         0.591 |                         0.532 |                         0.58  |
| (18, 'resid_split', 'mean') |                                 0.553 |                     0.55  |                     0.54  |                     0.539 |                                     0.609 |                         0.603 |                         0.595 |                         0.591 |

