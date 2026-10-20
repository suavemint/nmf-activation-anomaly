# Prior art (verified 2026-10-20)

Every entry was checked against a primary source:
- arXiv API (title, authors, first-submission date, comments field)
- Crossref (DOI metadata)
- the publisher's page

Relevance notes are based on the abstracts, not on memory. Venues are given only where the arXiv comments field, Crossref or the publisher states them; otherwise "arXiv". Read each paper before quoting anything beyond its abstract.

Legend: **[cite]** = should be cited in the write-up; **[context]** = worth a sentence or a related-work line; **[data]** = dataset/model source.

## 1. Methods this study uses directly

| | reference | verified via | why it matters here |
|---|---|---|---|
| [cite] | D. D. Lee, H. S. Seung. *Learning the parts of objects by non-negative matrix factorization.* Nature, 1999. doi:10.1038/44565 | Crossref | The NMF formulation used (parts-based, nonnegative factors). |
| [cite] | D. Kong, C. Ding, H. Huang. *Robust nonnegative matrix factorization using L21-norm.* CIKM 2011. doi:10.1145/2063576.2063676 | Crossref | The robust L2,1 NMF variant implemented in `src/detectors.py`. |
| [cite] | O. Ledoit, M. Wolf. *A well-conditioned estimator for large-dimensional covariance matrices.* J. Multivariate Analysis, 2004. doi:10.1016/S0047-259X(03)00096-4 | Crossref | Shrinkage covariance for the Mahalanobis baseline (D10). |
| [cite] | K. Lee, K. Lee, H. Lee, J. Shin. *A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks.* NeurIPS 2018 (arXiv comment: "Accepted in NIPS 2018"). arXiv:1807.03888 | arXiv | Mahalanobis distance on deep-network features as an OOD/adversarial detector: the lineage of the Mahalanobis baseline. |
| [cite] | A. Podolskiy et al. *Revisiting Mahalanobis Distance for Transformer-Based Out-of-Domain Detection.* AAAI 2021 (arXiv comment). arXiv:2101.03778 | arXiv | Mahalanobis on transformer representations for out-of-domain detection without OOD training data: closest baseline lineage for a transformer setting. |
| [cite] | L. Gao et al. *Scaling and evaluating sparse autoencoders.* 2024. arXiv:2406.04093 | arXiv | k-sparse (TopK) autoencoders: the SAE family used for the SAE baseline (EleutherAI's Pythia-410M SAEs). |

## 2. Closest work on NMF and LLM activations

| | reference | verified via | relevance |
|---|---|---|---|
| [cite] | O. Shafran, A. Geiger, M. Geva. *Constructing Interpretable Features from Compositional Neuron Groups.* ACL 2026 main conference (per arXiv page). arXiv:2506.10920. v1 (2025-06-12) was titled "Decomposing MLP Activations into Interpretable Features via Semi-Nonnegative Matrix Factorization"; the current v2 (2026-05-02) has the new title. | arXiv abs page | The closest published use of (semi-)NMF on LLM MLP activations. Their goal is **interpretability/steering** (SNMF features vs. SAEs), not anomaly detection. This is the main "NMF on activations" reference, and a reason the question here was plausible. Cite the v2 title. |

I did not find published work using an **NMF reconstruction residual as an anomaly score on LLM activations**. That absence is based on a limited search (a few web queries plus the references above), so treat it as "not found", not "doesn't exist", and re-search before claiming novelty.

## 3. Activation-based backdoor detection (classic, mostly vision)

| | reference | verified via | relevance |
|---|---|---|---|
| [cite] | B. Tran, J. Li, A. Madry. *Spectral Signatures in Backdoor Attacks.* NeurIPS 2018 (arXiv comment). arXiv:1811.00636 | arXiv | Backdoored examples leave a detectable "spectral signature" in learned representations, found with robust statistics. A spectral/PCA-family relative of the PCA-residual baseline. |
| [cite] | B. Chen et al. *Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering.* 2018. arXiv:1811.03728 | arXiv | Detects poisoned training data from activations (activation clustering). The standard "backdoors are visible in activations" reference. |
| [cite] | J. Hayase et al. *SPECTRE: Defending Against Backdoor Attacks Using Robust Statistics.* 2021. arXiv:2104.11315 | arXiv | Notes that representation-based defenses work only when the spectral signature is large enough, and strengthens them with robust statistics. |
| [context] | C. S. Sastry, S. Oore. *Detecting Out-of-Distribution Examples with In-distribution Examples and Gram Matrices.* NeurIPS 2019 Workshop on Safety and Robustness in Decision Making (arXiv comment). arXiv:1912.12510 | arXiv | Another activation-statistics OOD detector that needs only in-distribution data. |
| [context] | Y. Gao et al. *STRIP: A Defence Against Trojan Attacks on Deep Neural Networks.* 2019. arXiv:1902.06531 | arXiv | Input-level trojan detection; background for the backdoor setting. |

Note: these works mostly detect **poisoned training data** or backdoored inputs in vision classifiers. This study scores **individual prompts at inference time** in an LLM.

## 4. Backdoors / model organisms in LLMs and probes

| | reference | verified via | relevance |
|---|---|---|---|
| [cite] | E. Hubinger et al. *Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training.* 2024. arXiv:2401.05566 | arXiv | The "model organism" framing for a trained-in trigger behavior (type A). |
| [cite] | M. MacDiarmid et al. *Simple probes can catch sleeper agents.* Anthropic research note, 2024-04-23. https://www.anthropic.com/research/probes-catch-sleeper-agents | publisher page | Linear probes on residual-stream activations detect sleeper-agent defection at >99% AUROC from generic contrast pairs. The authors caution that the organisms are artificial. Directly comparable to this study's probe ceiling, and to its finding that the backdoor is far easier to detect than jailbreaks. |
| [cite] | L. Bailey et al. *Obfuscated Activations Bypass LLM Latent-Space Defenses.* 2024. arXiv:2412.09565 (no venue in the arXiv comments) | arXiv | Shows that SAEs, representation probes **and latent OOD detection** can be evaded by obfuscated activations. Belongs in Limitations: every detector here assumes a non-adaptive attacker. |
| [context] | J. Yan et al. *Rethinking Backdoor Detection Evaluation for Language Models.* EMNLP 2025 (arXiv comment). arXiv:2409.00399 | arXiv | Model-level backdoor detectors are less robust than benchmark results suggest when planting factors vary. A related evaluation-robustness point, though in a different setting (is the *model* backdoored, not is the *input* triggering). |
| [context] | T. Lasnier et al. *Language Triggers Hijack Language Circuits: A Mechanistic Analysis of Backdoor Behaviors in Large Language Models.* 2026. arXiv:2602.10382 | arXiv | Backdoor trigger heads overlap with heads that do the model's normal work. They argue this is a blind spot for out-of-distribution-activation detection. Relevant to interpreting where anomaly detectors can and can't work. |

## 5. Mechanistic anomaly detection (closest framing)

| | reference | verified via | relevance |
|---|---|---|---|
| [cite] | D. O. Johnston, A. Chakraborty, N. Belrose. *Mechanistic Anomaly Detection for "Quirky" Language Models.* ICLR 2025 Building Trust Workshop (arXiv comment). arXiv:2504.08812 | arXiv | Detectors on internal features flag test points that differ from a trusted environment. They find high discrimination on some tasks but no detector effective across all models and tasks. Same unsupervised-on-trusted-data framing as this study, by EleutherAI (who also made Pythia and the SAEs used). |
| [cite] | H. Lyons Keenan, C. Leckie, S. Erfani. *Mechanistic Anomaly Detection via Functional Attribution.* ICML 2026 (arXiv comment: "ICML '26 Camera Ready"). arXiv:2604.18970 | arXiv | Argues latent-space analysis is vulnerable to obfuscation, and instead uses influence-function attribution to a trusted set. Evaluated on LLM backdoors. An alternative to the activation-statistics detectors tested here. |

## 6. Activation-based jailbreak / misbehavior detection in LLMs

| | reference | verified via | relevance |
|---|---|---|---|
| [cite] | N. M. Min, L. H. Pham, J. Sun. *Layerwise Convergence Fingerprints for Runtime Misbehavior Detection in Large Language Models.* 2026. arXiv:2604.24542 | arXiv | Tuning-free runtime monitor using a diagonal Mahalanobis distance on inter-layer hidden-state differences, aimed at backdoors, jailbreaks and prompt injections. Very close in spirit (unsupervised, Mahalanobis-type, both threat types). Their results may contrast with this study's weak jailbreak detection. Read before writing. |
| [cite] | G. Azov, O. Rivlin, G. Shtar. *SALLIE: Generation-Free Hidden-State Detection of Jailbreaks and Prompt Injections Across Text and Vision.* 2026 (preprint under review). arXiv:2604.06247 | arXiv | Uses **last-token** residual-stream representations with layer-wise k-NN probes. Relevant to the last-token-vs-mean pooling result. |
| [context] | S. Liang et al. *Learning to Detect Unseen Jailbreak Attacks in Large Vision-Language Models.* 2025. arXiv:2508.09201 | arXiv | A learnable detector that needs no attack data (vision-language models). A counterpoint to this study's negative unsupervised jailbreak result. |
| [context] | S. Zhou et al. *Exposing the Ghost in the Transformer: Abnormal Detection for Large Language Models via Hidden State Forensics.* 2025. arXiv:2504.00446 | arXiv | Layer-specific activation patterns used to detect hallucinations, jailbreaks and backdoors. |
| [context] | Y. Jiang et al. *HiddenDetect: Detecting Jailbreak Attacks against Large Vision-Language Models via Monitoring Hidden States.* 2025. arXiv:2502.14744 | arXiv | Hidden-state jailbreak detection (vision-language models). |
| [context] | A. Zou et al. *Representation Engineering: A Top-Down Approach to AI Transparency.* 2023. arXiv:2310.01405; A. Arditi et al. *Refusal in Language Models Is Mediated by a Single Direction.* 2024. arXiv:2406.11717 | arXiv | Background for "safety-relevant properties are linearly represented", which is consistent with the in-type jailbreak probe reaching ~0.89 while unsupervised detectors don't. |

## 7. Data and model sources

| | reference | verified via |
|---|---|---|
| [data] | S. Biderman et al. *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling.* 2023. arXiv:2304.01373 | arXiv |
| [data] | X. Shen et al. *"Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models.* 2023. arXiv:2308.03825 (venue not confirmed here; check before citing one) | arXiv |
| [data] | A. Köpf et al. *OpenAssistant Conversations: Democratizing Large Language Model Alignment.* 2023. arXiv:2304.07327 | arXiv |
| [data] | S. Merity et al. *Pointer Sentinel Mixture Models.* 2016. arXiv:1609.07843 (WikiText) | arXiv |
| [data] | Alpaca-cleaned (`yahma/alpaca-cleaned`) and Dolly-15k (`databricks/databricks-dolly-15k`): cite their Hugging Face dataset cards/repositories. I did not verify a paper for either. | HF Hub (D2) |
| [data] | EleutherAI SAEs: `EleutherAI/sae-pythia-410m-65k` (Hugging Face model card) | HF Hub (D2) |

## Flags for the write-up

1. **Shafran, Geiger & Geva (ACL 2026)** is the reference a reviewer will ask about: "NMF on LLM activations has been done." The answer is that theirs is semi-NMF for interpretability, and this study is NMF residuals for anomaly detection. Cite the current title.
2. **Min, Pham & Sun (2026)** and **SALLIE (2026)** report hidden-state detection of jailbreaks. This study's jailbreak result is negative for *unsupervised, benign-only* detectors at matched length, with a far negative control. Read their setups (supervised vs. unsupervised, length control, negative controls) before contrasting.
3. **Bailey et al. (2024)** belongs in Limitations: no adaptive attacker was tested.
4. **MacDiarmid et al. (2024)** is the natural comparison for the probe ceiling and for "backdoors are easy, jailbreaks are hard".
5. Not included because they could not be verified quickly: Neural Cleanse, and venue details for Shen et al. and Bailey et al.
