# Bibliography verification pass

_Verified 2026-07-07 against the arXiv API (id_list query). All 14 IDs resolve; every first-author surname matches the manuscript's citation. Two year notes are venue-vs-preprint, not errors._

## Verified references

| arXiv | Authors (first) | Title | Preprint | Cite as | Note |
|---|---|---|---|---|---|
| 2001.04063 | Weizhen Qi | ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training | 2020-01 | (Qi et al., 2020) |  |
| 2404.19737 | Fabian Gloeckle | Better & Faster Large Language Models via Multi-token Prediction | 2024-04 | (Gloeckle et al., 2024) | published ICML 2024; separate per-horizon heads |
| 2412.19437 |  DeepSeek-AI | DeepSeek-V3 Technical Report | 2024-12 | (DeepSeek-AI, 2024) | 200-author tech report; sequential MTP modules |
| 2401.10774 | Tianle Cai | Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads | 2024-01 | (Cai et al., 2024) | Medusa; speculative-decoding heads |
| 2505.10518 | Anastasios Gerontopoulos | Multi-Token Prediction Needs Registers | 2025-05 | (Gerontopoulos et al., 2025) | MuToR = register tokens |
| 2508.19228 | Zayd M. K. Zuhri | Predicting the Order of Upcoming Tokens Improves Language Modeling | 2025-08 | (Zuhri et al., 2025) | = 'Token Order Prediction (TOP)'; same paper, method name |
| 2507.11851 | Mohammad Samragh | Your LLM Knows the Future: Uncovering Its Multi-Token Prediction Potential | 2025-07 | (Samragh et al., 2025) | gated LoRA |
| 1711.03953 | Zhilin Yang | Breaking the Softmax Bottleneck: A High-Rank RNN Language Model | 2017-11 | (Yang et al., 2018) | arXiv 2017-11; **published ICLR 2018** → 2018 correct |
| 2603.10145 | Nathan Godey | Lost in Backpropagation: The LM Head is a Gradient Bottleneck | 2026-03 | (Godey & Artzi, 2026) | 2 authors: Nathan Godey, Yoav Artzi |
| 1812.02224 | Yunshu Du | Adapting Auxiliary Losses Using Gradient Similarity | 2018-12 | (Du et al., 2018) | gradient-similarity aux weighting |
| 2001.06782 | Tianhe Yu | Gradient Surgery for Multi-Task Learning | 2020-01 | (Yu et al., 2020) | PCGrad; NeurIPS 2020 |
| 2010.05874 | Zirui Wang | Gradient Vaccine: Investigating and Improving Multi-task Optimization in Massively Multilingual Models | 2020-10 | (Wang et al., 2021) | arXiv 2020-10; **published ICLR 2021** → 2021 correct |
| 2305.19844 | Yi Sun | Learning Task-preferred Inference Routes for Gradient De-conflict in Multi-output DNNs | 2023-05 | (Sun et al., 2023) | DR-MGF |
| 2302.11289 | Guangyuan Shi | Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning | 2023-02 | (Shi et al., 2023) | Recon |

## Findings

- **No fabricated citations.** All 14 arXiv IDs resolve to real papers with matching first authors.
- **Two year notes are correct as cited** (venue year, not preprint year): Yang et al. "Breaking the Softmax Bottleneck" (arXiv 2017-11 → ICLR **2018**); Wang et al. "Gradient Vaccine" (arXiv 2020-10 → ICLR **2021**). Keep the manuscript's 2018/2021.
- **Godey & Artzi (2026)** confirmed 2-author ("Lost in Backpropagation: The LM Head is a Gradient Bottleneck") — the anchor for our output-head bridge. Cite as `& Artzi`, not `et al.`
- **Zuhri et al. (2025)** title is "Predicting the Order of Upcoming Tokens Improves Language Modeling"; the manuscript's shorthand "Token Order Prediction" is that paper's method name (TOP) — one citation, not two.
- **DeepSeek-V3** is a technical report with ~200 listed authors; `(DeepSeek-AI, 2024)` is the conventional corporate-author form — correct.

## Action items before submission
- Replace arXiv preprint entries with **published venue** entries where one exists (ICML/ICLR/NeurIPS above) in the final `.bib`.
- Confirm **MuToR / Samragh / Zuhri / Godey&Artzi / Sun / Shi** final venues (several may still be preprint-only as of mid-2026).
- The manuscript still carries ⟨verify⟩ tags inline; once the `.bib` is built from this table, strip them.
## Addendum (Part B) — 2 new citations verified against arXiv primary source

| Key | arXiv | Verified claim (from abstract) | Cited where |
|---|---|---|---|
| chen2018gradnorm | 1711.02257 | "automatically balances training ... by dynamically tuning gradient magnitudes" (Chen, Badrinarayanan, Lee, Rabinovich; ICML 2018) | related.tex — magnitude-vs-direction |
| jiang2023forkmerge | 2301.12618 | negative transfer "often attributed to gradient conflicts", but optimization-based methods "largely overlook the auxiliary-target generalization capability" (Jiang et al.; NeurIPS 2023) | main.tex intro + related.tex |

Both first-author surnames, titles, and venue years confirmed via export.arxiv.org API.
Only claims present in the abstract were used; the "L2-regularization control" detail
noted in the literature scan is NOT in the abstract and was NOT cited (would need PDF body).
