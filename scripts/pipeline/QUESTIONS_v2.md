# VLM question set v2 — pre-registration

Written and frozen on 2026-10-28, **before any v2 result was produced**. The run is executed once; all
results are reported whether they improve on v1 or not. No question, weight or threshold is changed after
the v2 results are seen.

## Why a v2
The v1 run (Qwen2.5-VL 7B and 32B, 2026-10-26/28) showed two design problems, found by inspecting
per-question answers, not by tuning against the human scores:

1. Identity: one global "same character?" question on the full image. The answer was driven by outfit,
   scene and body framing (32B: I_vlm correlates more with human P, rho 0.44, than with human I, rho 0.28).
2. Quality: three of four questions were answered identically for almost every image
   (2D 1.00, single picture 1.00, no anatomy errors 0.99 for 32B), so the axis carried no information.

## What changes (and only this)
| | v1 | v2 |
|---|---|---|
| System instruction | none | "You are a strict art reviewer checking AI-generated illustrations. Judge only what is visible in the images. Answer with a single word: yes or no." |
| Identity images | reference head + full generated image | reference head + **head crop** of the generated image (Grounding DINO box from identity_*.csv, 15 % margin; full image if no box) |
| Identity questions | "same character?" (w2) | "same character?" (w2) + hair (w1) + eyes (w1) + face / head accessories (w1), each telling the model to ignore drawing style and chibi proportions |
| Position bias | single order | every head-pair question asked in both image orders, p(yes) averaged |
| Checklist identity items | full image | unchanged (scarves, hats lie outside the head crop) |
| Quality | 2D, single picture, no anatomy errors, no border | six error checks, yes = error: hands, duplicated / merged objects, distorted face, watermark / logo / signature overlay (scene writing excluded), photo / 3D look, border |
| Prompt items, penalties, vqa_prompt | — | unchanged |
| Model, image size (768 px), aggregation, Eq. total | — | unchanged |

Exact texts: `ID_V2`, `Q_V2`, `SYSTEM_V2` in `evaluate_vlm.py`.

## Planned analysis (fixed in advance)
* Model: Qwen2.5-VL-32B-Instruct, ref images, all 162 human-scored images.
* Primary reference: Rater A (Louie) alone; secondary: mean of both raters.
* Report for v1 and v2 side by side: Spearman rho per axis, QWK per axis, rho **within each generator**
  (FLUX only, AV5 only), and per-group winner agreement.
* Robustness split: worlds w004 + w006 vs. w014 + w022 + w029 reported separately.
* Success criterion stated in advance: v2 counts as an improvement on an axis only if QWK **and** the
  within-generator rho both increase for Rater A.

## Known limitation
The v1 diagnosis used the same 162 images, so v2 is a post-hoc redesign, not a blind test. A truly held-out
check would need new human scores (e.g. a sample of the unscored no-ref images).
