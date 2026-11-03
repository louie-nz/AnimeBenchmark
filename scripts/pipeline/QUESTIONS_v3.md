# VLM judge v3 (rubric judge) — pre-registration

Written and frozen on 2026-10-29, **before any v3 output was produced**. Code: `evaluate_rubric.py`
(`SYSTEM_V3`, `RUBRIC`, `ANSWER`, `build_prompt`). Nothing in these texts, the model, the decoding or the
analysis below is changed after v3 results are seen. All results are reported, including failures.

## Motivation (from v1 / v2, decided before v3)
* v1 / v2 asked yes/no questions and read p(yes). Prompt following worked (QWK ≈ 0.80 vs Rater A), but
  Identity and Quality did not (v2, Rater A: QWK 0.27 and 0.00).
* v2's six error-check questions were answered "no error" for 0–10 % of images only → no signal.
* Human raters do not answer yes/no questions: they read a rubric with anchors and give one score per axis.
  v3 tests whether the VLM can do the same task as the humans (VIEScore-style "reason, then score").

## What v3 is
| | v3 |
|---|---|
| Model | Qwen2.5-VL-32B-Instruct (same as v1-32B / v2), bf16, images max 768 px (unchanged) |
| Input | reference head (image 1) + full generated image (image 2) |
| Text | task level, the SD prompt, identity cues, checklist MUST / SHOULD / PENALTY items, and the human rubric anchors (translated from the scoring sheet's 说明 page, unchanged in meaning) |
| Output | JSON: notes + integer 0–4 for identity, prompt, quality; per-item yes/no |
| Decoding | greedy (`do_sample=False`), max 700 new tokens → deterministic |
| Parsing | strict JSON, else field-by-field regex; `parse_ok` recorded; unparsable axes left empty and reported |
| Total | same geometric-mean formula as the human scores |

## Data split (to avoid tuning on the test set)
* **Development set** = the 162 human-scored ref images of FLUX.2-klein-4B and AV5 (already used to diagnose
  v1 / v2). v3 is run **once** on it; no prompt change follows.
* **Test set** = images of a generator never used during judge design (planned: FLUX.2-klein-9B, optionally one
  closed model), ref condition, of which Rater A scores a fixed random sample of 60 images **before** seeing the
  v3 scores for them. The frozen v3 judge is then run once on the test set.

## Planned analysis
* Reference: Rater A (Louie) alone; secondary: mean of both raters (development set only).
* Per axis: QWK, Spearman ρ, ρ within each generator, per-group winner agreement; parse rate.
* Compared side by side with v1-7B, v1-32B and v2-32B on the development set.

## Criteria (fixed now)
* **Improvement over v2** on an axis (development set): QWK *and* within-generator ρ (both generators) increase.
* **"Pass" for use as an automatic scorer** on an axis (test set): QWK ≥ 0.60 vs Rater A.
  (Reference points: Rater A vs B QWK 0.71 / 0.91 / 0.71 for I / P / Q; the published VLM-judge validation in
  Tong et al. 2026 reports ρ = 0.61.)
* If an axis passes on the development set but fails on the test set, it is reported as not passing.

## Known limitations
* One model and one rubric wording; no ensemble or self-consistency.
* The rubric is shown in English; the human raters used a Chinese sheet with the same anchors.
* Rater B is an AI model; Rater A alone is therefore the primary reference.
