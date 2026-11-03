# Held-out test on FLUX.2-klein-9B — analysis plan (frozen 2026-10-30, before any human test score exists)

## Test set
* 60 ref images of FLUX.2-klein-9B (a generator never used while designing any judge), sampled with
  `random.Random(2026)`: 2 per group for all 27 groups + 6 extra; list in `test_sample_60.json`.
* Human reference: Rater A (Louie) scores the 60 images blind (`scoring_test9b_blind.xlsx`, no Claude scores,
  automatic test results not opened). Rater B does not score the test set.
* Automatic scores already computed on the cluster on 2026-10-30 and not inspected by either rater:
  v1 yes/no judge (Qwen2.5-VL-32B), v3 rubric judge (32B), CLIP-T.

## Development-set outcome that is fixed before the test (for the record)
On the development set (162 images) v3 did **not** meet its pre-registered "improvement over v2" criterion on any
axis (Rater A: I QWK 0.27→0.21, P 0.80→0.67, Q 0.00→−0.03). The v3 test result is still reported, as registered.

## Questions and criteria
1. **Does automatic prompt-following scoring generalise to a new generator?**
   v1-32B `P_vlm` vs Rater A on the 60 test images: pass if QWK ≥ 0.60 (same threshold as QUESTIONS_v3.md).
   Also reported: Spearman ρ, and the same for v3 `P_vlm` and CLIP-T (ρ only).
2. **Does the automatic total score generalise?** v1-32B and v3 `total_vlm` vs Rater A total: Spearman ρ.
3. **Identity / Quality:** QWK and ρ for v1 and v3 I and Q, reported without a pass threshold beyond the 0.60 one.
4. **Model ranking with three generators:** mean human total of FLUX.2-klein-9B (60 test images) placed next to the
   development means of 4B and AV5 (Rater A), and whether the automatic judges reproduce the human ordering of
   the three generators (group-level winner agreement, 4B vs 9B, on the groups in the test sample).
* Caveat stated in advance: the test set has one generator only, so within-generator ρ is the only ρ reported for it.
