# Human scores
Rubric: Identity (I), Prompt following (P), Quality (Q), integers 0-4; total = 100 * (I*P*Q)^(1/3) / 4.
Rater A = the author (human). Rater B = Claude (an AI rater using the same rubric), not a second human.
* `dev_162_raterA_raterB.csv` — development images (FLUX.2-klein-4B, Anything V5 + IP-Adapter), both raters.
* `heldout_90_raterA.csv` — held-out generators, scored blind by Rater A: 60 FLUX.2-klein-9B
  (TEST_PLAN_test9b.md) + 30 Qwen-Image / SD3.5 + IPA / GPT Image 2 (TEST_PLAN_addendum_A.md).
* `identity_audit_20.csv` — attribute-level identity re-check of 20 9B images by both raters (addendum B).
* Rater C (second human, 40 images, addendum C) will be added when available.
