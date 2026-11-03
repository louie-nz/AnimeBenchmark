# TEST_PLAN addendum C — leaderboard, generator tests, bias correction, second human rater
(frozen 2026-11-03, before the leaderboard, the tests, the corrected scores or any second-rater score exist)

## C1. Official leaderboard (scorer = GPT-6.1-sol v4, the recommended main judge)
* All six generators, all generated images of the ref condition (27 groups x 3 seeds; GPT Image 2: 80) and of the
  noref condition (26 groups x 3; b16 has no noref), same rubric and settings as QUESTIONS_v4.md.
* Reported per generator: mean total S and mean I / P / Q, overall and per level (L1, L2, Task B) and per style type;
  95 % CI by bootstrap over the 27 groups (2 000 resamples); ref gain = S(ref) − S(noref) on the 26 shared groups.
* Human means (Rater A) are shown next to the judge means where they exist (dev: 81 per generator; test: 60 / 10).

## C2. Are generators different? (paired tests)
* Unit = group (27 matched cases): mean judge total over the seeds of each group.
* All 15 generator pairs: two-sided Wilcoxon signed-rank test, Holm correction over the 15 tests, alpha 0.05.
  Pairs not significant after correction are reported as ties on the leaderboard.

## C3. Bias correction of the judge
* Rule fixed here: offset_a = mean(judge_a − RaterA_a) on the 162 development images, per axis a in {I, P, Q};
  corrected_a = clip(judge_a − offset_a, 0, 4); total recomputed with Eq. (total).
* Evaluated once on the 90 held-out images against Rater A: within-1 %, MAE, bias, QWK (rounded), AC2.
* Disclosure: the held-out biases of the uncorrected judge (I −0.26, P +0.48, Q +0.57) were already known when this
  rule was written; the rule uses development data only, so it is not tuned on the held-out set.

## C4. Second human rater (Rater C)
* Sample (fixed, sample_raterC_40.json, random.Random(2029)): 20 development images (10 FLUX.2-klein-4B, 10 Anything V5)
  and 20 held-out images (10 FLUX.2-klein-9B from the 60, and 10 from the 30 of Qwen / SD3.5 / GPT Image 2),
  shuffled, ids C01–C40, generator hidden. Same rubric, reference heads and checklists as Rater A.
* Reported: Rater A vs Rater C (human–human) and Rater B vs C, per axis: exact %, within-1 %, MAE, bias, QWK, AC2.
  Then GPT-6.1-sol vs the mean of the two humans on the same 40 images.
* The human–human values replace the A–B (human–AI) values as the ceiling in the thesis wherever both exist;
  n = 40 is small, so all values carry bootstrap CIs.
