# TEST_PLAN addendum A — multi-generator held-out test (frozen 2026-11-02, before any human score of these images)

## Why
The original test (TEST_PLAN_test9b.md, 60 FLUX.2-klein-9B images) contains one generator only, so it can test
within-generator agreement but not pooled agreement, model ranking, or self-preference of the GPT judges.
Result of the original test, fixed here for the record: P QWK 0.02–0.13 for all judges (criterion ≥ 0.60 not met);
Rater A gave I = 4 to 58 / 60 images, so identity could not be tested there.

## Sample (fixed, `test_sample_30.json`, script `sample30.py`)
* Three generators never used while designing any judge: Qwen-Image (Edit-2511), SD3.5-Large + IP-Adapter,
  GPT Image 2.
* 10 of the 27 groups drawn with `random.Random(2027)`: al1, cb1, cc1, cl1, hb1, hr2, od1, rr1, tb1, vb1
  (the same 10 groups for all three generators, so their prompts are matched); one random ref seed per group and
  generator; 30 images in random order, shown with an anonymous id (T01–T30) and **without the generator name**.
* Random draw gave 4 Task B groups out of 10 (more than the 4 / 27 share); reported, not re-drawn.
* Combined test set = 60 (9B) + 30 = 90 images, four unseen generators.

## Blinding
Rater A has not seen any per-image automatic score for these images. He has seen the generator means of
gpt-6.1-sol for the six generators (reported to him on 2026-11-01); this partial knowledge is stated as a limitation.

## Sample size is fixed in advance
Every statistic is reported with a 95 % bootstrap confidence interval (2 000 resamples of images). A claim is made
only when the interval supports it; otherwise the result is reported as **inconclusive**. The sample size is part
of the design and is not used afterwards to explain away a result.

## Questions
A1. **Pooled agreement on unseen generators** (90 images): QWK and Spearman ρ for I, P, Q and total, per judge
    (Qwen2.5-VL-32B v1, v3; gpt-4o-2024-11-20; gpt-6.1-sol). P passes if QWK ≥ 0.60 and the CI lower bound > 0.40.
A2. **Within-generator agreement**: the 9B result (n = 60) plus, as exploratory only, the image-weighted mean of
    within-generator ρ over the four generators.
A3. **Model ranking**: human mean total per generator on the 10 matched groups (9B restricted to the same 10
    groups) vs each judge's mean on the same images; Kendall τ between the two orderings of the four generators.
A4. **Self-preference**: for each judge, bias = judge total − human total. GPT judges are flagged for possible
    self-preference if bias(GPT Image 2) − mean bias(Qwen, SD3.5) > 5 points and its CI excludes 0; the same
    quantity is reported for Qwen2.5-VL-32B as the non-OpenAI comparison.
A5. **Quality detection**: precision / recall of judge Q ≤ 2 against Rater A Q ≤ 2 on all 90 images.
