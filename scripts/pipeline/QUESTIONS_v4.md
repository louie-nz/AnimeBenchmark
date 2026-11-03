# v4 closed judges — pre-registration (frozen 2026-11-01, before any v4 score exists)

## Why
The open judges (Qwen2.5-VL 7B / 32B; question sets v1, v2, v3) reached the 0.60 QWK threshold only for prompt
following. Identity and Quality stayed far below it. v4 asks whether this is a limit of the **judge model** or of
the **task**: the v3 rubric is kept exactly as it is and only the judge model is replaced by stronger, closed
models.

## Judges (fixed; not changed during the run)
| Judge | Role | Settings |
|---|---|---|
| `gpt-4o` (snapshot recorded in every row) | same judge as Tong et al., *Trait Collapse* (our group) | temperature 0.2, JSON response format, image detail high, max 2000 tokens |
| `gpt-6.1-sol` | strongest OpenAI vision model available (Roboflow Vision Evals, Sept 2026) | reasoning model: no temperature, reasoning_effort low, JSON, detail high, max 2000 tokens |

If the API rejects a setting, the run stops; the setting that is finally used is reported, and it is the same
for every image of that judge.

## Fixed protocol (identical to the 32B v3 run)
* System prompt, rubric, checklist and answer format imported from `evaluate_rubric.py` without change.
* Same image pair (reference head, generated image), both downscaled to at most 768×768.
* All ref images of the six generators (27 groups × 3 seeds = 81 per generator, 486 per judge), one call per image.
* Same parser and total formula. Parse failures and refusals are counted as failures, never re-asked.

## Questions and criteria
Human reference: Rater A (Louie). Development set = 162 images (4B + AV5); test set = the 60 9B images in
`test_sample_60.json` (scored blind by Rater A before the v4 test file is opened).

Best earlier judge on the development set (Rater A, QWK): I 0.28 (32B v1), P 0.80 (32B v1), Q 0.31 (7B v1).

1. **Does a stronger judge score Identity and Quality?** Development set, per judge and axis: QWK with Rater A.
   *Usable* = QWK ≥ 0.60 (same threshold as QUESTIONS_v3.md). *Improvement* = QWK at least 0.10 higher than the
   best earlier judge on that axis.
2. **Does it hold on a new generator?** Test set: QWK and Spearman ρ for I, P, Q and total.
   P passes if QWK ≥ 0.60 (as in TEST_PLAN_test9b.md).
3. **Can it find quality problems?** Detection of images that Rater A scored Q ≤ 2: precision and recall of the
   judge's Q ≤ 2. Also the share of images given Q = 4, to measure leniency (32B v3: 152 / 162).
4. **Is the model ranking the same?** Mean total per generator for each judge (six generators) next to Qwen 32B v1
   and v3. Within-generator ρ is reported next to every pooled ρ.
5. **Self-preference (limitation check).** GPT judges scoring GPT Image 2: compare the rank of `gptimage2` among the
   six generators under the GPT judges with its rank under Qwen 32B v3. A difference of more than one rank
   position is reported as a possible self-preference effect. If Rater A scores a sample of GPT Image 2 images
   later, judge-minus-human differences for that sample are reported as well.

## Reported regardless of outcome
All of the above for both judges, tokens and cost, number of failures, and the full raw answers
(`*_raw.csv`).
