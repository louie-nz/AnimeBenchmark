# AnimeCharBench

A benchmark for **reference-guided anime and cartoon character generation in established story worlds**, with a
pre-registered study of whether it can be scored automatically.
CS792 dissertation project, School of Computer Science, The University of Auckland (2026).
Author: Louie Zhang. Supervisors: A/Prof. Jiamou Liu, Zhongsheng Wang.

## What the benchmark tests
* **27 cases** from 5 public-domain literary worlds (Wonderland, Oz, Cthulhu, Vampyre, R.U.R.). Each case = a
  reference head + a prompt + a checklist (must / should / penalty items).
* **Three task levels**: L1 reproduce the character, L2 change outfit / pose / scene, Task B put a *new* character
  into an established world.
* **Two conditions**: `ref` (reference head given) and `noref` (prompt only), seeds 42 / 123 / 777.
* **Scoring**: Identity, Prompt following, Quality (0-4 each); total = 100 · (I·P·Q)^(1/3) / 4.

## Leaderboard (judge: GPT-6.1-sol, v4 rubric; ref condition)
| Rank | Generator | Total [95 % CI] | L1 | L2 | Task B | No-ref | Ref gain | Human (n) |
|---|---|---|---|---|---|---|---|---|
| 1* | GPT Image 2 | 96.5 [94.4, 98.4] | 96.1 | 95.1 | 99.4 | 76.3 | +20.4 | 84.9 (10) |
| 2-4 | Qwen-Image (Edit-2511 / 2512) | 91.6 [89.2, 93.6] | 92.4 | 90.0 | 90.1 | 69.1 | +22.9 | 81.5 (10) |
| 2-4 | FLUX.2-klein-9B | 90.5 [87.8, 93.0] | 90.2 | 91.3 | 90.9 | 78.1 | +12.5 | 85.4 (60) |
| 2-4 | FLUX.2-klein-4B | 88.7 [84.4, 92.6] | 88.1 | 89.8 | 89.6 | 73.8 | +15.2 | 86.1 (81) |
| 5 | SD3.5-Large + IP-Adapter | 66.5 [59.1, 72.8] | 68.7 | 68.5 | 56.7 | 62.9 | +4.0 | 57.0 (10) |
| 6 | Anything V5 + IP-Adapter | 47.7 [38.3, 56.6] | 56.0 | 60.1 | 7.1 | 38.3 | +9.9 | 50.0 (81) |

Ranks 2-4 are ties (paired Wilcoxon over 27 groups, Holm-corrected). \*GPT Image 2 is first by the judge but tied
with FLUX.2-klein-9B by the human rater; the judge overrates it by about 12 points.

## Main findings on automatic scoring
* GPT-6.1-sol scores 89-99 % of images within one point of the human rater and is the only judge that finds quality
  errors, but it is about 0.5 point lenient on prompt following and quality.
* Pooled agreement over generators overstates judge quality (prompt-following QWK 0.74 on the development
  generators vs. 0.40 on unseen generators). Report within-generator and held-out agreement.
* CLIP-T does not track human prompt following on held-out images (rho = 0.04); CLIP-I / DINO-I reward copying the
  reference head.
* Recommended use: judge ranking with CIs + human spot-check of 10-15 % of images; Qwen2.5-VL-32B (v1) as a free
  alternative for prompt following.

## Repository layout
```
scripts/pipeline/          generation, evaluation, judges, leaderboard (see README_benchmark.md)
  cases/, cases_legacy/    the 27 case files (prompt templates, checklists, reference-head names)
  slurm/                   cluster scripts (run_benchmark.sh = one command per new generator)
  QUESTIONS_v*.md, TEST_PLAN_*.md   pre-registrations, frozen before each result
results/eval/              automatic scores (CSV) and the leaderboard
human_scores/              human ratings (CSV)
```
Not in the repository: generated images, reference-head images (from the source dataset; licence to be confirmed),
model weights.

## Reproduce
1. Cluster (GPU): `bash scripts/pipeline/slurm/run_benchmark.sh MODEL_KEY TAG`
2. With an OpenAI API key: `python scripts/pipeline/benchmark_judge.py --tags TAG1,TAG2,...`
3. Leaderboard: `results/eval/leaderboard/leaderboard.md`

Details: `scripts/pipeline/README_benchmark.md`.
