# AnimeCharBench — how to benchmark a new generator

Input: 27 cases (reference head + prompt + checklist) in `scripts/pipeline/cases*/`. Every generator gets the same
cases, prompt templates and seeds 42 / 123 / 777, under two conditions: **ref** (reference head given) and **noref**.

## One command per machine
1. Cluster (GPU):  `srun ... bash scripts/pipeline/slurm/run_benchmark.sh MODEL_KEY TAG`
   -> generated images, CLIP-I / DINO-I, CLIP-T, Qwen2.5-VL-32B v1 scores in `results/eval/models/TAG/`.
   New model = one class in `generate_all.py` (MODELS registry). API-only models: generate with `gen_openai.py`,
   then `SKIP_GEN=1`.
2. Laptop (OpenAI key): `py scripts/pipeline/benchmark_judge.py --tags TAG1,TAG2,...`
   -> GPT-6.1-sol v4 rubric judge on ref + noref (`results/eval/gpt_judge/`), then
   `results/eval/leaderboard/{leaderboard.md, leaderboard.csv, pairwise.csv}`.
   Bias-corrected totals: add `--offsets scripts/pipeline/judge_offsets.json` to `make_leaderboard.py`
   (offsets fixed on the development set, TEST_PLAN_addendum_C.md).

## Recommended use of the scores
* Rank generators with the GPT-6.1-sol leaderboard; pairs that are not significant after Holm correction are ties.
* The judge is ~0.5 point lenient on P and Q and overrates highly polished images: check a random 10-15 % of images
  by hand with the rubric (`scoring_*` sheets) and report within-1 %, MAE, bias and QWK.
* Free alternative judge: Qwen2.5-VL-32B v1 (prompt following only; it does not find quality errors).
* CLIP-I / DINO-I reward copying the reference head; use them as diagnostics only, together with P.

## Pre-registrations (frozen before each result)
QUESTIONS_v2/v3/v4.md, TEST_PLAN_test9b.md, TEST_PLAN_addendum_A/B/C.md.
