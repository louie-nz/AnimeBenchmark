# AnimeBenchmark pipeline v3 (2026-10-31)

One pipeline for every generator. Consistency rules (apply to every model, old and new):
same 27 case files, same reference heads, same instruction-style prompt templates (SD tag prompt only for SD-style
models), seeds 42 / 123 / 777 (3 API samples for GPT Image), 1024x1024 output, model-card sampler settings,
and the same evaluation: CLIP-I / DINO-I, CLIP-T, Qwen2.5-VL-32B judges v1 (yes/no) and v3 (rubric).

## Where each file goes (cluster: /data/lzha060/animebenchmark/, laptop: the same repo layout)
| File | Folder | What it does |
|---|---|---|
| generate_all.py | scripts/pipeline/ | generation; models: flux2 (4B), flux2_9b, qwen (Edit-2511 / 2512), av5 |
| gen_openai.py | scripts/pipeline/ | GPT Image generation through the official API (run on the laptop) |
| evaluate_all.py | scripts/pipeline/ | CLIP-I / DINO-I on Grounding-DINO head crops (needs eval_identity.py, unchanged) |
| evaluate_vlm.py | scripts/pipeline/ | VLM yes/no judge, question sets v1 / v2 |
| evaluate_rubric.py | scripts/pipeline/ | VLM rubric judge v3 |
| clip_t.py | scripts/pipeline/ | CLIP-T (image <-> prompt) |
| compare_auto_human.py | scripts/pipeline/ | automatic scores vs human scores (laptop) |
| QUESTIONS_v2.md, QUESTIONS_v3.md, TEST_PLAN_test9b.md | scripts/pipeline/ | pre-registrations (keep, cited in the thesis) |
| run_all.sbatch | scripts/pipeline/slurm/ | any combination of generation and evaluation steps in one job |
| run_model.sh | scripts/pipeline/slurm/ | one new generator: generate + v3 + v1 + CLIP-T |
| run_vlm.sh, run_9b_and_v3.sh | scripts/pipeline/slurm/ | earlier runs, kept for reproducibility |

Generator tags in file names: flux2klein4b, anythingv5_ipadapter, flux2klein9b, qwenimage, gptimage2, sd35ipa.
Output: results/generated/{world}/{group}_files/{group}_{tag}_{character}_{ref|noref}_seed{n}.png

## Order for the new generators
1. FLUX.2-klein-9B — done (2026-10-30), judges v1 / v3 and CLIP-T done.
2. Qwen-Image — `run_model.sh qwen qwenimage`.
3. GPT Image 2 — laptop `gen_openai.py`, upload images, then `SKIP_GEN=1 run_model.sh gptimage2 gptimage2`.
4. SD3.5-Large + IP-Adapter — to be added (tag sd35ipa).
5. At the end, CLIP-I / DINO-I for all new models in one job:
   `TAGS=flux2klein9b,qwenimage,gptimage2,sd35ipa` + run_all.sbatch step `eval`.
