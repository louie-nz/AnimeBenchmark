#!/bin/bash
# run_benchmark.sh MODEL_KEY TAG  -- one command on the cluster for a new generator:
#   generate (27 cases x ref/noref x 3 seeds) -> CLIP-I / DINO-I -> CLIP-T -> Qwen2.5-VL-32B v1 (free judge)
# then, on a machine with an OpenAI key:  python scripts/pipeline/benchmark_judge.py --tags TAG
#   (GPT-6.1-sol v4 judge on ref + noref, then the leaderboard). Everything is resumable (existing outputs skipped).
# Example (tmux on Server 1):
#   srun --gres=gpu:1 --cpus-per-task=4 --mem=128G --time=08:00:00 \
#     bash scripts/pipeline/slurm/run_benchmark.sh flux2_9b flux2klein9b 2>&1 | tee logs/bench_$(date +%m%d_%H%M).log
# SKIP_GEN=1 for a generator whose images were made elsewhere (e.g. GPT Image 2 via gen_openai.py).
KEY=$1; TAG=$2; [ -z "$TAG" ] && { echo "usage: run_benchmark.sh MODEL_KEY TAG"; exit 1; }
export HOME=/data/lzha060 TMPDIR=/data/lzha060/tmp HF_HOME=/data/lzha060/hf_cache PYTHONUNBUFFERED=1
export http_proxy=http://squid.auckland.ac.nz:3128 https_proxy=http://squid.auckland.ac.nz:3128
source /data/lzha060/miniconda3/etc/profile.d/conda.sh
cd /data/lzha060/animebenchmark || exit 1
CASES="scripts/pipeline/cases/w004/cl1*.json,scripts/pipeline/cases/w006/*.json,scripts/pipeline/cases/w014/*.json,scripts/pipeline/cases/w022/*.json,scripts/pipeline/cases/w029/*.json,scripts/pipeline/cases_legacy/w004/*.json"
OUT=results/eval/models/$TAG; mkdir -p $OUT logs; FAIL=0
step() { echo; echo "################ $1  $(date)"; }
if [ "${SKIP_GEN:-0}" != "1" ]; then
  conda activate ${ENV:-flux2}; step "1 generate $KEY -> $TAG"
  python scripts/pipeline/generate_all.py --model $KEY --cases "$CASES" || FAIL=1
fi
conda activate flux2
step "2 CLIP-I / DINO-I"; MODEL_TAGS=$TAG python scripts/pipeline/evaluate_all.py --cases "$CASES" --out $OUT/identity.csv --tags $TAG || FAIL=1
step "3 CLIP-T";         MODEL_TAGS=$TAG python scripts/pipeline/clip_t.py --cases "$CASES" --out $OUT/clip_t.csv || FAIL=1
step "4 Qwen-32B v1";    MODEL_TAGS=$TAG python scripts/pipeline/evaluate_vlm.py --cases "$CASES" --out $OUT/vlm32b_v1.csv \
                           --model-path Qwen/Qwen2.5-VL-32B-Instruct --batch 2 --qset v1 --cond ref,noref || FAIL=1
echo; echo "################ done $(date) FAIL=$FAIL   next: benchmark_judge.py --tags $TAG (laptop)"; exit $FAIL
