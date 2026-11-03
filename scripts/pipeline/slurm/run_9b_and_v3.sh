#!/bin/bash
# run_9b_and_v3.sh -- everything for the next step in ONE Slurm allocation (one A100 80GB on Server 1):
#   1. generate FLUX.2-klein-9B images for all 27 groups (ref + noref, seeds 42/123/777)    ~10-20 min
#   2. v3 rubric judge (QUESTIONS_v3.md) on the development set (4B + AV5, 162 ref images)  ~30-60 min
#   3. v3 rubric judge on the 9B ref images (test set; do NOT open these scores before scoring them by hand)
#   4. v1 yes/no judge (32B) on the 9B ref images, so v1 and v3 can be compared on the test set
#      (v2 needs Grounding-DINO head boxes from evaluate_all.py, which has not been run for 9B; v2 also failed its
#       pre-registered criterion, so it is not repeated)
#   5. CLIP-T for the 9B images
# Usage (in tmux on Server 1):
#   srun --gres=gpu:1 --cpus-per-task=4 --mem=96G --time=04:00:00 \
#     bash /data/lzha060/animebenchmark/scripts/pipeline/slurm/run_9b_and_v3.sh \
#     2>&1 | tee /data/lzha060/animebenchmark/logs/run_9b_v3_$(date +%m%d_%H%M).log
# Steps can be skipped: SKIP_GEN=1, SKIP_DEV=1, SKIP_TEST=1
export HOME=/data/lzha060 TMPDIR=/data/lzha060/tmp HF_HOME=/data/lzha060/hf_cache
export http_proxy=http://squid.auckland.ac.nz:3128 https_proxy=http://squid.auckland.ac.nz:3128 PYTHONUNBUFFERED=1
source /data/lzha060/miniconda3/etc/profile.d/conda.sh && conda activate flux2
cd /data/lzha060/animebenchmark || exit 1
CASES="scripts/pipeline/cases/w004/cl1*.json,scripts/pipeline/cases/w006/*.json,scripts/pipeline/cases/w014/*.json,scripts/pipeline/cases/w022/*.json,scripts/pipeline/cases/w029/*.json,scripts/pipeline/cases_legacy/w004/*.json"
JUDGE=Qwen/Qwen2.5-VL-32B-Instruct
FAIL=0
step() { echo; echo "################ $1  $(date)"; }

if [ "${SKIP_GEN:-0}" != "1" ]; then
  step "1 generate FLUX.2-klein-9B"
  python scripts/pipeline/generate_all.py --model flux2_9b --cases "$CASES" || FAIL=1
fi
if [ "${SKIP_DEV:-0}" != "1" ]; then
  step "2 v3 rubric judge, development set (4B + AV5)"
  MODEL_TAGS=flux2klein4b,anythingv5_ipadapter python scripts/pipeline/evaluate_rubric.py --cases "$CASES" \
    --out results/eval/vlm32b_v3/vlm_dev.csv --model-path $JUDGE --batch 2 || FAIL=1
fi
if [ "${SKIP_TEST:-0}" != "1" ]; then
  step "3 v3 rubric judge, test set (9B)"
  MODEL_TAGS=flux2klein9b python scripts/pipeline/evaluate_rubric.py --cases "$CASES" \
    --out results/eval/test9b/vlm32b_v3.csv --model-path $JUDGE --batch 2 || FAIL=1
  for Q in v1; do
    step "4 $Q yes/no judge, test set (9B)"
    MODEL_TAGS=flux2klein9b python scripts/pipeline/evaluate_vlm.py --cases "$CASES" \
      --out results/eval/test9b/vlm32b_$Q.csv --model-path $JUDGE --batch 2 --qset $Q || FAIL=1
  done
  step "5 CLIP-T, 9B images"
  MODEL_TAGS=flux2klein9b python scripts/pipeline/clip_t.py --cases "$CASES" --out results/eval/test9b/clip_t_9b.csv || FAIL=1
fi
echo; echo "################ done $(date)  FAIL=$FAIL"
exit $FAIL
