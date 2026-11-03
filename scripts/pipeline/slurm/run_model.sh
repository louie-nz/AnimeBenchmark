#!/bin/bash
# run_model.sh MODEL_KEY TAG -- generate one new generator and evaluate it exactly like FLUX.2-klein-9B, in ONE
# Slurm allocation. Same cases, same prompts, same seeds, same judges (Qwen2.5-VL-32B v1 yes/no + v3 rubric), CLIP-T.
#   srun --gres=gpu:1 --cpus-per-task=4 --mem=128G --time=06:00:00 \
#     bash /data/lzha060/animebenchmark/scripts/pipeline/slurm/run_model.sh qwen qwenimage \
#     2>&1 | tee /data/lzha060/animebenchmark/logs/run_qwenimage_$(date +%m%d_%H%M).log
# ENV=flux2 (conda env for generation; the judges always use flux2).  SKIP_GEN=1 to only evaluate
# (e.g. for GPT Image 2, whose images are generated on the laptop through the API and uploaded).
KEY=$1; TAG=$2
[ -z "$TAG" ] && { echo "用法: run_model.sh MODEL_KEY TAG"; exit 1; }
export HOME=/data/lzha060 TMPDIR=/data/lzha060/tmp HF_HOME=/data/lzha060/hf_cache
export http_proxy=http://squid.auckland.ac.nz:3128 https_proxy=http://squid.auckland.ac.nz:3128 PYTHONUNBUFFERED=1
source /data/lzha060/miniconda3/etc/profile.d/conda.sh
cd /data/lzha060/animebenchmark || exit 1
CASES="scripts/pipeline/cases/w004/cl1*.json,scripts/pipeline/cases/w006/*.json,scripts/pipeline/cases/w014/*.json,scripts/pipeline/cases/w022/*.json,scripts/pipeline/cases/w029/*.json,scripts/pipeline/cases_legacy/w004/*.json"
JUDGE=Qwen/Qwen2.5-VL-32B-Instruct
OUT=results/eval/models/$TAG; mkdir -p $OUT
FAIL=0
step() { echo; echo "################ $1  $(date)"; }
if [ "${SKIP_GEN:-0}" != "1" ]; then
  conda activate ${ENV:-flux2}
  step "1 generate $KEY ($TAG)"
  python scripts/pipeline/generate_all.py --model $KEY --cases "$CASES" || FAIL=1
fi
conda activate flux2
step "2 v3 rubric judge ($TAG)"
MODEL_TAGS=$TAG python scripts/pipeline/evaluate_rubric.py --cases "$CASES" --out $OUT/vlm32b_v3.csv --model-path $JUDGE --batch 2 || FAIL=1
step "3 v1 yes/no judge ($TAG)"
MODEL_TAGS=$TAG python scripts/pipeline/evaluate_vlm.py --cases "$CASES" --out $OUT/vlm32b_v1.csv --model-path $JUDGE --batch 2 --qset v1 || FAIL=1
step "4 CLIP-T ($TAG)"
MODEL_TAGS=$TAG python scripts/pipeline/clip_t.py --cases "$CASES" --out $OUT/clip_t.csv || FAIL=1
echo; echo "################ done $(date)  FAIL=$FAIL"
exit $FAIL
