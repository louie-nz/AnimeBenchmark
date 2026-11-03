#!/bin/bash
# run_vlm.sh -- VLM scoring (Qwen2.5-VL) for one or more worlds, inside ONE Slurm allocation.
# Usage (in tmux, from anywhere):
#   srun --gres=gpu:1 --cpus-per-task=4 --mem=48G --time=01:00:00 \
#     bash /data/lzha060/animebenchmark/scripts/pipeline/slurm/run_vlm.sh w006 w004 \
#     2>&1 | tee /data/lzha060/animebenchmark/logs/vlm_$(date +%m%d_%H%M).log
# World names: w006 ... or legacy_w004 (= scripts/pipeline/cases_legacy/w004).
# CLIP-T runs at the end (CLIPT=0 to skip).
# Optional: QSET=v2  OUTDIR=...  BATCH=2  COND=ref,noref  MODEL=/path/to/local/Qwen2.5-VL-7B-Instruct  bash run_vlm.sh ...
export HOME=/data/lzha060
export TMPDIR=/data/lzha060/tmp
export HF_HOME=/data/lzha060/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export http_proxy=http://squid.auckland.ac.nz:3128 https_proxy=http://squid.auckland.ac.nz:3128
export PYTHONUNBUFFERED=1
source /data/lzha060/miniconda3/etc/profile.d/conda.sh
conda activate flux2
cd /data/lzha060/animebenchmark || exit 1
COND=${COND:-ref}
MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
OUTDIR=${OUTDIR:-results/eval/vlm}      # e.g. OUTDIR=results/eval/vlm32b so the 7B results are kept
QSET=${QSET:-v1}                         # v1 (first run), v2 (pre-registered yes/no redesign), v3 (pre-registered rubric judge)
BATCH=${BATCH:-8}                        # questions per forward pass; use 2 for the 32B model
mkdir -p "$OUTDIR"
echo "model=$MODEL  outdir=$OUTDIR  batch=$BATCH  qset=$QSET"
FAIL=0
for W in "$@"; do
  echo "################ VLM $W  $(date)"
  CASES="scripts/pipeline/cases/$W/*.json"
  case "$W" in legacy_*) CASES="scripts/pipeline/cases_legacy/${W#legacy_}/*.json" ;; esac   # e.g. legacy_w004
  if [ "$QSET" = "v3" ]; then        # v3 = rubric judge (evaluate_rubric.py), see QUESTIONS_v3.md
    python scripts/pipeline/evaluate_rubric.py --cases "$CASES" \
      --out "$OUTDIR/vlm_$W.csv" --cond "$COND" --model-path "$MODEL" --batch "$BATCH" || FAIL=1
  else
    python scripts/pipeline/evaluate_vlm.py --cases "$CASES" \
      --out "$OUTDIR/vlm_$W.csv" --cond "$COND" --model-path "$MODEL" --batch "$BATCH" --qset "$QSET" || FAIL=1
  fi
done
# CLIP-T (image <-> prompt) for every case, ref + noref; a few minutes, runs after the VLM so it cannot block it
if [ "${CLIPT:-1}" = "1" ]; then
  echo "################ CLIP-T  $(date)"
  python scripts/pipeline/clip_t.py \
    --cases "scripts/pipeline/cases/*/*.json,scripts/pipeline/cases_legacy/**/*.json" \
    --out results/eval/clip_t.csv || FAIL=1
fi
echo "################ done $(date)  FAIL=$FAIL"
exit $FAIL
