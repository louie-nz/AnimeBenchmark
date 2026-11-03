#!/bin/bash
# run_worlds.sh -- run several worlds in ONE Slurm allocation, one world after another.
# Each world gets its own CSV: results/eval/{world}/identity_{world}.csv
# Usage (in tmux, from the project root):
#   export PYTHONUNBUFFERED=1
#   srun --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:40:00 \
#     bash scripts/pipeline/slurm/run_worlds.sh w014 w022 w029 2>&1 | tee logs/worlds_$(date +%m%d_%H%M).log
cd /data/lzha060/animebenchmark || exit 1
FAIL=0
for W in "$@"; do
  echo "################ $W  $(date)"
  mkdir -p results/eval/$W
  bash scripts/pipeline/slurm/run_all.sbatch "scripts/pipeline/cases/$W/*.json" "results/eval/$W/identity_$W.csv" || FAIL=1
done
echo "################ all done $(date)  FAIL=$FAIL"
exit $FAIL
