#!/usr/bin/env bash
# run_direct.sh —— 不经 Slurm、直接在 tmux 里跑（和 run_all.sbatch 步骤完全一样）
# 用法（项目根目录，先在 tmux 里）：
#   GPU=4 bash scripts/pipeline/slurm/run_direct.sh "scripts/pipeline/cases/w004/cl1_*.json" results/eval/w004/identity_cl1.csv
# 先用 nvidia-smi 确认该 GPU 显存基本空闲，再填 GPU=编号
CASES=${1:-scripts/pipeline/cases/**/*.json}
OUT=${2:-results/eval/all/identity_all.csv}
STEPS=${3:-flux2,av5,eval}
export CUDA_VISIBLE_DEVICES=${GPU:?请用 GPU=编号 指定一块空闲 GPU}

export HF_HOME=/data/lzha060/hf_cache
export TMPDIR=/data/lzha060/tmp; mkdir -p "$TMPDIR"
source /data/lzha060/miniconda3/etc/profile.d/conda.sh
cd /data/lzha060/animebenchmark
mkdir -p logs
LOG=logs/direct-$(date +%Y%m%d-%H%M%S).log
echo "GPU=$CUDA_VISIBLE_DEVICES CASES=$CASES OUT=$OUT STEPS=$STEPS  日志: $LOG"

{
if [[ ",$STEPS," == *",flux2,"* ]]; then
  echo "== [1/3] FLUX 生成 $(date)"; conda activate flux2
  python scripts/pipeline/generate_all.py --model flux2 --cases "$CASES"; conda deactivate
fi
if [[ ",$STEPS," == *",av5,"* ]]; then
  echo "== [2/3] AV5 生成 $(date)"; conda activate animebenchmark
  python scripts/pipeline/generate_all.py --model av5 --cases "$CASES"; conda deactivate
fi
if [[ ",$STEPS," == *",eval,"* ]]; then
  echo "== [3/3] 评估 $(date)"; conda activate flux2
  python scripts/pipeline/evaluate_all.py --cases "$CASES" --out "$OUT"; conda deactivate
fi
echo "== 结束 $(date)"
} 2>&1 | tee -a "$LOG"
