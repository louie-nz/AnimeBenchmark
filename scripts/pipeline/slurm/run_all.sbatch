#!/usr/bin/env bash
# ======== SLURM Job Configuration（参照 MMMU 跑通过的配置）========
#SBATCH --job-name=animebench
#SBATCH --time=06:00:00
#SBATCH --open-mode=append
#SBATCH --output=/data/lzha060/animebenchmark/logs/%x-%j.log
#SBATCH --error=/data/lzha060/animebenchmark/logs/%x-%j.err
#SBATCH --gres=gpu:1                       # 1 块 GPU（规定）；Server 1 = A100 80GB，Server 3 = H200
#SBATCH --partition=slurmpartition
#SBATCH --mem=128G                         # Qwen-Image（20B）和 32B 评审需要较大内存
#SBATCH --cpus-per-task=4
#
# 一个作业、一块 GPU，按顺序跑完所选步骤（2026-10-31 版，所有生成模型用同一套 case / prompt / seed / 评估）：
#   生成：flux2（FLUX.2-klein-4B）、flux2_9b（9B）、qwen（Qwen-Image-Edit-2511 / 2512）、sd35（SD3.5-Large + IP-Adapter）、av5
#   评估：eval（CLIP-I / DINO-I）、vlm1（Qwen2.5-VL-32B 是/否评审）、vlm3（32B rubric 评审）、clipt（CLIP-T）
#   评估步骤只评 TAGS 里的模型（环境变量，默认全部），例如 TAGS=flux2klein9b,qwenimage
#   GPT Image 2 的图在笔记本上用 gen_openai.py 生成后上传，这里只跑评估步骤。
# 用法（登录节点、项目根目录；第一次先 mkdir -p logs，否则 Slurm 写不了日志、作业会直接失败）：
#   sbatch scripts/pipeline/slurm/run_all.sbatch                                   # cases/ 下全部世界
#   sbatch scripts/pipeline/slurm/run_all.sbatch "scripts/pipeline/cases/w004/*.json" results/eval/w004/identity_w004.csv
#   sbatch scripts/pipeline/slurm/run_all.sbatch "scripts/pipeline/cases/w004/cl1_*.json" results/eval/w004/identity_cl1.csv
# 第 3 个参数 STEPS 可选，flux2,av5,eval 的任意组合（默认全部），用于只补跑某一步，例如只重做评估：
#   sbatch scripts/pipeline/slurm/run_all.sbatch "scripts/pipeline/cases/w004/cl1_*.json" results/eval/w004/identity_cl1.csv eval
# 给所有新模型补身份分（推荐，最后统一跑一次）：
#   TAGS=flux2klein9b,qwenimage,gptimage2,sd35ipa sbatch --export=ALL scripts/pipeline/slurm/run_all.sbatch \
#     "scripts/pipeline/cases/w004/cl1*.json,scripts/pipeline/cases/w0[0-9][0-9]/*.json,scripts/pipeline/cases_legacy/w004/*.json" \
#     results/eval/new_models/identity_new.csv eval
# 也可以用 srun 交互式跑同一个脚本（#SBATCH 行会被忽略）：srun --gres=gpu:1 --mem=128G --time=06:00:00 bash scripts/pipeline/slurm/run_all.sbatch ...

CASES=${1:-scripts/pipeline/cases/**/*.json}
OUT=${2:-results/eval/all/identity_all.csv}
STEPS=${3:-flux2,av5,eval}
TAGS=${TAGS:-flux2klein4b,anythingv5_ipadapter,flux2klein9b,qwenimage,gptimage2,sd35ipa}
EVDIR=$(dirname "$OUT")
JUDGE=Qwen/Qwen2.5-VL-32B-Instruct

# ======== 环境（集群规定：HOME/TMPDIR 放 /data，经 squid 代理）========
export HOME=/data/lzha060
export TMPDIR=/data/lzha060/tmp; mkdir -p "$TMPDIR"
export HF_HOME=/data/lzha060/hf_cache
export TRANSFORMERS_CACHE=/data/lzha060/hf_cache
export HUGGINGFACE_HUB_CACHE=/data/lzha060/hf_cache
export http_proxy=http://squid.auckland.ac.nz:3128
export https_proxy=http://squid.auckland.ac.nz:3128
source /data/lzha060/miniconda3/etc/profile.d/conda.sh
cd /data/lzha060/animebenchmark

echo "== Job $SLURM_JOB_ID on $(hostname)  GPU=$CUDA_VISIBLE_DEVICES  start $(date)"
echo "== CASES=$CASES  OUT=$OUT  STEPS=$STEPS  TAGS=$TAGS"
FAIL=0

if [[ ",$STEPS," == *",flux2,"* ]]; then
  echo "== FLUX.2-klein-4B 生成 $(date)"
  conda activate flux2
  python scripts/pipeline/generate_all.py --model flux2 --cases "$CASES" || FAIL=1
  conda deactivate
fi

for KEY in flux2_9b qwen sd35; do
  if [[ ",$STEPS," == *",$KEY,"* ]]; then
    echo "== 生成 $KEY $(date)"
    ENVK=flux2; [ "$KEY" = "qwen" ] && ENVK=${QWEN_ENV:-flux2}      # QWEN_ENV=qwenimg if a separate env was needed
    conda activate $ENVK
    python scripts/pipeline/generate_all.py --model $KEY --cases "$CASES" || FAIL=1
    conda deactivate
  fi
done

if [[ ",$STEPS," == *",av5,"* ]]; then
  echo "== Anything V5 + IP-Adapter 生成 $(date)"
  conda activate animebenchmark
  python scripts/pipeline/generate_all.py --model av5 --cases "$CASES" || FAIL=1
  conda deactivate
fi

if [[ ",$STEPS," == *",eval,"* ]]; then
  echo "== 身份评估 CLIP-I / DINO-I $(date)"
  conda activate flux2
  MODEL_TAGS=$TAGS python scripts/pipeline/evaluate_all.py --cases "$CASES" --out "$OUT" --tags "$TAGS" || FAIL=1
  conda deactivate
fi

if [[ ",$STEPS," == *",vlm1,"* ]]; then
  echo "== VLM v1（是/否）$(date)"; conda activate flux2
  MODEL_TAGS=$TAGS python scripts/pipeline/evaluate_vlm.py --cases "$CASES" --out "$EVDIR/vlm32b_v1.csv" --model-path $JUDGE --batch 2 --qset v1 || FAIL=1
  conda deactivate
fi
if [[ ",$STEPS," == *",vlm3,"* ]]; then
  echo "== VLM v3（rubric）$(date)"; conda activate flux2
  MODEL_TAGS=$TAGS python scripts/pipeline/evaluate_rubric.py --cases "$CASES" --out "$EVDIR/vlm32b_v3.csv" --model-path $JUDGE --batch 2 || FAIL=1
  conda deactivate
fi
if [[ ",$STEPS," == *",clipt,"* ]]; then
  echo "== CLIP-T $(date)"; conda activate flux2
  MODEL_TAGS=$TAGS python scripts/pipeline/clip_t.py --cases "$CASES" --out "$EVDIR/clip_t.csv" || FAIL=1
  conda deactivate
fi

echo "== 结束 $(date)  FAIL=$FAIL"
exit $FAIL
