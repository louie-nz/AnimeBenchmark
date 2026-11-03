# 总生成 / 总评估流程（cases/*.json）

本包含两个新组（Wonderland）：
- gl1 扑克正义卫士 du074（L1，anime）：ref = img01（持卡牌盾）头部，目标 = img02（持卡牌剑）
- cl1 柴郡猫 du087（L1，非人 Q版anime）：ref = img01（睁眼露齿笑）头部，目标 = img02（闭眼、漂浮、身体渐隐）

## 放置
- wl1_guard_ref_head.png、wl1_cat_ref_head.png -> data/characters/
- generate_all.py、evaluate_all.py、eval_identity.py -> scripts/pipeline/
- cases/ 整个文件夹 -> scripts/pipeline/cases/（按世界分子文件夹：cases/w004/、以后 cases/w006/ ...）

## 1. 先检查 prompt（不加载模型）
cd /data/lzha060/animebenchmark
python scripts/pipeline/generate_all.py --model flux2 --dry-run

## 2. 生成（每个模型一条命令，自动跑完所有 case × ref/noref × 3 seed）
export HF_HOME=/data/lzha060/hf_cache

conda activate flux2
CUDA_VISIBLE_DEVICES=4 python scripts/pipeline/generate_all.py --model flux2

conda activate animebenchmark
CUDA_VISIBLE_DEVICES=4 python scripts/pipeline/generate_all.py --model av5

输出：results/generated/w004/gl1_files/、results/generated/w004/cl1_files/
已存在的图会跳过；中断后重跑即可续上。

## 3. 评估（flux2 环境，一次评完所有组）
conda activate flux2
CUDA_VISIBLE_DEVICES=4 python scripts/pipeline/evaluate_all.py

输出：results/eval/all/identity_all.csv，裁剪图在 results/eval/all/crops/{group}/

## 以后加新组
只需在 cases/<世界>/ 里加一个 JSON（复制现有文件改字段），不用写新脚本。

## 两组的注意事项
- gl1：原 prompt 写 "3-head chibi"，但原图是正常比例 anime（Midjourney 没按头身比画）。
  这组实际在测"正常 anime 参考图 + Q版 prompt"，看模型跟哪个；身份评分不因比例变化扣分。
  参考头像皇冠上方有一小块红色残留（光环抠图残留）。
- cl1：prompt 写 "purple and gray"、"large green eyes"，参考图是粉紫色、紫黄色眼睛。
  眼睛颜色是很干净的检验：绿眼 = 跟 prompt，紫眼 = 跟参考图。
  头部检测文本用 "a cat head."

## 正式运行：Slurm（学校规定计算必须用 sbatch 提交）
slurm/run_all.sbatch 放到 scripts/pipeline/slurm/。一个作业、一块 GPU，按顺序：
FLUX 生成（ref+noref）-> AV5 生成（ref+noref）-> 身份评估。两个 conda 环境在作业里依次切换。

cd /data/lzha060/animebenchmark
mkdir -p logs                                   # 只需第一次；没有这个文件夹 Slurm 会直接失败

# 只跑柴郡猫
sbatch scripts/pipeline/slurm/run_all.sbatch "scripts/pipeline/cases/w004/cl1_*.json" results/eval/w004/identity_cl1.csv
# 跑整个仙境
sbatch scripts/pipeline/slurm/run_all.sbatch "scripts/pipeline/cases/w004/*.json" results/eval/w004/identity_w004.csv
# 只补跑某一步（第 3 个参数：flux2 / av5 / eval 的组合）
sbatch scripts/pipeline/slurm/run_all.sbatch "scripts/pipeline/cases/w004/cl1_*.json" results/eval/w004/identity_cl1.csv eval

squeue -u $USER
tail -f logs/animebench-<作业号>.log

说明：
- 不要再写 CUDA_VISIBLE_DEVICES=4；Slurm 用 --gres 分配 GPU
- 默认申请 H200（gpu:h200:1，和 MMMU 一样）；在 A100 服务器上改成 gpu:1
- 已生成的图自动跳过，作业中断后重新提交即可续上
