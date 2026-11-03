# Oz（w006）5 组测试包

## 放置

* oz\_dorothy\_ref\_head.png、oz\_lion\_ref\_head.png、oz\_wizard\_ref\_head.png -> data/characters/
（Task B 用的 tb1\_minguo\_ref\_head.png 已在 data/characters/，沿用）
* cases/w006/ 整个文件夹 -> scripts/pipeline/cases/w006/
* 脚本不用改（generate\_all.py / evaluate\_all.py / run\_direct.sh 都是现成的）

## 先检查 prompt（不加载模型）

cd /data/lzha060/animebenchmark
python scripts/pipeline/generate\_all.py --model flux2 --cases "scripts/pipeline/cases/w006/\*.json" --dry-run

## 运行（tmux 里；先 nvidia-smi 选一块显存充足的卡）

GPU=3 bash scripts/pipeline/slurm/run\_direct.sh "scripts/pipeline/cases/w006/\*.json" results/eval/w006/identity\_w006.csv

5 组 × 2 模型 × ref/noref × 3 seed = 60 张图
输出：results/generated/w006/{od1,od2,ol1,ow1,ob1}\_files/
评测：results/eval/w006/identity\_w006.csv，裁剪图 results/eval/w006/crops/

## 5 组

| 组 | 内容 | 参考头像 -> 目标 | 类型 |
| od1 | Dorothy L1 | du031 img02 -> img01 | Q版 anime |
| od2 | Dorothy L2 战士 | du031 头像 -> du055 | Q版 anime |
| ol1 | 胆小狮子 L1 | du065 img02 -> img01 | 非人 Q版 cartoon |
| ow1 | 奥兹巫师 L1 | du013 img01 -> img02 | cartoon（正常比例、老年男性）|
| ob1 | Task B 民国女学生 | 民国头像 -> 翡翠城访客 | 新角色 |

## 有记录的改动（都写在各 JSON 的 notes 里）

* ol1：prompt 里 "Princess Ozma" -> "the Cowardly Lion"；"biting" -> "hugging his own tail"（以图为准）
* ow1：中文 prompt 翻译成英文（AV5 读不懂中文）
* ob1：Task B prompt 按固定模板自写，世界观元素均为原著设定（绿色眼镜、黄砖路、翡翠城门），数据集 du030、du012 也有对应描绘
* od1、ob1 在 SD1.5 下可能超过 77 token，生成时会打印提示，按规则记为模型局限

