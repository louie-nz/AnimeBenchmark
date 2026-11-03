"""
generate_all.py —— 总生成脚本：读取 cases/*.json，一个模型加载一次，跑完所有 case × ref/noref × seed。

每个 case 文件包含：组名、参考头像、原 prompt 拆分字段(content)、各模型模板(templates)、原 negative。
新增测试组 = 只加一个 JSON 文件，不用再写脚本。

用法（项目根目录下）：
    export HF_HOME=/data/lzha060/hf_cache

    conda activate flux2
    python scripts/pipeline/generate_all.py --model flux2   # 正式运行请用 slurm/run_all.sbatch 提交

    conda activate animebenchmark
    python scripts/pipeline/generate_all.py --model av5

常用参数：
    --cases "scripts/pipeline/cases/w004/*.json"  只跑某个世界或某几个 case（默认 cases/ 下全部，含子文件夹）
    --cond ref            只跑 ref（默认 ref,noref 都跑）
    --seeds 42,123,777
    --dry-run             只打印 prompt 和输出路径，不加载模型（先检查用）
    已存在的输出图会自动跳过，中断后重跑不会重复生成；加 --overwrite 强制重画。

输出：results/generated/{world}/{group}_files/{group}_{model_tag}_{character}_{ref|noref}_seed{n}.png
"""
import argparse
import glob
import json
from pathlib import Path

ROOT = Path("/data/lzha060/animebenchmark")
CASE_DIR = ROOT / "scripts/pipeline/cases"
REF_DIR = ROOT / "data/characters"
OUT_ROOT = ROOT / "results/generated"

FLUX_DIR = ROOT / "models/FLUX.2-klein-4B"
AV5_PATH = ROOT / "models/AnythingV5V3_v5PrtRE.safetensors"
IPA_DIR = ROOT / "ip_adapter_plus_anime"
IPA_WEIGHT = "ip-adapter-plus_sd15.bin"
IPA_SCALE = 0.6

MODEL_TAG = {"flux2": "flux2klein4b", "av5": "anythingv5_ipadapter"}
SD_EXTRA_NEG = (", bad anatomy, worst quality, extra fingers, lowres, fused legs, extra legs, "
                "malformed legs, disconnected limbs, floating limbs")


def load_cases(pattern):
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise SystemExit(f"没有找到 case 文件：{pattern}")
    return [json.load(open(f, encoding="utf-8")) for f in files]


def build_prompt(case, model, cond):
    t = case["templates"]
    key = "sd" if model == "av5" else ("flux_ref" if cond == "ref" else "flux_noref")
    return t[key].format(**case["content"])


def out_path(case, model, cond, seed):
    d = OUT_ROOT / case["world"] / f"{case['group']}_files"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{case['group']}_{MODEL_TAG[model]}_{case['character']}_{cond}_seed{seed}.png"


def jobs(cases, model, conds, seeds):
    for cond in conds:            # 先跑完所有 ref，再跑 noref（AV5 需要在两者之间卸载 IP-Adapter）
        for c in cases:
            for s in seeds:
                yield c, cond, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["flux2", "av5"])
    ap.add_argument("--cases", default=str(CASE_DIR / "**" / "*.json"))
    ap.add_argument("--cond", default="ref,noref")
    ap.add_argument("--seeds", default="42,123,777")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    cases = load_cases(a.cases)
    conds = [x.strip() for x in a.cond.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    todo = [(c, cond, s) for c, cond, s in jobs(cases, a.model, conds, seeds)
            if a.overwrite or not out_path(c, a.model, cond, s).exists()]
    print(f"{len(cases)} 个 case，{len(todo)} 张待生成（已存在的跳过）")

    for c in cases:
        for cond in conds:
            p = build_prompt(c, a.model, cond)
            print(f"\n[{c['group']} {cond}] {p}")
    if a.dry_run or not todo:
        return

    import torch
    from diffusers.utils import load_image

    if a.model == "flux2":
        from diffusers import Flux2KleinPipeline
        pipe = Flux2KleinPipeline.from_pretrained(str(FLUX_DIR), torch_dtype=torch.bfloat16).to("cuda")
    else:
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_single_file(str(AV5_PATH), torch_dtype=torch.float16,
                                                        safety_checker=None).to("cuda")
        tok = pipe.tokenizer
    ipa_loaded = False

    for c, cond, seed in todo:
        prompt = build_prompt(c, a.model, cond)
        g = torch.Generator("cuda").manual_seed(seed)
        ref = load_image(str(REF_DIR / c["ref_head"])) if cond == "ref" else None

        if a.model == "flux2":
            kw = dict(prompt=prompt, height=1024, width=1024, guidance_scale=1.0,
                      num_inference_steps=4, generator=g)
            if ref is not None:
                kw["image"] = ref
        else:
            if cond == "ref" and not ipa_loaded:
                pipe.load_ip_adapter(str(IPA_DIR), subfolder="models", weight_name=IPA_WEIGHT)
                pipe.set_ip_adapter_scale(IPA_SCALE)
                ipa_loaded = True
            if cond == "noref" and ipa_loaded:
                pipe.unload_ip_adapter()
                ipa_loaded = False
            neg = c["negative_original"] + SD_EXTRA_NEG
            n_tok = len(tok(prompt).input_ids)
            if n_tok > 77:
                print(f"  注意：prompt {n_tok} token > 77，尾部被截断（记为 SD1.5 局限）")
            kw = dict(prompt=prompt, negative_prompt=neg, num_inference_steps=30, generator=g)
            if ref is not None:
                kw["ip_adapter_image"] = ref

        img = pipe(**kw).images[0]
        p = out_path(c, a.model, cond, seed)
        img.save(p)
        print("Saved:", p)


if __name__ == "__main__":
    main()
