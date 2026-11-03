"""
evaluate_all.py —— 身份自动分 CLIP-I / DINO-I（Grounding DINO 头部裁剪），所有生成模型用同一套方法。
2026-10-31 更新：
  * 生成模型由文件名里的标签识别（MODEL_TAGS，与 evaluate_vlm.py 相同），新模型（FLUX.2-klein-9B、Qwen-Image、
    GPT Image 2、SD3.5+IP-Adapter）自动计入；--tags 只评指定模型，已有的 4B / AV5 结果不必重算。
  * 旧版 Wonderland case（cases_legacy，image_glob）也能评：先找旧路径，找不到再找标准路径。
  * 参考头像按 ref_head_candidates 依次查找（data/characters，再 data/**）。
  方法、模型、裁剪参数完全不变，所以新旧结果可以直接比较。
读取 cases/*.json，自动知道每组用哪个参考头像、哪种检测文本，一次评完所有组，写一张总表。
以后 VQAScore、VLM 评分也加在这里。

用法（项目根目录，flux2 环境）：
    conda activate flux2
    export HF_HOME=/data/lzha060/hf_cache
    python scripts/pipeline/evaluate_all.py        # 正式运行请用 slurm/run_all.sbatch 提交
    # 只评某几组：--cases "scripts/pipeline/cases/w004/cl1_*.json"

输出：
    results/eval/all/identity_all.csv        每张图一行（含 group / model / ref_cond / seed）
    results/eval/all/crops/{group}/          头部裁剪图，请人工抽查
"""
import argparse
import csv
import os
import re
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_identity import HeadCropper, Embedder, crop_head, load, parse_name  # noqa: E402

ROOT = Path(os.environ.get("ANIMEBENCH_ROOT", "/data/lzha060/animebenchmark"))
MODEL_TAGS = os.environ.get("MODEL_TAGS", "flux2klein4b,anythingv5_ipadapter,flux2klein9b,qwenimage,gptimage2,sd35ipa").split(",")
CASE_DIR = ROOT / "scripts/pipeline/cases"
REF_DIR = ROOT / "data/characters"
GEN_ROOT = ROOT / "results/generated"


def find_ref(c):
    for n in c.get("ref_head_candidates") or [c["ref_head"]]:
        if (REF_DIR / n).exists():
            return REF_DIR / n
    for n in c.get("ref_head_candidates") or [c["ref_head"]]:
        hits = sorted(glob.glob(str(ROOT / "data" / "**" / n), recursive=True))
        if hits:
            return Path(hits[0])
    return None


def find_images(c, tag):
    """Same lookup as evaluate_vlm.find_files: legacy image_glob first, then the standard layout; ref + noref."""
    out = []
    for cond in ("ref", "noref"):
        pats = []
        g = (c.get("image_glob") or {}).get(cond, "__none__") if "image_glob" in c else "__none__"
        if g not in ("__none__", None):
            pats.append(str(ROOT / g.format(tag=tag, cond=cond)))
        if g is not None:
            pats.append(str(GEN_ROOT / c["world"] / f"{c['group']}_files" / f"{c['group']}_{tag}_{c['character']}_{cond}_seed*.png"))
        for p in pats:
            hits = sorted(f for f in glob.glob(p, recursive=True) if re.search(r"seed\d+\.png$", f))
            if hits:
                out += hits
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(CASE_DIR / "**" / "*.json"))
    ap.add_argument("--out", default=str(ROOT / "results/eval/all/identity_all.csv"))
    ap.add_argument("--tags", default=",".join(MODEL_TAGS), help="only these generator tags, comma-separated")
    a = ap.parse_args()
    tags = [t.strip() for t in a.tags.split(",") if t.strip()]

    cfiles = []
    for p in [x.strip() for x in a.cases.split(",") if x.strip()]:
        cfiles += sorted(glob.glob(p, recursive=True))
    cases = [json.load(open(f, encoding="utf-8")) for f in dict.fromkeys(cfiles)]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    emb = Embedder()
    croppers = {}
    rows = []

    for c in cases:
        text = c.get("head_text", "a head.")
        if text not in croppers:
            croppers[text] = HeadCropper(text)
        cropper = croppers[text]
        crop_dir = out.parent / "crops" / c["group"]
        crop_dir.mkdir(parents=True, exist_ok=True)

        ref_path = find_ref(c)
        if ref_path is None:
            print(f"!! {c['group']}: 找不到参考头像，跳过")
            continue
        ref_crop, ref_found, ref_score, _ = crop_head(cropper, load(ref_path))
        ref_crop.save(crop_dir / f"REF_{c['group']}.png")
        ref_clip, ref_dino = emb.clip_emb(ref_crop), emb.dino_emb(ref_crop)

        files = [f for t in tags for f in find_images(c, t)]
        print(f"\n== {c['group']} ({c['character']}, {c['style_type']}, {c['level']}): {len(files)} 张图")
        for f in files:
            try:
                img = load(f)
            except OSError as e:
                print("跳过（文件损坏）:", Path(f).name, e)
                continue
            crop, found, score, box = crop_head(cropper, img)
            crop.save(crop_dir / (Path(f).stem + "_head.png"))
            info = parse_name(Path(f).name)
            name = Path(f).name                      # model / condition / seed from the file name, for every generator
            info["model"] = next(t for t in tags if t in name)
            info["ref_cond"] = "noref" if "_noref_" in name else "ref"
            info["seed"] = re.search(r"seed(\d+)\.png$", name).group(1)
            info["group"] = c["group"]
            rows.append({"file": Path(f).name, **info, "character": c["character"],
                         "style_type": c["style_type"], "level": c["level"],
                         "head_found": found, "head_score": round(score, 3), "head_box": box,
                         "clip_i": round(float((emb.clip_emb(crop) @ ref_clip.T).item()), 4),
                         "dino_i": round(float((emb.dino_emb(crop) @ ref_dino.T).item()), 4)})

    if not rows:
        raise SystemExit("没有可评分的图片")
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    agg = defaultdict(list)
    for r in rows:
        agg[(r["group"], r["model"], r["ref_cond"])].append(r)
    print("\n===== 汇总（各 seed 平均）=====")
    for (g, m, rc), rs in sorted(agg.items()):
        n = len(rs)
        print(f"{g:5s} {m:22s} {rc:6s} n={n}  CLIP-I={sum(r['clip_i'] for r in rs)/n:.4f}  "
              f"DINO-I={sum(r['dino_i'] for r in rs)/n:.4f}  头部检测 {sum(r['head_found'] for r in rs)}/{n}")
    print(f"\n明细：{out}\n裁剪图：{out.parent / 'crops'}")


if __name__ == "__main__":
    main()
