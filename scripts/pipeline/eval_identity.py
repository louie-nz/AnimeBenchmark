"""
eval_identity.py
评测 v0 —— 身份（Identity）指标：生成图头部 vs 参考图头部，CLIP-I 与 DINO-I。

设计要点（来自之前的讨论）：
  1. 只比较头部区域：身份由参考图负责；服装由 prompt 负责，用检查清单另评，不在这里比较。
  2. 头部检测用 Grounding DINO（文本提示 "head"），人和动物角色都能用；
     检测框按 1.4 倍外扩并取正方形，保证完整发型被包含（动漫角色的身份很大程度在头发上）。
  3. 统一尺寸：在原始分辨率上检测并裁出头部，再把头部统一缩放到 512×512 后比较，
     排除 FLUX(1024) 与 SD1.5(512) 分辨率不同、全身图与半身图头部大小不同的影响。
  4. CLIP-I：openai/clip-vit-large-patch14 的图像嵌入余弦相似度；
     DINO-I：facebook/dinov2-base 的 CLS 嵌入余弦相似度（对结构/外观更敏感）。
  5. 已知局限：嵌入相似度会受表情、角度、画风影响（例如 prompt 要求愤怒，脸本来就该和参考图不同）。
     因此它只作为身份的自动参考分；正式的身份判断以人工/VLM 的 0–4 评分为准，
     评分说明中写明"看发型、发色、脸型、眼睛颜色，忽略表情与姿势"。
  6. 每张图的头部裁剪都会保存到 crops/ 目录，请人工检查裁得对不对；检测失败的会标记 head_found=0。

用法（建议在 flux2 环境，transformers 较新）:
    conda activate flux2
    CUDA_VISIBLE_DEVICES=4 python scripts/eval_identity.py \
        --ref data/characters/p2_w004_du001_img02_head.png \
        --images "results/generated/du003_img01_*_b16_*.png" \
        --out results/eval/identity_b16_alice.csv

    CUDA_VISIBLE_DEVICES=4 python scripts/eval_identity.py \
        --ref data/characters/tb1_minguo_ref_head.png \
        --images "results/generated/tb1_*.png" \
        --out results/eval/identity_tb1_minguo.csv
"""
import argparse
import csv
import glob
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from transformers import (AutoModel, AutoModelForZeroShotObjectDetection,
                          AutoProcessor, CLIPImageProcessor, CLIPModel)

EVAL_SIZE = 512          # 统一尺寸
EXPAND = 1.4             # 头部框外扩倍数（包含完整发型）
DET_MODEL = "IDEA-Research/grounding-dino-tiny"
CLIP_MODEL = "openai/clip-vit-large-patch14"
DINO_MODEL = "facebook/dinov2-base"
KNOWN_MODELS = ["anythingv5_ipadapter", "anythingv5", "flux2klein4b"]

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- 头部检测与裁剪 ----------------
class HeadCropper:
    def __init__(self, text="a head."):
        self.text = text
        self.proc = AutoProcessor.from_pretrained(DET_MODEL)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(DET_MODEL).to(device).eval()

    @torch.no_grad()
    def detect(self, img):
        inputs = self.proc(images=img, text=self.text, return_tensors="pt").to(device)
        outputs = self.model(**inputs)
        size = [img.size[::-1]]
        try:   # 新版 transformers
            res = self.proc.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=0.3, text_threshold=0.25, target_sizes=size)[0]
        except TypeError:   # 旧版参数名
            res = self.proc.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=0.3, text_threshold=0.25, target_sizes=size)[0]
        if len(res["scores"]) == 0:
            return None, 0.0
        i = int(res["scores"].argmax())
        return res["boxes"][i].tolist(), float(res["scores"][i])

    def crop(self, img):
        box, score = self.detect(img)
        if box is None:
            return img, 0, 0.0, None
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        side = max(x1 - x0, y1 - y0) * EXPAND
        l, t = int(cx - side / 2), int(cy - side / 2)
        r, b = int(cx + side / 2), int(cy + side / 2)
        canvas = Image.new("RGB", (r - l, b - t), "white")          # 超出边界部分补白
        canvas.paste(img.crop((max(l, 0), max(t, 0), min(r, img.width), min(b, img.height))),
                     (max(-l, 0), max(-t, 0)))
        return canvas, 1, score, [round(v) for v in box]


# ---------------- 嵌入 ----------------
class Embedder:
    def __init__(self):
        self.clip_proc = CLIPImageProcessor.from_pretrained(CLIP_MODEL)   # 无 torchvision 时自动用 PIL 版本，结果相同
        self.clip = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
        # DINOv2 的预处理手工实现（与官方处理器一致：缩放到 256、中心裁 224、ImageNet 归一化），
        # 这样不依赖 torchvision，避免为装 torchvision 而改动环境里的 torch 版本
        self.dino = AutoModel.from_pretrained(DINO_MODEL).to(device).eval()

    @torch.no_grad()
    def clip_emb(self, img):
        px = self.clip_proc(images=img, return_tensors="pt")["pixel_values"].to(device)
        # 直接走 vision_model + visual_projection，避开不同 transformers 版本
        # get_image_features 返回格式不一致的问题（week9 遇到过）
        pooled = self.clip.vision_model(pixel_values=px).pooler_output
        return F.normalize(self.clip.visual_projection(pooled), dim=-1)

    @torch.no_grad()
    def dino_emb(self, img):
        im = img.convert("RGB").resize((256, 256), Image.BICUBIC).crop((16, 16, 240, 240))
        arr = np.asarray(im, dtype=np.float32) / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        px = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        out = self.dino(pixel_values=px)
        cls = out.pooler_output if getattr(out, "pooler_output", None) is not None \
            else out.last_hidden_state[:, 0]
        return F.normalize(cls, dim=-1)


def parse_name(name):
    seed = re.search(r"seed(\d+)", name)
    # 组别 = 1~3 个字母 + 数字（b16 / tb1 / qa1 / al1 ...），排除 du003、img01 这种数据编号
    group = re.search(r"(?:^|_)((?!du|img)[a-z]{1,3}\d+)(?:_|$)", name)
    model = next((m for m in KNOWN_MODELS if m in name), "unknown")
    return {
        "group": group.group(1) if group else "",
        "model": model,
        "ref_cond": "noref" if "noref" in name else "ref",
        "seed": seed.group(1) if seed else "",
    }


def load(path):
    # 保持原始分辨率去检测和裁剪头部（全身图里的头部本来就小，先缩小会损失细节），
    # 裁出来之后再统一缩放到 EVAL_SIZE，保证不同模型、不同分辨率的头部以相同尺寸比较
    return Image.open(path).convert("RGB")


def crop_head(cropper, img):
    crop, found, score, box = cropper.crop(img)
    return crop.resize((EVAL_SIZE, EVAL_SIZE), Image.LANCZOS), found, score, box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="参考图（头像）路径")
    ap.add_argument("--images", required=True, help="生成图的通配符路径，记得加引号")
    ap.add_argument("--out", required=True, help="输出 CSV 路径")
    ap.add_argument("--head-text", default="a head.", help="Grounding DINO 的检测文本")
    args = ap.parse_args()

    files = sorted(glob.glob(args.images))
    if not files:
        raise SystemExit(f"没有找到图片：{args.images}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop_dir = out_path.parent / "crops"
    crop_dir.mkdir(exist_ok=True)

    cropper = HeadCropper(args.head_text)
    emb = Embedder()

    ref_img = load(args.ref)
    ref_crop, ref_found, ref_score, _ = crop_head(cropper, ref_img)
    ref_crop.save(crop_dir / ("REF_" + Path(args.ref).stem + ".png"))
    ref_clip, ref_dino = emb.clip_emb(ref_crop), emb.dino_emb(ref_crop)
    print(f"参考图头部检测: found={ref_found} score={ref_score:.2f}")

    rows = []
    for f in files:
        try:
            img = load(f)
        except OSError as e:
            print(f"跳过（文件损坏或不完整）: {Path(f).name}  原因: {e}")
            continue
        crop, found, score, box = crop_head(cropper, img)
        crop.save(crop_dir / (Path(f).stem + "_head.png"))
        clip_i = float((emb.clip_emb(crop) @ ref_clip.T).item())
        dino_i = float((emb.dino_emb(crop) @ ref_dino.T).item())
        row = {"file": Path(f).name, **parse_name(Path(f).name),
               "head_found": found, "head_score": round(score, 3), "head_box": box,
               "clip_i": round(clip_i, 4), "dino_i": round(dino_i, 4)}
        rows.append(row)
        print(f"{row['file']}: head={found} clip_i={clip_i:.4f} dino_i={dino_i:.4f}")

    if not rows:
        raise SystemExit("没有可评分的图片")
    with open(out_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 按 组别 × 模型 × 参考图条件 汇总（各 seed 取平均）
    agg = defaultdict(list)
    for r in rows:
        agg[(r["group"], r["model"], r["ref_cond"])].append(r)
    print("\n===== 汇总（各 seed 平均）=====")
    for (g, m, rc), rs in sorted(agg.items()):
        n = len(rs)
        print(f"{g:5s} {m:22s} {rc:6s} n={n}  "
              f"CLIP-I={sum(r['clip_i'] for r in rs)/n:.4f}  "
              f"DINO-I={sum(r['dino_i'] for r in rs)/n:.4f}  "
              f"头部检测成功 {sum(r['head_found'] for r in rs)}/{n}")
    print(f"\n明细已保存: {out_path}\n头部裁剪图: {crop_dir}（请人工检查）")


if __name__ == "__main__":
    main()
