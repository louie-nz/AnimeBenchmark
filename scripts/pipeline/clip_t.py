"""
clip_t.py -- CLIP-T: cosine similarity between a generated image and its text prompt (image <-> text).
Complements CLIP-I / DINO-I (image <-> reference head) as a cheap prompt-following baseline.

Two text variants per case (both truncated by CLIP at 77 tokens, as in the SD1.5 pipeline itself):
  clip_t_sd    the SD template filled with the case content (same text AV5 was given)
  clip_t_desc  the FLUX no-ref template filled with the content (plain description, no edit instruction)

Uses the same case files and image lookup as evaluate_vlm.py (must sit in the same folder).
    python scripts/pipeline/clip_t.py --cases "scripts/pipeline/cases/*/*.json" --out results/eval/clip_t.csv
    --cond ref,noref (default)   --model-path openai/clip-vit-large-patch14 (default)   --dry-run
"""
import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_vlm import MODEL_TAGS, ROOT, find_files, load_cases  # noqa: E402


def texts(case):
    out = {"sd": case["templates"]["sd"].format(**case["content"])}
    if "flux_noref" in case["templates"]:
        out["desc"] = case["templates"]["flux_noref"].format(**case["content"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, help="comma-separated glob patterns")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cond", default="ref,noref")
    ap.add_argument("--root", default=None)
    ap.add_argument("--model-path", default="openai/clip-vit-large-patch14")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root = Path(a.root) if a.root else ROOT
    gen_root = root / "results/generated"
    conds = [c.strip() for c in a.cond.split(",")]
    cases = load_cases(a.cases)

    jobs, seen = [], set()
    for c in cases:
        files = [f for cond in conds for tag in MODEL_TAGS for f in find_files(c, cond, tag, root, gen_root)]
        files = [f for f in files if Path(f).name not in seen]
        seen.update(Path(f).name for f in files)
        print(f"== {c['group']}: {len(files)} 张图", flush=True)
        jobs.append((c, files))
    if a.dry_run:
        for c, _ in jobs[:2]:
            for k, t in texts(c).items():
                print(f"  [{c['group']} {k}] {t[:160]}")
        print(f"共 {sum(len(f) for _, f in jobs)} 张图")
        return

    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(a.model_path).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(a.model_path)

    @torch.no_grad()
    def emb_text(ts):
        x = proc(text=ts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(dev)
        # text_model + projection works on transformers 4.x and 5.x (5.x get_text_features returns an output object)
        e = model.text_projection(model.text_model(input_ids=x["input_ids"], attention_mask=x["attention_mask"]).pooler_output)
        return torch.nn.functional.normalize(e, dim=-1)

    @torch.no_grad()
    def emb_img(path):
        x = proc(images=Image.open(path).convert("RGB"), return_tensors="pt").to(dev)
        e = model.visual_projection(model.vision_model(pixel_values=x["pixel_values"]).pooler_output)
        return torch.nn.functional.normalize(e, dim=-1)

    rows = []
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    for c, files in jobs:
        tx = texts(c)
        te = {k: emb_text([t]) for k, t in tx.items()}
        ntok = len(proc.tokenizer(tx["sd"])["input_ids"])
        for f in files:
            ie = emb_img(f)
            name = Path(f).name
            row = dict(file=name, world=c["world"], group=c["group"],
                       model=next(t for t in MODEL_TAGS if t in name),
                       ref_cond="noref" if "_noref_" in name else "ref",
                       seed=re.search(r"seed(\d+)\.png$", name).group(1), level=c["level"],
                       sd_tokens=ntok, sd_truncated=int(ntok > 77))
            for k in ("sd", "desc"):
                row[f"clip_t_{k}"] = round(float((ie @ te[k].T).item()), 4) if k in te else ""
            rows.append(row)
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\n{len(rows)} 张图 -> {out}")


if __name__ == "__main__":
    main()
