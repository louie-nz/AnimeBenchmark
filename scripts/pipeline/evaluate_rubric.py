"""
evaluate_rubric.py -- v3 "rubric judge": the VLM scores I / P / Q directly on the human 0-4 scale.

Why (see QUESTIONS_v3.md): the yes/no judges (v1, v2) could not score Identity and Quality -- the quality questions
were answered "no error" for almost every image, and identity yes/no answers did not match the human scale. Human
raters do not answer yes/no questions; they read a rubric with anchors and give one score per axis. v3 gives the
VLM exactly that rubric (the anchors from the human instruction sheet, translated), the case checklist and both
images, asks it to write short observations first and then the three integer scores as JSON (VIEScore-style
"reason, then score"). Decoding is greedy, so the run is deterministic.

Per image: I_vlm, P_vlm, Q_vlm (0-4 integers), total_vlm (same geometric-mean formula as the human scores),
parse_ok, and the full model answer in <out>_raw.csv for auditing.

Usage (project root, inside srun, conda env flux2; file sits next to evaluate_vlm.py):
    python scripts/pipeline/evaluate_rubric.py --cases "scripts/pipeline/cases/w006/*.json" \
        --out results/eval/vlm32b_v3/vlm_w006.csv --model-path Qwen/Qwen2.5-VL-32B-Instruct --batch 2
    --dry-run   print the full prompt for each case (no GPU)
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_vlm import MODEL_TAGS, ROOT, clean, find_files, find_ref, is_identity_item, load_cases  # noqa: E402

SYSTEM_V3 = ("You are an expert reviewer of anime and cartoon character illustrations. You apply the scoring "
             "rubric exactly as written and you answer only with one JSON object.")

LEVEL_TEXT = {
    "L1": "L1 (reproduce the reference character as described)",
    "L2": "L2 (keep the reference character but change outfit, pose or scene as described)",
    "TaskB": "Task B (a NEW character, shown in the reference head, visits an established story world)",
}

RUBRIC = """Score three axes, each an integer from 0 to 4.

IDENTITY - is the character in image 2 the same character as in image 1 (hairstyle and hair colour, face, eyes,
signature features, species)? Expression, pose and outfit do not count. A requested change of drawing style or of
head-to-body proportions (for example chibi) is not an identity error.
  4 = unmistakably the reference character (hair colour, hairstyle, face shape and eyes all match)
  3 = the main features match, drawing style or details changed
  2 = only the hairstyle / colours or other large features remain, the face is clearly different
  1 = only one or two features remain
  0 = not alike at all

PROMPT FOLLOWING - check every checklist item against image 2.
  0 = no MUST item is present (the picture only copies the reference head)
  1 = only one MUST item is present
  2 = about half of the MUST items are present
  3 = the MUST items are basically all present, some SHOULD items are missing
  4 = all MUST items and most SHOULD items are present
  Every PENALTY item that is present lowers the score.

QUALITY - judge only the picture itself: hands, limbs, extra or duplicated objects, border, blur, and an
unrequested photographic or 3D-rendered look.
  4 = clean overall
  3 = minor problems such as a slightly wrong hand
  2 = a structural error such as an extra tail or a duplicated sceptre
  an unrequested photograph or 3D render is at most 2"""

ANSWER = """Look carefully at both images first. Then answer with JSON only, in this order:
{"identity_notes": "<one or two sentences>", "identity": <0-4>,
 "items": {"<checklist item>": "yes" or "no", ...},
 "prompt_notes": "<one or two sentences>", "prompt": <0-4>,
 "quality_notes": "<one or two sentences>", "quality": <0-4>}"""


def build_prompt(case):
    items = [(x, "must") for x in case.get("checklist_must", [])] + [(x, "should") for x in case.get("checklist_should", [])]
    ident = [clean(x) for x, _ in items if is_identity_item(x)]
    keep = [(clean(x), k) for x, k in items if not is_identity_item(x) and not clean(x).lower().startswith("single non-photoreal")]
    must = [x for x, k in keep if k == "must"]
    should = [x for x, k in keep if k == "should"]
    pen = [clean(x.split(":", 1)[-1]) for x in case.get("checklist_penalty", [])]
    prompt = case["templates"]["sd"].format(**case["content"])
    lines = ["Image 1 is the reference head of a character. Image 2 is a new picture generated for this task:",
             f"Task level: {LEVEL_TEXT.get(case['level'], case['level'])}.",
             f'Text prompt given to the image generator: "{prompt}"', "",
             "Identity cues of the reference character (for IDENTITY only): " + ("; ".join(ident) or "see image 1"),
             "Checklist for prompt following:",
             "  MUST items: " + "; ".join(m for m in must if m),
             "  SHOULD items: " + ("; ".join(s for s in should if s) or "none")]
    if pen:
        lines.append("  PENALTY items (should NOT appear): " + "; ".join(pen))
    return "\n".join(lines) + "\n\n" + RUBRIC + "\n\n" + ANSWER


def parse(text):
    """Returns (I, P, Q, ok). Tries strict JSON first, then a field-by-field regex."""
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            vals = [int(d[k]) for k in ("identity", "prompt", "quality")]
            if all(0 <= v <= 4 for v in vals):
                return (*vals, True)
        except (ValueError, KeyError, TypeError):
            pass
    vals = []
    for k in ("identity", "prompt", "quality"):
        mm = re.search(rf'"{k}"\s*:\s*"?([0-4])', text)
        vals.append(int(mm.group(1)) if mm else None)
    return (*vals, False)


class RubricVLM:
    def __init__(self, path, max_pixels, max_new_tokens=700):
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as M
        except ImportError:
            from transformers import AutoModelForImageTextToText as M
        self.torch, self.max_new_tokens = torch, max_new_tokens
        try:
            self.model = M.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
        except TypeError:
            self.model = M.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda")
        self.model.eval()
        try:
            self.proc = AutoProcessor.from_pretrained(path, min_pixels=256 * 28 * 28, max_pixels=max_pixels)
        except (TypeError, ValueError):
            self.proc = AutoProcessor.from_pretrained(path)
        self.proc.tokenizer.padding_side = "left"

    def answer(self, image_pairs, texts, batch=2):
        out = []
        for s in range(0, len(texts), batch):
            msgs, flat = [], []
            for imgs, t in zip(image_pairs[s:s + batch], texts[s:s + batch]):
                conv = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_V3}]},
                        {"role": "user", "content": [{"type": "image"} for _ in imgs] + [{"type": "text", "text": t}]}]
                msgs.append(self.proc.apply_chat_template(conv, tokenize=False, add_generation_prompt=True))
                flat += imgs
            inp = self.proc(text=msgs, images=flat, padding=True, return_tensors="pt").to(self.model.device)
            with self.torch.no_grad():
                gen = self.model.generate(**inp, max_new_tokens=self.max_new_tokens, do_sample=False)
            gen = gen[:, inp["input_ids"].shape[1]:]
            out += self.proc.batch_decode(gen, skip_special_tokens=True)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cond", default="ref")
    ap.add_argument("--root", default=None)
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-VL-32B-Instruct")
    ap.add_argument("--max-pixels", type=int, default=768 * 768)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root = Path(a.root) if a.root else ROOT
    ref_dir, gen_root = root / "data/characters", root / "results/generated"
    conds = [c.strip() for c in a.cond.split(",")]
    cases = load_cases(a.cases)
    if a.dry_run:
        for c in cases:
            print(f"\n================ {c['group']} ({c['level']})\n{build_prompt(c)}")
        return

    from PIL import Image
    vlm = RubricVLM(a.model_path, a.max_pixels)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out.with_name(out.stem + "_raw.csv")
    side = int(a.max_pixels ** 0.5)
    rows, raws = [], []
    for c in cases:
        ref_path = find_ref(c, root, ref_dir)
        if ref_path is None:
            print(f"!! {c['group']}: 找不到参考头像，跳过", flush=True)
            continue
        ref = Image.open(ref_path).convert("RGB"); ref.thumbnail((side, side))
        files = [f for cond in conds for tag in MODEL_TAGS for f in find_files(c, cond, tag, root, gen_root)]
        if a.limit:
            files = files[:a.limit]
        text = build_prompt(c)
        print(f"== {c['group']}: {len(files)} 张图", flush=True)
        imgs = []
        for f in files:
            im = Image.open(f).convert("RGB"); im.thumbnail((side, side)); imgs.append([ref, im])
        answers = vlm.answer(imgs, [text] * len(files), a.batch)
        for f, ans in zip(files, answers):
            I, P, Q, ok = parse(ans)
            name = Path(f).name
            total = 100 * (I * P * Q) ** (1 / 3) / 4 if None not in (I, P, Q) else None
            rows.append(dict(file=name, world=c["world"], group=c["group"],
                             model=next(t for t in MODEL_TAGS if t in name),
                             ref_cond="noref" if "_noref_" in name else "ref",
                             seed=re.search(r"seed(\d+)\.png$", name).group(1), level=c["level"],
                             style_type=c["style_type"], supplementary=int(bool(c.get("supplementary"))),
                             qset="v3", parse_ok=int(ok), I_vlm=I, P_vlm=P, Q_vlm=Q,
                             total_vlm=round(total, 1) if total is not None else ""))
            raws.append(dict(file=name, group=c["group"], answer=ans))
            print(f"  {name}: I={I} P={P} Q={Q} ok={int(ok)}", flush=True)
        for path, data in [(out, rows), (raw_path, raws)]:
            if data:
                with open(path, "w", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)
    print(f"\n明细：{out}\n原始回答：{raw_path}")


if __name__ == "__main__":
    main()
