"""
evaluate_vlm.py -- automatic Identity / Prompt-following / Quality scores with an open vision-language model.

Why: CLIP-I / DINO-I only measure similarity to the reference head. They cannot tell whether the prompt was
followed or whether the picture is well drawn. This script asks a VLM (default Qwen2.5-VL-7B-Instruct) yes/no
questions built from each case's checklist (TIFA / DSG style) and reads the probability of "yes" from the logits
(VQAScore style), so every answer is a soft score in [0, 1].

Per image it produces
  I_vlm  identity   : reference head + generated image, "same character?" + every checklist item tagged (reference)
  P_vlm  prompt     : every other must / should item (must counts double); Task-B penalty items subtract (up to 1 point each, only when p(yes) > 0.5)
  Q_vlm  quality    : 2D illustration (not photo / not 3D render), single picture (no grid / sticker sheet),
                      no anatomy errors, no border / frame
  total_vlm         : 100 * (I*P*Q)^(1/3) / 4  -- the same formula as the human scores
  vqa_prompt        : VQAScore-style p(yes) for the whole original prompt ("Does this image show: <prompt>?")
Question sets (--qset):
  v1  the questions above (first run, 2026-10-26).
  v2  pre-registered redesign (QUESTIONS_v2.md, fixed before any v2 result was seen):
      Identity on HEAD CROPS (from identity_*.csv head_box) with one question per attribute (hair, eyes, face) plus
      "same character?", every pair question asked in both image orders and averaged (removes position bias);
      checklist identity items stay on the full image (scarves, hats are outside the head crop).
      Quality as six concrete error checks (hands, duplicated / merged objects, distorted face, watermark / overlay,
      photo / 3D look, border), each "yes = error".  Prompt items and vqa_prompt are unchanged.
      A short system instruction asks for a strict, literal answer.
All axes are on the human 0-4 scale. Every single question and its p(yes) is also written to <out>_questions.csv,
so any score can be audited.

Usage (project root, inside srun/sbatch, conda env flux2):
    python scripts/pipeline/evaluate_vlm.py --cases "scripts/pipeline/cases/w006/*.json" \
        --out results/eval/vlm/vlm_w006.csv
    --cond ref | ref,noref     (default ref: only ref images have human scores)
    --model-path PATH_OR_HF_ID (default Qwen/Qwen2.5-VL-7B-Instruct, cached in $HF_HOME)
    --dry-run                  print the questions only (no GPU, no model)
"""
import argparse
import collections
import csv
import glob
import json
import os
import re
from pathlib import Path

ROOT = Path(os.environ.get("ANIMEBENCH_ROOT", "/data/lzha060/animebenchmark"))
# generator tags looked for in file names; override with env MODEL_TAGS=flux2klein9b to score one generator only
MODEL_TAGS = os.environ.get("MODEL_TAGS", "flux2klein4b,anythingv5_ipadapter,flux2klein9b,qwenimage,gptimage2,sd35ipa").split(",")

QUALITY_Q = [
    ("q_2d", "Is this image a 2D drawn illustration, not a photograph and not a 3D render?", +1),
    ("q_single", "Is this a single picture of one scene, not a grid, collage or sticker sheet of several pictures?", +1),
    ("q_anatomy", "Is the image free of obvious drawing errors such as extra limbs, malformed hands, merged objects or duplicated props?", +1),
    ("q_border", "Does the image have a decorative border, frame or vignette around it?", -1),
]
SAME_CHAR_Q = ("The first image is the reference head of a character. Is the main character in the second image the same "
               "character as in the first image, with the same hair, eyes, face and signature features? "
               "Ignore differences in pose, expression, outfit and background.")


SYSTEM_V2 = ("You are a strict art reviewer checking AI-generated illustrations. Judge only what is visible in the "
             "images. Answer with a single word: yes or no.")
STYLE_NOTE = ("Ignore differences in drawing style, line thickness, colouring technique and head-to-body proportions "
              "(one may be chibi); ")
ID_V2 = [  # (id, weight, text) -- image 1 = reference head, image 2 = head of the generated character
    ("i_same", 2, "Image 1 is the reference head of a character, image 2 is the head of a character in a new drawing. "
                  + STYLE_NOTE + "do they show the same character?"),
    ("i_hair", 1, "Compare only the HAIR in the two images. " + STYLE_NOTE + "do they have the same hairstyle: "
                  "the same hair colour, length, bangs shape and braids / ponytail / buns?"),
    ("i_eyes", 1, "Compare only the EYES in the two images. " + STYLE_NOTE + "do they have the same eye colour and a "
                  "similar eye shape?"),
    ("i_face", 1, "Compare only the FACE in the two images. " + STYLE_NOTE + "do they share the same distinctive "
                  "facial features and accessories on the head (for example glasses, freckles, earrings, lip colour, "
                  "hat or hair ornament), with none missing and none added?"),
]
Q_V2 = [  # (id, text); every answer yes = an error
    ("q_hands", "Look at every visible hand. Does any hand have too many or too few fingers, or fused, twisted or "
                "melted fingers? If no hand is visible, answer no."),
    ("q_objects", "Is any object duplicated by mistake, or melted into another object or into the body "
                  "(for example two lamps fused together, or a book merging into the hands)?"),
    ("q_face", "Is any face distorted: eyes at different heights, eyes of clearly different sizes, a smeared or "
               "doubled mouth, or a melted face?"),
    ("q_text", "Is there a watermark, logo, signature or caption overlaid on the image? Writing that belongs to "
               "objects in the scene (books, papers, signs, carvings) does not count."),
    ("q_photo", "Does the image look like a photograph or a 3D render rather than a 2D drawing?"),
    ("q_border", "Does the image have a decorative border, frame or vignette around it?"),
]


# ----------------------------------------------------------------------------- questions
def clean(item):
    """'blonde twin braids, brown eyes (reference)' -> 'blonde twin braids, brown eyes'."""
    item = re.sub(r"\([^)]*\)", "", item)
    item = re.sub(r"^[^:]{0,40}\bkept:\s*", "", item, flags=re.I)      # 'identity kept: ...' -> '...' 
    item = re.sub(r"^overall coherence[^:]*:\s*", "", item, flags=re.I)
    item = item.replace(" -- not an identity error", "")
    return re.sub(r"\s+", " ", item).strip(" ,;:")


def is_identity_item(item):
    low = item.lower()
    return ("(reference" in low or re.search(r"only from (the )?reference", low) is not None
            or low.startswith("identity kept"))


def statement_q(text):
    return f'Look at the image. Is this description true for the image: "{text}"? Answer yes or no.' 


def build_questions(case, qset="v1"):
    """Returns list of dicts: id, axis, kind(single|pair|headpair), weight, sign, text.
    single = [generated image]; pair = [reference head, generated image]; headpair = [reference head, generated head]."""
    if qset == "v2":
        qs = [dict(id=i, axis="I", kind="headpair", weight=w, sign=+1, text=t + " Answer yes or no.") for i, w, t in ID_V2]
    else:
        qs = [dict(id="i_same", axis="I", kind="pair", weight=2, sign=+1, text=SAME_CHAR_Q + " Answer yes or no.")]
    for kind, items, w in [("must", case.get("checklist_must", []), 2), ("should", case.get("checklist_should", []), 1)]:
        for n, it in enumerate(items):
            c = clean(it)
            if not c:
                continue
            if is_identity_item(it):
                qs.append(dict(id=f"i_{kind}{n}", axis="I", kind="pair", weight=1, sign=+1,
                               text=f"The first image is the reference head of a character. Does the character in the "
                                    f"second image have {c}, as in the first image? Answer yes or no."))
            elif c.lower().startswith("single non-photoreal"):
                continue                      # covered by the quality questions
            else:
                qs.append(dict(id=f"p_{kind}{n}", axis="P", kind="single", weight=w, sign=+1, text=statement_q(c)))
    for n, it in enumerate(case.get("checklist_penalty", [])):
        c = clean(it.split(":", 1)[-1]) if ":" in it else clean(it)
        qs.append(dict(id=f"pen{n}", axis="PEN", kind="single", weight=1, sign=-1, text=statement_q(c)))
    if qset == "v2":
        for qid, text in Q_V2:
            qs.append(dict(id=qid, axis="Q", kind="single", weight=1, sign=-1, text=text + " Answer yes or no."))
    else:
        for qid, text, sign in QUALITY_Q:
            qs.append(dict(id=qid, axis="Q", kind="single", weight=1, sign=sign, text=text + " Answer yes or no."))
    sd_prompt = case["templates"]["sd"].format(**case["content"])
    qs.append(dict(id="vqa_prompt", axis="VQA", kind="single", weight=1, sign=+1,
                   text=f'Does this image show: "{sd_prompt}"? Answer yes or no.'))
    return qs


def aggregate(qs, p):
    """p: dict id -> p(yes). Returns I, P, Q (0-4), total, vqa."""
    def wmean(axis):
        rows = [(q["weight"], p[q["id"]] if q["sign"] > 0 else 1 - p[q["id"]]) for q in qs if q["axis"] == axis]
        return sum(w * v for w, v in rows) / sum(w for w, _ in rows) if rows else None
    I = 4 * wmean("I")
    P = 4 * wmean("P") if wmean("P") is not None else None
    pen = [p[q["id"]] for q in qs if q["axis"] == "PEN"]
    if P is not None and pen:
        P = max(0.0, P - sum(max(0.0, 2 * v - 1) for v in pen))   # a penalty costs up to 1 point, only when p(yes) > 0.5
    Q = 4 * wmean("Q")
    total = 100 * (I * P * Q) ** (1 / 3) / 4 if None not in (I, P, Q) else None
    return I, P, Q, total, p.get("vqa_prompt")


# ----------------------------------------------------------------------------- VLM
class YesNoVLM:
    def __init__(self, path, max_pixels, system=None):
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as M
        except ImportError:
            from transformers import AutoModelForImageTextToText as M
        self.torch = torch
        try:
            self.model = M.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
        except TypeError:
            self.model = M.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda")
        self.model.eval()
        self.system = system
        try:
            self.proc = AutoProcessor.from_pretrained(path, min_pixels=256 * 28 * 28, max_pixels=max_pixels)
        except (TypeError, ValueError):
            self.proc = AutoProcessor.from_pretrained(path)        # images are downscaled in main() anyway
        self.proc.tokenizer.padding_side = "left"
        tok = self.proc.tokenizer
        ids = lambda w: tok.encode(w, add_special_tokens=False)[0]
        self.yes = sorted({ids("Yes"), ids("yes")})
        self.no = sorted({ids("No"), ids("no")})

    def p_yes(self, images_per_q, texts, batch=8):
        """images_per_q: list of [PIL,...] (same length as texts). Returns list of p(yes)."""
        out = []
        for s in range(0, len(texts), batch):
            msgs, flat = [], []
            for imgs, t in zip(images_per_q[s:s + batch], texts[s:s + batch]):
                content = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": t}]
                conv = ([{"role": "system", "content": [{"type": "text", "text": self.system}]}] if self.system else [])
                conv.append({"role": "user", "content": content})
                msgs.append(self.proc.apply_chat_template(conv, tokenize=False, add_generation_prompt=True))
                flat += imgs
            inp = self.proc(text=msgs, images=flat, padding=True, return_tensors="pt").to(self.model.device)
            with self.torch.no_grad():
                logits = self.model(**inp).logits[:, -1, :].float()
            two = self.torch.stack([logits[:, self.yes].logsumexp(-1), logits[:, self.no].logsumexp(-1)], -1)
            out += two.softmax(-1)[:, 0].tolist()
        return out


def find_ref(case, root, ref_dir):
    """ref_head, or the first existing name in ref_head_candidates; searched in data/characters, then data/**."""
    names = case.get("ref_head_candidates") or [case["ref_head"]]
    for n in names:
        if (ref_dir / n).exists():
            return ref_dir / n
    for n in names:
        hits = sorted(glob.glob(str(root / "data" / "**" / n), recursive=True))
        if hits:
            return Path(hits[0])
    return None


def find_files(case, cond, tag, root, gen_root):
    """Pipeline layout by default; legacy cases can give image_glob = {"ref": "...{tag}...", "noref": null}."""
    pat = (case.get("image_glob") or {}).get(cond, "__default__") if "image_glob" in case else "__default__"
    if pat is None:
        return []
    default = str(gen_root / case["world"] / f"{case['group']}_files" /
                  f"{case['group']}_{tag}_{case['character']}_{cond}_seed*.png")
    pats = [default] if pat == "__default__" else [str(root / pat.format(tag=tag, cond=cond)), default]
    for pt in pats:          # legacy layout first; new generators (e.g. FLUX.2-klein-9B) write the default layout
        hits = sorted(f for f in glob.glob(pt, recursive=True) if re.search(r"seed\d+\.png$", f))
        if hits:
            return hits
    return []


def load_head_boxes(root):
    """file name -> [x1, y1, x2, y2] from every identity_*.csv under results/eval (written by evaluate_all.py)."""
    boxes = {}
    for f in glob.glob(str(root / "results" / "eval" / "**" / "identity_*.csv"), recursive=True):
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            try:
                if str(r.get("head_found", "1")) == "1" and r.get("head_box"):
                    boxes[r["file"]] = json.loads(r["head_box"])
            except (ValueError, KeyError):
                pass
    return boxes


def head_crop(img, box, pad=0.15):
    """Square-ish crop around the detected head box with some margin (hair and ears are often outside the box)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1, y1 = max(0, x1 - pad * w), max(0, y1 - pad * h)
    x2, y2 = min(img.width, x2 + pad * w), min(img.height, y2 + pad * h)
    return img.crop((int(x1), int(y1), int(x2), int(y2)))


# ----------------------------------------------------------------------------- main
def load_cases(patterns):
    files = []
    for p in [x.strip() for x in patterns.split(",") if x.strip()]:
        files += sorted(glob.glob(p, recursive=True))
    files = list(dict.fromkeys(files))
    if not files:
        raise SystemExit(f"没有找到 case 文件：{patterns}")
    return [json.load(open(f, encoding="utf-8")) for f in files]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cond", default="ref")
    ap.add_argument("--root", default=None)
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--max-pixels", type=int, default=768 * 768)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N images per case (smoke test)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--qset", default="v1", choices=["v1", "v2"], help="question set, see the header")
    a = ap.parse_args()

    root = Path(a.root) if a.root else ROOT
    ref_dir, gen_root = root / "data/characters", root / "results/generated"
    conds = [c.strip() for c in a.cond.split(",")]
    cases = load_cases(a.cases)

    if a.dry_run:
        for c in cases:
            qs = build_questions(c, a.qset)
            print(f"\n== {c['group']} ({c['level']}, {c['style_type']}): {len(qs)} questions")
            for q in qs:
                print(f"  [{q['axis']:3} w{q['weight']}{'-' if q['sign'] < 0 else ' '}] {q['id']:11} {q['text'][:150]}")
        return

    from PIL import Image
    vlm = YesNoVLM(a.model_path, a.max_pixels, SYSTEM_V2 if a.qset == "v2" else None)
    boxes = load_head_boxes(root) if a.qset == "v2" else {}
    if a.qset == "v2":
        print(f"head boxes loaded: {len(boxes)}", flush=True)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    qpath = out.with_name(out.stem + "_questions.csv")
    rows, qrows = [], []
    for c in cases:
        qs = build_questions(c, a.qset)
        side = int(a.max_pixels ** 0.5)
        ref_path = find_ref(c, root, ref_dir)
        if ref_path is None:
            print(f"!! {c['group']}: 找不到参考头像 {c.get('ref_head_candidates') or c['ref_head']}，跳过", flush=True)
            continue
        ref = Image.open(ref_path).convert("RGB"); ref.thumbnail((side, side))
        files = [f for cond in conds for tag in MODEL_TAGS for f in find_files(c, cond, tag, root, gen_root)]
        if a.limit:
            files = files[:a.limit]
        print(f"== {c['group']}: {len(files)} 张图 × {len(qs)} 问", flush=True)
        for f in files:
            full = Image.open(f).convert("RGB")
            head_src = ""
            if any(q["kind"] == "headpair" for q in qs):
                box = boxes.get(Path(f).name)
                head = head_crop(full, box) if box else full
                head_src = "box" if box else "full"
                head.thumbnail((side, side))
            img = full.copy(); img.thumbnail((side, side))
            # every question becomes one or two "asks"; headpair questions are asked in both image orders
            asks = []                                   # (question index, [images], text)
            for i, q in enumerate(qs):
                if q["kind"] == "pair":
                    asks.append((i, [ref, img], q["text"]))
                elif q["kind"] == "headpair":
                    asks.append((i, [ref, head], q["text"]))
                    swapped = (q["text"].replace("Image 1 is the reference head of a character, image 2 is the head "
                                                 "of a character in a new drawing.",
                                                 "Image 2 is the reference head of a character, image 1 is the head "
                                                 "of a character in a new drawing."))
                    asks.append((i, [head, ref], swapped))
                else:
                    asks.append((i, [img], q["text"]))
            p, acc = {}, collections.defaultdict(list)
            for n_img in sorted({len(x[1]) for x in asks}):   # batch by number of images
                sel = [x for x in asks if len(x[1]) == n_img]
                vals = vlm.p_yes([x[1] for x in sel], [x[2] for x in sel], a.batch)
                for (i, _, _), v in zip(sel, vals):
                    acc[i].append(v)
            p = {qs[i]["id"]: sum(v) / len(v) for i, v in acc.items()}
            I, P, Q, total, vqa = aggregate(qs, p)
            name = Path(f).name
            model = next(t for t in MODEL_TAGS if t in name)
            cond = "noref" if "_noref_" in name else "ref"
            seed = re.search(r"seed(\d+)\.png$", name).group(1)
            rows.append(dict(file=name, world=c["world"], group=c["group"], model=model, ref_cond=cond,
                             seed=seed, level=c["level"], style_type=c["style_type"],
                             supplementary=int(bool(c.get("supplementary"))), qset=a.qset, head_src=head_src,
                             I_vlm=round(I, 3), P_vlm=round(P, 3), Q_vlm=round(Q, 3),
                             total_vlm=round(total, 1), vqa_prompt=round(vqa, 4)))
            qrows += [dict(file=name, group=c["group"], qid=q["id"], axis=q["axis"], weight=q["weight"],
                           sign=q["sign"], p_yes=round(p[q["id"]], 4), question=q["text"]) for q in qs]
            print(f"  {name}: I={I:.2f} P={P:.2f} Q={Q:.2f} total={total:.1f} vqa={vqa:.3f}", flush=True)
        for path, data in [(out, rows), (qpath, qrows)]:
            if data:
                with open(path, "w", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)
    print(f"\n明细：{out}\n逐题：{qpath}")


if __name__ == "__main__":
    main()
