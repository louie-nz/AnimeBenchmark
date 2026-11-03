"""
generate_all.py (v2) -- reads cases/**/*.json, loads one model once, runs every case x ref/noref x seed.

What changed from v1 (outputs, file names and prompts are IDENTICAL, so existing images are still skipped):
  * --cases accepts several patterns separated by commas, e.g.
        --cases "scripts/pipeline/cases/w014/*.json,scripts/pipeline/cases/w022/*.json"
  * Models live in a small registry (MODELS below). Adding a new model = write one class with
    load() and generate(), add it to MODELS. Case JSON files and reference heads are reused as they are.
  * --root (or env ANIMEBENCH_ROOT) lets you dry-run the script outside the cluster.

Usage (project root, inside an srun/sbatch allocation):
    conda activate flux2
    python scripts/pipeline/generate_all.py --model flux2
    python scripts/pipeline/generate_all.py --model flux2_9b      # FLUX.2-klein-9B (same env)
    conda activate animebenchmark
    python scripts/pipeline/generate_all.py --model av5
Options:
    --cases PATTERN[,PATTERN...]   default: every case under cases/ (sub-folders included)
    --cond ref | noref | ref,noref (default both)
    --seeds 42,123,777
    --dry-run                      print prompts and output paths only, no model loading
    --overwrite                    regenerate even if the output exists
Output: results/generated/{world}/{group}_files/{group}_{model_tag}_{character}_{ref|noref}_seed{n}.png
"""
import argparse
import glob
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("ANIMEBENCH_ROOT", "/data/lzha060/animebenchmark"))


def paths(root):
    return dict(case_dir=root / "scripts/pipeline/cases", ref_dir=root / "data/characters",
                out_root=root / "results/generated")


# ----------------------------------------------------------------------------- models
class Flux2Klein:
    """FLUX.2-klein-4B: edit-style prompt; the reference head is passed as image=."""
    tag = "flux2klein4b"
    template_key = {"ref": "flux_ref", "noref": "flux_noref"}

    def __init__(self, root):
        self.model_dir = root / "models/FLUX.2-klein-4B"

    def load(self):
        import torch
        from diffusers import Flux2KleinPipeline
        self.torch = torch
        self.pipe = Flux2KleinPipeline.from_pretrained(str(self.model_dir), torch_dtype=torch.bfloat16).to("cuda")

    def generate(self, case, cond, prompt, ref, seed):
        g = self.torch.Generator("cuda").manual_seed(seed)
        kw = dict(prompt=prompt, height=1024, width=1024, guidance_scale=1.0, num_inference_steps=4, generator=g)
        if ref is not None:
            kw["image"] = ref
        return self.pipe(**kw).images[0]


class AnythingV5IPAdapter:
    """Anything V5 (SD1.5) + IP-Adapter Plus: comma-separated original prompt; reference via IP-Adapter."""
    tag = "anythingv5_ipadapter"
    template_key = {"ref": "sd", "noref": "sd"}
    IPA_WEIGHT = "ip-adapter-plus_sd15.bin"
    IPA_SCALE = 0.6
    EXTRA_NEG = (", bad anatomy, worst quality, extra fingers, lowres, fused legs, extra legs, "
                 "malformed legs, disconnected limbs, floating limbs")

    def __init__(self, root):
        self.ckpt = root / "models/AnythingV5V3_v5PrtRE.safetensors"
        self.ipa_dir = root / "ip_adapter_plus_anime"
        self.ipa_loaded = False

    def load(self):
        import torch
        from diffusers import StableDiffusionPipeline
        self.torch = torch
        self.pipe = StableDiffusionPipeline.from_single_file(str(self.ckpt), torch_dtype=torch.float16,
                                                             safety_checker=None).to("cuda")

    def generate(self, case, cond, prompt, ref, seed):
        if cond == "ref" and not self.ipa_loaded:
            self.pipe.load_ip_adapter(str(self.ipa_dir), subfolder="models", weight_name=self.IPA_WEIGHT)
            self.pipe.set_ip_adapter_scale(self.IPA_SCALE)
            self.ipa_loaded = True
        if cond == "noref" and self.ipa_loaded:
            self.pipe.unload_ip_adapter()
            self.ipa_loaded = False
        n_tok = len(self.pipe.tokenizer(prompt).input_ids)
        if n_tok > 77:
            print(f"  注意：prompt {n_tok} token > 77，尾部被截断（记为 SD1.5 局限）")
        g = self.torch.Generator("cuda").manual_seed(seed)
        kw = dict(prompt=prompt, negative_prompt=case["negative_original"] + self.EXTRA_NEG,
                  num_inference_steps=30, generator=g)
        if ref is not None:
            kw["ip_adapter_image"] = ref
        return self.pipe(**kw).images[0]


class Flux2Klein9B(Flux2Klein):
    """FLUX.2-klein-9B: same pipeline, prompts and settings as the 4B model (4 steps, guidance 1.0, 1024 px),
    so the only change is model size. Download (gated: accept the licence on Hugging Face first):
        hf download black-forest-labs/FLUX.2-klein-9B --local-dir models/FLUX.2-klein-9B"""
    tag = "flux2klein9b"

    def __init__(self, root):
        self.model_dir = root / "models/FLUX.2-klein-9B"

    def load(self):
        if not (self.model_dir / "model_index.json").exists():
            raise SystemExit(f"找不到模型：{self.model_dir}（先下载，见类说明）")
        super().load()


class QwenImage:
    """Qwen-Image family (Apache-2.0, LLM-conditioned MMDiT).
    ref   = Qwen-Image-Edit-2511 (QwenImageEditPlusPipeline, reference head passed as image=[ref])
    noref = Qwen-Image-2512      (QwenImagePipeline, text-to-image; the edit model cannot run without an image)
    Fairness rules, same as for every other generator: identical case files, prompts (instruction-style
    flux_ref / flux_noref templates, like FLUX), reference heads, seeds 42/123/777 and 1024x1024 output.
    Sampler settings follow each model card (Edit-2511: 40 steps, true_cfg 4.0; 2512: 50 steps, true_cfg 4.0),
    with a blank negative prompt, as FLUX also runs without negative prompts.
    Download:  hf download Qwen/Qwen-Image-Edit-2511 --local-dir models/Qwen-Image-Edit-2511
               hf download Qwen/Qwen-Image-2512      --local-dir models/Qwen-Image-2512"""
    tag = "qwenimage"
    template_key = {"ref": "flux_ref", "noref": "flux_noref"}
    SETTINGS = {"ref": dict(num_inference_steps=40, true_cfg_scale=4.0, guidance_scale=1.0),
                "noref": dict(num_inference_steps=50, true_cfg_scale=4.0)}

    def __init__(self, root):
        self.dirs = {"ref": root / "models/Qwen-Image-Edit-2511", "noref": root / "models/Qwen-Image-2512"}
        self.loaded = None

    def load(self):
        import torch
        self.torch = torch
        for cond, d in self.dirs.items():
            if not (d / "model_index.json").exists():
                print(f"注意：找不到 {d}（{cond} 条件会失败，先下载，见类说明）")

    def _switch(self, cond):
        if self.loaded == cond:
            return
        if self.loaded is not None:          # free the other 20B model before loading this one
            import gc
            self.pipe.to("cpu")              # move weights off the GPU first, then drop every reference
            del self.pipe
            gc.collect()
            self.torch.cuda.empty_cache()
        if cond == "ref":
            from diffusers import QwenImageEditPlusPipeline as P
        else:
            from diffusers import QwenImagePipeline as P
        self.pipe = P.from_pretrained(str(self.dirs[cond]), torch_dtype=self.torch.bfloat16).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        self.loaded = cond

    def generate(self, case, cond, prompt, ref, seed):
        self._switch(cond)
        g = self.torch.Generator("cuda").manual_seed(seed)
        kw = dict(prompt=prompt, negative_prompt=" ", height=1024, width=1024, generator=g, **self.SETTINGS[cond])
        if ref is not None:
            kw["image"] = [ref]
        with self.torch.inference_mode():
            return self.pipe(**kw).images[0]


class SD35IPAdapter:
    """SD3.5-Large + InstantX IP-Adapter (SigLIP image encoder) -- the MMDiT / T5 counterpart of AV5 + IP-Adapter.
    To isolate the effect of the base model and text encoder, everything else is kept as for AV5: the same SD
    tag prompt ("sd" template), the same negative prompt (dataset negative + AV5's extra negatives), IP-Adapter
    scale 0.6, reference via ip_adapter_image, noref = the same pipeline without the adapter. T5 reads up to 512
    tokens (max_sequence_length=512), so the prompt is no longer cut at 77 tokens for T5 (the two CLIP encoders
    still are). Sampler: 24 steps, guidance 5.0, 1024x1024 (IP-Adapter model card / diffusers docs).
    Download (SD3.5-Large is gated: accept the licence on Hugging Face first):
        hf download stabilityai/stable-diffusion-3.5-large --local-dir models/stable-diffusion-3.5-large \
            --exclude "sd3.5_large.safetensors" --exclude "text_encoders/*"   (single-file copies, not needed)
        hf download InstantX/SD3.5-Large-IP-Adapter        --local-dir models/SD3.5-Large-IP-Adapter
        hf download google/siglip-so400m-patch14-384       --local-dir models/siglip-so400m-patch14-384"""
    tag = "sd35ipa"
    template_key = {"ref": "sd", "noref": "sd"}
    IPA_SCALE = 0.6
    EXTRA_NEG = AnythingV5IPAdapter.EXTRA_NEG

    def __init__(self, root):
        self.base = root / "models/stable-diffusion-3.5-large"
        self.ipa = root / "models/SD3.5-Large-IP-Adapter"
        self.enc = root / "models/siglip-so400m-patch14-384"
        self.ipa_loaded = False

    def load(self):
        import torch
        from diffusers import StableDiffusion3Pipeline
        from transformers import SiglipImageProcessor, SiglipVisionModel
        self.torch = torch
        fe = SiglipImageProcessor.from_pretrained(str(self.enc))
        ie = SiglipVisionModel.from_pretrained(str(self.enc), torch_dtype=torch.float16).to("cuda")
        self.pipe = StableDiffusion3Pipeline.from_pretrained(str(self.base), torch_dtype=torch.float16,
                                                             feature_extractor=fe, image_encoder=ie).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)

    def generate(self, case, cond, prompt, ref, seed):
        if cond == "ref" and not self.ipa_loaded:
            names = [n for n in ("ip-adapter.safetensors", "ip-adapter.bin") if (self.ipa / n).exists()]
            if not names:
                raise SystemExit(f"找不到 IP-Adapter 权重：{self.ipa}")
            self.pipe.load_ip_adapter(str(self.ipa), weight_name=names[0])
            self.pipe.set_ip_adapter_scale(self.IPA_SCALE)
            self.ipa_loaded = True
        if cond == "noref" and self.ipa_loaded:
            try:
                self.pipe.unload_ip_adapter()
            except Exception:                    # older diffusers: reload a clean pipeline without the adapter
                del self.pipe; self.torch.cuda.empty_cache(); self.load()
            self.ipa_loaded = False
        g = self.torch.Generator("cuda").manual_seed(seed)
        kw = dict(prompt=prompt, negative_prompt=case["negative_original"] + self.EXTRA_NEG, height=1024, width=1024,
                  num_inference_steps=24, guidance_scale=5.0, max_sequence_length=512, generator=g)
        if ref is not None:
            kw["ip_adapter_image"] = ref
        return self.pipe(**kw).images[0]


# To add a model: copy one of the classes above, set a new tag, choose which case template it reads
# ("flux_ref"/"flux_noref" for instruction-style models, "sd" for tag-style SD models), implement
# load() and generate(), then register it here. Nothing else has to change.
MODELS = {"flux2": Flux2Klein, "flux2_9b": Flux2Klein9B, "qwen": QwenImage, "sd35": SD35IPAdapter,
          "av5": AnythingV5IPAdapter}


# ----------------------------------------------------------------------------- cases
def load_cases(patterns):
    files = []
    for p in [x.strip() for x in patterns.split(",") if x.strip()]:
        files += sorted(glob.glob(p, recursive=True))
    files = list(dict.fromkeys(files))          # drop duplicates, keep order
    if not files:
        raise SystemExit(f"没有找到 case 文件：{patterns}")
    return [json.load(open(f, encoding="utf-8")) for f in files]


def build_prompt(case, model, cond):
    return case["templates"][model.template_key[cond]].format(**case["content"])


def out_path(P, case, model, cond, seed, mkdir=True):
    d = P["out_root"] / case["world"] / f"{case['group']}_files"
    if mkdir:
        d.mkdir(parents=True, exist_ok=True)
    return d / f"{case['group']}_{model.tag}_{case['character']}_{cond}_seed{seed}.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--cases", default=None)
    ap.add_argument("--cond", default="ref,noref")
    ap.add_argument("--seeds", default="42,123,777")
    ap.add_argument("--root", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    root = Path(a.root) if a.root else ROOT
    P = paths(root)
    cases = load_cases(a.cases or str(P["case_dir"] / "**" / "*.json"))
    model = MODELS[a.model](root)
    conds = [x.strip() for x in a.cond.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]

    # missing reference heads are reported up front instead of crashing half-way (the w006 lesson)
    missing = sorted({c["ref_head"] for c in cases if "ref" in conds and not (P["ref_dir"] / c["ref_head"]).exists()})
    if missing:
        print("缺少参考头像（请先放进 data/characters/）：", ", ".join(missing))
        if not a.dry_run:
            raise SystemExit(1)

    # ref first, then noref (AV5 unloads IP-Adapter between the two)
    todo = [(c, cond, s) for cond in conds for c in cases for s in seeds
            if a.overwrite or not out_path(P, c, model, cond, s, mkdir=not a.dry_run).exists()]
    print(f"{len(cases)} 个 case，{len(todo)} 张待生成（已存在的跳过）")
    for c in cases:
        for cond in conds:
            print(f"\n[{c['group']} {cond}] {build_prompt(c, model, cond)}")
    if a.dry_run or not todo:
        return

    from diffusers.utils import load_image
    model.load()
    for c, cond, seed in todo:
        ref = load_image(str(P["ref_dir"] / c["ref_head"])) if cond == "ref" else None
        img = model.generate(c, cond, build_prompt(c, model, cond), ref, seed)
        p = out_path(P, c, model, cond, seed)
        img.save(p)
        print("Saved:", p)


if __name__ == "__main__":
    main()
