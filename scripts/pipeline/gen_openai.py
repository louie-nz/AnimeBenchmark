"""
gen_openai.py -- generate the benchmark images with OpenAI's GPT Image model through the official API.
Runs on the laptop (no GPU, no Slurm). Output uses exactly the same layout and file names as generate_all.py,
so the images are uploaded to the cluster and evaluated with the same judges as every other generator.

Fairness / consistency with the open models
  * same case files, same reference heads, same instruction-style prompts (flux_ref / flux_noref templates)
  * ref   = images.edit with the reference head as the only input image
  * noref = images.generate with the text prompt only
  * 3 images per case and condition, saved as seed42 / seed123 / seed777 (the API has no seed; the three labels
    are three independent samples, one API call each, so the file layout matches the other generators;
    a missing image is generated alone, existing images are never overwritten)
  * 1024x1024, quality fixed (default "medium") and recorded; no other options changed
  * refusals / errors are logged in openai_log.csv and reported as failures, never silently retried with a new prompt

Setup (once):  pip install openai pillow ;  set OPENAI_API_KEY=sk-...   (Windows: setx OPENAI_API_KEY "sk-...")
Usage (from the project root on the laptop, e.g. D:\\source\\AnimeBenchmark):
    python scripts/pipeline/gen_openai.py --root . --dry-run          # prompts, file names, estimated number of calls
    python scripts/pipeline/gen_openai.py --root . --limit 1          # one case first, look at the images
    python scripts/pipeline/gen_openai.py --root .                    # everything (skips existing files)
"""
import argparse
import base64
import csv
import glob
import json
import time
from pathlib import Path

CASES = ["scripts/pipeline/cases/w004/cl1*.json", "scripts/pipeline/cases/w006/*.json",
         "scripts/pipeline/cases/w014/*.json", "scripts/pipeline/cases/w022/*.json",
         "scripts/pipeline/cases/w029/*.json", "scripts/pipeline/cases_legacy/w004/*.json"]
SEEDS = [42, 123, 777]
TAG = "gptimage2"


def find_ref(root, case):
    for n in case.get("ref_head_candidates") or [case["ref_head"]]:
        p = root / "data/characters" / n
        if p.exists():
            return p
        hits = sorted(glob.glob(str(root / "data" / "**" / n), recursive=True))
        if hits:
            return Path(hits[0])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--model", default="gpt-image-2", help="exact model id as listed in your OpenAI account")
    ap.add_argument("--quality", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--cond", default="ref,noref")
    ap.add_argument("--limit", type=int, default=0, help="only the first N cases (trial run)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    files = []
    for p in CASES:
        files += sorted(glob.glob(str(root / p)))
    cases = [json.load(open(f, encoding="utf-8")) for f in dict.fromkeys(files)]
    if a.limit:
        cases = cases[:a.limit]
    conds = [c.strip() for c in a.cond.split(",")]
    jobs = []
    for c in cases:
        for cond in conds:
            outs = [root / "results/generated" / c["world"] / f"{c['group']}_files" /
                    f"{c['group']}_{TAG}_{c['character']}_{cond}_seed{s}.png" for s in SEEDS]
            key = "flux_ref" if cond == "ref" else "flux_noref"
            for o in outs:
                if not o.exists():
                    jobs.append((c, cond, c["templates"][key].format(**c["content"]), [o]))
    missing = [c["group"] for c in cases if "ref" in conds and find_ref(root, c) is None]
    print(f"{len(cases)} 个 case，还需 {len(jobs)} 张（每张一次 API 调用，已存在的跳过）；模型 {a.model}，质量 {a.quality}")
    if missing:
        print("找不到参考头像：", missing)
    if a.dry_run:
        for c, cond, prompt, outs in jobs[:4]:
            print(f"\n[{c['group']} {cond}] -> {outs[0].name}\n{prompt}")
        return
    if missing:
        raise SystemExit(1)

    from openai import OpenAI
    client = OpenAI()
    log_path = root / "results/generated" / "openai_log.csv"
    new_log = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as fh:
        log = csv.writer(fh)
        if new_log:
            log.writerow(["time", "group", "cond", "model", "quality", "status", "detail", "input_tokens", "output_tokens"])
        for c, cond, prompt, outs in jobs:
            for attempt in range(3):
                try:
                    if cond == "ref":
                        with open(find_ref(root, c), "rb") as img:
                            r = client.images.edit(model=a.model, image=[img], prompt=prompt, n=1,
                                                   size="1024x1024", quality=a.quality)
                    else:
                        r = client.images.generate(model=a.model, prompt=prompt, n=1,
                                                   size="1024x1024", quality=a.quality)
                    for o, d in zip(outs, r.data):
                        o.parent.mkdir(parents=True, exist_ok=True)
                        o.write_bytes(base64.b64decode(d.b64_json))
                    u = getattr(r, "usage", None)
                    log.writerow([time.strftime("%F %T"), c["group"], cond, a.model, a.quality, "ok", outs[0].name,
                                  getattr(u, "input_tokens", ""), getattr(u, "output_tokens", "")])
                    print(f"  ok   {outs[0].name}")
                    break
                except Exception as e:                      # refusals are logged, not rephrased
                    msg = str(e)[:300]
                    transient = any(k in msg.lower() for k in ("rate limit", "timeout", "temporarily", "502", "503"))
                    if transient and attempt < 2:
                        time.sleep(20 * (attempt + 1))
                        continue
                    log.writerow([time.strftime("%F %T"), c["group"], cond, a.model, a.quality, "error", msg, "", ""])
                    print(f"  FAIL {c['group']} {cond}: {msg}")
                    break
            fh.flush()
    print(f"\n完成。日志：{log_path}")


if __name__ == "__main__":
    main()
