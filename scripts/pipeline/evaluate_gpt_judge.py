"""
evaluate_gpt_judge.py -- v4 "closed judge": the v3 rubric (QUESTIONS_v3.md), unchanged, answered by an OpenAI vision
model through the official API. Pre-registration: QUESTIONS_v4.md.

Fairness / consistency with the open judge (Qwen2.5-VL-32B v3)
  * system prompt, rubric, checklist and answer format are IMPORTED from evaluate_rubric.py (no copy, no edit)
  * same two images (reference head, generated image), both downscaled to at most 768x768 exactly like the 32B run
  * same images: every ref image of the selected generator tags, same case files
  * same parser (evaluate_rubric.parse) and same total formula
  * one API call per image; transient errors are retried with the SAME request; refusals / parse failures are
    recorded, never re-asked with a different prompt
  * decoding settings are fixed per judge and written into every output row
      gpt-4o       : temperature 0.2, JSON response format, image detail high  (= Trait Collapse paper, App. B.4)
      gpt-6.1-sol  : reasoning model -> no temperature; reasoning_effort low; JSON response format; detail high

Runs on the laptop (no GPU). Run from the project root (the same layout as the cluster):
    py scripts/pipeline/evaluate_gpt_judge.py --judge gpt-4o --dry-run              # prompt, image count, estimate
    py scripts/pipeline/evaluate_gpt_judge.py --judge gpt-4o --tags flux2klein4b --limit 3     # trial
    py scripts/pipeline/evaluate_gpt_judge.py --judge gpt-4o                       # all 6 generators, ref images
    py scripts/pipeline/evaluate_gpt_judge.py --judge gpt-6.1-sol --temperature none --reasoning-effort low
Output: results/eval/gpt_judge/<judge>/<tag>.csv (+ <tag>_raw.csv with the full answers). Already judged images
are skipped, so the script can be stopped and restarted. flux2klein9b.csv is the TEST SET: do not open it before
the blind human scores of the 60 test images are finished.
"""
import argparse
import base64
import csv
import io
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_rubric import SYSTEM_V3, build_prompt, parse  # noqa: E402
from evaluate_vlm import find_files, find_ref, load_cases  # noqa: E402

CASES = ("scripts/pipeline/cases/w004/cl1*.json,scripts/pipeline/cases/w006/*.json,scripts/pipeline/cases/w014/*.json,"
         "scripts/pipeline/cases/w022/*.json,scripts/pipeline/cases/w029/*.json,scripts/pipeline/cases_legacy/w004/*.json")
TAGS = "flux2klein4b,anythingv5_ipadapter,flux2klein9b,qwenimage,sd35ipa,gptimage2"
FIELDS = ["file", "world", "group", "model", "ref_cond", "seed", "level", "style_type", "supplementary", "qset",
          "judge", "judge_model_returned", "settings", "parse_ok", "I_vlm", "P_vlm", "Q_vlm", "total_vlm",
          "input_tokens", "output_tokens", "status"]


def to_data_url(path, side):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((side, side))                       # identical downscaling to evaluate_rubric.py
    buf = io.BytesIO(); im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True, help="exact model id, e.g. gpt-4o, gpt-4o-2024-11-20, gpt-6.1-sol")
    ap.add_argument("--cases", default=CASES)
    ap.add_argument("--tags", default=TAGS)
    ap.add_argument("--cond", default="ref")
    ap.add_argument("--root", default=".")
    ap.add_argument("--temperature", default="0.2", help="'none' for reasoning models")
    ap.add_argument("--reasoning-effort", default="none", help="low / medium / high, or none")
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--max-pixels", type=int, default=768 * 768)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only N images per tag (trial)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    os.chdir(root)
    side = int(a.max_pixels ** 0.5)
    temp = None if a.temperature.lower() == "none" else float(a.temperature)
    effort = None if a.reasoning_effort.lower() == "none" else a.reasoning_effort
    settings = f"temperature={temp};reasoning_effort={effort};max_tokens={a.max_tokens};detail=high;json;side={side}"
    tags = [t.strip() for t in a.tags.split(",") if t.strip()]
    conds = [c.strip() for c in a.cond.split(",")]
    cases = load_cases(a.cases)
    outdir = root / "results/eval/gpt_judge" / a.judge.replace("/", "_")

    jobs, missing_ref = {t: [] for t in tags}, []
    for c in cases:
        ref = find_ref(c, root, root / "data/characters")
        if ref is None:
            missing_ref.append(c["group"]); continue
        for t in tags:
            for cond in conds:
                for f in find_files(c, cond, t, root, root / "results/generated"):
                    jobs[t].append((c, ref, Path(f)))
    done = {}
    for t in tags:
        p = outdir / f"{t}.csv"
        done[t] = {r["file"] for r in csv.DictReader(open(p, encoding="utf-8"))} if p.exists() else set()
        jobs[t] = [j for j in jobs[t] if j[2].name not in done[t]]
        if a.limit:
            jobs[t] = jobs[t][:a.limit]
    total = sum(len(v) for v in jobs.values())
    print(f"评委 {a.judge}；设置 {settings}")
    for t in tags:
        print(f"  {t:22s} 还需 {len(jobs[t]):3d} 张（已完成 {len(done[t])}）")
    print(f"共 {total} 次 API 调用")
    if missing_ref:
        print("找不到参考头像：", missing_ref)
    if not any(jobs.values()) and not a.dry_run:
        print("没有需要评的图（检查图片是否已从集群下载到 results/generated/）"); return
    if a.dry_run:
        c = cases[0]
        print(f"\n---- 示例 prompt（{c['group']}）----\nSYSTEM: {SYSTEM_V3}\n\n{build_prompt(c)}")
        return
    if missing_ref:
        raise SystemExit(1)

    from openai import OpenAI
    client = OpenAI()
    outdir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def call(c, ref, f):
        kw = dict(model=a.judge, max_completion_tokens=a.max_tokens, response_format={"type": "json_object"},
                  messages=[{"role": "system", "content": SYSTEM_V3},
                            {"role": "user", "content": [
                                {"type": "image_url", "image_url": {"url": to_data_url(ref, side), "detail": "high"}},
                                {"type": "image_url", "image_url": {"url": to_data_url(f, side), "detail": "high"}},
                                {"type": "text", "text": build_prompt(c)}]}])
        if temp is not None:
            kw["temperature"] = temp
        if effort is not None:
            kw["reasoning_effort"] = effort
        for attempt in range(4):
            try:
                return client.chat.completions.create(**kw), None
            except Exception as e:
                msg = str(e)[:400]
                low = msg.lower()
                if "unsupported" in low or "not supported" in low or "invalid" in low and "param" in low:
                    raise SystemExit(f"\nAPI 不接受这个参数设置，请按提示改 --temperature / --reasoning-effort：\n{msg}")
                if attempt < 3 and any(k in low for k in ("rate limit", "429", "timeout", "temporarily", "500", "502", "503", "overloaded")):
                    time.sleep(15 * (attempt + 1)); continue
                return None, msg

    def work(t, c, ref, f):
        r, err = call(c, ref, f)
        name = f.name
        row = dict(file=name, world=c["world"], group=c["group"], model=t,
                   ref_cond="noref" if "_noref_" in name else "ref",
                   seed=re.search(r"seed(\d+)\.png$", name).group(1), level=c["level"], style_type=c["style_type"],
                   supplementary=int(bool(c.get("supplementary"))), qset="v3", judge=a.judge, settings=settings)
        ans = ""
        if r is None:
            row.update(status="error: " + err, parse_ok=0)
        else:
            ans = r.choices[0].message.content or ""
            I, P, Q, ok = parse(ans)
            tot = 100 * (I * P * Q) ** (1 / 3) / 4 if None not in (I, P, Q) else None
            u = r.usage
            row.update(judge_model_returned=r.model, parse_ok=int(ok), I_vlm=I, P_vlm=P, Q_vlm=Q,
                       total_vlm=round(tot, 1) if tot is not None else "",
                       input_tokens=getattr(u, "prompt_tokens", ""), output_tokens=getattr(u, "completion_tokens", ""),
                       status="ok" if ans else "empty (refusal or token limit: " + str(r.choices[0].finish_reason) + ")")
        with lock:
            for path, fields, data in [(outdir / f"{t}.csv", FIELDS, row),
                                       (outdir / f"{t}_raw.csv", ["file", "group", "answer"],
                                        dict(file=name, group=c["group"], answer=ans))]:
                new = not path.exists()
                with open(path, "a", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=fields)
                    if new:
                        w.writeheader()
                    w.writerow(data)
            tag9b = t == "flux2klein9b"           # test set: do not print scores
            shown = "(测试集，不显示分数)" if tag9b else f"I={row.get('I_vlm')} P={row.get('P_vlm')} Q={row.get('Q_vlm')}"
            print(f"  {t:20s} {name}: {shown} {row['status'][:60]}", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, t, c, ref, f) for t in tags for c, ref, f in jobs[t]]
        for fu in futs:
            fu.result()
    print(f"\n完成。结果在 {outdir}")


if __name__ == "__main__":
    main()
