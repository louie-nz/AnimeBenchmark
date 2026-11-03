"""
compare_auto_human.py -- how well do automatic scores agree with the human ratings?

Inputs
  --xlsx      the scoring workbook (sheet '打分': Louie I/P/Q in cols I-K, Claude I/P/Q in cols M-O, file name in col G)
  --identity  identity_*.csv files from evaluate_all.py (clip_i, dino_i), comma-separated, optional
  --clipt     clip_t.csv from clip_t.py (clip_t_sd, clip_t_desc), optional
  --vlm       vlm_*.csv files from evaluate_vlm.py (I_vlm, P_vlm, Q_vlm, total_vlm, vqa_prompt), comma-separated, optional
  --rater     louie | claude | mean   (default mean of the raters who filled the row)
Output: Spearman rho of every automatic score against human I, P, Q, Total; quadratic-weighted kappa for the
VLM axes (rounded to 0-4); and, per group, whether each score picks the same winning model as the humans.
Runs on a laptop (no GPU):  python compare_auto_human.py --xlsx scoring_v3.xlsx --identity a.csv,b.csv --vlm v.csv
"""
import argparse
import collections
import csv

import numpy as np
import openpyxl
from scipy.stats import spearmanr


def read_csvs(paths):
    out = {}
    for p in [x for x in (paths or "").split(",") if x.strip()]:
        for r in csv.DictReader(open(p.strip(), encoding="utf-8-sig")):
            out[r["file"]] = r
    return out


def qwk(a, b, k=5):
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[int(x), int(y)] += 1
    W = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return 1 - (W * O).sum() / (W * E).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--identity", default="")
    ap.add_argument("--vlm", default="")
    ap.add_argument("--clipt", default="", help="clip_t.csv from clip_t.py, optional")
    ap.add_argument("--rater", default="mean", choices=["louie", "claude", "mean"])
    a = ap.parse_args()
    ident, vlm, clipt = read_csvs(a.identity), read_csvs(a.vlm), read_csvs(a.clipt)

    H, AUTO, LAB = [], collections.defaultdict(list), []
    for r in openpyxl.load_workbook(a.xlsx, data_only=True)["打分"].iter_rows(min_row=2, values_only=True):
        if not r[0] or not r[6]:
            continue
        L = r[8:11] if all(isinstance(v, (int, float)) for v in r[8:11]) else None
        C = r[12:15] if all(isinstance(v, (int, float)) for v in r[12:15]) else None
        pick = {"louie": [L], "claude": [C], "mean": [L, C]}[a.rater]
        pick = [np.array(x, float) for x in pick if x is not None]
        if not pick:
            continue
        ipq = np.mean(pick, 0)
        f = r[6]
        H.append([*ipq, 100 * np.cbrt(ipq.prod()) / 4])
        LAB.append((r[0], "FLUX" if "FLUX" in str(r[4]) else "AV5"))
        for k in ["clip_i", "dino_i"]:
            AUTO[k].append(float(ident[f][k]) if f in ident else np.nan)
        for k in ["clip_t_sd", "clip_t_desc"]:
            AUTO[k].append(float(clipt[f][k]) if f in clipt and clipt[f].get(k) not in (None, "") else np.nan)
        for k in ["I_vlm", "P_vlm", "Q_vlm", "total_vlm", "vqa_prompt"]:
            AUTO[k].append(float(vlm[f][k]) if f in vlm else np.nan)
    H = np.array(H)
    print(f"{len(H)} 张图有人工分（rater = {a.rater}）\n")
    print(f"{'score':11} {'n':>4}  " + "  ".join(f"rho_{n:5}" for n in ["I", "P", "Q", "Total"]))
    for k, v in AUTO.items():
        v = np.array(v); m = ~np.isnan(v)
        if m.sum() < 5:
            continue
        rhos = [spearmanr(v[m], H[m, i])[0] for i in range(4)]
        print(f"{k:11} {m.sum():4}  " + "  ".join(f"{x:+.2f}    " for x in rhos))
    for k, i in [("I_vlm", 0), ("P_vlm", 1), ("Q_vlm", 2)]:
        v = np.array(AUTO[k]); m = ~np.isnan(v)
        if m.sum() >= 5:
            print(f"QWK {k} vs human {['I','P','Q'][i]}: {qwk(np.clip(np.rint(v[m]),0,4), np.clip(np.rint(H[m,i]),0,4)):.2f}")
    print("\n每组胜者（FLUX / AV5）是否与人工总分一致：")
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for i, (g, mdl) in enumerate(LAB):
        groups[g][mdl].append(i)
    agree, tot = collections.Counter(), collections.Counter()
    for g, d in groups.items():
        if not d["FLUX"] or not d["AV5"]:
            continue
        win = lambda arr: "FLUX" if np.nanmean(arr[d["FLUX"]]) > np.nanmean(arr[d["AV5"]]) else "AV5"
        human = win(H[:, 3]); line = [f"{g:5} human={human}"]
        for k, v in AUTO.items():
            v = np.array(v)
            if np.isnan(v[d["FLUX"] + d["AV5"]]).all():
                continue
            w = win(v); agree[k] += (w == human); tot[k] += 1; line.append(f"{k}={w}")
        print("  " + "  ".join(line))
    print("\n一致组数：" + ", ".join(f"{k} {agree[k]}/{tot[k]}" for k in tot))


if __name__ == "__main__":
    main()
