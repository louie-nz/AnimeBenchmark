"""make_leaderboard.py -- official leaderboard from the GPT-6.1-sol v4 judge (TEST_PLAN addendum C).
Per generator: mean total S (ref) with a group-level bootstrap 95 % CI, mean I/P/Q, S per level (L1/L2/TaskB),
S(noref) and ref gain on shared groups, optional bias-corrected S (offsets fixed on the development set),
and Holm-corrected paired Wilcoxon tests between all generators (unit = group).
    py scripts/pipeline/make_leaderboard.py --tags ... [--offsets results/eval/judge_offsets.json]
Writes results/eval/leaderboard/{leaderboard.csv, leaderboard.md, pairwise.csv}."""
import argparse, itertools, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
ap = argparse.ArgumentParser(); ap.add_argument("--judge", default="gpt-6.1-sol"); ap.add_argument("--tags", required=True)
ap.add_argument("--root", default="."); ap.add_argument("--offsets", default=None); a = ap.parse_args()
root = Path(a.root); tags = [t.strip() for t in a.tags.split(",")]; rng = np.random.default_rng(0)
J = pd.concat([pd.read_csv(root / "results/eval/gpt_judge" / a.judge / f"{t}.csv") for t in tags])
J = J[J.status == "ok"].drop_duplicates("file")
J["lvl"] = J.level.astype(str).map(lambda x: "TaskB" if "TaskB" in x else ("L2" if "L2" in x else "L1"))
if a.offsets:
    off = json.load(open(a.offsets))
    for k in "IPQ": J[k + "_c"] = (J[k + "_vlm"] - off[k]).clip(0, 4)
    J["total_c"] = 100 * (J.I_c * J.P_c * J.Q_c) ** (1 / 3) / 4
def gmean(g, col="total_vlm"): return g.groupby("group")[col].mean()
rows = []
for t in tags:
    r, n = J[(J.model == t) & (J.ref_cond == "ref")], J[(J.model == t) & (J.ref_cond == "noref")]
    gm = gmean(r).values; b = [rng.choice(gm, len(gm)).mean() for _ in range(2000)]
    d = dict(generator=t, n_ref=len(r), S=round(gm.mean(), 1), CI95=f"[{np.percentile(b,2.5):.1f}, {np.percentile(b,97.5):.1f}]",
             I=round(r.I_vlm.mean(), 2), P=round(r.P_vlm.mean(), 2), Q=round(r.Q_vlm.mean(), 2))
    for l in ["L1", "L2", "TaskB"]: d["S_" + l] = round(gmean(r[r.lvl == l]).mean(), 1)
    if len(n):
        shared = gmean(n).index.intersection(gmean(r).index)
        d["S_noref"] = round(gmean(n)[shared].mean(), 1); d["ref_gain"] = round((gmean(r)[shared] - gmean(n)[shared]).mean(), 1)
    if a.offsets: d["S_corrected"] = round(gmean(r, "total_c").mean(), 1)
    rows.append(d)
L = pd.DataFrame(rows).sort_values("S", ascending=False)
G = J[J.ref_cond == "ref"].groupby(["model", "group"]).total_vlm.mean().unstack(0)
res = []
for x, y in itertools.combinations(tags, 2):
    p = G[[x, y]].dropna(); pv = wilcoxon(p[x], p[y]).pvalue if (p[x] != p[y]).any() else 1.0
    res.append([x, y, len(p), round((p[x] - p[y]).mean(), 1), pv])
R = pd.DataFrame(res, columns=["A", "B", "n_groups", "diff_A_minus_B", "p"]).sort_values("p")
R["p_holm"] = np.maximum.accumulate([min(1, (len(R) - i) * p) for i, p in enumerate(R.p)]); R["significant"] = R.p_holm < 0.05
out = root / "results/eval/leaderboard"; out.mkdir(parents=True, exist_ok=True)
L.to_csv(out / "leaderboard.csv", index=False); R.to_csv(out / "pairwise.csv", index=False)
md = lambda d: d.to_markdown(index=False) if __import__("importlib").util.find_spec("tabulate") else d.to_string(index=False)
(out / "leaderboard.md").write_text(md(L) + "\n\nPairs not significant after Holm correction are ties.\n\n" + md(R), encoding="utf-8")
print(L.to_string(index=False)); print(); print(R.to_string(index=False))
