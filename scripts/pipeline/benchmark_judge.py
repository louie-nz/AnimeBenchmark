"""benchmark_judge.py -- laptop half of the benchmark: GPT-6.1-sol v4 judge on ref + noref, then the leaderboard.
    py scripts/pipeline/benchmark_judge.py --tags flux2klein4b,anythingv5_ipadapter,flux2klein9b,qwenimage,sd35ipa,gptimage2
Needs OPENAI_API_KEY and the generated images under results/generated/. Resumable."""
import argparse, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--tags", required=True); ap.add_argument("--judge", default="gpt-6.1-sol")
ap.add_argument("--skip-judge", action="store_true"); a = ap.parse_args()
if not a.skip_judge:
    for cond in ["ref", "noref"]:
        subprocess.run([sys.executable, str(HERE / "evaluate_gpt_judge.py"), "--judge", a.judge, "--temperature", "none",
                        "--reasoning-effort", "low", "--tags", a.tags, "--cond", cond], check=True)
subprocess.run([sys.executable, str(HERE / "make_leaderboard.py"), "--judge", a.judge, "--tags", a.tags], check=True)
