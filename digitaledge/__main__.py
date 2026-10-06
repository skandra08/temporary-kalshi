"""python -m digitaledge study --start 2026-09-01 --end 2026-10-01"""
import argparse
import sys
import time
from . import study

ap = argparse.ArgumentParser(prog="digitaledge")
ap.add_argument("command", choices=["study"])
ap.add_argument("--start", required=True)
ap.add_argument("--end", required=True)
ap.add_argument("--out", default="results/report.json")
a = ap.parse_args()
t = time.time()
obs, rep = study.run(a.start, a.end, a.out)
obs.to_csv(a.out.replace(".json", "_obs.csv"), index=False)
print(f"done in {time.time()-t:.0f}s: {rep['n_obs']} observations, {rep['n_events']} events -> {a.out}")
