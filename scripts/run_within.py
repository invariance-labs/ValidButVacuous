"""Within-benchmark deployments for every cached score file (Section 4.1-4.4).

    python scripts/run_within.py --scores scores/*.npz --splits 1000 --out results/within.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vbv import run_within, summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--splits", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/within.json")
    args = ap.parse_args()

    files = sorted(f for pat in args.scores for f in glob.glob(pat))
    pooled, per_cell = [], {}
    for f in files:
        z = np.load(f)
        ds = run_within(z["conf"], z["correct"], n_splits=args.splits, seed=args.seed)
        pooled += ds
        per_cell[os.path.basename(f)] = summarize(ds)
        print(f"{f}: {len(ds)} deployment-budgets")
    res = dict(pooled=summarize(pooled), per_cell=per_cell)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print_table(res["pooled"])


def print_table(rows):
    print("alpha  spoke   broke  broke|all  speak   broke|spoke  delta/s   cov cert/plugin/oracle")
    for r in rows:
        bs = "--" if r["broke_when_spoke"] is None else f"{r['broke_when_spoke']:.3f}"
        lim = "none" if r["limit_delta_over_s"] is None else f"{r['limit_delta_over_s']:.3f}"
        print(f"{r['alpha']:.2f} {r['spoke']:7d} {r['broke']:7d}   {r['broke_all']:.4f}  "
              f"{r['speaking_rate']:.4f}  {bs:>10}  {lim:>7}   "
              f"{r['coverage_cert']:.3f}/{r['coverage_plugin']:.3f}/{r['coverage_oracle']:.3f}")


if __name__ == "__main__":
    main()
