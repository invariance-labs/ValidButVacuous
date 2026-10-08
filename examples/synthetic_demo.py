"""See all three facts in two minutes on a CPU, with no model and no GPU.

Builds a synthetic 'small medical model' (55% accurate, overconfident) on 700
questions and runs 300 simulated deployments per budget.

    python examples/synthetic_demo.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vbv import run_within, summarize  # noqa: E402
from scripts.run_within import print_table  # noqa: E402


def synthetic_cell(n=700, acc=0.55, seed=1):
    rng = np.random.default_rng(seed)
    correct = rng.random(n) < acc
    # confidence is informative but noisy and inflated
    logit = 0.9 * (correct - 0.5) + rng.normal(0, 1.0, n) + 0.8
    conf = 0.25 + 0.75 / (1 + np.exp(-logit))
    return conf, correct


if __name__ == "__main__":
    conf, correct = synthetic_cell()
    print(f"synthetic accuracy {correct.mean():.3f}, mean confidence {conf.mean():.3f}\n")
    print_table(summarize(run_within(conf, correct, n_splits=300, seed=0)))
