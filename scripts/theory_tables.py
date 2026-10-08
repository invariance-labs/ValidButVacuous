"""Print the closed-form numbers quoted in the paper (Tables 1, 2; Prop. 3c; Remark 7).

    python scripts/theory_tables.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vbv.theory import conditional_ceiling, gaussian_luck_factor, luck_ceiling, proof_floor  # noqa

ALPHAS = [0.05, 0.10, 0.20, 0.30, 0.40]

print("Table 1  delta/s once the model speaks (delta = 5%)")
for s in [1.0, 0.5, 0.25, 0.10, 0.05]:
    c = conditional_ceiling(0.05, s)
    print(f"  s = {s:4.0%}  ->  {'no limit' if c is None else f'{c:.0%}'}")

print("\nTable 2  the floor (fewest answers that could pass)")
print("  alpha       " + "  ".join(f"{a:>4.0%}" for a in ALPHAS))
print("  one test    " + "  ".join(f"{proof_floor(a):>4d}" for a in ALPHAS))
print("  20 cut-offs " + "  ".join(f"{proof_floor(a, n_tests=20):>4d}" for a in ALPHAS))

print("\nProposition 3c  Gaussian luck factor g(p)")
for p in [0.5, 0.1, 0.01]:
    print(f"  g({p:.0%}) = {gaussian_luck_factor(p):.2f}")

print("\nRemark 7  luck ceiling for a pass once in 100 tries")
for n in [30, 300]:
    print(f"  n = {n:3d}: up to {luck_ceiling(0.01, n):.0%} better than it is")
