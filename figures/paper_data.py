"""Numbers behind every figure in the rewritten paper.

Source: the vector figures of the previous compiled draft (selective_risk_13.pdf),
read at full plotting precision. Rates times the number of deployments land on
whole numbers (e.g. 0.030025 x 40,000 = 1,201 broken promises), so the counts
below are the counts the analysis produced. Re-derive them from the cached
split results before submission and replace this file if anything differs.

Every conditional rate is a POOLED ratio: (deployments that broke the promise)
divided by (deployments in which the certificate fired). Because a deployment
that answers nothing cannot break the promise, broken <= fired always holds.
"""

ALPHAS = [0.05, 0.10, 0.20, 0.30, 0.40]
DELTA = 0.05

# ---------------------------------------------------------------- within source
# 10 model configurations x 4 benchmarks x 1,000 random splits
N_WITHIN = 40_000
WITHIN_FIRED = [0, 218, 4_340, 10_314, 16_200]
WITHIN_BROKE = [0, 85, 573, 1_011, 1_201]
# 95% Wilson band shown in the previous draft (computed with n = 1,000 splits)
WITHIN_BROKE_CI = [(0.0, 0.00383), (0.00124, 0.00682), (0.00985, 0.02251),
                   (0.01804, 0.03615), (0.02204, 0.04161)]

# Share of questions answered (coverage), averaged over all deployments
COV_CERTIFIED = [0.0000, 0.0014, 0.0309, 0.1138, 0.2425]
COV_PLUGIN = [0.0087, 0.0713, 0.2349, 0.4139, 0.6010]
COV_PLUGIN_FULL = [0.0053, 0.0747, 0.2311, 0.4154, 0.6021]
COV_ORACLE = [0.0529, 0.1027, 0.2489, 0.4195, 0.6106]

# ---------------------------------------------------------------- survivorship
# Over deployments that fired (alpha = 0.1 .. 0.4): error on new questions minus
# error the check measured on its testing half, with 95% bootstrap intervals.
SURV_ALPHAS = [0.10, 0.20, 0.30, 0.40]
SURV_GAP = [0.11352, 0.17610, 0.13743, 0.11703]
SURV_GAP_LO = [0.1007, 0.1481, 0.1085, 0.0851]
SURV_GAP_HI = [0.1268, 0.2039, 0.1674, 0.1503]
SURV_SHARE_WORSE = [0.98, 0.88, 0.74, 0.68]
# per model x benchmark configuration: data/fig3c.json  {alpha, fire, gap}

# ---------------------------------------------------------------- transport
# 10 configurations x 12 ordered benchmark pairs x 300 splits
N_TRANSPORT = 36_000
TR_FIRED = [0, 208, 3_848, 9_185, 14_261]          # identical decision in both arms
TR_FIRED_CONTROL = [0, 208, 3_848, 9_185, 14_253]
TR_BROKE_NEW = [0, 159, 1_791, 3_995, 5_935]       # deploy on a different benchmark
TR_BROKE_SAME = [0, 84, 511, 970, 1_032]           # matched control, same benchmark
TR_FIRED_REWEIGHTED = [0, 0, 0, 0, 0]              # importance-weighted arm
# per pair at alpha = 0.4: data/fig4d.json {auc, shift}; shift = new - same,
# share of that pair's 300 deployments that broke the promise.


def within_conditional():
    return [b / f if f else None for b, f in zip(WITHIN_BROKE, WITHIN_FIRED)]


def ceiling():
    """Proposition 1: the most a valid certificate promises once it fires."""
    out = []
    for f in WITHIN_FIRED:
        rate = f / N_WITHIN
        out.append(DELTA / rate if rate else None)
    return out


if __name__ == "__main__":
    print("alpha  fired   broke   broke|all  fired/all  broke|fired  ceiling")
    for a, f, b, c, ce in zip(ALPHAS, WITHIN_FIRED, WITHIN_BROKE,
                              within_conditional(), ceiling()):
        print(f"{a:4.2f} {f:7d} {b:7d}   {b/N_WITHIN:7.4f}   {f/N_WITHIN:7.4f}"
              f"   {c if c is None else round(c,4)!s:>8}  {ce if ce is None else round(ce,4)!s:>8}")
    print()
    print("transport: alpha  new|all  same|all  new|fired  same|fired  ratio")
    for a, f, fc, bn, bs in zip(ALPHAS, TR_FIRED, TR_FIRED_CONTROL,
                                TR_BROKE_NEW, TR_BROKE_SAME):
        if not f:
            continue
        print(f"{a:4.2f}  {bn/N_TRANSPORT:.4f}  {bs/N_TRANSPORT:.4f}  "
              f"{bn/f:.4f}  {bs/fc:.4f}  {bn/bs:.2f}")
