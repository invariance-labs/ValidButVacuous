"""The selective-risk certificate tested in the paper (Section 2, Appendix A).

A deployment holds n_cal practice questions (560 in the paper) split into two
halves.  Half 1 turns answer counts k into confidence cut-offs.  Half 2 tests
each cut-off with a Hoeffding-Bentkus p-value.  Cut-offs are tested in a fixed
order from careful to bold (fixed-sequence Learn-then-Test); the certificate
returns the boldest cut-off that passed before the first failure, or silence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import binom

from .theory import proof_floor


# ------------------------------------------------------------------ p-values
def _kl_bernoulli(a: float, b: float) -> float:
    a = min(max(a, 0.0), 1.0)
    out = 0.0
    if a > 0:
        out += a * math.log(a / b)
    if a < 1:
        out += (1 - a) * math.log((1 - a) / (1 - b))
    return out


def hb_pvalue(errors: int, n: int, alpha: float) -> float:
    """Hoeffding-Bentkus p-value for H0: true risk >= alpha (Bates et al. 2021).

    With zero errors it equals (1 - alpha)^n exactly.
    """
    if n == 0:
        return 1.0
    rhat = errors / n
    hoeff = 1.0 if rhat >= alpha else math.exp(-n * _kl_bernoulli(rhat, alpha))
    bentkus = math.e * binom.cdf(math.ceil(n * rhat), n, alpha)
    return float(min(1.0, hoeff, bentkus))


def binomial_pvalue(errors: int, n: int, alpha: float) -> float:
    """Exact binomial p-value (robustness check in Appendix A)."""
    return 1.0 if n == 0 else float(binom.cdf(errors, n, alpha))


PVALUES = {"hb": hb_pvalue, "binomial": binomial_pvalue}


# ------------------------------------------------------------------ certificate
@dataclass
class Certificate:
    alpha: float
    delta: float = 0.05
    n_cutoffs: int = 20          # G
    pvalue: str = "hb"           # "hb" or "binomial"
    correction: str = "fixed_sequence"   # or "bonferroni"

    def answer_counts(self, n_half: int) -> np.ndarray:
        """Cut-offs listed by answer count, from the floor up to the half size."""
        floor = proof_floor(self.alpha, self.delta, self.n_cutoffs)
        floor = min(floor, n_half)
        return np.unique(np.linspace(floor, n_half, self.n_cutoffs).round().astype(int))

    def fit(self, conf1, correct1, conf2, correct2):
        """Return (cut-off, info) or (None, info) for silence.

        conf*/correct*: confidence and 0/1 correctness on the two halves.
        A question is answered when its confidence is >= the cut-off.
        """
        conf1 = np.asarray(conf1, float)
        conf2 = np.asarray(conf2, float)
        correct2 = np.asarray(correct2, bool)
        order = np.sort(conf1)[::-1]
        counts = self.answer_counts(len(conf1))
        cutoffs = order[counts - 1]                       # k-th highest confidence
        level = self.delta / self.n_cutoffs
        pfun = PVALUES[self.pvalue]

        chosen, tested = None, []
        for k, lam in zip(counts, cutoffs):
            mask = conf2 >= lam
            n = int(mask.sum())
            err = int((~correct2[mask]).sum())
            p = pfun(err, n, self.alpha)
            passed = p <= level
            tested.append(dict(k=int(k), cutoff=float(lam), n=n, errors=err, p=p, passed=passed))
            if passed:
                chosen = dict(cutoff=float(lam), n=n, errors=err,
                              practice_risk=err / n if n else 0.0)
            elif self.correction == "fixed_sequence":
                break
        if self.correction == "bonferroni":
            passing = [t for t in tested if t["passed"]]
            chosen = None
            if passing:
                t = min(passing, key=lambda t: t["cutoff"])   # boldest = lowest cut-off
                chosen = dict(cutoff=t["cutoff"], n=t["n"], errors=t["errors"],
                              practice_risk=t["errors"] / t["n"] if t["n"] else 0.0)
        return (chosen["cutoff"] if chosen else None), dict(chosen=chosen, tested=tested)


# ------------------------------------------------------------------ reference rules
def no_guarantee_cutoff(conf, correct, alpha):
    """Boldest cut-off whose practice selective risk is <= alpha, with no test."""
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, bool)
    idx = np.argsort(-conf)
    cum_err = np.cumsum(~correct[idx])
    risk = cum_err / np.arange(1, len(conf) + 1)
    ok = np.nonzero(risk <= alpha)[0]
    return float(conf[idx][ok[-1]]) if len(ok) else None


def oracle_cutoff(conf_new, correct_new, alpha):
    """Best possible: most answers on the new questions with risk <= alpha."""
    return no_guarantee_cutoff(conf_new, correct_new, alpha)


def selective_risk(conf, correct, cutoff):
    """(risk, coverage) of a cut-off; silence gives (0, 0)."""
    if cutoff is None:
        return 0.0, 0.0
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, bool)
    mask = conf >= cutoff
    if not mask.any():
        return 0.0, 0.0
    return float((~correct[mask]).mean()), float(mask.mean())
