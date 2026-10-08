"""The three facts of "Valid but Vacuous", as small numeric functions.

Fact 1 (Proposition 1)  silence hides failure:  P(B | S) <= delta / s
Fact 2 (Proposition 2)  strict promises force silence:  n0 = ln(1/delta) / ln(1/(1-alpha))
Fact 3 (Proposition 3)  passing is luck:  L <= sqrt(ln(G/p) / (2 n)),  L = sigma * g(p)
"""
from __future__ import annotations

import math

from scipy.stats import norm


# ------------------------------------------------------------------ Fact 1
def conditional_ceiling(delta: float, s: float) -> float | None:
    """Most a valid certificate promises once it speaks: delta / s.

    Returns None when the bound is vacuous (s <= delta, i.e. delta/s >= 1)
    or the model never speaks (s == 0).
    """
    if s <= 0:
        return None
    b = delta / s
    return None if b >= 1 else b


def conditional_failure(broke: int, spoke: int) -> float | None:
    """Pooled failure rate among deployments that spoke (broke <= spoke)."""
    if broke > spoke:
        raise ValueError("a silent deployment cannot break the promise: broke <= spoke")
    return broke / spoke if spoke else None


def silence_buys_validity(q: float, delta: float) -> float:
    """Remark 1: speaking rate at which a procedure that breaks the promise
    with probability q when it speaks becomes valid at level delta."""
    return min(1.0, delta / q) if q > 0 else 1.0


# ------------------------------------------------------------------ Fact 2
def proof_floor(alpha: float, delta: float = 0.05, n_tests: int = 1) -> int:
    """Fewest answers a monotone valid check can pass (Proposition 2).

    With n_tests cut-offs under a fixed-sequence / Bonferroni budget the
    level becomes delta / n_tests.  Rounded up to a whole question.
    """
    level = delta / n_tests
    n0 = math.log(1 / level) / math.log(1 / (1 - alpha))
    return math.ceil(n0 - 1e-12)


def rule_of_three(alpha: float) -> float:
    """The small-alpha approximation n0 ~ 3 / alpha."""
    return 3.0 / alpha


# ------------------------------------------------------------------ Fact 3
def luck_ceiling(p: float, n: int, n_cutoffs: int = 1) -> float:
    """Proposition 3a / Remark 6: the most a pass of probability p can
    overstate quality, for a cut-off answering n checked questions."""
    if not 0 < p <= 1:
        raise ValueError("p must be in (0, 1]")
    return math.sqrt(math.log(n_cutoffs / p) / (2 * n))


def gaussian_luck_factor(p: float) -> float:
    """g(p) = phi(Phi^{-1}(p)) / p from Proposition 3c (luck in units of sigma)."""
    if not 0 < p <= 1:
        raise ValueError("p must be in (0, 1]")
    if p == 1:
        return 0.0
    t = norm.ppf(p)
    return float(norm.pdf(t) / p)


def gaussian_luck(p: float, n: int, risk: float) -> float:
    """L = sigma * g(p), with sigma the binomial std of a risk estimate on n answers."""
    sigma = math.sqrt(risk * (1 - risk) / n)
    return sigma * gaussian_luck_factor(p)
