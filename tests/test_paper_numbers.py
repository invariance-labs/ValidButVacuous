"""Check the code against every closed-form number stated in the paper."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vbv import (Certificate, conditional_ceiling, gaussian_luck_factor, hb_pvalue,  # noqa: E402
                 luck_ceiling, proof_floor, run_within)

ALPHAS = [0.05, 0.10, 0.20, 0.30, 0.40]


def test_floor_one_test():
    assert [proof_floor(a) for a in ALPHAS] == [59, 29, 14, 9, 6]


def test_floor_twenty_cutoffs():
    assert [proof_floor(a, n_tests=20) for a in ALPHAS] == [117, 57, 27, 17, 12]


def test_hb_zero_errors_is_rule_of_three():
    for a in ALPHAS:
        for n in [10, 59, 200]:
            assert hb_pvalue(0, n, a) == pytest.approx((1 - a) ** n)


def test_hb_monotone_in_errors():
    ps = [hb_pvalue(k, 100, 0.2) for k in range(30)]
    assert all(x <= y + 1e-15 for x, y in zip(ps, ps[1:]))


def test_table1():
    assert [conditional_ceiling(0.05, s) for s in [1, .5, .25, .1]] == pytest.approx([.05, .1, .2, .5])
    assert conditional_ceiling(0.05, 0.05) is None


def test_gaussian_luck_values():
    assert gaussian_luck_factor(0.5) == pytest.approx(0.80, abs=0.005)
    assert gaussian_luck_factor(0.1) == pytest.approx(1.75, abs=0.005)
    assert gaussian_luck_factor(0.01) == pytest.approx(2.67, abs=0.005)
    assert gaussian_luck_factor(1.0) == 0.0


def test_remark7():
    assert round(luck_ceiling(0.01, 30), 2) == 0.28
    assert round(luck_ceiling(0.01, 300), 2) == 0.09


def test_paper_table_counts_respect_prop1():
    # Appendix C: spoke / broke counts over 40,000 deployments
    spoke = [0, 218, 4340, 10314, 16200]
    broke = [0, 85, 573, 1011, 1201]
    for k, b in zip(spoke, broke):
        assert b <= k
        if k:
            s = k / 40000
            lim = conditional_ceiling(0.05, s)
            assert lim is None or b / k <= lim


def test_certificate_silent_on_hopeless_model():
    rng = np.random.default_rng(0)
    conf = rng.random(560)
    correct = rng.random(560) < 0.3          # 70% wrong everywhere
    lam, _ = Certificate(alpha=0.05).fit(conf[:280], correct[:280], conf[280:], correct[280:])
    assert lam is None                        # Corollary 1: forced silence


def test_certificate_speaks_on_perfect_model():
    conf = np.linspace(0, 1, 560)
    correct = np.ones(560, bool)
    lam, info = Certificate(alpha=0.10).fit(conf[::2], correct[::2], conf[1::2], correct[1::2])
    assert lam is not None and info["chosen"]["errors"] == 0


def test_validity_and_silence_identity():
    rng = np.random.default_rng(3)
    correct = rng.random(700) < 0.6
    conf = np.clip(0.5 + 0.3 * (correct - 0.5) + rng.normal(0, 0.15, 700), 0, 1)
    ds = run_within(conf, correct, alphas=(0.2,), n_splits=200, seed=1)
    assert all(d.spoke or not d.broke for d in ds)   # B implies S
