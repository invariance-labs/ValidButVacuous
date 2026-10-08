"""Simulated deployments (Section 4).

Input: cached per-question scores for one (model, benchmark) cell:
    conf    float[n]  softmax probability of the chosen option letter
    correct bool[n]   whether the chosen option is right
Each deployment draws n_cal practice + n_test new questions without
replacement (560 + 140 of 700 in the paper), splits practice in half,
and runs the certificate plus the reference rules.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import norm

from .certificate import Certificate, no_guarantee_cutoff, oracle_cutoff, selective_risk

ALPHAS = (0.05, 0.10, 0.20, 0.30, 0.40)


@dataclass
class Deployment:
    alpha: float
    spoke: bool
    broke: bool
    practice_risk: float | None   # risk the check saw (half 2)
    new_risk: float               # risk on the new questions
    coverage_cert: float
    coverage_plugin: float
    coverage_plugin_full: float
    coverage_oracle: float


def run_within(conf, correct, alphas=ALPHAS, n_splits=1000, n_cal=560, n_test=140,
               delta=0.05, n_cutoffs=20, seed=0, **cert_kw):
    """Simulate deployments on one benchmark. Returns a list of Deployment."""
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, bool)
    rng = np.random.default_rng(seed)
    half = n_cal // 2
    out = []
    for _ in range(n_splits):
        perm = rng.permutation(len(conf))[: n_cal + n_test]
        i1, i2, it = perm[:half], perm[half:n_cal], perm[n_cal:]
        ical = perm[:n_cal]
        for a in alphas:
            cert = Certificate(alpha=a, delta=delta, n_cutoffs=n_cutoffs, **cert_kw)
            lam, info = cert.fit(conf[i1], correct[i1], conf[i2], correct[i2])
            r_new, cov = selective_risk(conf[it], correct[it], lam)
            spoke = lam is not None and cov > 0
            pl = no_guarantee_cutoff(conf[ical], correct[ical], a)
            plf = no_guarantee_cutoff(conf, correct, a)
            orc = oracle_cutoff(conf[it], correct[it], a)
            out.append(Deployment(
                alpha=a, spoke=spoke, broke=spoke and r_new > a,
                practice_risk=info["chosen"]["practice_risk"] if info["chosen"] else None,
                new_risk=r_new, coverage_cert=cov,
                coverage_plugin=selective_risk(conf[it], correct[it], pl)[1],
                coverage_plugin_full=selective_risk(conf[it], correct[it], plf)[1],
                coverage_oracle=selective_risk(conf[it], correct[it], orc)[1]))
    return out


def run_transport(conf_src, correct_src, conf_tgt, correct_tgt, alphas=ALPHAS,
                  n_splits=300, n_cal=560, n_test=140, delta=0.05, n_cutoffs=20, seed=0):
    """Certify on a source benchmark, deploy on a target benchmark.

    The control arm applies the same cut-off to held-out source questions,
    trimmed to the same number of answers, so both arms speak together.
    Returns dicts with spoke / broke_new / broke_same per deployment.
    """
    cs, ys = np.asarray(conf_src, float), np.asarray(correct_src, bool)
    ct, yt = np.asarray(conf_tgt, float), np.asarray(correct_tgt, bool)
    rng = np.random.default_rng(seed)
    half = n_cal // 2
    out = []
    for _ in range(n_splits):
        perm = rng.permutation(len(cs))
        i1, i2, ihold = perm[:half], perm[half:n_cal], perm[n_cal:n_cal + n_test]
        itgt = rng.permutation(len(ct))[:n_test]
        for a in alphas:
            lam, _ = Certificate(alpha=a, delta=delta, n_cutoffs=n_cutoffs).fit(
                cs[i1], ys[i1], cs[i2], ys[i2])
            r_new, cov_new = selective_risk(ct[itgt], yt[itgt], lam)
            spoke = lam is not None and cov_new > 0
            broke_same = False
            if spoke:
                m = np.nonzero(cs[ihold] >= lam)[0]
                n_ans = int(round(cov_new * n_test))
                if len(m):
                    top = m[np.argsort(-cs[ihold][m])][: max(1, n_ans)]
                    broke_same = (~ys[ihold][top]).mean() > a
            out.append(dict(alpha=a, spoke=spoke, broke_new=spoke and r_new > a,
                            broke_same=bool(broke_same)))
    return out


# ------------------------------------------------------------------ summaries
def wilson(k: int, n: int, level: float = 0.95):
    if n == 0:
        return (0.0, 1.0)
    z = norm.ppf(0.5 + level / 2)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def summarize(deployments, delta=0.05):
    """The four numbers every report should include (Section 5), per alpha."""
    rows = []
    for a in sorted({d.alpha for d in deployments}):
        ds = [d for d in deployments if d.alpha == a]
        n = len(ds)
        spoke = sum(d.spoke for d in ds)
        broke = sum(d.broke for d in ds)
        s = spoke / n
        gaps = [d.new_risk - d.practice_risk for d in ds if d.spoke]
        rows.append(dict(
            alpha=a, deployments=n, spoke=spoke, broke=broke,
            broke_all=broke / n, broke_all_ci=wilson(broke, n),
            speaking_rate=s,
            broke_when_spoke=broke / spoke if spoke else None,
            limit_delta_over_s=(delta / s if s > delta else None) if s else None,
            luck_gap_mean=float(np.mean(gaps)) if gaps else None,
            share_new_worse=float(np.mean(np.array(gaps) > 0)) if gaps else None,
            coverage_cert=float(np.mean([d.coverage_cert for d in ds])),
            coverage_plugin=float(np.mean([d.coverage_plugin for d in ds])),
            coverage_plugin_full=float(np.mean([d.coverage_plugin_full for d in ds])),
            coverage_oracle=float(np.mean([d.coverage_oracle for d in ds])),
        ))
    return rows


def to_records(deployments):
    return [asdict(d) for d in deployments]
