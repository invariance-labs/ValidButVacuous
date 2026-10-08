"""vbv: code for "Valid but Vacuous" (Swarna, Invariance Labs)."""
from .certificate import (Certificate, binomial_pvalue, hb_pvalue, no_guarantee_cutoff,
                          oracle_cutoff, selective_risk)
from .simulate import ALPHAS, run_transport, run_within, summarize, wilson
from .theory import (conditional_ceiling, conditional_failure, gaussian_luck_factor,
                     luck_ceiling, proof_floor, rule_of_three, silence_buys_validity)

__version__ = "0.1.0"
