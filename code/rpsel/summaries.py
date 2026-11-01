"""Posterior summaries: state probabilities, decisions and effect summaries."""

from __future__ import annotations

import numpy as np


def state_probabilities(trace) -> np.ndarray:
    """Array (p, 3) with P(E), P(F), P(R) for each candidate (eq. 9 of the manuscript)."""
    g, d = trace["gamma"], trace["delta"]
    pE = np.mean(~g & ~d, axis=0)
    pF = np.mean(g & ~d, axis=0)
    pR = np.mean(d, axis=0)
    return np.column_stack([pE, pF, pR])


def decide_states(trace, rule: str = "median") -> np.ndarray:
    """Selected state per candidate: 0 = excluded, 1 = fixed, 2 = random.

    rule="median": median-probability rule applied to each indicator
                   (random if P(delta=1) > 0.5, else fixed if P(gamma=1) > 0.5).
    rule="argmax": the state with the largest posterior probability.
    """
    if rule == "argmax":
        return np.argmax(state_probabilities(trace), axis=1)
    pg = trace["gamma"].mean(axis=0)
    pd = trace["delta"].mean(axis=0)
    s = np.where(pg > 0.5, 1, 0)
    s[pd > 0.5] = 2
    return s


def effect_summaries(trace, level: float = 0.95) -> dict:
    """Model-averaged posterior means and equal-tailed intervals for beta_j and |sigma_j|."""
    a = (1 - level) / 2
    beta, asig = trace["beta"], np.abs(trace["sigma"])
    return {
        "beta_mean": beta.mean(axis=0),
        "beta_lo": np.quantile(beta, a, axis=0),
        "beta_hi": np.quantile(beta, 1 - a, axis=0),
        "sigma_mean": asig.mean(axis=0),
        "sigma_lo": np.quantile(asig, a, axis=0),
        "sigma_hi": np.quantile(asig, 1 - a, axis=0),
    }
