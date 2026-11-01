"""Data-generating processes for the simulation study (Section 5 of the manuscript).

Covariates
    p = 20 candidates: x1..x10 continuous, x11..x20 binary (0-indexed 0..9 and 10..19).
    Continuous covariates are 1 + u with u multivariate normal, unit variances and
    AR(1) correlation rho across covariates, so they have nonzero means and are not centered.
    Binary covariates threshold correlated latent normals at their 70th percentile.
    In panel designs, continuous covariates x4..x8 vary over time and all others are
    constant within a site.

States (DGP-A)
    random : x1 (0.3, 0.25), x2 (-0.3, 0.25), x11 (0.4, 0.5)   as (beta, sigma)
             (sigma on the continuous covariates is 0.25 so that the random part has an
             average standard deviation of about 0.6 on the log scale and counts have
             realistic upper tails, see Section 5 of the manuscript)
    fixed  : x3 = 0.3, x4 = -0.2, x12 = 0.3, x13 = -0.3
    excluded: the other 13 candidates
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

P_CONT = 10
P_BIN = 10
P = P_CONT + P_BIN
TIME_VARYING = np.array([3, 4, 5, 6, 7])  # x4..x8


@dataclass
class DGPConfig:
    dgp: str = "A"
    n: int = 300
    T: int = 1
    r: float = 1.0
    rho: float = 0.5
    mean_count: float = 2.0


def _truth(dgp: str):
    beta = np.zeros(P)
    sigma = np.zeros(P)
    beta[[0, 1, 10]] = [0.3, -0.3, 0.4]
    sigma[[0, 1, 10]] = [0.25, 0.25, 0.5]
    beta[[2, 3, 11, 12]] = [0.3, -0.2, 0.3, -0.3]
    quad = np.zeros(P)
    if dgp == "B":  # omitted convex curvature on x1, no random parameters
        sigma[:] = 0.0
        quad[0] = 0.15
    elif dgp == "C":  # varying dispersion, no random parameters
        sigma[:] = 0.0
    elif dgp == "D":  # zero-mean heterogeneity on x5
        sigma[4] = 0.25
    return beta, sigma, quad


def true_states(dgp: str = "A") -> np.ndarray:
    """Integer states per candidate: 0 = excluded, 1 = fixed, 2 = random."""
    beta, sigma, _ = _truth(dgp)
    s = np.zeros(P, dtype=int)
    s[beta != 0] = 1
    s[sigma != 0] = 2
    return s


def _ar1_corr(k: int, rho: float) -> np.ndarray:
    idx = np.arange(k)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def simulate(cfg: DGPConfig, seed: int | None = None) -> dict:
    """Simulate one data set. Returns a dict with y, X, site, offset and the truth."""
    rng = np.random.default_rng(seed)
    n, T = cfg.n, cfg.T
    N = n * T
    site = np.repeat(np.arange(n), T)

    Lc = np.linalg.cholesky(_ar1_corr(P_CONT, cfg.rho))
    Lb = np.linalg.cholesky(_ar1_corr(P_BIN, cfg.rho))

    # continuous: site component plus, for time-varying covariates, a period component
    u_site = rng.standard_normal((n, P_CONT)) @ Lc.T
    u = u_site[site].copy()
    if T > 1:
        u_time = rng.standard_normal((N, P_CONT)) @ Lc.T
        tv = TIME_VARYING
        u[:, tv] = np.sqrt(0.5) * u[:, tv] + np.sqrt(0.5) * u_time[:, tv]
    Xc = 1.0 + u

    # binary: constant within site
    latent = rng.standard_normal((n, P_BIN)) @ Lb.T
    thr = 0.5244005127080407  # standard normal 70th percentile
    Xb = (latent > thr).astype(float)[site]

    X = np.hstack([Xc, Xb])
    beta, sigma, quad = _truth(cfg.dgp)

    z = rng.standard_normal((n, P))
    eps = (X * (sigma * z[site])).sum(axis=1)
    lin = X @ beta + (quad * X**2).sum(axis=1)

    seg_len = rng.uniform(0.5, 2.0, size=n)
    offset = np.log(seg_len)[site]

    # intercept chosen so the average expected count is cfg.mean_count
    v = (X**2 * sigma**2).sum(axis=1)
    alpha = np.log(cfg.mean_count) - np.log(np.mean(np.exp(offset + lin + v / 2)))
    lam = np.exp(offset + alpha + lin + eps)

    if cfg.dgp == "C":
        r_obs = cfg.r * np.exp(-0.7 * X[:, 10])
    else:
        r_obs = np.full(N, cfg.r)
    y = rng.poisson(rng.gamma(r_obs, lam / r_obs))

    return {
        "y": y.astype(np.int64),
        "X": X,
        "site": site,
        "offset": offset,
        "alpha": alpha,
        "beta": beta,
        "sigma": sigma,
        "quad": quad,
        "states": true_states(cfg.dgp),
        "config": cfg,
    }


def add_quadratic_candidates(X: np.ndarray, n_cont: int = P_CONT) -> np.ndarray:
    """Append squares of the continuous covariates as fixed-only candidates (method M5)."""
    return np.hstack([X, X[:, :n_cont] ** 2])
