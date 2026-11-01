"""Information criterion search over effect structures (method M3 of the manuscript).

A two-stage greedy search that minimizes the Bayesian information criterion (BIC),
standing in for the metaheuristic search of Ahern et al. (2024):

Stage A. Bidirectional stepwise search over excluded/fixed states with the fixed
         parameters NB model fitted by maximum likelihood.
Stage B. Among the included covariates, a forward then backward greedy search over
         fixed/random states with the random parameters NB model fitted by simulated
         maximum likelihood (scrambled Halton draws). At each step every single change is
         evaluated and the best improving one is taken.

BIC = -2 loglik + K log N, with N the number of observations and K the number of
estimated parameters (intercept, means, standard deviations, dispersion).
Data must be sorted by site.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

from .sml import _fixed_nb_negll, fit_rpnb


def _fixed_ll(y, X, off, cols, theta0=None):
    Xa = X[:, cols]
    k = len(cols)
    if theta0 is None:
        theta0 = np.concatenate([[np.log(max(y.mean(), 0.1)) - off.mean()], np.zeros(k), [0.0]])
    res = minimize(_fixed_nb_negll, theta0, args=(y, Xa, off), jac=True, method="BFGS",
                   options={"gtol": 1e-6, "maxiter": 2000})
    return -res.fun, res.x


def _bic(ll, K, N):
    return -2.0 * ll + K * np.log(N)


def stage_a(y, X, off):
    """Bidirectional stepwise BIC over excluded/fixed with the fixed NB model."""
    N, p = X.shape
    cur = []
    ll, th = _fixed_ll(y, X, off, cur)
    best = _bic(ll, 2, N)
    fits = 1
    while True:
        cands = []
        for j in range(p):
            cols = sorted(set(cur) ^ {j})  # add if absent, remove if present
            llj, thj = _fixed_ll(y, X, off, cols)
            fits += 1
            cands.append((_bic(llj, len(cols) + 2, N), cols, thj))
        b, cols, thj = min(cands, key=lambda c: c[0])
        if b < best - 1e-9:
            best, cur, th = b, cols, thj
        else:
            return cur, th, best, fits


def bic_search(y, X, site, off, R=200, seed=0, sig0=0.1):
    """Run M3. Returns states, estimates with Wald 95% intervals, BIC and run time."""
    t0 = time.perf_counter()
    y = np.asarray(y, dtype=np.float64)
    N, p = X.shape
    active, th_fixed, best, fits = stage_a(y, X, off)
    states = np.zeros(p, dtype=int)
    beta = np.zeros(p)
    sigma = np.zeros(p)
    beta_se = np.full(p, np.nan)
    sigma_se = np.full(p, np.nan)
    if not active:
        return dict(states=states, beta=beta, sigma=sigma, beta_se=beta_se, sigma_se=sigma_se,
                    bic=best, seconds=time.perf_counter() - t0, n_fits=fits)
    Xa = X[:, active]
    k = len(active)

    def theta_for(rand, base_theta, base_rand):
        """Warm start: means from the current fit, sd of newly random parameters = sig0."""
        sig = [base_theta[k + 1 + base_rand.index(i)] if i in base_rand else sig0 for i in rand]
        return np.concatenate([base_theta[:k + 1], sig, [base_theta[-1]]])

    def fit(rand, start):
        th, se, ll = fit_rpnb(y, Xa, site, off, rand, start, R=R, seed=seed)
        return th, se, ll, _bic(ll, k + len(rand) + 2, N)

    rand = []
    theta = np.concatenate([th_fixed[:k + 1], [th_fixed[-1]]])
    th_cur, se_cur, _, best = fit(rand, theta)
    fits += 1
    # forward over fixed -> random, then backward over random -> fixed
    for direction in ("forward", "backward"):
        while True:
            pool = [i for i in range(k) if i not in rand] if direction == "forward" else list(rand)
            if not pool:
                break
            cands = []
            for i in pool:
                new = sorted(rand + [i]) if direction == "forward" else [x for x in rand if x != i]
                thc, sec, _, b = fit(new, theta_for(new, th_cur, rand))
                fits += 1
                cands.append((b, new, thc, sec))
            b, new, thc, sec = min(cands, key=lambda c: c[0])
            if b < best - 1e-9:
                best, rand, th_cur, se_cur = b, new, thc, sec
            else:
                break
    for pos, j in enumerate(active):
        states[j] = 1
        beta[j] = th_cur[1 + pos]
        beta_se[j] = se_cur[1 + pos]
    for t_idx, pos in enumerate(rand):
        j = active[pos]
        states[j] = 2
        sigma[j] = abs(th_cur[k + 1 + t_idx])
        sigma_se[j] = se_cur[k + 1 + t_idx]
    return dict(states=states, beta=beta, sigma=sigma, beta_se=beta_se, sigma_se=sigma_se,
                bic=best, seconds=time.perf_counter() - t0, n_fits=fits)
