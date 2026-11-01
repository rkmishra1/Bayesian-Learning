"""Sequential Bayesian specification (method M2 of the manuscript).

Mimics common applied practice with full Bayesian estimation:

Step 1. Bayesian fixed parameters NB model with vague normal priors, reduced by backward
        elimination: the covariate whose 95% credible interval most clearly includes zero
        (largest two-sided tail probability above 0.05) is dropped and the model refitted.
Step 2. Each retained coefficient is tried as random, one at a time, in a random parameters
        NB model with the conventional centered parameterization and vague inverse-gamma
        priors on the variances, IG(0.001, 0.001). A coefficient is made random if the
        deviance information criterion falls by more than `dic_threshold` relative to the
        fixed parameters model.
Step 3. The final model, with all selected coefficients random, is refitted for estimation.

The DIC is the conditional DIC reported by WinBUGS/OpenBUGS: the deviance is evaluated
given the site-specific coefficients, and pD = mean deviance - deviance at the posterior
means of (alpha, fixed coefficients, site-specific coefficients, r).

Data must be sorted by site.
"""

from __future__ import annotations

import time

import numpy as np
from polyagamma import random_polyagamma
from scipy.special import gammaln

from .sampler import _crt

PRIOR_VAR = 1000.0  # dnorm(0, 0.001) in BUGS notation
IG_A = IG_B = 0.001
R_A = R_B = 0.01


def _nb_dev(y, psi, r):
    """-2 x NB log likelihood in the log-odds parameterization (lambda = r exp(psi))."""
    ll = (gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
          + y * psi - (y + r) * np.logaddexp(0.0, psi))
    return -2.0 * ll.sum()


def fit_bayes_nb(y, XF, site, off, XR=None, n_iter=1500, burn=500, seed=0):
    """Gibbs sampler for an NB model with fixed columns XF and (optionally) random columns XR.

    Returns posterior draws of the fixed coefficients (intercept first), the means and
    standard deviations of the random coefficients, and the conditional DIC.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.float64)
    N = y.size
    n = int(site.max()) + 1
    starts = np.r_[0, np.nonzero(np.diff(site))[0] + 1]
    A = np.column_stack([np.ones(N), XF]) if XF is not None and XF.shape[1] else np.ones((N, 1))
    kF = A.shape[1]
    kR = 0 if XR is None else XR.shape[1]

    theta = np.zeros(kF)
    theta[0] = np.log(max(y.mean(), 0.1)) - off.mean()
    r = 1.0
    if kR:
        mu = np.zeros(kR)
        s2 = np.full(kR, 0.1)
        b = np.zeros((n, kR))
        # per-observation outer products of the random columns, reused every iteration
        XRo = XR[:, :, None] * XR[:, None, :]
    rand_part = np.zeros(N)
    keep = n_iter - burn
    out_theta = np.empty((keep, kF))
    out_mu = np.empty((keep, kR))
    out_sd = np.empty((keep, kR))
    dev = np.empty(keep)
    sum_b = np.zeros((n, kR))
    sum_theta = np.zeros(kF)
    sum_r = 0.0
    psi = off + A @ theta
    for it in range(n_iter):
        L = _crt(y.astype(np.int64), r, rng)
        r = rng.gamma(R_A + L.sum(), 1.0 / (R_B + np.logaddexp(0.0, psi).sum()))
        omega = np.maximum(random_polyagamma(y + r, psi, random_state=rng), 1e-12)
        ytil = (y - r) / (2.0 * omega)
        # fixed block
        resp = ytil - off - rand_part
        Q = (A * omega[:, None]).T @ A + np.eye(kF) / PRIOR_VAR
        m = np.linalg.solve(Q, A.T @ (omega * resp))
        Lc = np.linalg.cholesky(Q)
        theta = m + np.linalg.solve(Lc.T, rng.standard_normal(kF))
        fixed_part = A @ theta
        if kR:
            # site-specific coefficients, batched over sites
            e = ytil - off - fixed_part
            Pi = np.add.reduceat(XRo * omega[:, None, None], starts, axis=0)
            Pi[:, np.arange(kR), np.arange(kR)] += 1.0 / s2
            rhs = np.add.reduceat(XR * (omega * e)[:, None], starts, axis=0) + mu / s2
            Lb = np.linalg.cholesky(Pi)
            mean_b = np.linalg.solve(Pi, rhs[:, :, None])[:, :, 0]
            zz = rng.standard_normal((n, kR, 1))
            b = mean_b + np.linalg.solve(np.transpose(Lb, (0, 2, 1)), zz)[:, :, 0]
            # hyperparameters
            prec_mu = n / s2 + 1.0 / PRIOR_VAR
            mu = (b.sum(axis=0) / s2) / prec_mu + rng.standard_normal(kR) / np.sqrt(prec_mu)
            s2 = 1.0 / rng.gamma(IG_A + n / 2.0, 1.0 / (IG_B + 0.5 * ((b - mu) ** 2).sum(axis=0)))
            rand_part = (XR * b[site]).sum(axis=1)
        psi = off + fixed_part + rand_part
        if it >= burn:
            k = it - burn
            out_theta[k] = theta
            if kR:
                out_mu[k] = mu
                out_sd[k] = np.sqrt(s2)
                sum_b += b
            sum_theta += theta
            sum_r += r
            dev[k] = _nb_dev(y, psi, r)
    # conditional DIC
    th_bar = sum_theta / keep
    psi_bar = off + A @ th_bar
    if kR:
        b_bar = sum_b / keep
        psi_bar = psi_bar + (XR * b_bar[site]).sum(axis=1)
    d_bar = dev.mean()
    pD = d_bar - _nb_dev(y, psi_bar, sum_r / keep)
    return dict(theta=out_theta, mu=out_mu, sd=out_sd, dic=d_bar + pD, pD=pD)


def sequential_bayes(y, X, site, off, dic_threshold=5.0, seed=0, n_iter=1500, burn=500,
                     screen_iter=1000, screen_burn=300):
    """Run M2. Returns states, estimates with 95% credible intervals, and run time."""
    t0 = time.perf_counter()
    p = X.shape[1]
    y = np.asarray(y, dtype=np.float64)
    # Step 1: backward elimination on 95% credible intervals
    active = list(range(p))
    while active:
        fit = fit_bayes_nb(y, X[:, active], site, off, n_iter=screen_iter, burn=screen_burn, seed=seed)
        draws = fit["theta"][:, 1:]
        tail = 2.0 * np.minimum((draws > 0).mean(axis=0), (draws < 0).mean(axis=0))
        worst = int(np.argmax(tail))
        if tail[worst] <= 0.05:
            break
        active.pop(worst)
    states = np.zeros(p, dtype=int)
    out = dict(beta=np.zeros(p), beta_lo=np.zeros(p), beta_hi=np.zeros(p),
               sigma=np.zeros(p), sigma_lo=np.zeros(p), sigma_hi=np.zeros(p))
    n_fits = 0
    if not active:
        return dict(states=states, **out, seconds=time.perf_counter() - t0, n_fits=n_fits)
    # Step 2: one-at-a-time random specification compared by DIC
    base = fit_bayes_nb(y, X[:, active], site, off, n_iter=n_iter, burn=burn, seed=seed + 1)
    n_fits += 1
    rand = []
    for pos, j in enumerate(active):
        fixed_cols = [a for a in active if a != j]
        fit = fit_bayes_nb(y, X[:, fixed_cols], site, off, XR=X[:, [j]], n_iter=n_iter, burn=burn,
                           seed=seed + 2 + pos)
        n_fits += 1
        if base["dic"] - fit["dic"] > dic_threshold:
            rand.append(j)
    # Step 3: final model
    fixed_cols = [a for a in active if a not in rand]
    if rand:
        fit = fit_bayes_nb(y, X[:, fixed_cols], site, off, XR=X[:, rand], n_iter=n_iter, burn=burn,
                           seed=seed + 999)
        n_fits += 1
    else:
        fit = base
    q = lambda a, lo: np.quantile(a, 0.025 if lo else 0.975, axis=0)
    th = fit["theta"][:, 1:]
    for pos, j in enumerate(fixed_cols):
        states[j] = 1
        out["beta"][j] = th[:, pos].mean()
        out["beta_lo"][j], out["beta_hi"][j] = q(th[:, pos], True), q(th[:, pos], False)
    for pos, j in enumerate(rand):
        states[j] = 2
        mu, sd = fit["mu"][:, pos], fit["sd"][:, pos]
        out["beta"][j] = mu.mean()
        out["beta_lo"][j], out["beta_hi"][j] = q(mu, True), q(mu, False)
        out["sigma"][j] = sd.mean()
        out["sigma_lo"][j], out["sigma_hi"][j] = q(sd, True), q(sd, False)
    return dict(states=states, **out, seconds=time.perf_counter() - t0, n_fits=n_fits)
