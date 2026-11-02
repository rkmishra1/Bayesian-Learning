"""Conventional sequential specification (method M1 of the manuscript).

Step 1. Fixed parameters NB model, backward elimination at the 5% level (Wald tests,
        observed-information standard errors).
Step 2. Random parameters NB model by simulated maximum likelihood (scrambled Halton
        draws), with every retained coefficient random. The random parameter whose
        standard deviation has the smallest |t| is fixed and the model re-estimated,
        until every remaining standard deviation has |t| >= 1.96 (BHHH standard errors).

Data must be sorted by site.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, gammaln
from scipy.stats import norm, qmc

Z_CRIT = 1.959963984540054


def _nb_logpmf(y, mu, r):
    return (gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
            + r * np.log(r) - (r + y) * np.log(r + mu) + y * np.log(mu))


# ---------------------------------------------------------------------- fixed NB
def _fixed_nb_negll(theta, y, X, off):
    k = X.shape[1]
    eta = off + theta[0] + X @ theta[1:k + 1]
    r = np.exp(theta[-1])
    mu = np.exp(eta)
    ll = _nb_logpmf(y, mu, r).sum()
    g_eta = r * (y - mu) / (r + mu)
    d_r = digamma(y + r) - digamma(r) + np.log(r) + 1.0 - np.log(r + mu) - (r + y) / (r + mu)
    grad = np.concatenate([[g_eta.sum()], X.T @ g_eta, [(r * d_r).sum()]])
    return -ll, -grad


def fit_fixed_nb(y, X, off, theta0=None):
    k = X.shape[1]
    if theta0 is None:
        theta0 = np.concatenate([[np.log(max(y.mean(), 0.1)) - off.mean()], np.zeros(k), [0.0]])
    res = minimize(_fixed_nb_negll, theta0, args=(y, X, off), jac=True, method="BFGS",
                   options={"gtol": 1e-6, "maxiter": 2000})
    theta = res.x
    # observed information by central differences of the analytic gradient
    h = 1e-5
    H = np.empty((theta.size, theta.size))
    for a in range(theta.size):
        e = np.zeros(theta.size)
        e[a] = h
        H[:, a] = (_fixed_nb_negll(theta + e, y, X, off)[1] - _fixed_nb_negll(theta - e, y, X, off)[1]) / (2 * h)
    H = 0.5 * (H + H.T)
    se = np.sqrt(np.clip(np.diag(np.linalg.pinv(H)), 1e-300, None))
    return theta, se, -res.fun


def backward_elimination(y, X, off, level=0.05):
    active = list(range(X.shape[1]))
    while True:
        theta, se, _ = fit_fixed_nb(y, X[:, active], off)
        if not active:
            return active, theta
        z = np.abs(theta[1:len(active) + 1] / se[1:len(active) + 1])
        pvals = 2 * norm.sf(z)
        worst = int(np.argmax(pvals))
        if pvals[worst] <= level:
            return active, theta
        active.pop(worst)


# ---------------------------------------------------------------------- RPNB by SML
class _SML:
    def __init__(self, y, X, site, off, rand_idx, R, seed, Xr=None):
        self.y, self.X, self.site, self.off = y, X, site, off
        self.Xr = X if Xr is None else Xr  # columns that may carry random parameters
        self.rand = np.asarray(rand_idx, dtype=int)
        self.n = int(site.max()) + 1
        m = self.rand.size
        self.starts = np.r_[0, np.nonzero(np.diff(site))[0] + 1]
        if m > 0:
            u = qmc.Halton(d=m, scramble=True, seed=seed).random(self.n * R)
            self.H = norm.ppf(u).reshape(self.n, R, m)
        else:
            self.H = np.zeros((self.n, R, 0))
        self.R = R
        self.Hs = self.H[site]  # N x R x m, precomputed

    def scores(self, theta):
        """Per-site log likelihood and score matrix (n x k)."""
        y, X, site, off = self.y, self.X, self.site, self.off
        k = X.shape[1]
        m = self.rand.size
        alpha, beta = theta[0], theta[1:k + 1]
        sig = theta[k + 1:k + 1 + m]
        r = np.exp(theta[-1])
        Hs = self.Hs
        eta = off + alpha + X @ beta
        eps = np.einsum("nrm,nm->nr", Hs, self.Xr[:, self.rand] * sig) if m else 0.0
        mu = np.exp(eta[:, None] + eps)  # N x R
        yy = y[:, None]
        lp = _nb_logpmf(yy, mu, r)
        s_ir = np.add.reduceat(lp, self.starts, axis=0)  # n x R
        mx = s_ir.max(axis=1, keepdims=True)
        w = np.exp(s_ir - mx)
        tot = w.sum(axis=1, keepdims=True)
        ll_i = (mx + np.log(tot / self.R)).ravel()
        w = w / tot  # n x R weights
        A = (r * (yy - mu) / (r + mu)) * w[site]  # N x R
        d_r = digamma(yy + r) - digamma(r) + np.log(r) + 1.0 - np.log(r + mu) - (r + yy) / (r + mu)
        cols = [np.add.reduceat(A.sum(axis=1), self.starts)]
        a_sum = A.sum(axis=1)
        cols += [np.add.reduceat(a_sum * X[:, j], self.starts) for j in range(k)]
        for t, j in enumerate(self.rand):
            cols.append(np.add.reduceat((A * Hs[:, :, t]).sum(axis=1) * self.Xr[:, j], self.starts))
        cols.append(np.add.reduceat((r * d_r * w[site]).sum(axis=1), self.starts))
        return ll_i, np.column_stack(cols)

    def negll(self, theta):
        ll_i, S = self.scores(theta)
        return -ll_i.sum(), -S.sum(axis=0)


def fit_rpnb(y, X, site, off, rand_idx, theta0, R=200, seed=0, Xr=None):
    model = _SML(y, X, site, off, rand_idx, R, seed, Xr=Xr)
    res = minimize(model.negll, theta0, jac=True, method="BFGS", options={"gtol": 1e-5, "maxiter": 1000})
    _, S = model.scores(res.x)
    se = np.sqrt(np.clip(np.diag(np.linalg.pinv(S.T @ S)), 1e-300, None))
    return res.x, se, -res.fun


def sequential_sml(y, X, site, off, R=200, seed=0, sig0=0.1, rand_const=False):
    """Run M1. Returns states (0/1/2), beta and sigma estimates with Wald 95% intervals.

    With rand_const=True the constant is also a candidate random parameter in step 2 and is
    tested and fixed like the others. Its result is returned as const_sigma (0 if fixed).
    """
    t0 = time.perf_counter()
    p = X.shape[1]
    active, theta_fixed = backward_elimination(y, X, off)
    states = np.zeros(p, dtype=int)
    beta = np.zeros(p)
    sigma = np.zeros(p)
    beta_se = np.full(p, np.nan)
    sigma_se = np.full(p, np.nan)
    if not active:
        return dict(states=states, beta=beta, sigma=sigma, beta_se=beta_se, sigma_se=sigma_se,
                    seconds=time.perf_counter() - t0, n_refits=0)
    Xa = X[:, active]
    k = len(active)
    Xr = np.hstack([Xa, np.ones((Xa.shape[0], 1))]) if rand_const else None
    rand = list(range(k + 1 if rand_const else k))  # positions within Xa (k is the constant)
    theta = np.concatenate([theta_fixed[:k + 1], np.full(len(rand), sig0), [theta_fixed[-1]]])
    n_refits = 0
    while True:
        theta, se, ll = fit_rpnb(y, Xa, site, off, rand, theta, R=R, seed=seed, Xr=Xr)
        n_refits += 1
        if not rand:
            break
        t = np.abs(theta[k + 1:k + 1 + len(rand)] / se[k + 1:k + 1 + len(rand)])
        worst = int(np.argmin(t))
        if t[worst] >= Z_CRIT:
            break
        # fix the weakest random parameter and re-estimate
        keep = [i for i in range(len(rand)) if i != worst]
        theta = np.concatenate([theta[:k + 1], theta[k + 1:k + 1 + len(rand)][keep], [theta[-1]]])
        rand = [rand[i] for i in keep]
    for pos, j in enumerate(active):
        states[j] = 1
        beta[j] = theta[1 + pos]
        beta_se[j] = se[1 + pos]
    const_sigma, const_sigma_se = 0.0, np.nan
    for t_idx, pos in enumerate(rand):
        if pos == k:  # the constant
            const_sigma, const_sigma_se = abs(theta[k + 1 + t_idx]), se[k + 1 + t_idx]
            continue
        j = active[pos]
        states[j] = 2
        sigma[j] = abs(theta[k + 1 + t_idx])
        sigma_se[j] = se[k + 1 + t_idx]
    out = dict(states=states, beta=beta, sigma=sigma, beta_se=beta_se, sigma_se=sigma_se,
               seconds=time.perf_counter() - t0, n_refits=n_refits)
    if rand_const:
        out.update(const_sigma=const_sigma, const_sigma_se=const_sigma_se, alpha=theta[0],
                   r=float(np.exp(theta[-1])), loglik=float(ll), theta=theta, rand=rand, active=active)
    return out
