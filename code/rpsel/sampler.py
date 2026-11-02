"""Three-state spike-and-slab Gibbs sampler for random parameters negative binomial models.

Implements Box 1 of the manuscript:

  psi_it   = o_it + alpha + x_it' beta + sum_j sigma_j x_itj z_ij        (log odds)
  lambda_it = r exp(psi_it),   y_it ~ NB(mean lambda_it, dispersion r)

  beta_j  | gamma_j ~ gamma_j N(0, tau_b^2) + (1 - gamma_j) delta_0
  sigma_j | delta_j ~ delta_j N(0, tau_s^2) + (1 - delta_j) delta_0
  gamma_j ~ Bern(pi_b),  delta_j | gamma_j ~ Bern(gamma_j pi_s1 + (1 - gamma_j) pi_s0)

Updates: CRT draw for r (Polya-Gamma variables integrated out), Polya-Gamma draws,
Gaussian intercept, collapsed (gamma_j, beta_j), and the block (delta_j, sigma_j, z_.j)
with z_.j integrated out by the closed form of Proposition 3.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numba import njit
from polyagamma import random_polyagamma
from scipy.special import digamma, gammaln, polygamma


@dataclass
class SamplerConfig:
    n_iter: int = 4000
    burn: int = 1500
    thin: int = 1
    heredity: str = "strong"  # "strong" or "weak"
    a_b: float = 1.0
    b_b: float = 1.0
    a_s: float = 1.0
    b_s: float = 1.0
    a_s0: float = 1.0  # weak heredity only
    b_s0: float = 9.0
    # Slab scales. By default they are fixed: learning them jointly with the inclusion
    # probabilities lets the slab collapse toward the spike in small samples, so that
    # "many small effects" and "few large effects" become nearly indistinguishable.
    learn_tau: bool = False
    tau_b: float = 0.5
    tau_s: float = 0.5
    A_b: float = 1.0  # half-Cauchy scales, used only if learn_tau=True
    A_s: float = 0.5
    s_alpha: float = 10.0
    a_r: float = 1.0
    b_r: float = 0.1
    grid_coarse: int = 48
    grid_fine: int = 80
    # Dispersion selection (method M6), used only when RPSelector receives W:
    # log r_it = rho0 + w_it' rho, spike-and-slab on each rho_k.
    tau_rho: float = 0.5
    a_rho: float = 1.0
    b_rho: float = 1.0
    s_rho0: float = 2.0
    r_floor: float = 0.01  # dispersion bounded below: r_it = max(exp(rho0 + w'rho), r_floor)
    # Random intercept (application): the constant is always in the model and is fixed or
    # random. It is appended to X as a column of ones whose mean stays zero (alpha is the
    # mean) and whose indicator delta has prior probability pi_ri, outside the beta-binomial.
    random_intercept: bool = False
    pi_ri: float = 0.5
    # Translation moves (Liu and Sabatti, 2000) for every random parameter: shift the mean
    # (beta_j, or alpha for the random intercept) by c and the site draws z_.j by -c/sigma_j.
    # The likelihood is unchanged and c is drawn from its Gaussian conditional. They remove the
    # slow trade-off between a mean and the average of its site draws when n is large.
    shift_moves: bool = False
    seed: int | None = None
    verbose: bool = False


@njit(cache=True)
def _ell_on_grid(sig, q, b):
    """ell_j(sigma) of eq. (13) evaluated at each grid point (sites with q = 0 removed)."""
    G = sig.shape[0]
    n = q.shape[0]
    out = np.empty(G)
    for g in range(G):
        s2 = sig[g] * sig[g]
        acc = 0.0
        for i in range(n):
            d = 1.0 + s2 * q[i]
            acc += -0.5 * np.log(d) + 0.5 * s2 * b[i] * b[i] / d
        out[g] = acc
    return out


@njit(cache=True)
def _trigamma(x):
    """Trigamma function for positive arguments: recurrence to x >= 6, then asymptotic series."""
    out = np.empty(x.shape[0])
    for i in range(x.shape[0]):
        z = x[i]
        acc = 0.0
        while z < 6.0:
            acc += 1.0 / (z * z)
            z += 1.0
        iz = 1.0 / z
        iz2 = iz * iz
        acc += iz + 0.5 * iz2 + iz * iz2 * (1.0 / 6.0 - iz2 * (1.0 / 30.0 - iz2 * (1.0 / 42.0 - iz2 / 30.0)))
        out[i] = acc
    return out


def _log_trapz(logf, x):
    """log of the trapezoid integral of exp(logf) over grid x."""
    dx = np.diff(x)
    a = np.logaddexp(logf[:-1], logf[1:]) - np.log(2.0)
    return np.logaddexp.reduce(a + np.log(dx))


def _crt(y, r, rng):
    """Chinese restaurant table counts L_it ~ CRT(y_it, r), vectorized."""
    ymax = int(y.max())
    if ymax == 0:
        return np.zeros_like(y)
    pos = y > 0
    yy = y[pos]
    ls = np.arange(1, ymax + 1)
    probs = r / (r + ls - 1.0)
    u = rng.random((yy.shape[0], ymax))
    hits = (u < probs[None, :]) & (ls[None, :] <= yy[:, None])
    out = np.zeros_like(y)
    out[pos] = hits.sum(axis=1)
    return out


class RPSelector:
    """Gibbs sampler for three-state selection of excluded, fixed and random parameters."""

    def __init__(self, y, X, site, offset=None, eligible=None, config: SamplerConfig | None = None, W=None,
                 on_keep=None):
        self.cfg = config or SamplerConfig()
        # optional callback, called at every kept iteration with the current state (e.g. to
        # accumulate a hold-out predictive density without storing the site draws)
        self.on_keep = on_keep
        self.W = None if W is None else np.ascontiguousarray(W, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.X = np.ascontiguousarray(X, dtype=np.float64)
        self.N, self.p = self.X.shape
        site = np.asarray(site)
        _, self.site = np.unique(site, return_inverse=True)
        self.n = int(self.site.max()) + 1
        self.offset = np.zeros(self.N) if offset is None else np.asarray(offset, dtype=np.float64)
        self.eligible = np.ones(self.p, bool) if eligible is None else np.asarray(eligible, bool)
        self.ri = -1  # index of the random-intercept column, -1 if none
        if self.cfg.random_intercept:
            self.X = np.ascontiguousarray(np.hstack([self.X, np.ones((self.N, 1))]))
            self.eligible = np.r_[self.eligible, True]
            self.ri = self.p
            self.p += 1
        self.rng = np.random.default_rng(self.cfg.seed)

    # ------------------------------------------------------------------ helpers
    def _bf_sigma(self, q, b, tau2):
        """log Bayes factor for delta_j = 1 and the fine grid/log density for drawing |sigma_j|."""
        cfg = self.cfg
        keep = q > 0
        q, b = q[keep], b[keep]
        tau = np.sqrt(tau2)
        smax = max(6.0 * tau, 2.5)
        log_prior = lambda s: -0.5 * np.log(2 * np.pi * tau2) - 0.5 * s * s / tau2
        g1 = np.linspace(0.0, smax, cfg.grid_coarse)
        lf1 = _ell_on_grid(g1, q, b) + log_prior(g1)
        # refine on the region holding essentially all the mass
        top = lf1.max()
        ok = np.nonzero(lf1 > top - 30.0)[0]
        lo = g1[max(ok[0] - 1, 0)]
        hi = g1[min(ok[-1] + 1, g1.size - 1)]
        g2 = np.linspace(lo, hi, cfg.grid_fine)
        lf2 = _ell_on_grid(g2, q, b) + log_prior(g2)
        log_bf = np.log(2.0) + _log_trapz(lf2, g2)  # integrand symmetric in sigma
        return log_bf, g2, lf2

    def _draw_abs_sigma(self, grid, logf):
        """Draw |sigma| from the gridded density: pick a cell by trapezoid mass, then interpolate."""
        rng = self.rng
        dx = np.diff(grid)
        f = np.exp(logf - logf.max())
        mass = 0.5 * (f[:-1] + f[1:]) * dx
        k = rng.choice(mass.size, p=mass / mass.sum())
        f0, f1 = f[k], f[k + 1]
        u = rng.random()
        # inverse CDF of a linear density on the cell
        if abs(f1 - f0) < 1e-12 * max(f0, f1):
            t = u
        else:
            t = (-f0 + np.sqrt(f0 * f0 + u * (f1 * f1 - f0 * f0))) / (f1 - f0)
        return grid[k] + t * dx[k]

    def _shift_moves(self, alpha, beta, sigma, gam, dlt, Z, ri, tau_b2):
        """Translation moves for every active random parameter (see SamplerConfig.shift_moves)."""
        cfg, rng = self.cfg, self.rng
        n = Z.shape[0]
        for j in np.nonzero(dlt)[0]:
            s = sigma[j]
            if s == 0.0:
                continue
            if j == ri:
                mean0, prec0 = alpha, 1.0 / cfg.s_alpha**2
            elif gam[j]:
                mean0, prec0 = beta[j], 1.0 / tau_b2
            else:  # weak heredity: the mean is held at zero, no move
                continue
            zbar = Z[:, j].mean()
            prec = n / (s * s) + prec0
            c = (n * zbar / s - mean0 * prec0) / prec + rng.standard_normal() / np.sqrt(prec)
            Z[:, j] -= c / s
            if j == ri:
                alpha = alpha + c
            else:
                beta[j] += c
        return alpha

    # ------------------------------------------------------------------ dispersion block
    # The dispersion is bounded below, r_it = max(exp(rho0 + w'rho), r_floor). The bound is part
    # of the model: it is applied identically in the likelihood, its derivatives and the
    # Polya-Gamma step. With r_floor = 0.01 the variance is at most mu + 100 mu^2, beyond any
    # realistic crash data, and Polya-Gamma draws with parameter y + r stay well defined.
    LOGR_MAX = 30.0

    def _nb_ll_r(self, y, logr, lamlog):
        """NB log likelihood per observation as a function of log r (terms free of r dropped)."""
        logr = np.clip(logr, self._logr_min, self.LOGR_MAX)
        r = np.exp(logr)
        return gammaln(y + r) - gammaln(r) + r * logr - (r + y) * np.logaddexp(logr, lamlog)

    def _nb_derivs(self, y, logr, lamlog):
        """Derivatives of the NB log likelihood with respect to the unbounded log r.

        Below the floor the likelihood does not depend on log r, so both derivatives are zero.
        """
        inside = logr > self._logr_min
        logr = np.clip(logr, self._logr_min, self.LOGR_MAX)
        r = np.exp(logr)
        lse = np.logaddexp(logr, lamlog)
        rl = np.exp(lse)  # r + lambda
        d1r = digamma(y + r) - digamma(r) + logr + 1.0 - lse - (r + y) / rl
        d2r = _trigamma(y + r) - _trigamma(r) + 1.0 / r - 2.0 / rl + (r + y) / rl**2
        return r * d1r * inside, (r * d1r + r * r * d2r) * inside

    def _laplace_from(self, d1, d2, wk, tau2):
        """One Newton step from rho_k = 0 given log-likelihood derivatives on the support of w_k."""
        h = (wk * wk * d2).sum() - 1.0 / tau2
        if h >= 0:
            return 0.0, tau2
        return (wk * d1).sum() / (-h), -1.0 / h

    def _dispersion_block(self, st, lamlog, adapt):
        """Update rho0 and the (eta_k, rho_k) pairs given the mean on the log scale.

        Switching moves are reversible-jump moves whose proposal for rho_k is a Gaussian
        (Laplace) approximation computed from the state with rho_k = 0, so the same proposal
        serves both directions. A second Newton step is taken when the first suggests a clear
        effect, which is also a function of that state only. Binary candidates are evaluated
        on their nonzero observations only, and derivatives at the current state are cached.
        """
        cfg, rng, y, W = self.cfg, self.rng, self.y, self.W
        tau2 = cfg.tau_rho**2
        supp = self._supp
        logr = st["rho0"] + W @ st["rho"]
        llv = self._nb_ll_r(y, logr, lamlog)
        # rho0: random-walk Metropolis on all observations
        prop = st["rho0"] + st["sd0"] * rng.standard_normal()
        llv_p = self._nb_ll_r(y, logr + (prop - st["rho0"]), lamlog)
        la = llv_p.sum() - llv.sum() - 0.5 * (prop**2 - st["rho0"] ** 2) / cfg.s_rho0**2
        acc = np.log(rng.random()) < la
        if acc:
            logr = logr + (prop - st["rho0"])
            st["rho0"], llv = prop, llv_p
        if adapt:
            st["sd0"] *= np.exp(0.05 * (acc - 0.44))
        cache = None  # derivatives at the current logr, full length
        lprior = lambda v: -0.5 * np.log(2 * np.pi * tau2) - 0.5 * v * v / tau2
        for kk in rng.permutation(W.shape[1]):
            ix = supp[kk]
            wk = W[ix, kk]
            yk, lamk = y[ix], lamlog[ix]
            rk = st["rho"][kk]
            loff = logr[ix] - wk * rk  # log r on the support with rho_k = 0
            if rk == 0.0:
                if cache is None:
                    cache = self._nb_derivs(y, logr, lamlog)
                d1, d2 = cache[0][ix], cache[1][ix]
            else:
                d1, d2 = self._nb_derivs(yk, loff, lamk)
            m, v = self._laplace_from(d1, d2, wk, tau2)
            if 1.5 * np.sqrt(v) < abs(m) <= 3.0:  # second Newton step for a clear, plausible effect
                e1, e2 = self._nb_derivs(yk, loff + wk * m, lamk)
                h = (wk * wk * e2).sum() - 1.0 / tau2
                if h < 0:
                    m, v = m - ((wk * e1).sum() - m / tau2) / h, -1.0 / h
            lq = lambda val: -0.5 * np.log(2 * np.pi * v) - 0.5 * (val - m) ** 2 / v
            pi = st["pi_rho"]
            ll_cur = llv[ix].sum()
            if not st["eta"][kk]:
                new = m + np.sqrt(v) * rng.standard_normal()
                lnew = self._nb_ll_r(yk, loff + wk * new, lamk)
                la = lnew.sum() + lprior(new) + np.log(pi) - ll_cur - np.log1p(-pi) - lq(new)
                if np.log(rng.random()) < la:
                    st["eta"][kk], st["rho"][kk] = True, new
                    logr[ix] = loff + wk * new
                    llv[ix] = lnew
                    cache = None
            else:
                loffv = self._nb_ll_r(yk, loff, lamk)
                la = loffv.sum() + np.log1p(-pi) + lq(rk) - ll_cur - lprior(rk) - np.log(pi)
                if np.log(rng.random()) < la:
                    st["eta"][kk], st["rho"][kk] = False, 0.0
                    logr[ix] = loff
                    llv[ix] = loffv
                    cache = None
            if st["eta"][kk]:  # within-model random-walk move, composed after the switch move
                rk = st["rho"][kk]
                base = logr[ix] - wk * rk
                new = rk + st["sdk"][kk] * rng.standard_normal()
                lnew = self._nb_ll_r(yk, base + wk * new, lamk)
                acc = np.log(rng.random()) < lnew.sum() + lprior(new) - llv[ix].sum() - lprior(rk)
                if acc:
                    st["rho"][kk] = new
                    logr[ix] = base + wk * new
                    llv[ix] = lnew
                    cache = None
                if adapt:
                    st["sdk"][kk] *= np.exp(0.05 * (acc - 0.44))
        return logr

    def _run_dispersion(self):
        """Main loop with covariate-dependent dispersion selection (method M6).

        Mean parameterization: log lambda_it = o + alpha + x'beta + eps, r_it = exp(rho0 + w'rho),
        and the Polya-Gamma working model uses psi_it = log lambda_it - log r_it.
        """
        cfg, rng = self.cfg, self.rng
        y, X, W, site, n, N, p = self.y, self.X, self.W, self.site, self.n, self.N, self.p
        qd = W.shape[1]
        self._supp = [np.nonzero(W[:, kk] != 0)[0] for kk in range(qd)]
        self._logr_min = float(np.log(cfg.r_floor))
        floor_hits = 0
        elig = self.eligible
        strong = cfg.heredity == "strong"
        alpha = np.log(max(y.mean(), 0.1)) - np.mean(self.offset)
        beta = np.zeros(p)
        sigma = np.zeros(p)
        gam = np.zeros(p, bool)
        dlt = np.zeros(p, bool)
        ri = self.ri
        cand = np.ones(p, bool)  # covariate candidates, excluding the random intercept
        if ri >= 0:
            gam[ri] = True  # the constant is always in the model, its mean is alpha
            cand[ri] = False
        n_cand = int(cand.sum())
        Z = rng.standard_normal((n, p))
        pi_b, pi_s1, pi_s0 = 0.5, 0.5, (0.0 if strong else 0.1)
        tau_b2, tau_s2 = cfg.tau_b**2, cfg.tau_s**2
        st = dict(rho0=0.0, rho=np.zeros(qd), eta=np.zeros(qd, bool), pi_rho=0.5,
                  sd0=0.05, sdk=np.full(qd, 0.1))
        logr = np.zeros(N)

        def mean_log():
            return self.offset + alpha + X @ beta + (X * (sigma * Z[site])).sum(axis=1)

        lamlog = mean_log()
        n_keep = (cfg.n_iter - cfg.burn) // cfg.thin
        tr = {
            "alpha": np.empty(n_keep), "rho0": np.empty(n_keep),
            "beta": np.empty((n_keep, p)), "sigma": np.empty((n_keep, p)),
            "gamma": np.empty((n_keep, p), bool), "delta": np.empty((n_keep, p), bool),
            "rho": np.empty((n_keep, qd)), "eta": np.empty((n_keep, qd), bool),
            "pi_b": np.empty(n_keep), "pi_s1": np.empty(n_keep), "pi_rho": np.empty(n_keep),
        }
        t0 = time.perf_counter()
        k = 0
        for it in range(cfg.n_iter):
            # 1. dispersion block, Polya-Gamma variables integrated out
            logr = self._dispersion_block(st, lamlog, adapt=it < cfg.burn)
            if logr.min() < self._logr_min:
                floor_hits += 1
            logr_eff = np.clip(logr, self._logr_min, self.LOGR_MAX)
            r = np.exp(logr_eff)
            psi = lamlog - logr_eff
            # 2. Polya-Gamma variables and working response
            omega = np.maximum(random_polyagamma(y + r, psi, random_state=rng), 1e-12)
            ytil = (y - r) / (2.0 * omega)
            # 3. intercept (mean scale)
            prec = omega.sum() + 1.0 / cfg.s_alpha**2
            mean = (omega * (ytil - psi + alpha)).sum() / prec
            new = mean + rng.standard_normal() / np.sqrt(prec)
            psi += new - alpha
            alpha = new
            # 4. (gamma_j, beta_j)
            for j in rng.permutation(p):
                if j == ri:
                    continue
                xj = X[:, j]
                e = ytil - psi + xj * beta[j]
                wx = omega * xj
                S = wx @ xj
                B = wx @ e
                prec_j = S + 1.0 / tau_b2
                log_bf = -0.5 * np.log1p(tau_b2 * S) + 0.5 * B * B / prec_j
                if elig[j]:
                    h1, h0 = (pi_s1, pi_s0) if dlt[j] else (1.0 - pi_s1, 1.0 - pi_s0)
                else:
                    h1 = h0 = 1.0
                if h0 <= 0.0:
                    g_new = True
                else:
                    lo = np.log(pi_b) - np.log1p(-pi_b) + log_bf + np.log(h1) - np.log(h0)
                    g_new = rng.random() < 1.0 / (1.0 + np.exp(-lo))
                b_new = (B / prec_j + rng.standard_normal() / np.sqrt(prec_j)) if g_new else 0.0
                psi += xj * (b_new - beta[j])
                beta[j], gam[j] = b_new, g_new
            # 5. (delta_j, sigma_j, z_.j)
            for j in rng.permutation(np.nonzero(elig)[0]):
                xj = X[:, j]
                u_old = sigma[j] * xj * Z[site, j]
                pij = cfg.pi_ri if j == ri else (pi_s1 if gam[j] else pi_s0)
                if pij <= 0.0:
                    psi -= u_old
                    sigma[j], dlt[j] = 0.0, False
                    Z[:, j] = rng.standard_normal(n)
                    continue
                e = ytil - psi + u_old
                q = np.bincount(site, weights=omega * xj * xj, minlength=n)
                b = np.bincount(site, weights=omega * xj * e, minlength=n)
                log_bf, grid, logf = self._bf_sigma(q, b, tau_s2)
                lo = np.log(pij) - np.log1p(-pij) + log_bf
                d_new = rng.random() < 1.0 / (1.0 + np.exp(-np.clip(lo, -700, 700)))
                if d_new:
                    s = self._draw_abs_sigma(grid, logf) * (1.0 if rng.random() < 0.5 else -1.0)
                    den = 1.0 + s * s * q
                    Z[:, j] = s * b / den + rng.standard_normal(n) / np.sqrt(den)
                else:
                    s = 0.0
                    Z[:, j] = rng.standard_normal(n)
                psi += s * xj * Z[site, j] - u_old
                sigma[j], dlt[j] = s, d_new
            if cfg.shift_moves:
                alpha = self._shift_moves(alpha, beta, sigma, gam, dlt, Z, ri, tau_b2)
            # 6. hyperparameters
            kb = int((gam & cand).sum())
            pi_b = rng.beta(cfg.a_b + kb, cfg.b_b + n_cand - kb)
            e1 = elig & gam & cand
            pi_s1 = rng.beta(cfg.a_s + (dlt & e1).sum(), cfg.b_s + (~dlt & e1).sum())
            if not strong:
                e0 = elig & ~gam & cand
                pi_s0 = rng.beta(cfg.a_s0 + (dlt & e0).sum(), cfg.b_s0 + (~dlt & e0).sum())
            ke = int(st["eta"].sum())
            st["pi_rho"] = rng.beta(cfg.a_rho + ke, cfg.b_rho + qd - ke)
            lamlog = mean_log()
            if it >= cfg.burn and (it - cfg.burn) % cfg.thin == 0 and k < n_keep:
                tr["alpha"][k], tr["rho0"][k] = alpha, st["rho0"]
                tr["beta"][k], tr["sigma"][k] = beta, sigma
                tr["gamma"][k], tr["delta"][k] = gam, dlt
                tr["rho"][k], tr["eta"][k] = st["rho"], st["eta"]
                tr["pi_b"][k], tr["pi_s1"][k], tr["pi_rho"][k] = pi_b, pi_s1, st["pi_rho"]
                if self.on_keep is not None:
                    self.on_keep(dict(kind="mean", alpha=alpha, beta=beta, sigma=sigma, Z=Z,
                                      rho0=st["rho0"], rho=st["rho"]))
                k += 1
            if cfg.verbose and (it + 1) % 500 == 0:
                print(f"iter {it + 1}: rho0={st['rho0']:.2f} disp={ke} fixed={kb} random={int(dlt.sum())} "
                      f"{time.perf_counter() - t0:.1f}s", flush=True)
        tr["r"] = np.exp(tr["rho0"])
        tr["floor_hits"] = floor_hits  # sweeps in which the dispersion floor was reached
        tr["seconds"] = time.perf_counter() - t0
        return tr

    # ------------------------------------------------------------------ main loop
    def run(self):
        if self.W is not None:
            return self._run_dispersion()
        cfg, rng = self.cfg, self.rng
        y, X, site, n, N, p = self.y, self.X, self.site, self.n, self.N, self.p
        elig = self.eligible
        strong = cfg.heredity == "strong"

        # initial state: everything excluded
        r = 1.0
        alpha = np.log(max(y.mean(), 0.1)) - np.mean(self.offset) - np.log(r)
        beta = np.zeros(p)
        sigma = np.zeros(p)
        gam = np.zeros(p, bool)
        dlt = np.zeros(p, bool)
        ri = self.ri
        cand = np.ones(p, bool)  # covariate candidates, excluding the random intercept
        if ri >= 0:
            gam[ri] = True  # the constant is always in the model, its mean is alpha
            cand[ri] = False
        n_cand = int(cand.sum())
        Z = rng.standard_normal((n, p))
        pi_b, pi_s1, pi_s0 = 0.5, 0.5, (0.0 if strong else 0.1)
        tau_b2, xi_b = cfg.tau_b**2, 1.0
        tau_s2, xi_s = cfg.tau_s**2, 1.0

        def full_psi():
            return self.offset + alpha + X @ beta + (X * (sigma * Z[site])).sum(axis=1)

        psi = full_psi()
        n_keep = (cfg.n_iter - cfg.burn) // cfg.thin
        tr = {
            "alpha": np.empty(n_keep),
            "r": np.empty(n_keep),
            "beta": np.empty((n_keep, p)),
            "sigma": np.empty((n_keep, p)),
            "gamma": np.empty((n_keep, p), bool),
            "delta": np.empty((n_keep, p), bool),
            "tau_b": np.empty(n_keep),
            "tau_s": np.empty(n_keep),
            "pi_b": np.empty(n_keep),
            "pi_s1": np.empty(n_keep),
        }
        t0 = time.perf_counter()
        k = 0
        for it in range(cfg.n_iter):
            # 1. dispersion via CRT, Polya-Gamma variables integrated out
            L = _crt(y.astype(np.int64), r, rng)
            r = rng.gamma(cfg.a_r + L.sum(), 1.0 / (cfg.b_r + np.logaddexp(0.0, psi).sum()))
            # 2. Polya-Gamma variables and working response
            omega = random_polyagamma(y + r, psi, random_state=rng)
            omega = np.maximum(omega, 1e-12)
            ytil = (y - r) / (2.0 * omega)
            # 3. intercept
            prec = omega.sum() + 1.0 / cfg.s_alpha**2
            mean = (omega * (ytil - psi + alpha)).sum() / prec
            new = mean + rng.standard_normal() / np.sqrt(prec)
            psi += new - alpha
            alpha = new

            # 4. (gamma_j, beta_j), beta_j integrated out for the indicator
            for j in rng.permutation(p):
                if j == ri:
                    continue
                xj = X[:, j]
                e = ytil - psi + xj * beta[j]
                wx = omega * xj
                S = wx @ xj
                B = wx @ e
                prec_j = S + 1.0 / tau_b2
                log_bf = -0.5 * np.log1p(tau_b2 * S) + 0.5 * B * B / prec_j
                if elig[j]:
                    if dlt[j]:
                        h1, h0 = pi_s1, pi_s0
                    else:
                        h1, h0 = 1.0 - pi_s1, 1.0 - pi_s0
                else:
                    h1 = h0 = 1.0
                if h0 <= 0.0:
                    g_new = True
                else:
                    lo = np.log(pi_b) - np.log1p(-pi_b) + log_bf + np.log(h1) - np.log(h0)
                    g_new = rng.random() < 1.0 / (1.0 + np.exp(-lo))
                b_new = (B / prec_j + rng.standard_normal() / np.sqrt(prec_j)) if g_new else 0.0
                psi += xj * (b_new - beta[j])
                beta[j], gam[j] = b_new, g_new

            # 5. (delta_j, sigma_j, z_.j), z_.j integrated out for the indicator and sigma_j
            for j in rng.permutation(np.nonzero(elig)[0]):
                xj = X[:, j]
                u_old = sigma[j] * xj * Z[site, j]
                pij = cfg.pi_ri if j == ri else (pi_s1 if gam[j] else pi_s0)
                if pij <= 0.0:
                    psi -= u_old
                    sigma[j], dlt[j] = 0.0, False
                    Z[:, j] = rng.standard_normal(n)
                    continue
                e = ytil - psi + u_old
                q = np.bincount(site, weights=omega * xj * xj, minlength=n)
                b = np.bincount(site, weights=omega * xj * e, minlength=n)
                log_bf, grid, logf = self._bf_sigma(q, b, tau_s2)
                lo = np.log(pij) - np.log1p(-pij) + log_bf
                d_new = rng.random() < 1.0 / (1.0 + np.exp(-np.clip(lo, -700, 700)))
                if d_new:
                    s = self._draw_abs_sigma(grid, logf) * (1.0 if rng.random() < 0.5 else -1.0)
                    den = 1.0 + s * s * q
                    Z[:, j] = s * b / den + rng.standard_normal(n) / np.sqrt(den)
                else:
                    s = 0.0
                    Z[:, j] = rng.standard_normal(n)
                psi += s * xj * Z[site, j] - u_old
                sigma[j], dlt[j] = s, d_new

            if cfg.shift_moves:
                alpha = self._shift_moves(alpha, beta, sigma, gam, dlt, Z, ri, tau_b2)
            # 6. hyperparameters
            kb = int((gam & cand).sum())
            pi_b = rng.beta(cfg.a_b + kb, cfg.b_b + n_cand - kb)
            e1 = elig & gam & cand
            pi_s1 = rng.beta(cfg.a_s + (dlt & e1).sum(), cfg.b_s + (~dlt & e1).sum())
            if not strong:
                e0 = elig & ~gam & cand
                pi_s0 = rng.beta(cfg.a_s0 + (dlt & e0).sum(), cfg.b_s0 + (~dlt & e0).sum())
            ks = int(dlt.sum())
            if cfg.learn_tau:
                tau_b2 = 1.0 / rng.gamma((kb + 1) / 2.0, 1.0 / (1.0 / xi_b + 0.5 * (beta[gam] ** 2).sum()))
                xi_b = 1.0 / rng.gamma(1.0, 1.0 / (1.0 / cfg.A_b**2 + 1.0 / tau_b2))
                tau_s2 = 1.0 / rng.gamma((ks + 1) / 2.0, 1.0 / (1.0 / xi_s + 0.5 * (sigma[dlt] ** 2).sum()))
                xi_s = 1.0 / rng.gamma(1.0, 1.0 / (1.0 / cfg.A_s**2 + 1.0 / tau_s2))

            psi = full_psi()  # guard against drift from incremental updates

            if it >= cfg.burn and (it - cfg.burn) % cfg.thin == 0 and k < n_keep:
                tr["alpha"][k] = alpha
                tr["r"][k] = r
                tr["beta"][k] = beta
                tr["sigma"][k] = sigma
                tr["gamma"][k] = gam
                tr["delta"][k] = dlt
                tr["tau_b"][k] = np.sqrt(tau_b2)
                tr["tau_s"][k] = np.sqrt(tau_s2)
                tr["pi_b"][k] = pi_b
                tr["pi_s1"][k] = pi_s1
                if self.on_keep is not None:
                    self.on_keep(dict(kind="logodds", alpha=alpha, beta=beta, sigma=sigma, Z=Z, r=r))
                k += 1
            if cfg.verbose and (it + 1) % 500 == 0:
                print(f"iter {it + 1}: r={r:.2f} fixed={kb} random={ks} "
                      f"{time.perf_counter() - t0:.1f}s", flush=True)
        tr["seconds"] = time.perf_counter() - t0
        return tr
