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

    def __init__(self, y, X, site, offset=None, eligible=None, config: SamplerConfig | None = None):
        self.cfg = config or SamplerConfig()
        self.y = np.asarray(y, dtype=np.float64)
        self.X = np.ascontiguousarray(X, dtype=np.float64)
        self.N, self.p = self.X.shape
        site = np.asarray(site)
        _, self.site = np.unique(site, return_inverse=True)
        self.n = int(self.site.max()) + 1
        self.offset = np.zeros(self.N) if offset is None else np.asarray(offset, dtype=np.float64)
        self.eligible = np.ones(self.p, bool) if eligible is None else np.asarray(eligible, bool)
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

    # ------------------------------------------------------------------ main loop
    def run(self):
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
                pij = pi_s1 if gam[j] else pi_s0
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

            # 6. hyperparameters
            kb = int(gam.sum())
            pi_b = rng.beta(cfg.a_b + kb, cfg.b_b + p - kb)
            e1 = elig & gam
            pi_s1 = rng.beta(cfg.a_s + (dlt & e1).sum(), cfg.b_s + (~dlt & e1).sum())
            if not strong:
                e0 = elig & ~gam
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
                k += 1
            if cfg.verbose and (it + 1) % 500 == 0:
                print(f"iter {it + 1}: r={r:.2f} fixed={kb} random={ks} "
                      f"{time.perf_counter() - t0:.1f}s", flush=True)
        tr["seconds"] = time.perf_counter() - t0
        return tr
