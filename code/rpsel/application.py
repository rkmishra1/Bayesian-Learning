"""Design matrix and hold-out scoring for the British major-road application (Section 6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gammaln

# name, label, column builder, time-varying?
CONTINUOUS = [
    ("log_aadf", "log AADF", lambda d: np.log(d.all_motor_vehicles), True),
    ("hgv_share", "HGV share", lambda d: d.all_HGVs / d.all_motor_vehicles, True),
    ("lgv_share", "LGV share", lambda d: d.LGVs / d.all_motor_vehicles, True),
    ("bus_share", "Bus and coach share", lambda d: d.buses_and_coaches / d.all_motor_vehicles, True),
    ("mc_share", "Motorcycle share", lambda d: d.two_wheeled_motor_vehicles / d.all_motor_vehicles, True),
    ("log_cycles", "log(1 + pedal cycles)", lambda d: np.log1p(d.pedal_cycles), True),
    ("log_length", "log link length", lambda d: np.log(d.link_length_km), False),
    ("curvature", "Curvature index", lambda d: np.log(np.maximum(d.link_length_km / d.chord_km, 1.0)), False),
    ("node_degree", "Junction degree", lambda d: d.node_degree.astype(float), False),
    ("road_density", "Major-road density", lambda d: d.road_density, False),
]
BINARY = [
    ("motorway", "Motorway", lambda d: d.road_category.isin(["TM", "PM"]).astype(float)),
    ("trunk_a", "Trunk A road", lambda d: (d.road_category == "TA").astype(float)),
    ("london", "London", lambda d: (d.region_name == "London").astype(float)),
    ("scotland", "Scotland", lambda d: (d.region_name == "Scotland").astype(float)),
    ("wales", "Wales", lambda d: (d.region_name == "Wales").astype(float)),
]
NAMES = [c[0] for c in CONTINUOUS] + [b[0] for b in BINARY]
LABELS = [c[1] for c in CONTINUOUS] + [b[1] for b in BINARY]
P_CONT = len(CONTINUOUS)
WINSOR = (0.005, 0.995)


def build_design(panel: pd.DataFrame, train_years, test_years, outcome="y"):
    """Standardized candidates (continuous: winsorized, then centered and scaled on the
    training years), scaled squares of the continuous candidates, offset log(length)."""
    d = panel.sort_values(["cp", "year"]).reset_index(drop=True)
    raw = np.column_stack([f(d) for _, _, f, _ in CONTINUOUS] + [f(d) for _, _, f in BINARY]).astype(float)
    tr = d.year.isin(train_years).values
    te = d.year.isin(test_years).values
    X = raw.copy()
    lo, hi = np.quantile(raw[tr, :P_CONT], WINSOR, axis=0)
    X[:, :P_CONT] = np.clip(X[:, :P_CONT], lo, hi)
    mu, sd = X[tr, :P_CONT].mean(axis=0), X[tr, :P_CONT].std(axis=0)
    X[:, :P_CONT] = (X[:, :P_CONT] - mu) / sd
    Q = X[:, :P_CONT] ** 2
    Q = Q / Q[tr].std(axis=0)
    site = pd.factorize(d.cp, sort=True)[0]
    off = np.log(d.link_length_km.values)
    y = d[outcome].values.astype(float)
    return dict(y=y, X=X, Q=Q, site=site, off=off, train=tr, test=te, names=NAMES, labels=LABELS,
                center=mu, scale=sd, winsor=(lo, hi), frame=d)


def nb_logpmf(y, loglam, r):
    loglr = np.logaddexp(loglam, np.log(r))
    return gammaln(y + r) - gammaln(r) - gammaln(y + 1.0) + r * (np.log(r) - loglr) + y * (loglam - loglr)


class Holdout:
    """Accumulates the posterior predictive density of hold-out observations of the same sites.

    Called by RPSelector at each kept iteration. Site draws are those of the current state, so the
    prediction conditions on everything the training years say about each site.
    """

    def __init__(self, X, site, off, y, W=None, r_floor=0.01):
        self.X, self.site, self.off, self.y, self.W = X, site, off, y, W
        self.r_floor = r_floor
        self.acc = np.full(y.size, -np.inf)
        self.mean_sum = np.zeros(y.size)
        self.k = 0

    def __call__(self, s):
        X = self.X
        beta, sigma, Z = s["beta"], s["sigma"], s["Z"]
        if beta.size == X.shape[1] + 1:  # random intercept column appended by the sampler
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        eta = self.off + s["alpha"] + X @ beta + (X * (sigma * Z[self.site])).sum(axis=1)
        if s["kind"] == "logodds":
            r = np.full(eta.size, s["r"])
            loglam = eta + np.log(s["r"])
        else:
            r = np.maximum(np.exp(s["rho0"] + self.W @ s["rho"]), self.r_floor)
            loglam = eta
        self.acc = np.logaddexp(self.acc, nb_logpmf(self.y, loglam, r))
        self.mean_sum += np.exp(loglam)
        self.k += 1

    def summary(self):
        lpd = self.acc - np.log(self.k)
        pred = self.mean_sum / self.k
        return dict(lpd=float(lpd.sum()), lpd_mean=float(lpd.mean()), mae=float(np.abs(pred - self.y).mean()),
                    rmse=float(np.sqrt(((pred - self.y) ** 2).mean())), n=int(self.y.size)), lpd, pred
