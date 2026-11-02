"""Application: British motorway and A-road links, 2015-2019 (Section 6).

Fits on 2015-2018 and scores the 2019 counts of the same links. One run per call:

    python scripts/run_uk.py --method M4 --chain 1
    python scripts/run_uk.py --method M1

Methods: M4 (linear mean), M5 (squares of the continuous candidates as fixed-only candidates),
M6 (M5 plus dispersion selection), all with a random intercept candidate, and M1 (sequential
simulated maximum likelihood with a random constant candidate).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpsel import RPSelector, SamplerConfig, decide_states, state_probabilities  # noqa: E402
from rpsel.application import Holdout, build_design, nb_logpmf  # noqa: E402
from rpsel.sml import _SML, sequential_sml  # noqa: E402

TRAIN, TEST = range(2015, 2019), [2019]


def run_bayes(D, method, chain, n_iter, burn, out):
    tr, te = D["train"], D["test"]
    p = D["X"].shape[1]
    XX = D["X"] if method == "M4" else np.hstack([D["X"], D["Q"]])
    elig = np.r_[np.ones(p, bool), np.zeros(XX.shape[1] - p, bool)]
    W = D["X"] if method == "M6" else None
    ho = Holdout(XX[te], D["site"][te], D["off"][te], D["y"][te], W=None if W is None else W[te])
    cfg = SamplerConfig(n_iter=n_iter, burn=burn, seed=1000 * chain + {"M4": 4, "M5": 5, "M6": 6}[method],
                        random_intercept=True, verbose=True)
    trace = RPSelector(D["y"][tr], XX[tr], D["site"][tr], D["off"][tr], eligible=elig, config=cfg,
                       W=None if W is None else W[tr], on_keep=ho).run()
    pred, lpd, mean = ho.summary()
    keys = ["alpha", "beta", "sigma", "gamma", "delta", "pi_b", "pi_s1"] + (
        ["rho0", "rho", "eta", "pi_rho"] if method == "M6" else ["r"])
    np.savez_compressed(out / f"{method}_c{chain}_trace.npz", **{k: trace[k] for k in keys})
    np.save(out / f"{method}_c{chain}_lpd.npy", lpd)
    Pm = state_probabilities(trace)
    rec = dict(method=method, chain=chain, n_iter=n_iter, burn=burn, seconds=float(trace["seconds"]),
               names=D["names"], P=Pm.round(4).tolist(), states=decide_states(trace).tolist(),
               P_ri=float(trace["delta"][:, -1].mean()), predict=pred)
    if method == "M6":
        rec["P_disp"] = trace["eta"].mean(axis=0).round(4).tolist()
        rec["floor_hits"] = int(trace["floor_hits"])
    return rec


def m1_predict(D, res, R, seed):
    """Predictive density of the 2019 counts given the 2015-2018 data, by importance weights
    over the Halton draws of each site (the simulated posterior of the site's coefficients)."""
    tr, te = D["train"], D["test"]
    active, rand, theta = res["active"], res["rand"], res["theta"]
    k = len(active)
    Xa = D["X"][:, active]
    Xr = np.hstack([Xa, np.ones((Xa.shape[0], 1))])
    model = _SML(D["y"][tr], Xa[tr], D["site"][tr], D["off"][tr], rand, R, seed, Xr=Xr[tr])
    alpha, beta, sig, r = theta[0], theta[1:k + 1], theta[k + 1:k + 1 + len(rand)], np.exp(theta[-1])
    eps = np.einsum("nrm,nm->nr", model.Hs, Xr[tr][:, rand] * sig)
    loglam = (D["off"][tr] + alpha + Xa[tr] @ beta)[:, None] + eps
    s_ir = np.add.reduceat(nb_logpmf(D["y"][tr][:, None], loglam, r), model.starts, axis=0)
    w = np.exp(s_ir - s_ir.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    site_te = D["site"][te]
    eps_te = np.einsum("nrm,nm->nr", model.H[site_te], Xr[te][:, rand] * sig)
    loglam_te = (D["off"][te] + alpha + Xa[te] @ beta)[:, None] + eps_te
    yt = D["y"][te]
    lp = nb_logpmf(yt[:, None], loglam_te, r) + np.log(w[site_te])
    mx = lp.max(axis=1, keepdims=True)
    lpd = (mx + np.log(np.exp(lp - mx).sum(axis=1, keepdims=True))).ravel()
    mean = (np.exp(loglam_te) * w[site_te]).sum(axis=1)
    pred = dict(lpd=float(lpd.sum()), lpd_mean=float(lpd.mean()), mae=float(np.abs(mean - yt).mean()),
                rmse=float(np.sqrt(((mean - yt) ** 2).mean())), n=int(yt.size))
    return pred, lpd


def run_m1(D, R, seed, out):
    tr = D["train"]
    res = sequential_sml(D["y"][tr], D["X"][tr], D["site"][tr], D["off"][tr], R=R, seed=seed, rand_const=True)
    pred, lpd = m1_predict(D, res, R, seed)
    np.save(out / "M1_c1_lpd.npy", lpd)
    p = D["X"].shape[1]
    rec = dict(method="M1", chain=1, seconds=float(res["seconds"]), names=D["names"],
               states=res["states"].tolist(), beta=res["beta"].tolist(), sigma=res["sigma"].tolist(),
               beta_se=np.nan_to_num(res["beta_se"], nan=-1).tolist(),
               sigma_se=np.nan_to_num(res["sigma_se"], nan=-1).tolist(),
               const_sigma=float(res["const_sigma"]), const_sigma_se=float(np.nan_to_num(res["const_sigma_se"], nan=-1)),
               alpha=float(res["alpha"]), r=res["r"], loglik=res["loglik"], n_refits=res["n_refits"],
               R=R, predict=pred, p=p)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["M1", "M4", "M5", "M6"])
    ap.add_argument("--chain", type=int, default=1)
    ap.add_argument("--n_iter", type=int, default=3000)
    ap.add_argument("--burn", type=int, default=1000)
    ap.add_argument("--R", type=int, default=200)
    ap.add_argument("--data", default="data/uk_links_2015_2019.csv.gz")
    ap.add_argument("--out", default="results/uk")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{a.method}_c{a.chain}.json"
    if target.exists():
        print(f"{target} exists")
        return
    D = build_design(pd.read_csv(a.data), TRAIN, TEST)
    t0 = time.time()
    if a.method == "M1":
        rec = run_m1(D, a.R, seed=2015, out=out)
    else:
        rec = run_bayes(D, a.method, a.chain, a.n_iter, a.burn, out)
    rec["wall_seconds"] = time.time() - t0
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    tmp.rename(target)
    print(json.dumps(rec["predict"]), f"{rec['wall_seconds']:.0f}s")


if __name__ == "__main__":
    main()
