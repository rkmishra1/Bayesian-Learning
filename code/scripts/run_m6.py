"""Simulation driver for M6: squared covariates as fixed-only mean candidates (as in M5) plus
selection of dispersion covariates, log r_it = rho0 + x_it' rho with spike-and-slab rho.
Uses the same seeds as scripts/run_sim.py, so M6 sees the same data sets as M1 to M5.

Example:
    python scripts/run_m6.py --dgp C --cells 2,3 --reps 100 --jobs 4 --out results/dgpC/raw_m6
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
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpsel import DGPConfig, RPSelector, SamplerConfig, decide_states, simulate, state_probabilities  # noqa: E402
from rpsel.dgp import P, P_CONT  # noqa: E402
from rpsel.summaries import effect_summaries  # noqa: E402

CELLS = [(300, 1), (300, 5), (1000, 1), (1000, 5)]


def run_one(dgp, cell_id, n, T, rep, outdir, n_iter, burn):
    path = outdir / f"cell{cell_id}_rep{rep:03d}.json"
    if path.exists():
        return str(path)
    seed = 100_000 * (ord(dgp) - 64) + 1_000 * cell_id + rep  # identical to run_sim.py
    d = simulate(DGPConfig(dgp=dgp, n=n, T=T, r=1.0, rho=0.5), seed=seed)
    X = d["X"]
    Q = X[:, :P_CONT] ** 2
    Q = Q / Q.std(axis=0)
    XX = np.hstack([X, Q])
    elig = np.r_[np.ones(P, bool), np.zeros(P_CONT, bool)]
    cfg = SamplerConfig(n_iter=n_iter, burn=burn, seed=seed + 4)
    tr = RPSelector(d["y"], XX, d["site"], d["offset"], eligible=elig, config=cfg, W=X).run()
    Pm = state_probabilities(tr)
    st = decide_states(tr)
    es = effect_summaries(tr)
    eta = tr["eta"]
    rho_mean = np.where(eta.any(axis=0), (tr["rho"] * eta).sum(axis=0) / np.maximum(eta.sum(axis=0), 1), 0.0)
    m6 = {
        "states": st[:P].tolist(),
        "P": Pm[:P].round(4).tolist(),
        "quad_states": st[P:].tolist(),
        "P_disp": eta.mean(axis=0).round(4).tolist(),
        "rho_mean_given_in": rho_mean.round(4).tolist(),
        "rho0_mean": float(tr["rho0"].mean()),
        "floor_hits": int(tr["floor_hits"]),
        "seconds": float(tr["seconds"]),
    }
    for key, val in es.items():
        m6[key] = np.asarray(val)[:P].round(5).tolist()
    rec = {"dgp": dgp, "cell": cell_id, "n": n, "T": T, "rep": rep, "seed": seed, "M6": m6}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    tmp.rename(path)
    return str(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dgp", default="C")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--n_iter", type=int, default=3000)
    ap.add_argument("--burn", type=int, default=1000)
    ap.add_argument("--cells", default="0,1,2,3")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    cells = [int(c) for c in a.cells.split(",")]
    tasks = sorted([(a.dgp, c, *CELLS[c], rep) for c in cells for rep in range(a.reps)],
                   key=lambda t: -(t[2] * t[3]))
    t0 = time.time()
    done = 0
    for _ in Parallel(n_jobs=a.jobs, return_as="generator_unordered")(
        delayed(run_one)(*t, outdir, a.n_iter, a.burn) for t in tasks
    ):
        done += 1
        if done % 10 == 0 or done == len(tasks):
            print(f"{done}/{len(tasks)} done, {time.time() - t0:.0f}s elapsed", flush=True)


if __name__ == "__main__":
    main()
