"""Simulation driver for the competing procedures M2 (sequential Bayesian with DIC) and
M3 (two-stage BIC search). Uses the same seeds as scripts/run_sim.py, so every method is
applied to identical data sets. Each (cell, replication) is written to its own JSON file.

Example:
    python scripts/run_m23.py --dgp A --reps 100 --jobs 4 --out results/dgpA/raw_m23
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

from rpsel import DGPConfig, simulate  # noqa: E402
from rpsel.bayes_seq import sequential_bayes  # noqa: E402
from rpsel.bic_search import bic_search  # noqa: E402

CELLS = [(300, 1), (300, 5), (1000, 1), (1000, 5)]


def _clean(d):
    return {k: (np.asarray(v).round(5).tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}


def run_one(dgp, cell_id, n, T, rep, outdir, R):
    path = outdir / f"cell{cell_id}_rep{rep:03d}.json"
    if path.exists():
        return str(path)
    seed = 100_000 * (ord(dgp) - 64) + 1_000 * cell_id + rep  # identical to run_sim.py
    d = simulate(DGPConfig(dgp=dgp, n=n, T=T, r=1.0, rho=0.5), seed=seed)
    y = d["y"].astype(float)
    rec = {"dgp": dgp, "cell": cell_id, "n": n, "T": T, "rep": rep, "seed": seed}
    rec["M2"] = _clean(sequential_bayes(y, d["X"], d["site"], d["offset"], seed=seed + 3))
    rec["M3"] = _clean(bic_search(y, d["X"], d["site"], d["offset"], R=R, seed=seed))
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    tmp.rename(path)
    return str(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dgp", default="A")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--R", type=int, default=200)
    ap.add_argument("--cells", default="0,1,2,3")
    ap.add_argument("--out", default="results/dgpA/raw_m23")
    a = ap.parse_args()
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    cells = [int(c) for c in a.cells.split(",")]
    tasks = sorted([(a.dgp, c, *CELLS[c], rep) for c in cells for rep in range(a.reps)],
                   key=lambda t: -(t[2] * t[3]))
    t0 = time.time()
    done = 0
    for _ in Parallel(n_jobs=a.jobs, return_as="generator_unordered")(
        delayed(run_one)(*t, outdir, a.R) for t in tasks
    ):
        done += 1
        if done % 10 == 0 or done == len(tasks):
            print(f"{done}/{len(tasks)} done, {time.time() - t0:.0f}s elapsed", flush=True)


if __name__ == "__main__":
    main()
