"""Simulation driver: runs M1 (sequential SML), M4 (proposed, linear mean) and
M5 (proposed, quadratic fixed-only candidates) on replicated data sets.

Each (cell, replication) is written to its own JSON file so runs can be resumed.

Example:
    python scripts/run_sim.py --dgp A --reps 100 --jobs 4 --out results/dgpA
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
from rpsel.dgp import P, P_CONT, add_quadratic_candidates  # noqa: E402
from rpsel.sml import sequential_sml  # noqa: E402
from rpsel.summaries import effect_summaries  # noqa: E402

CELLS = [(300, 1), (300, 5), (1000, 1), (1000, 5)]


def _bayes(d, quad: bool, seed: int, n_iter: int, burn: int):
    X = d["X"]
    elig = np.ones(P, bool)
    if quad:
        Q = X[:, :P_CONT] ** 2
        Q = Q / Q.std(axis=0)  # scaled, not centered (Remark 1)
        X = np.hstack([X, Q])
        elig = np.r_[elig, np.zeros(P_CONT, bool)]
    cfg = SamplerConfig(n_iter=n_iter, burn=burn, seed=seed)
    tr = RPSelector(d["y"], X, d["site"], d["offset"], eligible=elig, config=cfg).run()
    Pm = state_probabilities(tr)
    st = decide_states(tr)
    es = effect_summaries(tr)
    out = {
        "states": st[:P].tolist(),
        "P": Pm[:P].round(4).tolist(),
        "r_mean": float(tr["r"].mean()),
        "seconds": float(tr["seconds"]),
    }
    for key, val in es.items():
        out[key] = np.asarray(val)[:P].round(5).tolist()
    if quad:
        out["quad_states"] = st[P:].tolist()
        out["quad_P"] = Pm[P:].round(4).tolist()
    return out


def run_one(dgp, cell_id, n, T, rep, outdir, n_iter, burn, R):
    path = outdir / f"cell{cell_id}_rep{rep:03d}.json"
    if path.exists():
        return str(path)
    seed = 100_000 * (ord(dgp) - 64) + 1_000 * cell_id + rep
    d = simulate(DGPConfig(dgp=dgp, n=n, T=T, r=1.0, rho=0.5), seed=seed)
    rec = {
        "dgp": dgp, "cell": cell_id, "n": n, "T": T, "rep": rep, "seed": seed,
        "true_states": d["states"].tolist(),
        "true_beta": d["beta"].tolist(), "true_sigma": d["sigma"].tolist(),
        "y_mean": float(d["y"].mean()), "y_max": int(d["y"].max()),
    }
    t0 = time.perf_counter()
    m1 = sequential_sml(d["y"].astype(float), d["X"], d["site"], d["offset"], R=R, seed=seed)
    rec["M1"] = {k: (np.asarray(v).round(5).tolist() if isinstance(v, np.ndarray) else v) for k, v in m1.items()}
    rec["M4"] = _bayes(d, quad=False, seed=seed + 1, n_iter=n_iter, burn=burn)
    rec["M5"] = _bayes(d, quad=True, seed=seed + 2, n_iter=n_iter, burn=burn)
    rec["wall_seconds"] = time.perf_counter() - t0
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    tmp.rename(path)
    return str(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dgp", default="A")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--n_iter", type=int, default=3000)
    ap.add_argument("--burn", type=int, default=1000)
    ap.add_argument("--R", type=int, default=200)
    ap.add_argument("--cells", default="0,1,2,3")
    ap.add_argument("--out", default="results/dgpA/raw")
    a = ap.parse_args()
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    cells = [int(c) for c in a.cells.split(",")]
    # largest cells first so the pool finishes evenly
    tasks = sorted(
        [(a.dgp, c, *CELLS[c], rep) for c in cells for rep in range(a.reps)],
        key=lambda t: -(t[2] * t[3]),
    )
    t0 = time.time()
    done = 0
    for path in Parallel(n_jobs=a.jobs, return_as="generator_unordered")(
        delayed(run_one)(*t, outdir, a.n_iter, a.burn, a.R) for t in tasks
    ):
        done += 1
        if done % 10 == 0 or done == len(tasks):
            print(f"{done}/{len(tasks)} done, {time.time() - t0:.0f}s elapsed", flush=True)


if __name__ == "__main__":
    main()
