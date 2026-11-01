"""Between-chain agreement of posterior state probabilities with the final sampler settings.

For one data set per design (replication 0, same seeds as the simulation), run two chains
from different seeds and report the largest absolute difference in P(E), P(F), P(R).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rpsel import DGPConfig, RPSelector, SamplerConfig, simulate, state_probabilities  # noqa: E402

CELLS = [(300, 1), (300, 5), (1000, 1), (1000, 5)]


def one(cell_id, chain_seed):
    n, T = CELLS[cell_id]
    seed = 100_000 * 1 + 1_000 * cell_id + 0
    d = simulate(DGPConfig(dgp="A", n=n, T=T), seed=seed)
    tr = RPSelector(d["y"], d["X"], d["site"], d["offset"],
                    config=SamplerConfig(n_iter=3000, burn=1000, seed=chain_seed)).run()
    return cell_id, chain_seed, state_probabilities(tr)


if __name__ == "__main__":
    jobs = [(c, s) for c in range(4) for s in (901, 902)]
    out = Parallel(n_jobs=4)(delayed(one)(c, s) for c, s in jobs)
    res = {(c, s): P for c, s, P in out}
    for c in range(4):
        diff = np.abs(res[(c, 901)] - res[(c, 902)])
        print(f"n={CELLS[c][0]} T={CELLS[c][1]}: max |dP| = {diff.max():.3f}, mean |dP| = {diff.mean():.3f}")
