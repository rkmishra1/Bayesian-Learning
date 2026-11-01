# Code: Bayesian selection of fixed and random parameters

Python implementation of the three-state sampler in the manuscript (`../manuscript`), the
conventional sequential baseline, and the simulation study.

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Layout

| Path | Content |
|---|---|
| `rpsel/dgp.py` | Data-generating processes A to D (Section 5) |
| `rpsel/sampler.py` | Three-state spike-and-slab Gibbs sampler (Box 1): CRT and Pólya-Gamma augmentation, collapsed fixed-effect update, random-parameter block with the site draws integrated out (Proposition 3) |
| `rpsel/sml.py` | Method M1: backward elimination in a fixed NB model, then random parameters NB by simulated maximum likelihood (scrambled Halton draws), fixing the weakest random parameter until all remaining standard deviations have \|t\| >= 1.96 |
| `rpsel/summaries.py` | Posterior state probabilities, decision rules, effect summaries |
| `scripts/run_sim.py` | Simulation driver (parallel, resumable, one JSON per replication) |
| `scripts/analyze.py` | Metrics, LaTeX table rows and figures |
| `scripts/check_formulas.py` | Monte Carlo and quadrature checks of Propositions 1 to 3 and the Bayes factors |
| `results/dgpA/` | Raw replications, summary CSVs and the run log for DGP-A |

## Reproduce DGP-A

```bash
python scripts/run_sim.py --dgp A --reps 100 --jobs 4 --out results/dgpA/raw
python scripts/analyze.py --raw results/dgpA/raw --out results/dgpA \
    --fig ../manuscript/figures --tab ../manuscript/tables
```

Seeds are derived from the DGP, cell and replication index, so reruns reproduce the
stored results. With 4 cores the full DGP-A run takes about two hours.

## Settings used

* Sampler: 3,000 iterations, 1,000 burn-in, one chain, strong heredity, slab scales fixed
  at 0.5 for both the means and the signed standard deviations, Beta(1, 1) inclusion
  priors, median probability rule.
* M1: 200 scrambled Halton draws per site, BHHH standard errors, 5% level for screening.
* M5 adds the squares of the ten continuous covariates (scaled, not centered) as
  fixed-only candidates.

## Decisions taken during implementation

1. **Slab scales are fixed, not learned.** With half-Cauchy priors on the slab scales,
   pilot runs at n = 300 included almost every candidate: the slab shrank toward the
   spike, so many small effects fitted as well as a few large ones. Fixing the scales at
   0.5 removed this. The learned version remains available (`learn_tau=True`).
2. **sigma = 0.25 on the continuous random parameters in DGP-A.** Because covariates are
   not centered, sigma_j x_ij z_ij grows with |x_ij|. With sigma = 0.5 the simulated
   counts reached 2,000 against a mean of 2. With 0.25 the 99.9th percentile is below 50.
   This was decided on realism grounds before any method comparison.
