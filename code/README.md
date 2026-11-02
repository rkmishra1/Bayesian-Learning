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
| `rpsel/bayes_seq.py` | Method M2: Bayesian backward elimination on 95% credible intervals, then one-at-a-time random specification by conditional DIC |
| `rpsel/bic_search.py` | Method M3: two-stage greedy BIC search (excluded/fixed, then fixed/random) |
| `scripts/run_m23.py`, `scripts/analyze_bc.py` | Driver for M2/M3, analysis of DGP-B and DGP-C |
| `scripts/run_m6.py` | Driver for M6: squared candidates plus selection of dispersion covariates, log r = rho0 + x'rho |
| `scripts/check_formulas.py` | Monte Carlo and quadrature checks of Propositions 1 to 3 and the Bayes factors |
| `results/dgpA/` | Raw replications, summary CSVs and the run log for DGP-A |

## Reproduce DGP-A

```bash
python scripts/run_sim.py --dgp A --reps 100 --jobs 4 --out results/dgpA/raw
python scripts/analyze.py --raw results/dgpA/raw --out results/dgpA \
    --fig ../manuscript/figures --tab ../manuscript/tables
```

Competing procedures M2 (sequential Bayes with DIC) and M3 (BIC search) use the same
seeds and are merged by `analyze.py`:

```bash
python scripts/run_m23.py --dgp A --reps 100 --jobs 4 --out results/dgpA/raw_m23
```

Omitted curvature (DGP-B) and varying dispersion (DGP-C), n = 1000 only:

```bash
python scripts/run_sim.py --dgp B --reps 100 --jobs 4 --cells 2,3 --out results/dgpB/raw
python scripts/run_sim.py --dgp C --reps 100 --jobs 4 --cells 2,3 --out results/dgpC/raw
python scripts/analyze_bc.py --fig ../manuscript/figures --tab ../manuscript/tables
```

M6 (dispersion selection) on the same data sets, before running the two analysis scripts:

```bash
python scripts/run_m6.py --dgp A --reps 100 --jobs 4 --out results/dgpA/raw_m6
python scripts/run_m6.py --dgp B --reps 100 --jobs 4 --cells 2,3 --out results/dgpB/raw_m6
python scripts/run_m6.py --dgp C --reps 100 --jobs 4 --cells 2,3 --out results/dgpC/raw_m6
```

Seeds are derived from the DGP, cell and replication index, so reruns reproduce the
stored results, and every driver skips replications that already have a result file.
With 4 cores the DGP-A run takes about 1.5 hours, M2/M3 about 2.5 hours (M2 needs about
ten Bayesian fits per data set), and DGP-B and DGP-C about 75 minutes each.

## Settings used

* Sampler: 3,000 iterations, 1,000 burn-in, one chain, strong heredity, slab scales fixed
  at 0.5 for both the means and the signed standard deviations, Beta(1, 1) inclusion
  priors, median probability rule.
* M1: 200 scrambled Halton draws per site, BHHH standard errors, 5% level for screening.
* M5 adds the squares of the ten continuous covariates (scaled, not centered) as
  fixed-only candidates.
* M6 is M5 plus log r_it = rho0 + x_it' rho with point-mass spike-and-slab priors on rho
  (slab scale 0.5, Beta(1, 1) inclusion, rho0 ~ N(0, 2^2)). Each (eta_k, rho_k) pair is
  updated by a reversible jump move with a Laplace proposal built from the rho_k = 0 state,
  plus a random-walk move when active. The dispersion is bounded below at 0.01 in the
  likelihood, its derivatives and the Polya-Gamma step, and the trace records how many
  sweeps reached the bound (`floor_hits`).

## Decisions taken during implementation

1. **Slab scales are fixed, not learned.** With half-Cauchy priors on the slab scales,
   pilot runs at n = 300 included almost every candidate: the slab shrank toward the
   spike, so many small effects fitted as well as a few large ones. Fixing the scales at
   0.5 removed this. The learned version remains available (`learn_tau=True`).
2. **sigma = 0.25 on the continuous random parameters in DGP-A.** Because covariates are
   not centered, sigma_j x_ij z_ij grows with |x_ij|. With sigma = 0.5 the simulated
   counts reached 2,000 against a mean of 2. With 0.25 the 99.9th percentile is below 50.
   This was decided on realism grounds before any method comparison.
3. **Lower bound on the dispersion in M6.** In a few replications with 300 sites observed
   once, the dispersion drifted toward 1e-4 at sites with no crashes, where the Polya-Gamma
   sampler cannot draw with so small a shape parameter. A bound of 0.01 (variance up to
   mu + 100 mu^2) removed the failures. It was reached in 49% of the n = 300, T = 1 DGP-A
   replications (median 4 of 3,000 sweeps), in 2% at n = 1000, T = 1, and never in the panel
   designs or in DGP-C, whose results are identical with and without the bound.
