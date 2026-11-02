# Code: Bayesian selection of fixed and random parameters

Python implementation of the three-state sampler in the manuscript (`../manuscript`), the
conventional sequential baseline, the simulation study and the application to British
major-road links.

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
| `scripts/build_uk_data.py`, `data/` | Builds the British major-road panel from public DfT data (see `data/README.md`) |
| `rpsel/application.py` | Application design matrix and hold-out predictive density |
| `scripts/run_uk.py`, `scripts/analyze_uk.py` | Application driver (one method and chain per call) and its tables, figure and diagnostics |
| `results/uk_long/` | Application results reported in Section 6 (8,000 sweeps). `results/uk_pilot/` holds the 3,000-sweep pilot runs |

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
ten Bayesian fits per data set), and DGP-B and DGP-C about 75 minutes each. M6 takes
about one hour for DGP-A and 45 minutes each for DGP-B and DGP-C.

## Reproduce the application

Download the three DfT files listed in `data/README.md` into a folder of their own, then

```bash
python scripts/build_uk_data.py --collisions RAW/dft-road-casualty-statistics-collision-1979-latest-published-year.csv \
    --aadf RAW/dft_traffic_counts_aadf.csv --mrdb RAW/MRDB_2019_published.shp --out data/uk_links_2015_2019.csv.gz
python scripts/run_uk.py --method M6 --chain 1 --n_iter 8000 --burn 4000
python scripts/run_uk.py --method M6 --chain 2 --n_iter 8000 --burn 4000
python scripts/run_uk.py --method M4 --chain 1 --n_iter 8000 --burn 4000
python scripts/run_uk.py --method M5 --chain 1 --n_iter 8000 --burn 4000
python scripts/run_uk.py --method M1
python scripts/analyze_uk.py --fig ../manuscript/figures --tab ../manuscript/tables
```

Each M6 chain takes about 85 minutes on one core, M4 and M5 about 65 and M1 about 26.
Sensitivity runs use `--tag` and `--set`, for example
`--tag _tau025 --set tau_s=0.25 --n_iter 6000 --burn 3000`.

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
4. **Random intercept and centering in the application.** In panel data a random slope on
   a covariate far from zero can mimic a site effect, so the application adds a random
   intercept candidate (`random_intercept=True`, prior inclusion 1/2) and centers the
   continuous covariates. The option is off in the simulations, which reproduce exactly.
5. **Run length in the application.** With 17,139 links, 3,000-sweep pilot chains of M6
   disagreed on four candidates. The reported runs use 8,000 sweeps with 4,000 burn-in,
   after which two M6 chains agree on every state. Optional translation moves
   (`shift_moves=True`) shift a mean and its site draws together, leave the posterior
   unchanged and doubled the effective sample size of the intercept in a test, but were not
   used in the reported runs.
