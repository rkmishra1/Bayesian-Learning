# Research Ideas: New Bayesian Variable Selection Methods for Transport Data

**Prepared:** 1 November 2026
**Scope:** New Bayesian variable-selection (BVS) methodology, built on the papers in this folder, with transport applications as the motivating data. Kept separate from the graphical-model ideas in `Research-Ideas-Bayesian-Graphical-Shrinkage.md`.

---

## 0. Where the folder stands and what transport needs

The folder provides the prior machinery but is almost entirely **Gaussian linear regression**. Transport outcomes are not:

| Transport problem | Outcome / likelihood | Machinery already in the folder | What is missing |
|---|---|---|---|
| Crash frequency (segments, intersections) | Overdispersed counts: NB, Poisson-lognormal, with exposure offset | Bayesian lasso and adaptive lasso (Kyung et al. 2010, Leng, Tran & Nott 2014), SSL (Ročková & George 2018) | Selection of **how** an effect enters (fixed vs. random parameter), not only **whether** |
| Crash injury severity | Ordered / multinomial logit with random parameters | Regularized logistic via data augmentation (Gramacy & Polson 2012) | Same as above, for discrete-choice models |
| Safety countermeasure evaluation (CMFs) | NB counts, treatment effect | Confounder-aware continuous spike-and-slab (Antonelli, Parmigiani & Dominici 2019) | Count outcome, heterogeneous CMFs, multiple treatments |
| Real-time crash risk / speed prediction from loop detectors | Binary / continuous, many sensor features | Robust and shrinkage priors, plus your cellwise-robust graphical work | Sensor faults are **cellwise** contamination of X |
| Travel-time reliability | Upper quantiles (95th-percentile travel time, buffer index, planning-time index) | Bayesian adaptive lasso QR (Alhamzawi et al. 2012), fused adaptive lasso QR (Sun, Wang & Fuentes 2016), graph-Laplacian prior (Liu et al. 2014) | Joint selection across quantile levels and road links with valid uncertainty |

**How transport currently selects variables.** Practice relies on one-at-a-time significance tests, information-criterion search and machine-learning importance scores:
- Random-parameter models fix a coefficient when its variance is "insignificant at the 95% credibility level" (Zeng et al., 2017, *AA&P*).
- Specifications are found by metaheuristic BIC search over an NP-hard space (Ahern et al., 2024, *Heliyon*).
- Real-time crash models select inputs with random-forest importance (You et al., 2017) or conditional-logistic significance (Kwak et al., 2016).

**Searches run on 1 November 2026** (arXiv via alphaXiv, Consensus, Wiley Scholar Gateway) found almost no use of modern BVS (SSL, global-local, group/bi-level priors) for **covariate** selection in crash models. The closest is arXiv 2609.36268 (Sept 2026), which uses a continuous spike-and-slab to select **spatial spectral components** of an NB crash model, not covariates or effect structure.

**Technical bridge.** Pólya-Gamma augmentation (Polson, Scott & Windle 2013) makes negative-binomial and logistic likelihoods conditionally Gaussian. The Gibbs samplers and EM algorithms of the Bayesian lasso, Bayesian adaptive lasso and SSL papers in this folder therefore carry over to crash data almost unchanged. That is what makes the ideas below tractable.

---

## Idea 1: Bayesian selection of fixed vs. random parameters in crash models ★ strongest

**The gap.** Random-parameter NB and logit models (with heterogeneity in means and variances, and correlated random parameters) are the dominant specification in safety research. *Analytic Methods in Accident Research* publishes many of them each year. Yet the specification is chosen by hand:
- Each coefficient is tried as random and kept fixed if its standard deviation is not significant (Zeng et al. 2017).
- Alternatively, very large BIC searches over the specification space are run with metaheuristics (Ahern et al. 2024).

Neither approach accounts for the multiplicity of the search or quantifies specification uncertainty.

**What you'd do.** Give each covariate *j* three levels: excluded, fixed effect, or random effect (optionally with heterogeneity-in-means terms δ_j).
- Use the non-centered form β_ij = β_j + σ_j z_ij with z_ij ~ N(0,1) and σ_j allowed to take either sign (Frühwirth-Schnatter & Wagner 2010, Kinney & Dunson 2007). The random-vs-fixed question becomes a sparsity question on σ_j, which sits in the interior of the parameter space rather than on the boundary.
- Treat (β_j, σ_j, δ_j) as a **group with heredity**: σ_j ≠ 0 only if β_j ≠ 0. This is exactly the bi-level selection problem that Bai et al. (SSGL, 2021) and Yang & Narisetty (2020) both list as future work, and that Xu & Ghosh (2015) and Liquet et al. (2017) address in the Gaussian case.
- Fit with an SSL-style EM along a λ₀ path for speed, plus Gibbs or BB-SSL (Nie & Ročková 2022) for uncertainty. Pólya-Gamma augmentation handles the NB likelihood.
- Theory: selection consistency for the three-level structure, including when observation-level random parameters are and are not identifiable.

**The statistically interesting wrinkle.** In NB models, observation-level random parameters and the NB dispersion both generate extra-Poisson variation, so they partly compete to explain the same thing. Characterizing when σ_j is identifiable (it needs within-covariate variation interacting with x_ij) is itself a contribution, and it explains why practitioners see unstable random-parameter results.

**Why you're well placed.** You already hold the group and bi-level selection literature (Xu & Ghosh, Yang & Narisetty, SSGL, Liquet et al., and the Lai & Chen review) and the adaptive and global-local priors. The new ingredient is the variance-component group.

**Related statistics work to position against.**
- SSVS for GLMMs (arXiv 2412.01084).
- Empirical Bayes factor for variance components (arXiv 2410.14459).
- Joint fixed/random shrinkage in LMMs (arXiv 2604.24430).

All three handle **cluster-level** random effects, not the observation-level random parameters (with heterogeneity in means) used in transport.

**Data.**
- UK STATS19 (public, suits severity models).
- US FARS/CRSS (public).
- HSIS segment-level crash data with geometry and AADT (free on request from FHWA).

**Risk.** Moderate. The identifiability analysis is the hard part. The applied method paper stands even with only simulation-based evidence on identifiability.

**Target.** *Analytic Methods in Accident Research* or *AA&P* for the applied version. *JCGS*, *Bayesian Analysis* or *Statistics and Computing* for the method.

---

## Idea 2: Cellwise-robust Bayesian variable selection for detector-based traffic models

**The gap.** Real-time crash-risk and speed/travel-time models use dozens of loop-detector features: volume, speed, occupancy and speed variance at upstream and downstream stations. Detectors fail **individually**, so contamination hits single cells of X, not whole rows. With p columns and cell-error rate ε, only a fraction (1−ε)^p of rows is clean, so casewise-robust methods break down. Transport papers handle this by dropping rows, imputing, or switching to derived features (Fang et al. 2016 use eigenvectors partly to cope with missing loop data).
- Frequentist cellwise-robust sparse regression exists: CR-Lasso (arXiv 2307.05234), cellHandler, cellwise-robust M regression.
- Bayesian robust variable selection exists for heavy tails and casewise outliers (horseshoe family, arXiv 2507.10975).
- No Bayesian **cellwise**-robust variable selection turned up in these searches.

**What you'd do.** Write a joint model:
- Observed x_ij = x*_ij + e_ij, with a spike-and-slab indicator on each cell error e_ij.
- Clean x* ~ N(μ, Ω⁻¹), with a graphical-horseshoe prior on Ω. **This is your cellwise-robust graphical model, reused as the X-model.**
- Response y | x* with SSL or horseshoe on β (logistic for crash/no-crash via Pólya-Gamma, Gaussian for speed).

The posterior delivers selected predictors, flagged faulty cells, imputed values and their uncertainty in one fit.

**The point that makes it a paper, not a plug-in.** Genuine crash precursors (shockwaves, sudden speed drops, high speed variance) **look like outlying cells**. A filter that flags cells using X alone deletes the signal you want. Because the response is part of the joint model, a cell is flagged only if it is unexplained by both the network structure and y. Showing this on data is the main empirical result.

**Why you're well placed.** It directly extends your `Cellwise-robust-Bayesian-graphical-models-with-global-local-shrinkage` work, so the X-model and much of the sampler already exist.

**Data.**
- Caltrans PeMS (free account). It publishes detector-health diagnostics, which give partial ground truth for which cells are faulty, a rare validation opportunity.
- Crash records from California SWITRS/TIMS.
- Alternatively, UK motorway loop data paired with the national crash database, as in Cai et al. (2024).

**Risk.** Moderate. The computation is heavier (n×p cell indicators), but the conditional updates are conjugate given your graphical sampler.

**Target.** *Transportation Research Part C* for the application. *Technometrics*, *JCGS* or *Annals of Applied Statistics* for the method.

---

## Idea 3: Confounder-aware Bayesian selection for crash modification factors

**The gap.** Crash modification factors (CMFs, as used in the Highway Safety Manual) are often estimated cross-sectionally from NB models.
- Lan et al. (2020, *TRR*) note that this is biased by confounding and that propensity-score methods are still rarely used.
- Al-Marafi et al. (2023 review) flag **combined CMFs** for multiple simultaneous treatments as an under-studied gap.
- Standard prediction-oriented selection shrinks covariates that are strongly related to treatment siting but only weakly to crashes. That is exactly the mechanism Antonelli, Parmigiani & Dominici (2019) show causes confounding bias.

**What you'd do.** Extend Antonelli et al.'s continuous spike-and-slab (Gaussian outcome, single homogeneous treatment) in the three directions their own Discussion lists as open:
1. A count outcome (NB with offset for length and AADT).
2. Heterogeneous CMFs, varying by site type or traffic regime, with shrinkage toward the homogeneous model.
3. Covariate-specific treatment-informed inclusion weights w_j instead of a common δ.

Then add **multiple treatments** with shared confounders, which gives combined CMFs with joint uncertainty.

**Validation.** Use a "no-treatment" placebo analysis on before-period data, where the CMF should be 1 (the design used by Li et al. 2020 for Bayesian propensity-score matching). Also compare against empirical-Bayes before-after CMFs from the FHWA CMF Clearinghouse.

**Why you're well placed.** Antonelli et al. (2019) and the SSL and variance-prior papers (Moran, Ročková & George 2019) are already in the folder. This is the most bounded of the four projects.

**Risk.** Low to moderate. Treatment-assignment data (countermeasure inventories with installation dates) are the practical bottleneck. Line up a data source before starting.

**Target.** *AA&P* or *AMAR*. *Annals of Applied Statistics* or *JRSS-C* for a methods-forward version.

---

## Idea 4: Selecting tail-only effects for travel-time reliability across quantiles and links

**The gap.** Reliability is measured by percentiles: 95th-percentile travel time, buffer index and planning-time index. The key applied question is **which factors matter only in the tail** (incidents, weather, work zones) versus those that shift the whole distribution (demand).
- Sun, Wang & Fuentes (2016) fuse quantiles across space and time, and say the method "can be generalized … over multiple quantile levels". They also note that BIC theory for p ≥ n penalized QR is open.
- Alhamzawi et al. (2012) give Bayesian adaptive-lasso QR at a single quantile.
- Credible intervals from the asymmetric-Laplace working likelihood are miscalibrated (Yang, Wang & He 2016, and arXiv 2407.09772).

**What you'd do.** Fit a joint Bayesian QR over τ ∈ {0.5, …, 0.95} with a three-level prior on each coefficient path β_j(τ): excluded, constant in τ (a location shift), or τ-varying (a tail effect).
- Use fused adaptive shrinkage across adjacent τ, and a graph-Laplacian prior (Liu et al. 2014) across adjacent road links.
- Calibrate the uncertainty with a sandwich-type adjustment.

**Caution.** The statistics side is active:
- Joint quantile shrinkage (arXiv 2506.13257).
- Decision-analytic subset selection for QR (arXiv 2311.02043).
- Effect selection in additive QR (arXiv 2606.12164).
- Consistent selection across quantiles (arXiv 2505.16019).

Methodological novelty has to come from the network fusion and calibrated three-level selection. Novelty on the transport application side looks high: no Bayesian variable-selection QR for travel-time reliability turned up.

**Data.** PeMS travel times, and NYC TLC trip records (public, with zone-to-zone travel times).

**Risk.** Moderate, mostly from crowding on the methods side.

**Target.** *Transportation Research Part B or C*, or *Technometrics*.

---

## Also considered (lighter checks)

- **Network-informed sparse Bayesian VAR for short-term traffic forecasting.** Combine Gefang's (2014) doubly adaptive elastic-net VAR with direction-aware shrinkage scaled by road-network distance, and a graphical-horseshoe error precision. It is attractive, but the field is dominated by graph neural networks, so it would need strong forecast and calibration results. Not literature-checked as thoroughly as Ideas 1 to 4.
- **Variable selection in AFT models for traffic-incident clearance duration**, extending Khan & Shaw (2016) to a Bayesian adaptive elastic net with censoring. Not checked, because the search hit a rate limit.
- **Frequentist coverage of SSL credible sets.** Ročková & George (2018) list this as future work. Combining BB-SSL (Nie & Ročková 2022) with Chatterjee & Lahiri's (2013) higher-order bootstrap refinements would be pure theory, with no transport component.

---

## A unifying thread and suggested sequencing

Ideas 1 and 4 (and spatially varying coefficients for transit ridership, which uses the same template) share one structure. For each covariate, the selection is **excluded / constant effect / heterogeneous effect**, with heterogeneity across observations, quantiles or space. A single prior framework for this "effect-structure selection", with one theory paper and several applied papers, would give a distinctive research identity in transport statistics rather than another variant in a crowded table.

- **Start with:** Idea 1 (clearest transport demand, top applied venue) and, in parallel, Idea 2 (reuses existing code).
- **Fastest to a submission:** Idea 3.
- **Most distinctive given your background:** Idea 2.
- **Most likely to be widely used in transport:** Idea 1. Every random-parameter crash paper currently selects its specification by hand.

**On Indian data.** Public Indian road-safety data (MoRTH annual reports, data.gov.in) are mostly state- or city-level aggregates. Segment-level crash data with geometry usually needs a partnership with a state PWD, NHAI or city police. Start the methods on public US/UK data and add an Indian application through such a partnership.

---

## Read first (beyond the folder)

1. Polson, Scott & Windle (2013, *JASA*): Pólya-Gamma augmentation. Makes everything in this folder work for NB and logit likelihoods.
2. Frühwirth-Schnatter & Wagner (2010, *J. Econometrics*): non-centered parameterization and spike-and-slab on ±σ (the engine of Idea 1).
3. Mannering, Shankar & Bhat (2016, *AMAR*), "Unobserved heterogeneity and the statistical analysis of highway accident data": what transport reviewers expect from a random-parameter model.
4. Ahern et al. (2024, *Heliyon*) and Zeng et al. (2017, *AA&P*): the current specification-search practice Idea 1 replaces.
5. CR-Lasso (arXiv 2307.05234, 2023) and Raymaekers & Rousseeuw, "Challenges of cellwise outliers" (arXiv 2302.02156): the frequentist baselines for Idea 2.
6. Lan et al. (2020, *TRR*) and Al-Marafi et al. (2023 review): the CMF confounding problem framed by transport researchers (Idea 3).
