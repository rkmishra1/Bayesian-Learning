# Research Ideas: Shrinkage Priors for Precision Matrices

**Prepared:** 2 August 2026
**Scope:** Open problems connecting global–local / adaptive shrinkage priors (your existing adaptive lasso, spike-and-slab, horseshoe work) to Gaussian graphical models.

---

## 0. Where the field currently stands

The element-wise prior route — putting a shrinkage prior directly on the off-diagonals of Ω rather than a structured G-Wishart on the whole matrix — is now the dominant Bayesian approach. The landscape:

| Strand | Key work | Status |
|---|---|---|
| Base estimators | Bayesian graphical lasso (Wang 2012); Graphical Horseshoe (Li, Craig, Bhadra 2019) | Mature |
| Theory | Sagar et al. (2021) oracle contraction rate via prior–penalty dual; Mai (2024) tempered-posterior concentration + misspecification oracle inequality; Zhang et al. (2021) quasi-Bayes contraction | Recently filled in, but **only for the base priors** |
| Computation | ECM mode-finding (Lingjærde et al. 2022); telescoping block decomposition for evidence (Bhadra et al. 2022); reverse telescoping sampler giving O(p³)/iteration (Gao et al. 2025) | Active, fast-moving |
| Multiple graphs | Joint graphical horseshoe (Lingjærde 2022); mGHS (Busatto et al. 2023); Bayesian joint spike-and-slab graphical lasso | Crowded |
| Missing/censored data | Censored Graphical Horseshoe (Mai et al. 2026) — note: co-authored at IIM Indore | Just done |
| Selection / testing | Model cuts, post-hoc 2-means clustering, FDR-assisted horseshoe (regression only) | **Weakest link — see Idea 1** |

The pattern worth noticing: computation and asymptotics have caught up fast, but the *decision-theoretic* layer (how do you actually declare an edge present, with error control?) and the *adaptivity* layer (the graphical analogue of the adaptive lasso) are both thin. That is exactly where your existing work gives you leverage.

---

## Idea 1 — Edge selection with finite-sample error control under continuous shrinkage priors ★ strongest

**The gap.** Continuous shrinkage priors put zero mass on exactly zero, so every published graphical horseshoe paper falls back on an ad-hoc thresholding rule: symmetric 50% credible intervals, post-hoc 2-means clustering of shrinkage weights, or "model cuts." None of these carries a guarantee. Meanwhile the regression-side literature has moved on — frequentist-assisted horseshoe for FDR control (2025) and multiple testing under global–local priors (2026) — but nobody has transported this to the graph, where the multiplicity problem is *harder*: p(p−1)/2 hypotheses, dependent test statistics, and a positive-definiteness constraint coupling them.

**What you'd do.** Define a selection rule on the posterior shrinkage weight κ_ij = 1/(1+λ²_ij τ²) — the "shrinkage factor" — and establish (i) a Bayesian-FDR guarantee analogous to Storey's, accounting for the induced dependence across edges, and (ii) graph-selection consistency. The natural technical hook is that κ_ij concentrates near 0 or 1, so the posterior mean of κ behaves like a local fdr statistic.

**Why you're well placed.** This is the two-groups/local-fdr machinery from the horseshoe literature applied to a matrix parameter. Your Bayesian adaptive lasso and spike-and-slab reading covers the prior side; the missing piece is dependence-aware multiplicity, which is learnable.

**Risk.** Moderate. Handling dependence across edges rigorously is the hard part; a defensible fallback is an asymptotic FDR statement plus strong empirical calibration.

**Target.** *JCGS*, *Electronic Journal of Statistics*, or *Bayesian Analysis*. A sharper theory version could go to *Bernoulli*.

---

## Idea 2 — The genuinely adaptive graphical horseshoe

**The gap.** The graphical horseshoe uses one global τ for all p(p−1)/2 edges. This is the exact defect Zou (2006) identified in the plain lasso and fixed with data-driven weights — and the fix has never been done properly on the graph side. Real networks are not homogeneously sparse: hubs, modules, and degree heterogeneity mean the appropriate shrinkage level varies systematically by *node*, not just by edge.

**What you'd do.** Put a node-level shrinkage layer in: λ²_ij ~ C⁺(0, τ · ν_i · ν_j) with ν_i node-specific. This is a hierarchical decomposition that automatically produces hub-tolerant shrinkage without hard-coding a scale-free assumption. Two versions worth comparing:
- **Empirical-Bayes weights** — plug in a pilot estimate (nodewise lasso, CLIME), the direct analogue of adaptive lasso weights.
- **Fully Bayes** — half-Cauchy on ν_i, which makes the multiplicity adjustment automatic.

Establish the oracle property analogue: contraction at the rate that knows the true support, plus selection consistency, and show the standard GHS *fails* to adapt under degree heterogeneity (a negative result here is publishable on its own).

**Why you're well placed.** Directly your territory — you have the adaptive lasso oracle-property literature, the doubly-adaptive elastic net, and the Bayesian adaptive lasso all in your library. This is the transplant.

**Risk.** Low-moderate. The Gibbs sampler follows the existing GHS structure with one extra block. Sagar et al.'s prior–penalty dual gives you a proof template.

**Target.** *JCGS* or *Statistics and Computing*; a strong theory result would reach *EJS* or *Annals of Statistics*.

---

## Idea 3 — Cross-validation-free tuning: marginal-likelihood calibration of the global parameter

**The gap.** Tuning-parameter selection is a running theme in your library (you have the *Survey of Tuning Parameter Selection in High-Dimensional Statistics*). In graphical models it is unusually painful — CV over a matrix-valued penalty is expensive and known to over-select. Bhadra et al. (2022) solved evidence computation for element-wise graphical priors via telescoping block decomposition, which means marginal-likelihood-based tuning is now *computationally feasible* but, as far as I can find, nobody has actually built and studied the resulting empirical-Bayes procedure.

**What you'd do.** Use the telescoping evidence estimator to do empirical-Bayes selection of τ (and of the adaptive weights in Idea 2). Compare against CV, StARS, and extended BIC on selection accuracy and Frobenius/spectral risk. The theoretical question worth asking: does EB-τ adapt to the unknown sparsity level, in the sense that the EB posterior contracts at the same rate as the oracle-τ posterior?

**Why you're well placed.** It combines your tuning-parameter interest with the Bayesian machinery, and it's a well-defined, bounded project.

**Risk.** Low methodologically; the adaptivity theorem is the ambitious part. Even the purely methodological version is a solid applied-stats paper.

**Target.** *Computational Statistics and Data Analysis*, *Statistics and Computing*.

---

## Idea 4 — Robust / heavy-tailed graphical shrinkage

**The gap.** Every result above assumes Gaussianity. Financial returns, the obvious application at an IIM, are emphatically not Gaussian. Existing robust options are either frequentist (nonparanormal SKEPTIC, weighted graphical lasso) or Bayesian-but-unshrunk. A t-distributed or scale-mixture likelihood combined with a graphical horseshoe prior is a clean gap — and note the 2026 arXiv work on robust high-dimensional Bayesian regression under global–local priors with non-Gaussian errors shows the technique transfers.

**What you'd do.** Ω-estimation under a multivariate-t or general scale-mixture-of-normals likelihood with GHS prior. The scale mixture keeps the Gibbs sampler conjugate (one extra latent weight per observation), so this is computationally cheap. Establish contraction under the misspecified-Gaussian case — Mai's (2024) oracle inequality for misspecification is a direct entry point. Application: a partial-correlation network of Indian sectoral returns or a systemic-risk network.

**Why you're well placed.** Strong applied hook for a business-school setting, and the theory borrows an existing template rather than needing a new one.

**Risk.** Low. The main risk is that the contribution reads as incremental — mitigate by carrying the theory, not just the sampler.

**Target.** *Computational Statistics and Data Analysis*, *Journal of Multivariate Analysis*, or a finance-adjacent outlet.

---

## Idea 5 — Prior–penalty duals as a general recipe (higher risk, higher reward)

**The gap.** Sagar et al. (2021) built one prior–penalty dual approximating the horseshoe closely enough to get *both* good posterior contraction and MAP consistency. That was constructed for a single prior. The general question is open: for which global–local priors does such a dual exist, and can you characterise the class?

**What you'd do.** A unifying treatment — conditions on the mixing density under which a closed-form penalty exists with matching Bayesian and frequentist optimality, covering horseshoe, horseshoe+, Dirichlet–Laplace, normal–gamma, and the spike-and-slab lasso. This would be the graphical-model analogue of the "Bayesian Regularization: From Tikhonov to Horseshoe" synthesis you already have.

**Risk.** High — this is a genuine theory paper and could stall. But it is the one that would establish a distinct research identity rather than adding another variant to a crowded table.

**Target.** *Annals of Statistics*, *Bernoulli*, *Statistical Science* (if written as a review-plus-unification).

---

## Suggested sequencing

**Idea 2 → Idea 1 → Idea 3** is the coherent programme: build the adaptive prior, then solve selection for it, then solve tuning for it. Three linked papers, each standalone, sharing one codebase and one simulation harness. Idea 4 is the parallel applied track that needs the least new theory and can be run alongside. Idea 5 is the long-horizon bet — worth keeping as a background project, not a first move.

**Fastest to a submission:** Idea 3, then Idea 4.
**Most distinctive given your background:** Idea 2.
**Most likely to be highly cited:** Idea 1 — everyone using these methods currently has no principled selection rule.

---

## What to read first

1. Li, Craig & Bhadra (2019), *The Graphical Horseshoe Estimator for Inverse Covariance Matrices* — the base model.
2. Sagar, Banerjee, Datta & Bhadra (2021), *EJS* — the prior–penalty dual; template for Ideas 2 and 5.
3. Mai (2024), *Stat* — concentration + misspecification oracle inequality; template for Idea 4.
4. Gao et al. (2025), reverse telescoping sampler — the current computational state of the art; whatever you build should use it.
5. Bhadra et al. (2022), *JMLR* — evidence estimation; the engine for Idea 3.
6. The 2025 frequentist-assisted horseshoe FDR paper — the regression-side result you'd generalise in Idea 1.

**Note.** The Censored Graphical Horseshoe paper (Jan 2026) lists IIM Indore among its affiliations — if that is your group, several of these ideas are natural follow-ons and you may already have the codebase.
