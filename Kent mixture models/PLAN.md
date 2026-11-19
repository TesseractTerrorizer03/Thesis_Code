# Original-space FX generalized Kent anisotropy pilot

## 1. Deliverable and agreed experiment

Create an independent `Thesis/Kent mixture models/` folder containing an executed notebook, a reusable fitting module, numerical tests, and local result/figure directories. Preserve the existing experiments and archived hierarchy.

The notebook will test **full tangent anisotropy on the original FX directions \(S^8\)**, with:

- The original FX data snapshot, nine-dimensional Helmert transformation, direction/log-magnitude decomposition, and chronological splits.
- No dimensionality reduction, additional features, or hierarchy.
- **\(K\in\{1,3,6,10\}\)** for every family.
- **Ten initializations per configuration**.
- The original **shared concentration and shared magnitude variance** convention.
- Linked and unlinked versions of both vMF–Gaussian and generalized Kent–Gaussian.
- Original Cartesian GMM controls with spherical, diagonal, tied, and full covariance and the original regularization grid.

This is a limited pilot. No automatic expansion of component counts, sharing configurations, or anisotropic families will occur.

## 2. Statistical specification and fair comparisons

### Generalized Kent model

For \(u\in S^8\), use

\[
f_k(u)=Z(\kappa,\lambda_k)^{-1}
\exp\left[
\kappa\mu_k^\top u+
\sum_{j=1}^{8}\lambda_{kj}(\gamma_{kj}^{\top}u)^2
\right].
\]

The frame \([\mu_k,\gamma_{k1},\ldots,\gamma_{k8}]\) is orthonormal, \(\sum_j\lambda_{kj}=0\), and the quadratic eigenvalues are component-specific. Concentration \(\kappa\) is shared.

Enforce unimodality using \(2\max_j\lambda_{kj}\leq(1-10^{-6})\kappa\), with the original concentration cap. Treat the margin as a documented numerical safeguard, not a validation-selected hyperparameter. Record boundary hits and exclude boundary fits from selection, consistently with the baseline’s concentration/variance boundary policy. Include the uniform limit explicitly.

The joint component is

\[
f_k(u)\,\mathcal N(m;a_k+b_k^\top u,\sigma^2).
\]

Set \(b_k=0\) for unlinked models; estimate it for linked models. Share \(\sigma^2\) across components.

Document and independently verify these total parameter counts:

| Family | Parameters |
|---|---:|
| Unlinked vMF–Gaussian | \(10K+1\) |
| Linked vMF–Gaussian | \(19K+1\) |
| Unlinked generalized Kent–Gaussian | \(45K+1\) |
| Linked generalized Kent–Gaussian | \(54K+1\) |
| Spherical GMM | \(11K-1\) |
| Diagonal GMM | \(19K-1\) |
| Tied-covariance GMM | \(10K+44\) |
| Full-covariance GMM | \(55K-1\) |

### Comparison views

- **Same components:** report all five families at each \(K\in\{1,3,6,10\}\); select GMM covariance/regularization using validation only.
- **Exact parameter matches:** report every feasible cross-family parameter-count match, clearly identifying differences in component counts.
- **Approximate parameter matches:** enumerate feasible budget windows from the declared parameter counts, require maximum/minimum count ≤1.10, and select candidates by validation within each window. Deduplicate equivalent comparisons.
- **Best within pilot grid:** select each family’s validation winner and explicitly identify unequal parameter budgets.

Do not manufacture an all-family matched-budget comparison. Under the agreed grid, the unlinked Kent–vMF comparison has no match within 10%. Mark it unavailable. The only linked Kent–vMF match within 10% is Kent K=1 versus vMF K=3 (55 versus 58 parameters). The earlier 109-versus-115 comparison is no longer available because K=2 is outside the grid. Evaluate only the four declared counts, not every integer between 1 and 10.

Retain the original validation tie-breaking rule. Freeze all choices before development refitting and test scoring; a failed refit remains unavailable rather than being replaced using test performance.

## 3. Numerical implementation and fitting

### Normalization

Implement deterministic Fisher–Bingham normalization and derivatives using the Fourier-integral/continuous-Euler approach of [Chen and Tanaka](https://arxiv.org/abs/2004.14660), specialized to the constrained Kent family.

- Evaluate in a scaled/logarithmic form with adaptive quadrature refinement.
- Target successive-refinement agreement of \(10^{-10}\) in log normalization; check gradient accuracy separately.
- Independently validate representative cases using adaptive contour integration, analytic vMF values, and low-dimensional direct integration.
- Stop and report numerical failure if refinement cannot meet tolerance. Do not silently substitute a saddlepoint or tangent-Gaussian approximation.
- Record quadrature resolution, estimated numerical error, evaluation counts, and normalization time.

Use NumPy/SciPy rather than an \(S^2\)-only Kent package as the fitting backend.

### Joint mixture fitting

Compute responsibilities from the complete directional–magnitude likelihood. Retain the original weighted Gaussian regression and variance updates.

For generalized Kent, use generalized EM with alternating numerical updates:

- Optimize component orientation frames using orthogonality-preserving rotation updates.
- Jointly optimize shared concentration and component shape coefficients under the declared constraints.
- Accept updates only when the accurately evaluated weighted objective does not decrease.
- Check observed likelihood as well; tighten normalization accuracy before classifying a small apparent decrease as failure.
- Distinguish optimizer failure or stalling from successful EM convergence.

Retain the baseline’s 500-iteration limit, \(10^{-6}\) mean-log-likelihood stopping tolerance, variance floor, concentration cap, minimum effective membership, and deterministic initialization seeds. Numerical solver tolerances are separate from statistical hyperparameter selection and must be recorded.

Use matched baseline initialization responsibilities. Within the ten-start Kent budget, reserve one start for the corresponding vMF solution with zero anisotropy; for linked Kent, also reserve one for the fitted unlinked Kent solution with zero regression slopes. Document this warm-start advantage and report its contribution and computational cost separately.

### Data and likelihood compatibility

Reuse the original loader without refreshing data. Verify input hashes, coordinates, dates, split boundaries, and polar reconstruction.

Score every model on the same nine-dimensional Cartesian return-density measure:

\[
\log p_X(x)=\log p_{U,M}(u,m)-9m.
\]

Here m = log(||x||₂), and 9 is the ambient dimension of the original FX Helmert vector. With r = exp(m), Cartesian volume transforms as dx = r⁸ dr dω = r⁹ dm dω, where dω is sphere surface area. Thus −9m is a coordinate Jacobian correction, not a complexity penalty. Apply it once to directional–magnitude densities; the Cartesian GMM already uses this measure. The correction cancels in comparisons between vMF and Kent on identical observations, but is necessary when comparing either against the Cartesian GMM. Compute AIC = 2p − 2ℓ and BIC = p log(N) − 2ℓ from the same corrected, total development-sample likelihood; report held-out mean log likelihood separately.

Reuse historical baseline fits only when inputs, implementation, and fitting settings match exactly. Otherwise refit using unchanged baseline conventions. Keep all new caches and artifacts inside the new folder.

Expose reusable fitting, per-observation scoring, posterior, and diagnostics interfaces; the notebook orchestrates these without modifying the original baseline modules.

## 4. Validation gate, metrics, and visualizations

### Numerical gate before market comparisons

Test:

- Zero-anisotropy agreement with vMF densities, normalizers, and derivatives across weak through high concentration.
- Low-dimensional Kent normalization against direct integration.
- Nonzero-anisotropy \(S^8\) normalizers against the independent reference.
- Finite-difference checks of shape, concentration, and orientation derivatives.
- Rotation invariance, orthonormality, trace constraints, and permutation invariance of equivalent axes.
- Nondecreasing generalized-EM likelihood and valid posterior probabilities.
- Correct radial Jacobian and nominal parameter counts.
- Explicit handling of collapsed components, repeated eigenvalues, nearly isotropic shapes, boundaries, and failed optimizations.

Use simulated isotropic and anisotropic directional data to check numerical behavior and density fitting. Label these as implementation validation—not recovery of economic regimes. If the gate fails, produce a diagnostic notebook and withhold substantive model-ranking claims.

### Evaluation

Preserve baseline metrics and uncertainty conventions:

- Held-out original-space mean log likelihood.
- Development-sample AIC and BIC using nominal parameter counts.
- Paired circular block-bootstrap 95% intervals: **1,500 repetitions**, primary block length **20**, sensitivity lengths **5 and 60**.
- Planned contrasts: Kent versus vMF separately for linked/unlinked fits; linked versus unlinked within each directional family; comparisons with the validation-selected GMM.
- Descriptive directional and conditional-magnitude score decomposition for the directional families.

Intervals are pointwise and conditional on fitted models. Do not attach test-bootstrap intervals to fitted-sample AIC/BIC or interpret them as uncertainty over the entire selection process.

Measure fresh fitting time on the same machine with a consistent thread limit. Report per-start time, iterations, failures, accepted-start rate, inner-optimization work, normalization cost, and warm-start preparation cost. Cache retrieval is not fitting time. Flag that a Kent iteration costs more than a vMF iteration and that regularized GMM updates have different monotonicity properties.

### Notebook presentation

Include:

- Explicit model definitions, constraints, parameter counts, and protocol.
- Validation curves against component count and parameter count.
- Same-component and matched-budget score tables and confidence-interval plots.
- Likelihood-versus-time convergence plots and numerical-failure summaries.
- Anisotropy spectra and local tangent-spread ratios, annotated by component support.
- A caveat table covering search-budget disparity, warm starts, unavailable parameter matches, boundary solutions, retrospective test reuse, and static-mixture interpretation.

Do not portray projections of \(S^8\) as faithful three-dimensional Kent contours.

The completed pilot must either deliver reproducible comparisons with the numerical gate passed or clearly document why reliable comparison was unavailable. Expansion beyond the agreed experiment requires a separate decision.
