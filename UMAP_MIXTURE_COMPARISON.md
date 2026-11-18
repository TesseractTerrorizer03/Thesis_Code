# Mixture replication in the spherical UMAP notebook

Sections 15–19 of `Spherical_UMAP_Analysis.ipynb` extend the existing analysis using
`GMM/Size_Direction_Comparison.ipynb` and its two fitting modules. The reference
notebook and its results are preserved. This extension selects the best validation
configuration of each family/matrix setting; the original matched-parameter-budget
views remain in the reference notebook.

For equities and FX separately, compare original daily log-return vectors with
`original vector magnitude × spherical UMAP direction`. Original return magnitude
is retained to floating-point precision; fitting a Cartesian Gaussian to unit-radius
points would instead introduce a singular support mismatch. FX retains the original
orthonormal nine-contrast construction. The UMAP output has three Cartesian
coordinates and two angular degrees of freedom; adding varying magnitude gives a
three-dimensional modeled vector.

The shared encoder is the previous cosine-return S2 experiment, fixed at 15
neighbors, seed 42, and 300 epochs. Training-only coordinates support mixture
validation. A development refit of the encoder supports final mixture refits and
test transforms. Encoding is never fitted on test observations. Historical test
dates have already been inspected; the extension is retrospective.

## Covariance toggle for all three families

The ordinary GMM uses diagonal or full Cartesian covariance. The reference
unlinked/linked model is

    sum_k weight_k × vMF(u; mu_k, kappa_k) × N(log(radius); a_k + B_k u, variance_k)

with B=0 for unlinked, and the reference shared/component-specific variance and
concentration options. Its Gaussian variable is scalar. Diagonal and full matrices
are consequently the same 1×1 covariance for these two families. Both requested
toggles are exported and verified identical, but fitted once and not counted as
independent comparisons. Direction retains vMF concentration κ; the link remains
the reference row vector B. No matrix-valued angular distribution is substituted.

## Selection and scores

Use the original split dates and all original return features. Search K=1..20,
ten deterministic starts, 500 iterations, and original numerical guards. Validate
GMM covariance and ridge (1e-8/1e-7/1e-6); for cylinders validate shared versus
component-specific variance and κ. Numerical κ estimates are training maximum
likelihood estimates; validation selects among the fitted specifications, not
individual starts or test-estimated concentrations. Boundary K=20 winners are
flagged as unresolved outside the declared component range.

Freeze selections before development refits and test evaluation. Starts are
chosen using training likelihood. Models that fail every convergence check remain
unavailable. A selected failed refit stops evaluation; no test-based replacement
is made. Exact input/engine-matched raw-data searches and development refits are
reused from the reference. Their provenance is recorded in the reuse ledger.

Test mean log density, paired gains against the validation-selected normal GMM,
and full-versus-diagonal normal-GMM gains have circular block-bootstrap intervals:
1,500 replicates, 20 sessions primary, 5/60 sensitivity for family comparisons.
AIC=2p−2ℓ and BIC=p log(n)−2ℓ use development fitted-sample total likelihood and
nominal free-parameter counts. Cylinder scores include −d log(radius), with the
actual ambient dimension d of each representation. Criteria are checked against
independent recomputation and sklearn for Gaussian mixtures.

Compare densities and information criteria only within a market/representation.
Absolute raw-versus-UMAP or equities-versus-FX likelihood/AIC/BIC differences are
not meaningful. UMAP criteria condition on the common fixed encoder and omit its
nonparametric fitting capacity. Bare κ is dimension-dependent; expected cosine
A_d(κ) accompanies it. Full-versus-diagonal GMM results include separately selected
component counts. Best linked/unlinked comparisons also include selected sharing
and K differences, so they do not isolate a causal effect of the link alone.

## Reproduce

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 GMM/.venv/bin/python umap_mixture_comparison.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 GMM/.venv/bin/python -m unittest discover -s tests -p 'test_umap_mixture_comparison.py' -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 GMM/.venv/bin/python -m unittest discover -s GMM -p 'test_size_direction*.py' -v
```

Then run the notebook to render the completed experiment. Outputs live in a
new timestamped directory under `umap_mixture_results/`; fit caches use exact
training arrays and reference-engine fingerprints. Outputs include grid trials,
freeze manifests, test scores and intervals, actual covariance matrices, selected
κ diagnostics, figures, independent verification, and data-derived conclusions.

## Common-target transformation effect

The notebook also evaluates the same retained log-magnitude variable across raw
and UMAP pipelines. This secondary assessment uses lower-better CRPS, estimated
from 60,000 generative samples per frozen model (three fixed seeds), with paired
5/20/60-session bootstrap intervals and Monte Carlo seed sensitivity. No model is
reselected using that score. This supplies a comparable cross-representation
assessment of magnitude calibration; its scope excludes recovery of the original
directional distribution. The loss-form proper scoring rule follows
[Gneiting and Raftery](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf).
