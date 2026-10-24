# Experiment: spherical feature subsets

## Scope and hypothesis

The question is whether some subset of market observations, or unconstrained reduced coordinates of that subset, is well approximated by a noisy spherical **surface** in a stated metric. A sphere in q ambient coordinates has intrinsic dimension q−1. The code supports q≥2; it is not limited to 2D or 3D visualizations.

This is a geometric experiment. There is no GMM, Student-t model, clustering, regime classification, or predictive-return objective. Observations are never row-normalized before fitting or comparison. Unit directions appear only after fitting a sphere, as an angular-coverage diagnostic. They do not establish the spherical hypothesis.

This hypothesis is stronger than “direction contains useful information while magnitude varies.” A dataset can fail the noisy-shell hypothesis and still have a useful directional/radial decomposition. The notebook does not test that separate claim.

## Files and execution

- `Spherical_Feature_Discovery.ipynb`: configuration, chronological experiment, saved outputs, interpretation.
- `geometry.py`: coordinates, sphere/plane fitting, closest-point ellipsoid fitting/distances.
- `experiment.py`: automatic subsets, chronological validation, surrogate-search controls, selection, final confirmation.
- `tests/`: numerical and synthetic end-to-end checks.
- `data/market_features.csv`: copied input data. The original RegimeDetection CSV is not modified.
- `data/provenance.json`: original location and checksum.
- `results/`: independent timestamped experiment records.

Open the notebook with the existing Python environment that already has the dependencies. From this project, the available interpreter is `../RegimeDetection/.venv/bin/python`; it is used as an existing runtime, not as an output location. All new code, data copies, results, and verification files are in Thesis. `requirements.txt` lists dependencies if you later create a separate environment here.

Run notebook cells top to bottom. The delivered outputs leave the final market-data period unevaluated.

## The declared geometry and metric

Mixed market units require a deliberate metric. Default `scaling="standard"` uses per-feature training means and training standard deviations. A spherical result therefore means spherical geometry **in that train-fitted standardized metric**, not automatically in raw financial units. `scaling="none"` centers without rescaling. Changing the metric changes the hypothesis and is a separate experiment.

Whitening is not used. Freely choosing a transformation to make a shape round and then calling it evidence would be circular. If adding economic fixed scales, write them down before searching. In every case, the same transformation is applied to training and later observations without refitting on validation/test.

Optional PCA is unconstrained, fitted on training only, and applied separately for each subset. It never imposes a radius. Full-dimensional PCA is omitted because an orthogonal rotation cannot change distances to a sphere. Reducing to q coordinates changes the object under study: a successful PCA candidate supports the retained coordinates, not a spherical shape for the full original subset. Retained variance is reported; no original-coordinate likelihood is calculated or required.

## Candidate generation without manual category assignments

Default `seed_neighbors` starts from ETF, factor, and macro seed sets, then automatically adds one available feature or removes one seed feature. It searches **86** raw/PCA candidates with the supplied settings. This is a bounded first neighborhood, not a global optimum over all subsets.

Other modes:

| Mode | What is searched | When to use |
|---|---|---|
| `seeds` | Only declared seed subsets | Fast debugging and a small, interpretable initial experiment |
| `seed_neighbors` | Seeds plus every one-feature addition/deletion | Default automated local subset search |
| `exhaustive` | All configured sizes within `subset_pool` | A small, deliberately constrained pool |

`max_candidates` prevents accidental combinatorial explosions. The search is fully declared before scoring. Extending to multiple forward-selection rounds requires rerunning that **entire adaptive algorithm** for every surrogate; simply calibrating the final winner is insufficient.

## Chronological setup

For the current 5,010-row input:

1. Reserve the last 756 observations before any search. The boundary is 2023-08-25.
2. Use only the earlier 4,254 observations for development.
3. Construct three expanding folds, each validating on 252 later observations. The first training window has 3,498 observations when GAP=0.
4. Fit scaling, optional PCA, and surfaces inside each training window.
5. Fit surfaces on the same seeded random sample of up to 400 training rows. Scaling/PCA use the full training window. This is an editable runtime choice; check stability with larger samples before interpreting a discovery.
6. Evaluate all alternatives on identical later rows within each candidate.
7. Select using development results and surrogate calibration only.

The code uses positional trading observations, not calendar-day horizons. `gap=0` is appropriate because no forward label is constructed. Dependence is addressed approximately by block-bootstrap uncertainty, time-based validation, and surrogate controls. These do not repair point-in-time data errors. The copied CSV does not verify release lags or revised macro vintages.

If forward outcomes are added later, remove training origins whose full labels have not matured at validation start. They are not part of this geometry experiment.

## Surface fits and scoring

### Sphere

The fitted center c and radius r minimize training squared distances `(norm(x−c)−r)^2`. Algebraic sphere fitting supplies only an initial guess; the optimized loss is geometric. Two initializations are evaluated.

Very large spheres approximate planes. Radius is bounded relative to training RMS spread, center displacement is checked, and boundary solutions cannot support a spherical claim. A boundary result means this finite-curvature hypothesis was not supported by the fit; it is not proof that no sphere of any radius or local spherical patch exists. Never enlarge bounds repeatedly just to obtain a passing label.

The corrected sphere solver uses SLSQP with the explicit constraint `norm(center − training_sample_mean) / training_sample_RMS <= max_center_rms`. This is the same Euclidean norm reported by `center_rms` and used during acceptance, so several individually bounded center coordinates cannot accumulate into an inadmissible total displacement. Parameters are expressed in RMS units for numerical stability; observations are not row-normalized. Both initial centers are projected into the feasible ball. Feasibility is verified before selecting a solution, with only roundoff-size violations projected back to the boundary. Successful feasible initializations take precedence over unconverged ones.

The existing 1% near-upper-bound rejection remains an intentional safety buffer: the optimizer may reach a constraint, but a near-limit result still cannot support discovery. Radius contact at the lower bound is also rejected. The legacy configuration name `sphere_max_nfev` is retained to avoid breaking notebooks; for the new sphere SLSQP solver it specifies an iteration budget. Ellipsoid `max_nfev` still means a function-evaluation budget.

### Affine alternative

Fit a q−1-dimensional affine hyperplane. A q−1 sphere and q−1 plane in q coordinates have the same codimension, so held-out perpendicular residuals have a meaningful reconstruction comparison. Comparing a sphere to a full q-dimensional space by raw reconstruction error would be vacuous: full-dimensional space has zero error by definition.

### Ellipsoid control

Fit center and positive semiaxes using true orthogonal closest-point distances, including points inside the surface. Orientation is fixed to the training covariance/PCA axes. This control contains spheres but is not a search over all possible ellipsoid orientations. An arbitrary rotated ellipsoid can require a stronger orientation-fitting implementation; failure against this particular control cannot establish superiority over every ellipsoid or manifold.

Ellipsoid fitting is more costly, so the predeclared top three fit-and-coverage-eligible candidates receive it. Untested candidates cannot pass the final development gate. Increase the shortlist if warranted before final evaluation; do not choose it based on test results.

### Search statistic

Divide squared surface distances by the same training coordinate energy. Within a candidate all models use identical observations, metric, and normalization. The search statistic is:

```
T = log(mean validation plane loss / mean validation sphere loss)
```

Larger T favors the spherical approximation relative to an affine surface. It is not a spherical probability or proof of geometry. Candidate dimensions can differ; repeated complete-search surrogate calibration explicitly includes those dimension choices. The code is not comparing densities from different-dimensional spaces.

## Gates required for a development-supported candidate

1. Sphere optimization converges in every fold and does not hit radius/center limits.
2. Validation directions have broad enough coverage: mean resultant at most 0.60 and q times the smallest second-moment eigenvalue at least 0.05.
3. Held-out radial RMSE divided by the fitted radius is at most 0.15 in every fold. This is a declared thin-shell/noise tolerance, **not an estimated market measurement-noise level**. Change it only with a stated scientific reason, and report sensitivity.
4. The complete-search tail fraction is at most alpha under **both** declared surrogate families.
5. The one-sided paired bootstrap upper bound on sphere-minus-plane loss is negative.
6. The sphere is noninferior to the ellipsoid: the upper bound on extra error is at most `ellipsoid_margin × plane loss` (default margin 10%). Ellipsoid fits must converge and avoid bounds.

Choose the largest T among supported candidates. A practical noninferiority margin expresses tolerance for a simpler shape; it does not establish mathematically identical axes. Numerical failure is unresolved evidence, not a vote for another geometry.

If no candidate passes, report **no spherical subset established under this protocol**. Do not automatically label remaining columns Euclidean. A feature can appear in multiple overlapping candidate subsets, and a PCA-coordinate result cannot assign each original feature to an exclusive geometric category.

## Why surrogate calibration is necessary

High-dimensional solid clouds can occupy a thin-looking radial band. Searching many subsets can also find apparent curvature by chance. A sphere beating a plane on ordinary reconstruction error does not eliminate either explanation.

For each surrogate, rerun all candidate subsets, PCA dimensions, preprocessing fits, chronological folds, and sphere/plane fits. Store the largest eligible T across the declared search. Compare observed candidate T values against these maxima, using `(1 + exceedances)/(B + 1)`.

- `phase`: multiply each frequency by a random unit complex phase shared across all features. This preserves means, covariance, and multivariate cross-spectra while changing higher-order temporal structure. DC/Nyquist components remain real.
- `volatility_phase`: remove a trailing common RMS volatility envelope, phase-randomize residuals, and restore the envelope. Final marginal rescaling matches means and variances. The envelope control does not preserve raw cross-spectra or marginal tails exactly.

Coverage and thickness filters are not applied to null maxima, which is conservative relative to applying all observed gates. Boundary fits are ineligible in real and surrogate searches. A numerically unresolved surrogate search receives an infinite maximum, rather than silently shrinking its candidate set. Counts of eligible and unresolved null fits are exported; too many unresolved fits require numerical repair.

These are **conditional surrogate tail fractions**, not distribution-free p-values for “all nonspherical data.” Shared-phase stationarity and the envelope construction are modeling assumptions. The controls can alter heavy tails and nonlinear cross-feature structure. Rejecting them can reflect those changes rather than a unique spherical mechanism. Conversely, some periodic spherical structures are retained by phase randomization. More realistic domain-specific controls and outer time-period replication may be needed.

Nineteen repetitions give only 0.05 resolution and are intended for a working pilot. Use at least 199, examine Monte Carlo variability, and predeclare the null mechanisms before thesis-level claims. A revised search must be recalibrated, not just compared with previous null maxima.

## Uncertainty and final confirmation

The block bootstrap resamples paired error differences in observation blocks, separately inside each validation fold. It does not wrap blocks across refit boundaries. Intervals condition on fitted geometries and are approximate under dependence/nonstationarity. Validation intervals are not adjusted for candidate selection; the untouched final sample is therefore important.

Leave `RUN_FINAL_TEST=False` during development. Once the search space, thresholds, numerical settings, and candidate are frozen:

1. Enable final evaluation.
2. Refit the single selected candidate's coordinates and surfaces on development, respecting GAP.
3. Evaluate those fixed surfaces on the final period.
4. Apply the same geometric and coverage gates.
5. Generate final-period surrogate controls and score them using the **same frozen** transformation/surfaces; do not refit or search candidate subsets again.

If development selected nothing, final evaluation is skipped. The final-period surrogate procedure conditions on final-period properties as a shape-control experiment; it is not a forecasting backtest. If settings are changed after seeing final results, a fresh period or predeclared outer chronological replication is required. Disabling the flag does not restore an inspected holdout.

## Iteration sequence

| Iteration | Change | Keep fixed | Expected insight |
|---|---|---|---|
| 0 | Run the delivered pilot | Data, chronology, hypothesis | Verify search, diagnostics, and whether anything passes |
| 1 | Increase fit sample size and solver budgets | Metric and candidate space | Distinguish numerical artifacts from geometry |
| 2 | Revise pool, subset sizes, or seed neighborhoods | Metric, PCA choices, dates | Test a new declared collection of feature subsets |
| 3 | Add/remove unconstrained PCA dimensions | Raw candidate sets and dates | Separate evidence in reduced coordinates from raw subsets |
| 4 | Change metric with an economic rationale | Search and dates | Establish dependence on scaling/units |
| 5 | Increase surrogate repetitions; inspect envelope and block-length sensitivity | Scientific hypothesis | Assess control assumptions and uncertainty |
| 6 | Shift development fold spans | Final boundary | Check robustness across market periods |
| Final | Enable the holdout flag | All selected choices | Confirm the frozen claim |

Suggested research log:

```
Hypothesis / geometry and coordinate metric:
One change from prior run:
Settings held fixed:
Expected effect and reason:
Convergence / boundary / coverage / thickness diagnostics:
Surrogate control diagnostics and tail fractions:
Affine and ellipsoid comparisons:
What is supported, unresolved, or rejected under this protocol:
Next planned change:
Final test inspected? yes/no
```

Repeated validation-driven development can overfit validation. Timestamped records help track it but do not remove the statistical problem. For a very large or adaptive research program, introduce outer chronological replications of the entire selection pipeline and reserve fresh future data for confirmation.

## Where to modify the implementation

- Notebook `CONFIG`: search space, metric, PCA, dates, fitting budget, controls, thresholds.
- `geometry.Coordinates`: scaling or another unconstrained DR method. Do not impose a spherical output before testing it.
- `geometry.Sphere`: objective, initialization, finite-curvature restrictions.
- `geometry.Ellipsoid`: stronger ellipsoid orientation fitting or another geometric alternative; retain true orthogonal distances for comparison.
- `experiment.make_candidates`: additional search strategies. Reproduce all adaptive steps in surrogate runs.
- `experiment.phase_surrogate`: domain-specific conditional control processes.
- `experiment.final_evaluation`: frozen confirmation procedure, not feature reselection.

Run numerical and synthetic checks from Thesis with the available interpreter:

```sh
../RegimeDetection/.venv/bin/python -m unittest discover -s tests -v
```

Tests cover shifted noisy spheres, affine data, elongated ellipsoids, exact versus independently computed closest-point distances, frozen preprocessing, preserved cross-spectra, chronological splits, automatic subsets, complete-search calibration, and the enabled final-confirmation branch on synthetic inputs. Synthetic inputs do not consume the market holdout.

## Delivered verification

All 13 tests pass after the center-constraint correction, including new checks that a rotated near-planar cloud cannot exceed the total center-displacement limit and that fitting/acceptance are rotation invariant. The synthetic discovery and frozen-final-confirmation test also passes with the corrected solver. A prior separate full notebook execution on synthetic observations exercised automatic one-feature neighborhoods, optional PCA, export-disabled operation, and the enabled final-confirmation branch. A high-dimensional full-dimensional cloud does not pass the surrogate gate in its seeded negative-control test.

The original pre-correction run (`20261024T193110_022432Z`) searched the default 86 candidates and both 19-repetition surrogate controls. It rejected every sphere on its center/radius checks. That historical result used inconsistent coordinate-wise fitting bounds and total-displacement acceptance; do not interpret it as evidence that spherical subsets are absent, or pool its calibration with corrected runs. Its code hashes identify the old implementation.

New timestamped runs record the corrected code hashes and settings. Final-period confirmation remains disabled and unevaluated. These experiments establish neither an exclusive Euclidean partition nor the absence of all possible local spherical geometry.

Corrected run `20261024T201757_924290Z` refreshes the notebook outputs. All 258 observed candidate-fold fits converged, and every fitted center satisfies the total-displacement limit to floating-point precision (maximum `center_rms` 1.5000000000000002 for limit 1.5). All still trigger the intentional boundary exclusion, so no subset is selected. Unlike the original run, both surrogate families have zero unresolved sphere fits. The market holdout remains unevaluated.

## Primary methodological references

- [Li, Mukhopadhyay, and Dunson: sphere-based manifold approximation](https://arxiv.org/abs/1706.08263). Motivation for cross-validated spherical versus affine approximation, not a claim that this notebook reproduces their full method.
- [Eberly: distance to ellipses, ellipsoids, and hyperellipsoids](https://geometrictools.com/Documentation/DistancePointEllipseEllipsoid.pdf). Closest-point geometric comparison.
- [Prichard and Theiler: multivariate phase surrogates](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.73.951).
- [Schreiber and Schmitz: surrogate-data assumptions and limitations](https://arxiv.org/abs/chao-dyn/9909037).
