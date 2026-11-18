# Spherical UMAP representation pilot

Open **Spherical_UMAP_Analysis.ipynb**. The notebook is a representation experiment,
not a natural-sphere fitting test or a forecasting backtest. Each point is a date.
Equities and FX, and daily-return and trailing-trend inputs, are fitted separately.

## Reproduction

Use Python 3.11 in `GMM/.venv`. The existing numerical dependencies are recorded
in `GMM/requirements_size_direction.txt`; the additional pinned packages are in
`requirements_spherical_umap.txt`. The existing dependency file is unchanged.

```sh
python3 -m pip --python GMM/.venv/bin/python install -r requirements_spherical_umap.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 GMM/.venv/bin/python -m unittest discover -s tests -p 'test_spherical_umap.py' -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 GMM/.venv/bin/python spherical_umap.py --workers 4
```

The CLI creates a timestamped run under `spherical_umap_results/`. Open the notebook
and run all cells to reuse a verified complete run and render the report. The
configuration cell controls all scientific settings. Changed inputs, fitting
source, settings, or dependency versions invalidate the relevant caches. No
market data are downloaded by the CLI itself. The notebook downloads missing
inputs with the shared GMM acquisition routine before calling this experiment;
existing caches are verified without refreshing. See `REPRODUCING_NOTEBOOKS.md`
for a clean-checkout setup.

## Inputs and timing

The frozen inputs are the existing 50-stock adjusted-price cache and nine ECB
foreign-currency-per-EUR quote histories, ending October 30, 2026. Hashes must
match the original provenance files. The equity cohort has survivorship/current
adjustment limitations; ECB fixes are not executable quotes.

FX starts with EUR strength zero and negative daily changes in log foreign-per-EUR
quotes, removes the cross-currency mean, and uses nine orthonormal Helmert
contrasts. Pair differences and Euclidean norms are independent of the original
numeraire. Trend descriptors are constructed in the ten centered currency
strengths, not arbitrary labeled Helmert axes.

Trailing features are, per stock/currency, log momentum over 21/63/126 sessions,
price divided by its trailing 200-session arithmetic average minus one, and
21-session sample return standard deviation times sqrt(252). Price paths are
exponentiated cumulative log returns and are invariant to the starting price.
The FX path is a synthetic basket-relative currency value.

Original split dates are retained: equities validation begins February 5, 2016,
and retrospective testing begins June 17, 2021; FX boundaries are February 9,
2016, and June 23, 2021. The 199-session trend warmup reduces training rows only.
Stock/FX calendars are not forced to overlap. Historical test results have already
been inspected in earlier research; this is retrospective validation.

## Embedding and scores

Trend standardization uses training means/standard deviations; daily returns
are unstandardized. Cosine experiments normalize the preprocessed input rows;
Euclidean experiments retain magnitude. Pre-normalization norms are exported.
Undefined zero directions raise a dated error instead of changing the sample.

Spherical UMAP uses two periodic angular coordinates and a unit-sphere geodesic
output distance. The distance, analytic gradient, and Cartesian display share
one latitude/longitude convention. Tests check finite-difference gradients,
wrapping, poles, coincidence, and antipodes. No physical radius is estimated.
Planar UMAP and PCA2 have the same intrinsic output dimension; PCA3 is a secondary
three-dimensional capacity reference. The training neighbor graph is shared
across output geometries and fixed during layout-seed comparisons.

Held-out dates are transformed into frozen training maps. Recall @5/15/50 compares
original-space and embedded neighbors among training dates. Inductive
trustworthiness ranks the same training candidates; sampled Spearman correlation
compares held-out pairwise distances. The primary metric is recall @15. UMAP
neighborhood settings 15 and 50 are selected on validation; ties favor 15.
All choices are written before refitting on development and test scoring.

Intervals use paired circular block bootstrap draws, 1,500 replicates, with
5/20/60-session blocks. These are pointwise, conditional intervals; validation
intervals are post-selection descriptions. A two-point recall margin is a
predeclared practical noninferiority tolerance, not proof of equivalence.
Independent seeds and validation thirds measure layout stability and drift.
Spherical alignment is an orthogonal rotation/reflection fitted on training
anchors, with held-out angular error in radians.

## Surrogate controls and interpretation

Nineteen shared-phase and nineteen volatility-envelope surrogate searches are
run for each market. Daily inputs are randomized before rolling features are
rebuilt; FX independent contrasts are reconstructed into ten currency strengths.
Training and validation segments are randomized separately. The artificial join
can affect rolling descriptors and is a limitation of the conditional control.

Shared phases preserve multivariate cross-spectra within each segment. The second
family removes/restores a common trailing RMS envelope and only approximately
preserves the original spectra. Spectral errors are recorded. Every surrogate
repeats parameter selection for both feature families and distance choices.
The maximum spherical-minus-planar recall gain calibrates the four observed
tracks per market. Unresolved searches count as exceedances, never exclusions.
The smallest attainable tail fraction is 0.05. These are conditional surrogate
tail fractions, not universal topology tests or multiplicity-adjusted final-test
claims; the two market families remain separate.

Date, raw return magnitude, equity return/breadth/sector contrasts, and currency
strength/dispersion colors are applied after fitting. Local descriptor coherence
uses training-neighbor descriptor means to describe held-out same-day observations;
it does not forecast the future. These input-derived descriptors are not
independent evidence. Synthetic sphere/plane/cloud examples demonstrate that
spherical UMAP produces unit radius regardless of source geometry.

## Artifacts

Each run preserves input/code hashes, exact environment versions, configuration,
feature definitions, splits, selections, grid trials, per-date scores, norms,
paired comparisons, query coordinates (including canonical spherical angles), seed/period diagnostics, descriptive coherence, surrogate
trials/maxima, synthetic scores, and conclusions. A numerical report bundle without serialized UMAP reducers lets
the notebook render results without rerunning fits. PNG and PDF figures accompany
the embedded notebook outputs.

Resumable caches live under `.cache/spherical_umap/`. Each numerical fit records
settings, timing, warnings, and frozen-transform verification; numerical failures
have separate JSON records. A failed grid member makes selection unresolved.
Neither rejected geometry nor an inconclusive control is relabeled as success.

Sources: [UMAP](https://arxiv.org/abs/1802.03426),
[non-Euclidean embedding documentation](https://umap-learn.readthedocs.io/en/latest/embedding_space.html),
[distance implementation](https://github.com/lmcinnes/umap/blob/master/umap/distances.py),
and the original `EXPERIMENT_NOTES.md` and `Spherical_Feature_Discovery.ipynb`.
