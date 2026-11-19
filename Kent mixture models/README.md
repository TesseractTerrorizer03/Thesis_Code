# Kent mixture models: original-space FX pilot

Open **[Kent_FX_Anisotropy_Pilot.ipynb](Kent_FX_Anisotropy_Pilot.ipynb)** for the mathematical specification, numerical gate, executed comparisons, and caveats.

This separate experiment uses original nine-dimensional FX Helmert returns and log magnitudes. It compares shared-concentration/shared-variance vMF and constrained generalized Kent mixtures, linked and unlinked, against the original Cartesian GMM specifications. The declared grid is **K = [1, 3, 6, 10]**, with ten starts per configuration. There is no dimensionality reduction, added context, or hierarchical model.

## Completed pilot outcome

All 64 search configurations (640 starts) finished. **Kent was fitted, but excluded before test scoring:** 77 of 80 Kent starts converged, all 80 reached the imposed shape margin, and there were zero normalization failures. The other three starts reached the iteration limit. The frozen protocol's automatic rejection of shape-boundary fits prevented the intended Kent–vMF held-out comparison. A valid boundary density is not automatically a numerical failure, and this result does not establish that Kent performs worse.

The executed notebook contains the diagnostics, seven figure sets, and completed vMF/GMM comparisons on 1,374 test dates. All 19 selected baseline refits were available, and their scores, AIC/BIC, data hashes, and dates were verified. A revised analysis retaining converged, valid shape-boundary fits has **not** been performed. The original acceptance policy and its consequence remain explicit rather than being changed after fitting without documentation.

## Reproduce

From this folder, using the existing Python 3.11 GMM environment:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 ../GMM/.venv/bin/python -m unittest test_kent test_protocol -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 ../GMM/.venv/bin/python kent_experiment.py --stage all
../GMM/.venv/bin/python build_notebook.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 ../GMM/.venv/bin/python execute_notebook.py
```

Alternatively, open the notebook with the same environment and run all cells. The numerical gate runs before market comparisons. Checkpoints are keyed by input data, fitting source, settings, and dependency versions. No missing data is silently downloaded and no existing data is refreshed. `--stage gate` or `--stage search` runs only that stage and its prerequisites. Failed fits stay in the ledgers; no component grid or shape constraint is silently relaxed.

## Dependencies and isolation

The pilot uses NumPy, SciPy, pandas, scikit-learn, joblib, matplotlib, IPython, and nbformat already present in `../GMM/.venv`. The original `../GMM/size_direction_experiment.py` and frozen data under `../GMM/data/size_direction_2000` remain shared read-only dependencies. Earlier notebooks and the `Nov. 5-12th` archive are untouched.

`kent_numerics.py` implements deterministic Fourier/continuous-Euler normalization and derivatives. `kent_models.py` implements generalized EM, and `kent_experiment.py` handles protocol, fits, selection, and evaluation. `kent_report.py` generates figures/tables. The in-process notebook executor exists because sandboxed Jupyter kernels may be unable to bind sockets; it executes the actual cells and captures rich outputs.

## Artifacts

`results/latest.json` identifies the current run. Its directory contains the protocol and hashes, numerical gate log, all 64 search configurations, per-start diagnostic checkpoints, frozen validation selections, development refits, per-date test scores, AIC/BIC, paired block-bootstrap intervals, runtime records, anisotropy diagnostics, and PNG/PDF figures. Numerical failures and boundary-rejected fits are retained. A rejected fit is not promoted into a test comparison.

The original-space score includes the correct `−9 log(radius)` Jacobian for directional–magnitude models; the Cartesian GMM is not corrected again. Parameter-matching is exact where feasible and otherwise within the original 10% tolerance. Some matched comparisons are unavailable under the agreed grid. The test period is retrospective; nominal AIC/BIC and pointwise block intervals have the limitations stated in the notebook.

## Numerical references

- Chen & Tanaka, [Maximum likelihood estimation of the Fisher–Bingham distribution via efficient calculation of its normalizing constant](https://arxiv.org/abs/2004.14660).
- Wood, [Fisher–Bingham and Kent distributions on Stiefel manifolds](https://conferences.leeds.ac.uk/wp-content/uploads/2019/01/Wood.pdf).

The normalizer uses a real saddle only to choose a stable contour. It does not replace the density with a saddlepoint or tangent-Gaussian approximation. Quadrature refinement agreement is validated independently and is not claimed to be a rigorous global error bound.
