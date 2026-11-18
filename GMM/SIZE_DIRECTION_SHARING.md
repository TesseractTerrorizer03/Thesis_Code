# Variance/concentration sharing follow-up

Sections 11–14 of `Size_Direction_Comparison.ipynb` extend the original executed experiment. Both partial-sharing choices and fully separate variance/concentration are fitted, each linked and unlinked. All use the existing data, split, initialization seeds, component grid, ten-start rule, convergence rules, and decimal-return density correction.

Run in the existing `.venv`:

```sh
.venv/bin/python -m unittest test_size_direction_sharing -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python size_direction_sharing.py all --workers 6
```

Then run the notebook to render the combined report. Inputs are read from the original immutable caches; no network refresh is performed. New fit caches are keyed by training/validation arrays, fitting source, original engine identity, and its dependency versions. Per-start diagnostics remain in the joblib records.

`outputs/size_direction_sharing/test_results.csv` contains 90 comparison slots: two datasets, five views, and nine families. An unavailable candidate/refit is explicitly reported. Original validation choices and trained refits are reused exactly. Same-component anchors remain K=19 for equities and K=12 for FX. All new validation choices are frozen before refitting and test scoring. The medium matched budget remains primary.

AIC and BIC use total likelihood on the training-plus-validation refit sample. Test mean likelihood uses the later held-out observations and 1,500 paired circular block-bootstrap replicates, lengths 5/20/60. Intervals are conditional on fitted models, pointwise, and do not incorporate selection/refit uncertainty. No test-bootstrap intervals are attached to fitted-sample AIC/BIC. Information criteria use nominal parameter counts; ordinary mixture/serial-dependence caveats apply.

The same directory contains frozen selections, new and combined validation trials, per-day test scores, verification results, conclusions, and PNG/PDF performance figures. The pre-extension notebook is archived there as `before_sharing_extension.ipynb`.

This extension is exploratory because its design follows inspection of the previous test results. Validation selection remains isolated from test likelihoods; this does not make the historical test period newly untouched.
