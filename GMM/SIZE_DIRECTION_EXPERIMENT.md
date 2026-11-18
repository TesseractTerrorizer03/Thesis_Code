# Equity and FX size/direction experiment

Open `Size_Direction_Comparison.ipynb` and run it from this directory, `Thesis`, or the workspace root with the existing `.venv` Python environment. `requirements_size_direction.txt` records the exact delivered dependencies. The complete notebook is executed and includes its final figures and conclusions.

The fixed period is January 2000–October 30, 2026. Fifty raw stock log returns and nine orthonormal FX currency contrasts are separate datasets. No volatility standardization or feature selection is performed. Each dataset uses chronological 60/20/20 training/validation/test splits; 2007–2009 is in training. PNC replaces BK solely to obtain public coverage starting in 2000.

The companion module uses scikit-learn GMMs and SciPy distribution/numerical routines. Its joint cylindrical EM retains the shared-variance/shared-concentration formulation. Each dataset searches 280 configurations, with ten starts per configuration. The medium matched-parameter budget is primary; other budgets, unrestricted selection, and a same-component check are secondary. All configuration choices use validation only.

## Reproduce or resume

```sh
.venv/bin/python -m unittest test_size_direction_experiment.py -v
.venv/bin/python size_direction_experiment.py acquire
.venv/bin/python size_direction_experiment.py search --dataset equities
.venv/bin/python size_direction_experiment.py search --dataset fx
```

Then run the notebook to freeze selections, refit, evaluate, and render the report. `finalize` is also available through the module CLI after both complete searches. Public network access is needed only for missing inputs or an explicit `acquire --refresh`. Refreshing changes the experiment's provenance and may invalidate prior results; it is never automatic.

Search checkpoints are resumable and keyed by exact train/validation data, fitting code/settings, and library versions. Initializations are cached per component count. Every start's convergence, warnings, and failures are retained in the corresponding local joblib checkpoint. The selected final-test architecture is never replaced using test scores if its refit fails.

## Artifacts

- `data/size_direction_2000/`: immutable adjusted-price/reference-quote caches and provenance hashes.
- `outputs/size_direction_2000/test_results.csv`: the single final results table, with 30 rows across two datasets and five comparison views, plus numeric confidence bounds for all block lengths.
- The same output directory includes validation trial ledgers, selected configurations, frozen manifest, split dates, final daily log densities, extreme-return and network diagnostics, complete primary-model link coefficients, environment versions, and brief conclusions.
- `outputs/size_direction_comparison/legacy_8feature_notebook.ipynb`: the previous notebook, archived before replacement.

Stock histories have survivorship and current-vintage adjustment limitations; the fixed groups are not historical index membership. ECB reference fixings are not executable prices. Normalization is a coordinate representation, not evidence that the original financial data are naturally spherical. Density comparisons are within a dataset and common measure; absolute log densities across equities and FX are not comparable. Test intervals are conditional, pointwise, and not trading-performance claims.

The subsequent sharing ablation is documented in `SIZE_DIRECTION_SHARING.md` and notebook sections 11–14. It adds both partial-sharing choices and component-specific variance plus concentration, compares nine families, and provides uniform test likelihood intervals and refit-sample AIC/BIC. Its combined results are in `outputs/size_direction_sharing/test_results.csv`; the original experiment artifacts above remain available.
