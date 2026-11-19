# Original-space equity anisotropy experiment

The independent `Kent_Equities_Anisotropy_Pilot.ipynb` implements the boundary-inclusive FX methodology on the original 50 equity returns, with four starts per configuration and no development refits. GMM uses K={1,3,5}. Directional component counts additionally target comparable full-GMM parameter budgets. Full specifications and caveats are in the notebook.

Run from this directory with the existing environment:

```sh
../../GMM/.venv/bin/python build_equity_notebook.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/kent-mpl IPYTHONDIR=/private/tmp/kent-ipython ../../GMM/.venv/bin/python execute_equity_notebook.py
```

Use the absolute interpreter path if launching from another working directory.

Numerical tests run before any market fits. Completed starts are saved individually, including rejected fits. The results directory is keyed by inputs, settings and implementation. Bootstrap intervals use cached per-date scores and do not fit models. Restarting the same run reuses completed checkpoints. Do not run two copies concurrently.

Read-only dependencies are the parent FX numerical/fitting modules and the original GMM loader. No original FX files or data are modified. Numerical shape boundaries are accepted when converged and feasible; the stricter absolute-eigenvalue condition is reported separately.
