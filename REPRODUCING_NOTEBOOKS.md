# Reproduce the size/direction and spherical UMAP notebooks

The source distribution includes these two executed notebooks and their Python
helpers, tests, and dependency records:

- `GMM/Size_Direction_Comparison.ipynb`: raw-return Gaussian and Gaussian–vMF
  mixtures, matched parameter budgets, variance/concentration sharing, validation
  selection, and chronological test evaluation.
- `Spherical_UMAP_Analysis.ipynb`: separate equity/FX return/trend representation
  experiments, followed by raw-versus-radius-preserving-UMAP mixture comparisons.

Notebook tables and figures remain embedded for reading on GitHub. Downloaded
data, provenance files generated during acquisition, serialized models, fit
caches, standalone figures, result tables, environments, and temporary files are
excluded from this source upload. Running the notebooks regenerates them locally.

## Install

From the repository root, use Python 3.11 and a fresh environment:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r GMM/requirements_size_direction.txt -r requirements_spherical_umap.txt
```

The two requirements files record the delivered base and additional UMAP/notebook
execution dependencies. They do not replace the older experiment's
`requirements.txt`.

## Run from a clean checkout

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python run_research_notebooks.py --notebook all
```

Use `--notebook gmm` or `--notebook umap` to execute either notebook independently.
The runner starts a fresh kernel using its own Python interpreter, executes from
the repository root, and saves executed outputs back to the notebook. It stops on
errors and saves partial execution for diagnosis. No private absolute paths or
pre-existing result folders are required. Alternatively, open either notebook
with this environment's kernel and run all cells.

Both notebooks first call the shared acquisition routine with `refresh=False`.
It downloads missing adjusted stock prices from Yahoo Finance through yfinance
and daily currency reference quotes from the ECB, using the declared historical
window ending October 30, 2026. Network access is needed on the first run. Cached
files are subsequently verified against their local provenance hashes; existing
inputs are never silently refreshed. A missing provenance file or checksum
mismatch stops execution.

The UMAP notebook fits a new representation experiment when no matching completed
run exists. It also runs its mixture search when no matching completed mixture
study exists. The GMM notebook regenerates its own original and sharing searches.
Expensive fits are cached and reused when their input/code/settings identities
match. The UMAP notebook can be run without first executing the GMM notebook.

The full protocol is computationally expensive: the UMAP pilot includes 76
surrogate searches, and mixture searches use ten starts and up to 20 components.
Execution has no imposed cell timeout. Downloads or numerical fitting failures
are reported rather than silently replaced with a smaller experiment.

## Checks

```sh
python GMM/size_direction_experiment.py acquire
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python -m unittest discover -s GMM -p 'test_size_direction*.py' -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_*umap*.py' -v
```

## What can and cannot be reproduced exactly

Fresh downloads reproduce the procedure, universe, historical cutoff, and
validation rules. They may not reproduce the embedded historical numbers bit for
bit: providers can revise adjusted histories and historical observations. A new
download gets new hashes/provenance; split dates follow the documented 60/20/20
rule on the available verified calendar. Exact historical-input reproduction
requires the original caches, which are intentionally not uploaded. Runtime,
hardware, and library differences may also affect numerical results.

Every new run records inputs and settings in local output folders. Comparisons
remain retrospective, and absolute density/AIC/BIC values are comparable only
within a market and representation. See `GMM/SIZE_DIRECTION_EXPERIMENT.md`,
`GMM/SIZE_DIRECTION_SHARING.md`, `SPHERICAL_UMAP_RESEARCH.md`, and
`UMAP_MIXTURE_COMPARISON.md` for the scientific protocol and limitations.
