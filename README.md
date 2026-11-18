# Thesis experiments

The two size/direction research notebooks include executed tables and figures.
See [REPRODUCING_NOTEBOOKS.md](REPRODUCING_NOTEBOOKS.md) for installation,
first-run downloads, execution, and regeneration of excluded data and caches.

- **[Spherical_UMAP_Analysis.ipynb](Spherical_UMAP_Analysis.ipynb)**: separate equities/FX representation experiments and validation-selected mixture-model comparisons. See [SPHERICAL_UMAP_RESEARCH.md](SPHERICAL_UMAP_RESEARCH.md) and [UMAP_MIXTURE_COMPARISON.md](UMAP_MIXTURE_COMPARISON.md).
- **[GMM/Size_Direction_Comparison.ipynb](GMM/Size_Direction_Comparison.ipynb)**: original return-based Gaussian and size/direction mixture experiments, including variance/concentration sharing comparisons.
- **[Spherical_Feature_Discovery.ipynb](Spherical_Feature_Discovery.ipynb)**: original spherical-surface feature search. Supporting mathematics and experiment code are in `geometry.py` and `experiment.py`; see [EXPERIMENT_NOTES.md](EXPERIMENT_NOTES.md).

The UMAP implementation builds its own trailing descriptors from the GMM price caches. Its feature construction is independent of the removed trend-geometry experiment. Result directories retain the historical protocols and verification records for the experiments they belong to.
