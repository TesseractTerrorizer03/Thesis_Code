"""Build the readable pilot notebook; execution is a separate explicit step."""
from pathlib import Path
import nbformat as nb

ROOT=Path(__file__).resolve().parent
cells=[]
def md(text):cells.append(nb.v4.new_markdown_cell(text))
def code(text):cells.append(nb.v4.new_code_cell(text))

md(r'''# Original-space FX: generalized Kent anisotropy pilot

**Question:** Does replacing the rotationally symmetric vMF direction distribution with a constrained generalized Kent distribution improve joint direction–magnitude modeling of the original FX data?

This experiment uses the original nine-dimensional FX return representation: **no spherical UMAP, PCA, sector hierarchy, or added context variables**. A date is a nine-coordinate vector; its normalized direction lies on $S^8$. $K$ denotes mixture components, not currencies or dimensions.

**Agreed grid: $K\in\{1,3,6,10\}$; ten starts per configuration.** All directional families share concentration and radial variance across components. The Kent family estimates a full component-specific tangent shape. This is the initial shared-parameter benchmark, not the later concentration/variance-sharing extension.

This notebook runs from this folder, `Thesis`, or its parent workspace with the existing `GMM/.venv` environment. All new artifacts stay in this folder. Original data and estimator helpers are reused without modification.
''')
code('''from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from threadpoolctl import threadpool_limits

cwd = Path.cwd().resolve()
candidates = [cwd, cwd / 'Kent mixture models', cwd / 'Thesis' / 'Kent mixture models']
archive = next((p for p in candidates if (p / 'kent_experiment.py').is_file()), None)
if archive is None:
    raise FileNotFoundError('Run from Kent mixture models, Thesis, or its parent workspace.')
sys.path.insert(0, str(archive))
import kent_experiment as experiment
import kent_report as report

plt.rcParams.update({'figure.dpi': 110, 'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False})
data = experiment.b.load_data()['fx']
run_dir = experiment.run_directory(data)
display(pd.DataFrame(experiment.protocol(data)['splits']))
print('Run directory:', run_dir)
print('FX input hash:', experiment.protocol(data)['data_hash'])
print('Component grid:', experiment.km.GRID)
''')
md(r'''## 1. Data, fixed baseline, and density measure

The frozen ECB dataset includes EUR and nine other currencies. Negative log changes in foreign-currency units per EUR represent currency appreciation against EUR. Cross-currency demeaning removes the common numeraire, and the original orthonormal Helmert basis produces nine coordinates of that subspace. Orthogonality does not imply statistical independence. This basis change is not a lossy dimensionality reduction. No feature standardization, volatility scaling, filling, or new data acquisition is performed.

The original chronological 60/20/20 boundaries are retained. Validation selects configurations; selected models are refitted on training plus validation (the **development sample**) before evaluation on the final period. That period has been inspected in earlier work, so this is retrospective evaluation.

For an original return vector $x\in\mathbb R^9$, write $r=\|x\|_2$, $u=x/r$, and $m=\log r$. A directional–magnitude model supplies a density relative to sphere surface area and $dm$. The Cartesian GMM supplies a density relative to $dx$. Because $dx=r^8dr\,d\omega=r^9dm\,d\omega$,

$$\log p_X(x)=\log p_{U,M}(u,m)-9m.$$

The $-9m$ term is a **Jacobian coordinate correction**, not a complexity penalty. It is applied once to vMF/Kent models and never reapplied to the Cartesian GMM. It cancels between directional families on identical observations.
''')
md(r'''## 2. Model definitions and constraints

For either directional family, the joint model is

$$p(u,m)=\sum_{k=1}^K\pi_k f_k(u)\,\mathcal N(m;a_k+b_k^\top u,\sigma^2).$$

**Unlinked** sets $b_k=0$; **linked** estimates all nine Cartesian slopes. The linked families implement the **Type 1** direction-conditioned Gaussian structure discussed earlier. The unit-norm restriction is nonlinear and does not remove a linear regression coefficient. Magnitude variance $\sigma^2$ and concentration $\kappa$ are shared. Mixture membership can induce direction–magnitude dependence even in the unlinked model. The original Cartesian GMM directly models all nine return coordinates with a mixture of multivariate Gaussians.

For vMF, $f_k(u)\propto\exp(\kappa\mu_k^\top u)$. For generalized Kent,

$$f_k(u)=Z(\kappa,\lambda_k)^{-1}\exp\left[\kappa\mu_k^\top u+
\sum_{j=1}^8\lambda_{kj}(\gamma_{kj}^\top u)^2\right].$$

The nine-column frame is orthonormal, $\sum_j\lambda_{kj}=0$, and
$2\max_j\lambda_{kj}\leq(1-10^{-6})\kappa$. This is a constrained centered/unimodal Fisher–Bingham subfamily, not unrestricted Fisher–Bingham. At zero anisotropy it reduces to vMF. The small margin is a numerical constraint, not a tuned statistical hyperparameter.

A frame has 36 degrees of freedom and the trace-zero tangent spectrum has seven. Adding component intercepts/slopes, mixture weights, shared concentration, and shared variance gives:

| Family | Nominal parameters |
|---|---:|
| vMF unlinked | $10K+1$ |
| vMF linked | $19K+1$ |
| Kent unlinked | $45K+1$ |
| Kent linked | $54K+1$ |
| GMM spherical | $11K-1$ |
| GMM diagonal | $19K-1$ |
| GMM tied | $10K+44$ |
| GMM full | $55K-1$ |

GMM covariance options and regularization values $10^{-8},10^{-7},10^{-6}$ are inherited unchanged. There are **64 search configurations and 640 starts**. The Kent shape constraints restrict the allowable model but do not reduce its generic interior parameter count.
''')
md(r'''## 3. Numerical method and acceptance gate

The directional normalizer uses a one-dimensional inverse-Laplace/Fourier integral with continuous Euler damping, following [Chen and Tanaka](https://arxiv.org/abs/2004.14660). A real saddle locates a stable integration contour; **no saddlepoint approximation to the density is substituted**. The contour sign is derived directly from Laplace inversion, avoiding inconsistent signs in the manuscript's later expressions. An analytic vMF limit provides a separate check.

Refinement requires agreement below $10^{-10}$ in log normalization and $2\times10^{-10}$ in its derivatives. These are refinement diagnostics, not rigorous error bounds. Independent undamped adaptive contour integration, direct $S^2$ integration, finite differences, and the analytic vMF limit provide additional checks. No S²-only Kent package is used for the S⁸ fit.

The E-step uses **both** direction and magnitude. Gaussian updates retain the original weighted regression. Generalized EM alternates orthogonality-preserving orientation optimization with joint shared-concentration/eigenvalue optimization. Updates must improve the weighted objective; observed likelihood is also checked. An inner solver stall is not silently declared successful EM convergence.

The original limits are retained: 500 outer iterations, mean-log-likelihood tolerance $10^{-6}$, variance floor $10^{-6}$, concentration cap $10^4$, and minimum effective membership 5. Fits reaching variance, concentration, or the declared shape boundary are excluded from selection. Boundary rejection is a protocol choice, not proof that every boundary density is invalid.

All families have ten starts. Kent reserves a start for zero-anisotropy vMF initialization; linked Kent also reserves a start for an admissible zero-link unlinked Kent fit. If that prerequisite is unavailable, the slot uses its matched deterministic initialization. Warm starts have a computational advantage, so preparation costs and origins are recorded.

The gate checks normalization, gradients, rotations, axis permutations, repeated eigenvalues, the Jacobian, parameter counts, synthetic fits, collapse, and failure handling. Synthetic examples assess implementation behavior, not recovery of economic regimes.
''')
code('''# A failed numerical gate prevents market-model rankings.
gate_passed = experiment.run_gate(run_dir)
display(pd.DataFrame([json.loads((run_dir / 'numerical_gate.json').read_text())]))
print((run_dir / 'numerical_gate.txt').read_text())

if gate_passed and not (run_dir / 'completion.json').exists():
    experiment.run()
ready = gate_passed and (run_dir / 'completion.json').exists()
if ready:
    results = report.load(run_dir)
    kent_trials = results['search'][results['search'].family.str.startswith('Kent')]
    if not (kent_trials.status == 'ok').any():
        display(Markdown('**Outcome: Kent was fitted but not test-scored.** All eight Kent configurations '
            '(linked/unlinked at four K values, ten starts each) were excluded by the declared acceptance rules. '
            'A shape-boundary solution can remain a valid density; automatic boundary rejection was a conservative '
            'protocol choice. This run therefore does **not** answer whether Kent beats vMF. See Sections 7–9 for the evidence.'))
    with threadpool_limits(limits=1):
        display(pd.DataFrame([report.verify(results)]))
        display(report.numerical_diagnostics(results))
else:
    display(Markdown('**Pilot incomplete: see the numerical gate and fitting ledgers. No performance conclusion is available.**'))
''')
md(r'''## 4. Validation selection and fit availability

Starts are selected using training likelihood. Only admissible starts participate. Configuration choices use validation likelihood, with the original tie rule: within $10^{-6}$ of the maximum, prefer fewer parameters, then fewer components, then declared configuration order.

The same-component view selects a GMM specification within each K. The best-in-grid view selects K separately for each family. These views do not equalize model size. A winner at K=10 is a search-boundary result.

Selection is frozen before development refits or test scoring. Failed refits remain unavailable. The ledger includes every declared configuration and every rejected start.
''')
code('''if ready:
    display(results['search'][['family','k','covariance','regularization','parameters',
                              'accepted_starts','status','validation_mean_loglik','elapsed_seconds',
                              'warm_preparation_seconds']].round(5))
    display(results['selection'][results['selection'].view == 'best_in_grid'])
    report.validation_figure(results)
    plt.show()
''')
md(r'''## 5. Same-component held-out comparisons

**Log score:** mean log density of the original FX return vectors; higher is better, in natural-log units (nats) per date.

**AIC/BIC:** $2p-2\ell$ and $p\log N-2\ell$, using total development likelihood, nominal parameter count $p$, and development observations $N$. Lower is better. They describe fitted-sample fit/complexity, not test accuracy. No test-bootstrap intervals are attached to these criteria.

**Uncertainty:** 1,500 paired circular block-bootstrap draws over complete test dates. The primary block is 20 sessions; 5 and 60 are sensitivity checks. Paired differences retain the same resampled dates for both models. Intervals are pointwise, conditional on fitted models, and not adjusted for model selection or multiple comparisons.

No ordinary chi-square likelihood-ratio test is used for zero anisotropy: tangent axes become unidentified at the isotropic limit, and finite mixtures introduce further nonregularity. The declared held-out paired log-score comparisons are the primary evidence.

Kent-minus-vMF isolates directional-family changes at the same K and radial specification. Linked-minus-unlinked assesses the direction–magnitude regression. Comparisons with GMM use the same Cartesian density measure. None of these scores establishes true regime labels or forecasting profitability.
''')
code('''if ready:
    display(report.same_component_table(results).round(5))
    report.same_k_figure(results)
    plt.show()
    report.baseline_figure(results)
    plt.show()
    display(results['contrasts'][results['contrasts'].view.str.startswith('same_K')].round(5))
''')
md(r'''## 6. Equal-parameter and approximately matched comparisons

Exact matches are shown where feasible. Approximate budget windows require maximum/minimum parameter count ≤1.10; validation selects the available family configuration in each window, and duplicate selections are removed. A window need not contain every family.

For this grid, **no unlinked Kent–vMF match exists within 10%**. The only linked match is Kent K=1 (55 parameters) versus vMF K=3 (58 parameters), and is usable only if both fits are admissible. No extra K values are fitted to fill missing comparisons. Exact matching of nominal counts does not make the statistical models equally expressive.
''')
code('''if ready:
    display(report.comparison_feasibility(results).round(5))
    display(report.budget_table(results).round(5))
    report.budget_figure(results)
    plt.show()
    display(results['contrasts'][results['contrasts'].view.str.startswith(('budget10','exact'))].round(5))
''')
md(r'''## 7. Runtime, convergence, and numerical rejection

Training fits are timed freshly with one numerical-library thread. Baselines call the original estimator with timing/trace hooks verified to preserve fitted scores. GMM traces require an additional score evaluation per iteration, so recorded runtimes include that instrumentation overhead. GMM covariance regularization need not give strictly monotone unpenalized likelihood.

The runtime chart includes rejected starts; GMM uses its validation-selected covariance/regularization at each K. Per-start timings cover the instrumented fit routines and exclude shared K-means responsibility construction. Separate ledgers retain configuration totals and warm-start preparation costs. An EM iteration is not a fixed unit of work: Kent additionally performs numerical normalization, constrained shape optimization, and frame updates.

Convergence curves show within-start gaps to final training likelihood. Dashed curves are rejected training fits, displayed diagnostically rather than promoted to evaluated candidates. Development fit timings are also in the main score table.
''')
code('''if ready:
    starts = report.search_starts(results)
    display(starts.groupby(['family','k']).agg(starts=('start','size'),converged=('converged','sum'),accepted=('accepted','sum'),
             median_seconds=('seconds','median'),median_iterations=('iterations','median'),
             normalization_seconds=('normalizer_seconds','sum')).round(4))
    rejected = starts[~starts.accepted]
    display(rejected.groupby(['family','k','status','boundary_reasons'],dropna=False).size().rename('starts').reset_index())
    display(starts[starts.family.str.startswith('Kent')].groupby(['family','k'])[
        ['shape_iterations','shape_evaluations','frame_iterations','frame_evaluations',
         'normalizer_calls','normalizer_failures']].sum())
    display(report.normalization_run_summary(results))
    report.runtime_figure(results)
    plt.show()
    report.convergence_figure(results)
    plt.show()
''')
md(r'''## 8. What anisotropy was fitted?

The diagnostic heatmaps show $2\lambda_j/\kappa$ for ordered tangent eigenvalues. Zero means isotropy in that quadratic coefficient; the largest permitted ratio is approximately one. Rows are annotated by training mixture weight, not economic labels.

For a concentrated interior component, the local tangent Gaussian approximation has variances $(\kappa-2\lambda_j)^{-1}$. The exported largest/smallest variance ratio is approximate and becomes unreliable near the unimodality boundary. The heatmap can include the best rejected training fit, explicitly labeled. Such a fit is not evaluated on test data.

Directional and conditional-magnitude log scores decompose the joint cylindrical score for admissible directional models. They describe which distributional block changes; they are not directly comparable with a GMM Cartesian joint score.
''')
code('''if ready:
    report.anisotropy_figure(results)
    plt.show()
    display(pd.read_csv(run_dir / 'anisotropy_summary.csv').round(5))
    display(results['decomposition'].round(5))
''')
md(r'''## 9. Findings and limits

| Issue | Interpretation |
|---|---|
| Original-space data | No embedding distortion is introduced; FX coordinates and original log magnitudes are unchanged. |
| Full anisotropy | Kent has substantially more parameters per component; equal K does not control capacity. |
| Shared concentration/variance | This pilot matches the original shared baseline, not the later sharing ablation. |
| Numerical optimization | Kent is generalized EM with local numerical updates; global optimality is not guaranteed. |
| Boundary rejection | A constrained optimum at the shape boundary can be statistically meaningful, but is inadmissible under this declared pilot. No silent relaxation occurs. |
| Initialization | Matched seeds do not eliminate optimization differences; Kent has explicit nested warm starts. |
| Search budget | Each directional family has four configurations; GMM has 48 covariance/regularization/K configurations. Ten starts per configuration does not equalize total search effort. |
| Parameter matching | Some comparisons are structurally unavailable; no grid or tolerance is widened after seeing outcomes. |
| AIC/BIC | Nominal counts, mixture singularities, and serial dependence limit classical interpretations. |
| Test reuse | The historical period was previously inspected; this is retrospective evidence. |
| Static mixtures | This is density estimation, without regime transitions or a trading strategy. |

The statements below are generated from saved results. If the numerical gate passes but Kent fits are inadmissible, that is a limitation of this declared pilot—not evidence that vMF universally dominates anisotropic distributions.
''')
code('''if ready:
    display(Markdown(report.conclusions(results)))
    print('Full artifacts:', run_dir)
''')

notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3 (GMM environment)',
    'language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}})
nb.validate(notebook)
nb.write(notebook,ROOT/'Kent_FX_Anisotropy_Pilot.ipynb')
print('Built',ROOT/'Kent_FX_Anisotropy_Pilot.ipynb')
