from pathlib import Path
import nbformat as nb
ROOT=Path(__file__).resolve().parent
cells=[]
def md(s):cells.append(nb.v4.new_markdown_cell(s))
def code(s):cells.append(nb.v4.new_code_cell(s))
md(r'''# Original-space equities: generalized Kent versus vMF and GMM

This notebook is the equity equivalent of the **boundary-inclusive, training-only FX comparison**. Each observation is the original vector of 50 stock log returns. Direction $u=x/\|x\|$ lies on $S^{49}$; magnitude $m=\log\|x\|$. The frozen data, chronological 60/20/20 splits and baseline fitting conventions are reused. No UMAP, scaling, context variable or hierarchy is introduced.

The five families are Cartesian GMM, unlinked vMF–Gaussian, linked vMF–Gaussian, unlinked generalized Kent–Gaussian and linked generalized Kent–Gaussian. “Linked” is the Type 1 specification: conditional magnitude mean $a_k+b_k^Tu$; unlinked sets $b_k=0$. Concentration and magnitude variance are shared across components. Every configuration receives **four initialization starts total**, fitted on training observations only. Validation selects configurations. There are no development refits and no bootstrap refits.
''')
code('''from pathlib import Path
import sys
cwd=Path.cwd()
root=next(p for p in [cwd,cwd/'Equities',cwd/'Kent mixture models'/'Equities',cwd/'Thesis'/'Kent mixture models'/'Equities'] if (p/'equity_experiment.py').exists())
sys.path.insert(0,str(root))
import equity_experiment as experiment
import equity_report as report
import pandas as pd
from IPython.display import display, Markdown
data=experiment.b.load_data()['equities']
display(experiment.b.split_table({'equities':data}))
display(pd.DataFrame(experiment.specs(50)).groupby('family').agg(K=('k',lambda s:sorted(set(s))),configurations=('id','size')))
''')
md(r'''## Model definitions and budgets

$$p(u,m)=\sum_k\pi_k f_k(u)\mathcal N(m;a_k+b_k^Tu,\sigma^2).$$

The directional Kent extension uses $f_k(u)\propto\exp[\kappa\mu_k^Tu+\sum_{j=1}^{49}\lambda_{kj}(\gamma_{kj}^Tu)^2]$, an orthonormal frame and trace-zero tangent spectrum. It retains the FX **one-sided** restriction $2\lambda_{\max}\le(1-10^{-6})\kappa$. Converged, feasible shape-boundary fits are accepted. Variance-floor, concentration-cap, collapsed and nonconverged fits are rejected. The stronger $2|\lambda_j|<\kappa$ restriction is **diagnosed, not imposed**; these are the same generalized Fisher–Bingham models as the evaluated FX pilot, not a new strict-Kent experiment.

Nominal parameter counts at dimension 50 are:

| Family | Parameters |
|---|---:|
| vMF unlinked | $51K+1$ |
| vMF linked | $101K+1$ |
| Kent unlinked | $1275K+1$ |
| Kent linked | $1325K+1$ |
| GMM spherical / diagonal | $52K-1$ / $101K-1$ |
| GMM tied / full | $51K+1274$ / $1326K-1$ |

GMM K is restricted to **{1,3,5}**, with its original four covariance options and three regularizations. Directional grids include {1,3,5} and the nearest integer K to each full-GMM parameter budget. Additional unlinked vMF configurations provide same-K nested warm starts for the expanded linked grid. These dependencies also receive only four starts and are recorded.

Same-K views use validation-selected GMM covariance/regularization. Budget views target full-GMM K=1,3,5 and admit counts within $[P/\sqrt{1.1},P\sqrt{1.1}]$, ensuring any two compared counts differ by at most 10%. Exact equality is usually arithmetically impossible. Each family is selected by validation within each window; missing admissible families remain missing. Best-in-grid selection also uses validation only.
''')
md(r'''## Numerical gate and execution

The gate verifies normalization on $S^{49}$ against the analytic vMF limit and independent contour integration, checks derivatives and orientation updates, verifies zero-anisotropy joint densities, polar reconstruction and nondecreasing synthetic GEM likelihood. It must pass before market fitting.

The inherited fitting limits are 500 outer iterations, mean log-likelihood tolerance $10^{-6}$, minimum effective membership 5, variance floor $10^{-6}$ and concentration cap $10^4$. Kent alternates frame BFGS and constrained shared-shape SLSQP updates, with the same Fourier normalizer and numerical tolerances as FX. Four starts replace ten; Kent reserves its final start for vMF and its penultimate linked start for unlinked Kent when available. Every completed start, including failures, is cached. Restarting resumes those caches without rerunning them.

Full S⁴⁹ Kent frames have 1,225 orientation coordinates per component versus 36 in FX. The full grid may take many hours. No reliable total runtime follows from dimension alone. Expanded linked vMF mixtures may have too few effective observations for stable 51-coefficient regressions. Failures are retained, without adding regularization or widening acceptance rules.
''')
code('''out=experiment.run()
display(Markdown(f"Results: `{out}`"))
report.show(out)
''')
md(r'''## Scores, uncertainty and interpretation

All models use Cartesian return-density scores. Directional scores include the Jacobian **$-50m$**. Training AIC/BIC use the training likelihood and training observation count; they must not be confused with development-sample criteria from the original FX notebook. Mean test log likelihood is reported separately.

Paired circular block bootstrap uses 1,500 resamples of existing per-date scores, with block lengths 5,20,60 (20 primary). **It fits no models.** Intervals condition on the selected fitted models and do not capture uncertainty over initialization or model selection. Four starts reduce optimization coverage relative to the ten-start FX run; confidence intervals cannot compensate for missed local optima.

Equal parameter counts do not imply equal flexibility or equal compute. GMM searches additional covariance and regularization choices, while vMF searches much higher K. Boundary fits, mixture singularity and serial dependence complicate AIC/BIC interpretation. Large-K models can overfit, collapse or have unstable assignments. This evaluates retrospective density fit, not recovery of true regimes or trading profitability. A component is not automatically an economic regime.
''')
book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3 (GMM environment)','language':'python','name':'python3'}})
nb.validate(book);nb.write(book,ROOT/'Kent_Equities_Anisotropy_Pilot.ipynb')
