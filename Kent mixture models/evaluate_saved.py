"""Boundary-inclusive, training-only comparison; never fits a model."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nbformat
from threadpoolctl import threadpool_limits
import kent_experiment as e
from kent_report import COLORS

ROOT=Path(__file__).resolve().parent

def run():
    source=ROOT/'results'/'a9f01047734a8126'
    out=source/'saved_boundary_evaluation';out.mkdir(exist_ok=True)
    data=e.b.load_data()['fx'];trials=pd.read_csv(source/'search_trials.csv')
    rows=[];scores={};checks=[]
    for k in e.km.GRID:
        for family in e.FAMILIES:
            group=trials[(trials.k==k)&(trials.family==family)]
            selected=e.b.best_validation(group[group.status=='ok']) if family=='GMM' else group.iloc[0]
            rec=joblib.load(source/'search'/f'{selected.id}.joblib')
            assert rec['input_hash']==e.b.array_digest(data.train)
            if family.startswith('Kent'):
                fit=rec['best_rejected_fit']
                assert fit['converged'], 'Saved candidate did not converge'
                e.km.constraints(fit)
                rec['fit']=fit
                for component,lam in enumerate(fit['lambda']):
                    checks.append(dict(id=selected.id,k=k,component=component,kappa=fit['kappa'],
                        max_ratio=2*lam.max()/fit['kappa'],absolute_ratio=2*np.abs(lam).max()/fit['kappa'],
                        strict_positive=bool(2*lam.max()<fit['kappa']),
                        strict_absolute=bool(2*np.abs(lam).max()<fit['kappa'])))
            test=e.score_record(data.test,rec);train=e.score_record(data.train,rec)
            assert np.isfinite(test).all()
            scores[selected.id]=test
            p=int(selected.parameters)
            rows.append(dict(id=selected.id,family=family,k=k,parameters=p,
                test_log_score=test.mean(),training_aic=2*p-2*train.sum(),
                training_bic=p*np.log(len(train))-2*train.sum()))
    results=pd.DataFrame(rows);daily=pd.DataFrame(scores,index=data.frame.index[data.b:])
    intervals=[]
    pairs=[('Kent linked','vMF linked'),('Kent unlinked','vMF unlinked'),
           ('Kent linked','Kent unlinked'),('vMF linked','vMF unlinked')]
    pairs += [(f,'GMM') for f in e.FAMILIES[1:]]
    for block in (5,20,60):
        boot=e.b.bootstrap_means(daily.to_numpy(),block)
        if block==20:
            bounds=np.quantile(boot,[.025,.975],axis=0)
            results['lower']=bounds[0];results['upper']=bounds[1]
        for k in e.km.GRID:
            mapping=results[results.k==k].set_index('family').id.to_dict()
            for left,right in pairs:
                a=daily.columns.get_loc(mapping[left]);b=daily.columns.get_loc(mapping[right])
                low,high=np.quantile(boot[:,a]-boot[:,b],[.025,.975])
                intervals.append(dict(k=k,comparison=f'{left} minus {right}',block=block,
                    gain=float((daily.iloc[:,a]-daily.iloc[:,b]).mean()),lower=low,upper=high))
    contrasts=pd.DataFrame(intervals);constraints=pd.DataFrame(checks)
    for name,table in [('scores',results),('contrasts',contrasts),('constraints',constraints)]:
        table.to_csv(out/f'{name}.csv',index=False)
    daily.to_csv(out/'daily_scores.csv')
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for family in e.FAMILIES:
        g=results[results.family==family]
        axes[0].errorbar(g.k,g.test_log_score,yerr=[g.test_log_score-g.lower,g.upper-g.test_log_score],
                         fmt='o-',label=family,color=COLORS[family],capsize=3)
        axes[1].plot(g.parameters,g.test_log_score,'o-',label=family,color=COLORS[family])
    axes[0].set(xlabel='Components K',ylabel='Test log density (nats/date)',xticks=e.km.GRID,
                title='All five families · 95% block intervals')
    axes[1].set(xlabel='Nominal parameters',ylabel='Test log density (nats/date)',title='Same saved training sample')
    axes[1].legend(fontsize=8)
    fig.savefig(out/'five_families.png',dpi=170);fig.savefig(out/'five_families.pdf')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,(left,right) in zip(axes.flat,pairs[:4]):
        g=contrasts[(contrasts.block==20)&(contrasts.comparison==f'{left} minus {right}')]
        ax.errorbar(g.gain,g.k,xerr=[g.gain-g.lower,g.upper-g.gain],fmt='o',capsize=3)
        ax.axvline(0,color='gray');ax.set(title=f'{left} minus {right}',yticks=e.km.GRID,
            ylabel='K',xlabel='Paired test gain (nats/date), 95% interval')
    fig.savefig(out/'paired_comparisons.png',dpi=170);fig.savefig(out/'paired_comparisons.pdf')
    book=nbformat.read(ROOT/'Kent_FX_Anisotropy_Pilot.ipynb',as_version=4)
    marker='## 10. Saved-model boundary-inclusive test evaluation'
    for i,cell in enumerate(book.cells):
        if marker in cell.source:
            book.cells=book.cells[:i];break
    book.cells.append(nbformat.v4.new_markdown_cell(marker+'''

This user-requested follow-up accepts the eight saved converged Kent fits at the imposed shape margin. No fitting was performed. Full parameters were saved for one best training-likelihood Kent fit per configuration, not all 80 starts. All eight saved candidates converged.

The primary comparison uses training-only checkpoints for **all five families**, with identical original FX inputs and 1,374 test dates. GMM covariance/regularization uses the original validation selection within K. Kent candidates were selected by training likelihood. The earlier main tables use development-refitted baselines and must not be directly compared with these training-only Kent scores.

Intervals use 1,500 paired circular block-bootstrap draws, with 20 sessions primary and 5/60 sensitivity. AIC/BIC here use **training likelihood and training N**, not development likelihood. Nominal parameter counts and boundary estimates limit classical interpretations. This amendment follows inspection of boundary outcomes; the test period is retrospective. There is no claim of a fresh confirmatory test.

The strict positive test is $2\\max_j\\lambda_j<\\kappa$. The stronger all-axis test is $2\\max_j|\\lambda_j|<\\kappa$. These are different restrictions. Passing the first does not establish monotonic decrease along every great circle. K counts mixture components, not verified economic regimes.
'''))
    cell=nbformat.v4.new_code_cell("from evaluate_saved import show_saved\nshow_saved()")
    book.cells.append(cell)
    nbformat.write(book,ROOT/'Kent_FX_Anisotropy_Pilot.ipynb')
    print(results.to_string(index=False));print(constraints.groupby('k')[['strict_positive','strict_absolute']].sum())

def show_saved():
    from IPython.display import display,Image
    out=ROOT/'results'/'a9f01047734a8126'/'saved_boundary_evaluation'
    for name in ('scores','contrasts','constraints'):
        display(pd.read_csv(out/f'{name}.csv'))
    for name in ('five_families','paired_comparisons'):
        display(Image(filename=str(out/f'{name}.png')))

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
