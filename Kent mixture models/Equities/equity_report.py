from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display,Markdown
import equity_experiment as e
from kent_report import COLORS

def show(out):
    out=Path(out);figures=out/'figures';figures.mkdir(exist_ok=True)
    scores=pd.read_csv(out/'scores.csv');selection=pd.read_csv(out/'selections.csv')
    trials=pd.read_csv(out/'search.csv');contrasts=pd.read_csv(out/'contrasts.csv')
    constraints=pd.read_csv(out/'constraints.csv')
    display(pd.read_json(out/'completion.json',typ='series').to_frame('count'))
    display(trials);display(selection.merge(scores,on=['id','family','k','parameters']))
    def save(fig,name):
        fig.savefig(figures/f'{name}.png',dpi=170,bbox_inches='tight')
        fig.savefig(figures/f'{name}.pdf',bbox_inches='tight');plt.show()
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for family in e.FAMILIES:
        s=selection[selection.view.str.startswith('same_K')&selection.family.eq(family)].merge(scores,on=['id','family','k','parameters']).sort_values('k')
        if len(s):axes[0].errorbar(s.k,s.test_log_score,yerr=[s.test_log_score-s.lower,s.upper-s.test_log_score],fmt='o-',color=COLORS[family],label=family,capsize=3)
        g=scores[scores.family==family].sort_values('parameters')
        axes[1].scatter(g.parameters,g.test_log_score,color=COLORS[family],label=family,alpha=.65)
    axes[0].set(xlabel='K',xticks=e.BASE,ylabel='Test log density, nats/date',title='Same K · 95% block intervals')
    axes[1].set(xlabel='Nominal parameters',ylabel='Test log density, nats/date',title='All admissible configurations')
    axes[1].legend(fontsize=8);save(fig,'01_five_families')
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for ax,k in zip(axes,e.BASE):
        g=selection[selection.view.eq(f'budget_full_GMM_K{k}')].merge(scores,on=['id','family','k','parameters'])
        for i,f in enumerate(e.FAMILIES):
            s=g[g.family==f]
            if s.empty:ax.text(.02,i,'Unavailable',transform=ax.get_yaxis_transform());continue
            r=s.iloc[0];ax.plot([r.lower,r.upper],[i,i],color=COLORS[f]);ax.scatter(r.test_log_score,i,color=COLORS[f])
            ax.annotate(f'K={r.k}, p={r.parameters}',(r.test_log_score,i),xytext=(0,7),textcoords='offset points',fontsize=7)
        ax.set(yticks=range(5),yticklabels=e.FAMILIES,ylim=(-.6,4.6),xlabel='Test log density, nats/date',title=f'Full-GMM K={k} parameter budget')
    save(fig,'02_matched_budgets')
    primary=contrasts[contrasts.block==20]
    fig,axes=plt.subplots(1,2,figsize=(14,7),layout='constrained')
    for ax,prefix,title in zip(axes,['same_K','budget'],['Same K','Matched parameter budgets']):
        g=primary[primary.view.str.startswith(prefix)]
        for i,r in enumerate(g.itertuples()):
            ax.plot([r.lower,r.upper],[i,i],color='#2477a4');ax.scatter(r.gain,i,color='#2477a4')
        ax.axvline(0,color='gray');ax.set(yticks=range(len(g)),yticklabels=[f'{r.view}: {r.comparison}' for r in g.itertuples()],title=title,xlabel='Paired gain, nats/date (95% interval)')
        ax.tick_params(axis='y',labelsize=6)
    save(fig,'03_paired_intervals');display(contrasts)
    starts=[]
    for path in sorted((out/'fits').glob('*.joblib')):
        r=joblib.load(path);d=r['diagnostic'];s=r['pilot_spec']
        starts.append(dict(id=s['id'],family=s['family'],k=s['k'],start=d['start'],status=d['status'],
            accepted=d['accepted'],seconds=d['elapsed_seconds'],iterations=d['iterations'],origin=d.get('origin',''),
            normalization_seconds=d.get('normalizer',{}).get('seconds',0)))
    starts=pd.DataFrame(starts);starts.to_csv(out/'starts.csv',index=False);display(starts.groupby(['family','k']).agg(starts=('start','size'),accepted=('accepted','sum'),median_seconds=('seconds','median'),median_iterations=('iterations','median')))
    assert (starts.groupby('id').size()==4).all()
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for f in e.FAMILIES:
        g=starts[starts.family==f].groupby('k').agg(seconds=('seconds','median'),accepted=('accepted','mean'))
        axes[0].plot(g.index,g.seconds,'o-',label=f,color=COLORS[f]);axes[1].plot(g.index,g.accepted,'o-',color=COLORS[f])
    axes[0].set(xlabel='K',ylabel='Median seconds per start',yscale='log');axes[0].legend(fontsize=8)
    axes[1].set(xlabel='K',ylabel='Accepted fraction',ylim=(-.05,1.05));save(fig,'04_runtime_acceptance')
    display(constraints)
    if len(constraints):
        fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
        for i,(ident,g) in enumerate(constraints.groupby('id')):
            ax.scatter(np.full(len(g),i)-.08,g.max_ratio,label='2λmax/κ' if i==0 else None,color='#187e65')
            ax.scatter(np.full(len(g),i)+.08,g.absolute_ratio,label='2 max|λ|/κ' if i==0 else None,color='#9b5b9c')
        ax.axhline(1,color='gray');ax.set(xticks=range(constraints.id.nunique()),xticklabels=sorted(constraints.id.unique()),ylabel='Ratio; strict pass below 1')
        ax.tick_params(axis='x',labelrotation=35);ax.legend();save(fig,'05_constraints')
    data=e.b.load_data()['equities'];daily=pd.read_csv(out/'daily_scores.csv',index_col=0)
    for r in scores.itertuples():
        rec=joblib.load(out/f'{r.id}.joblib')
        np.testing.assert_allclose(e.fx.score_record(data.test,rec),daily[r.id],atol=1e-10,rtol=0)
    e.fx.save_json(out/'verification.json',dict(test_scores_recomputed=True,four_starts_per_configuration=True))
    findings=[]
    for r in primary[primary.comparison.isin(['Kent linked minus vMF linked','Kent unlinked minus vMF unlinked'])].itertuples():
        findings.append(f'- {r.view}, {r.comparison}: {r.gain:+.4f} nats/date, 95% interval [{r.lower:+.4f}, {r.upper:+.4f}].')
    for view in ['budget_full_GMM_K1','budget_full_GMM_K3','budget_full_GMM_K5']:
        absent=set(e.FAMILIES)-set(selection[selection.view==view].family)
        if absent:findings.append(f'- {view}: unavailable families: {", ".join(sorted(absent))}. The budget was not widened.')
    findings.append(f'- {int((trials.status=="ok").sum())}/{len(trials)} configurations produced an admissible fit; each had four starts.')
    if len(constraints):findings.append(f'- Saved Kent components passing positive-eigenvalue check: {int(constraints.positive_pass.sum())}/{len(constraints)}; absolute-eigenvalue check: {int(constraints.absolute_pass.sum())}/{len(constraints)}.')
    summary='\n'.join(findings);(out/'findings.md').write_text(summary+'\n');display(Markdown(summary))
    display(Markdown('All reported test scores were recomputed from saved parameters. No fit was repeated for reporting or confidence intervals. Missing families indicate unavailable fits, not zero gains.'))
