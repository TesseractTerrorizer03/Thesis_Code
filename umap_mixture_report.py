"""Within-representation diagnostics for the replicated mixture experiment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import ive
import matplotlib.pyplot as plt
import umap_mixture_comparison as m


def load():
    for path in sorted(m.OUT.glob('*/status.json'),reverse=True):
        status=json.loads(path.read_text());protocol=json.loads((path.parent/'protocol.json').read_text())
        if status.get('evaluation')=='complete' and protocol.get('evaluation_code_sha256',protocol['study_sha256'])==m.s.sha(m.__file__):
            folder=path.parent
            return folder,{name:pd.read_csv(folder/(name+'.csv')) for name in ['validation_trials','frozen_selections','test_results','test_daily','paired_gains','kappas']}
    raise FileNotFoundError('Mixture study is not yet complete; run umap_mixture_comparison.py.')


def save(fig,folder,name):
    directory=folder/'figures';directory.mkdir(exist_ok=True)
    fig.savefig(directory/(name+'.png'),dpi=150,bbox_inches='tight');fig.savefig(directory/(name+'.pdf'),bbox_inches='tight');plt.show()


def unique(frame):return frame.drop_duplicates(['case','candidate'])


def validation_plot(tables,folder):
    trials=tables['validation_trials'];fig,axes=plt.subplots(2,2,figsize=(14,8))
    for ax,(case,g) in zip(axes.flat,trials.groupby('case')):
        for family in ['GMM','Unlinked','Linked']:
            groups=g[g.family.eq(family)].groupby('covariance') if family=='GMM' else [('scalar',g[g.family.eq(family)])]
            for cov,rows in groups:
                curve=rows.groupby('k').validation_mean_loglik.max()
                ax.plot(curve.index,curve.values,marker='.',label=family+(' '+cov if family=='GMM' else ''))
        ax.set_title(case);ax.set_xlabel('Components (1–20)');ax.set_ylabel('Validation mean log density');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Validation selects components, Gaussian covariance/ridge, and variance/κ sharing\nEach panel has its own density measure; do not compare absolute heights between panels',fontsize=12)
    fig.tight_layout();save(fig,folder,'01_validation_selection')


def result_plot(tables,folder):
    frame=unique(tables['test_results'])
    fig,axes=plt.subplots(4,3,figsize=(17,15))
    for row,(case,g) in enumerate(frame.groupby('case')):
        labels=[r.family+(' '+r.matrix_toggle if r.family=='GMM' else ' / '+r.sharing) for r in g.itertuples()]
        for col,metric in enumerate(['test_mean_loglik','aic','bic']):
            ax=axes[row,col];values=g[metric].to_numpy()
            if col:
                values=values-values.min();ax.barh(range(len(g)),values,color='#3b82f6');ax.set_xlabel('Δ'+metric.upper()+' from best in this panel')
            else:
                for j,(_,r) in enumerate(g.iterrows()):
                    ax.plot([r.test_L20_low,r.test_L20_high],[j,j],color='#2563eb',lw=2)
                    ax.scatter(r.test_mean_loglik,j,color='#2563eb')
                ax.set_xlabel('Test log density; 95% block interval')
            ax.set_yticks(range(len(g)),labels if col==0 else ['']*len(g),fontsize=8);ax.set_title(case+' · '+metric);ax.grid(axis='x',alpha=.2)
    fig.suptitle('Validation-selected models: held-out test scores and development-fit information criteria\nUMAP criteria condition on a fixed encoder. All comparisons are within each panel.',fontsize=13)
    fig.tight_layout();save(fig,folder,'02_test_aic_bic')


def sharing_plot(tables,folder):
    trials=tables['validation_trials'];fig,axes=plt.subplots(2,2,figsize=(14,8))
    for ax,(case,g) in zip(axes.flat,trials.groupby('case')):
        for (family,mode),rows in g[g.family.ne('GMM')].groupby(['family','sharing']):
            ax.plot(rows.k,rows.validation_mean_loglik,label=family+' / '+mode,lw=1)
        ax.set_title(case);ax.set_xlabel('Components');ax.set_ylabel('Validation mean log density');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7,ncol=2)
    fig.suptitle('κ and variance sharing are selected on validation; numerical κ estimates use training only',fontsize=13)
    fig.tight_layout();save(fig,folder,'03_kappa_variance_selection')


def kappa_table(tables):
    frame=tables['kappas'].drop_duplicates(['case','candidate','component']).copy()
    frame['dimensions']=frame.case.map(lambda c:3 if c.endswith('/umap') else 50 if c.startswith('equities') else 9)
    frame['expected_direction_cosine']=[0. if k==0 else float(ive(d/2,k)/ive(d/2-1,k)) for k,d in zip(frame.kappa,frame.dimensions)]
    return frame


def verify(tables,folder):
    result=tables['test_results'];daily=tables['test_daily']
    assert len(result)==24
    for (case,family),g in result[result.family.ne('GMM')].groupby(['case','family']):
        assert g.matrix_toggle.nunique()==2 and g.candidate.nunique()==1
        assert g[['test_mean_loglik','aic','bic','parameters']].nunique().eq(1).all()
    for row in unique(result).itertuples():
        values=daily[(daily.case==row.case)&(daily.candidate==row.candidate)].loglik
        np.testing.assert_allclose(values.mean(),row.test_mean_loglik,atol=1e-10)
        selection=tables['validation_trials'].query('case == @row.case and family == @row.family')
        if row.family=='GMM':selection=selection[selection.covariance.eq(row.matrix_toggle)]
        best=selection.validation_mean_loglik.max()
        assert row.validation_mean_loglik>=best-1e-6
    cases,_=m.prepare_cases()
    for row in unique(result).itertuples():
        data=cases[row.case];record=m.joblib.load(m.OUT/'cache'/m.signature(data['development'])/'refit'/(row.candidate+'.joblib'))
        ll=m.score(data['development'],record).sum()
        np.testing.assert_allclose(row.aic,2*row.parameters-2*ll)
        np.testing.assert_allclose(row.bic,row.parameters*np.log(len(data['development']))-2*ll)
        if row.family=='GMM':
            np.testing.assert_allclose([row.aic,row.bic],[record['fit'].aic(data['development']),record['fit'].bic(data['development'])])
    record=dict(rows=24,distinct_fitted_comparisons=len(unique(result)),matrix_aliases_verified=True,
                validation_selection_verified=True,aic_bic_verified=True,test_daily_verified=True,
                test_used_for_selection=False,study_sha256=m.s.sha(m.__file__))
    m.s.write_json(folder/'verification.json',record);return record


def conclusions(tables):
    lines=[];results=unique(tables['test_results']);paired=tables['paired_gains']
    for case,g in results.groupby('case'):
        normal=g[g.family.eq('GMM')].sort_values('validation_mean_loglik',ascending=False).iloc[0]
        text=f'**{case}:** validation selected {normal.matrix_toggle} covariance for normal GMM (K={normal.k}). '
        for family in ['Unlinked','Linked']:
            r=g[g.family.eq(family)].iloc[0]
            interval=paired[(paired.case==case)&(paired.candidate==r.candidate)&(paired.block==20)].iloc[0]
            text+=f'{family}: K={r.k}, {r.sharing}; test gain versus the validation-selected normal GMM {interval["mean"]:+.3f} nats (95% interval {interval.lower:+.3f}, {interval.upper:+.3f}). '
        def label(r):return r.family+'/'+(r.matrix_toggle if r.family=='GMM' else r.sharing)
        text+=f'Lowest development AIC: {label(g.loc[g.aic.idxmin()])}; lowest BIC: {label(g.loc[g.bic.idxmin()])}.'
        if g.component_boundary.any():text+=' At least one winner reaches K=20; its component optimum is unresolved beyond the declared grid.'
        lines.append('- '+text)
    for market in ['equities','fx']:
        winners={}
        for representation in ['raw','umap']:
            g=results[results.case.eq(market+'/'+representation)]
            r=g.loc[g.test_mean_loglik.idxmax()]
            winners[representation]=r.family+('/'+r.matrix_toggle if r.family=='GMM' else '/'+r.sharing)
        lines.append(f'**{market} transformation comparison:** highest observed test-density model within raw space: {winners["raw"]}; within UMAP space: {winners["umap"]}. This is a descriptive ranking of frozen candidates, not a test-based hyperparameter choice.')
    for r in covariance_effects(tables).itertuples():
        lines.append(f'**{r.case}, full versus diagonal normal GMM:** validated K={r.full_components} (p={r.full_parameters}) versus K={r.diagonal_components} (p={r.diagonal_parameters}); test gain {r.mean:+.3f} nats (95% interval {r.lower:+.3f}, {r.upper:+.3f}). This comparison includes the separately selected component counts.')
    lines.append('These within-space rankings and selected complexities describe the effect of changing representation. '
        'It does not establish improved density modeling of the original returns: UMAP discards directional dimensions, and cross-space likelihood/AIC/BIC differences are not comparable. '
        'Differences between best linked/unlinked fits also include selected K and sharing changes, so they do not isolate the link coefficient alone. '
        'Diagonal/full cylinders are identical scalar-covariance models. κ values across dimensions should be interpreted using the reported expected direction cosine, not their numerical sizes alone. '
        'These are retrospective, conditional model comparisons, with serial dependence and mixture singularities limiting classical AIC/BIC interpretation.')
    return '\n\n'.join(lines)


def covariance_effects(tables):
    rows=[];daily=tables['test_daily'];result=unique(tables['test_results'])
    for case,g in result[result.family.eq('GMM')].groupby('case'):
        a=g[g.matrix_toggle.eq('full')].iloc[0];b=g[g.matrix_toggle.eq('diag')].iloc[0]
        full=daily[(daily.case==case)&(daily.candidate==a.candidate)].set_index('date').loglik
        diagonal=daily[(daily.case==case)&(daily.candidate==b.candidate)].set_index('date').loglik
        assert full.index.equals(diagonal.index)
        rows.append(dict(case=case,full_components=a.k,diagonal_components=b.k,full_parameters=a.parameters,
                         diagonal_parameters=b.parameters,**m.s.block_interval((full-diagonal).to_numpy(),20,1500)))
    return pd.DataFrame(rows)


def sample_log_magnitude(record,n=20000,seed=701):
    """Sample the fitted marginal of log original magnitude, without refitting."""
    from scipy.stats import vonmises_fisher
    rng=np.random.default_rng(seed)
    gaussian='sharing' not in record and record['spec']['family']=='GMM'
    f=record['fit'];weights=f.weights_ if gaussian else f['weights']
    group=rng.choice(len(weights),n,p=weights);out=np.empty(n)
    for j in range(len(weights)):
        positions=np.flatnonzero(group==j);count=len(positions)
        if not count:continue
        if gaussian:
            if f.covariance_type=='diag':
                draws=f.means_[j]+rng.normal(size=(count,f.means_.shape[1]))*np.sqrt(f.covariances_[j])
            else:draws=rng.multivariate_normal(f.means_[j],f.covariances_[j],size=count,check_valid='raise')
            out[positions]=np.log(np.linalg.norm(draws,axis=1))
        else:
            variance=np.atleast_1d(f['variance']);kap=np.atleast_1d(f['kappa'])
            kappa=kap[j] if len(kap)>1 else kap[0]
            if not np.any(f['coef'][j,1:]):conditional=np.full(count,f['coef'][j,0])
            else:
                if kappa==0:
                    u=rng.normal(size=(count,f['mu'].shape[1]));u/=np.linalg.norm(u,axis=1)[:,None]
                else:u=np.atleast_2d(vonmises_fisher(f['mu'][j],float(kappa)).rvs(size=count,random_state=rng))
                conditional=f['coef'][j,0]+u@f['coef'][j,1:]
            out[positions]=conditional+rng.normal(size=count)*np.sqrt(variance[j] if len(variance)>1 else variance[0])
    if not np.isfinite(out).all():raise ValueError('Invalid generative log-magnitude sample.')
    return out


def empirical_crps(samples,observations):
    """Lower-better, unbiased ensemble estimate E|S-y| - .5 E|S-S'|.

    Sorting/prefix sums avoid allocating a sample-by-observation distance array.
    The pair term excludes self-pairs (N*(N-1) denominator).
    """
    x=np.sort(np.asarray(samples,float));y=np.asarray(observations,float);n=len(x)
    if n<2 or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Finite samples and observations required.')
    sums=np.r_[0.,np.cumsum(x)];k=np.searchsorted(x,y)
    absolute=(y*k-sums[k]+sums[-1]-sums[k]-y*(n-k))/n
    half_pair=np.dot(2*np.arange(n)-n+1,x)/(n*(n-1))
    return absolute-half_pair


def common_magnitude(tables,folder):
    cases,_=m.prepare_cases();selected=unique(tables['test_results']);daily=[];summary=[];contrasts=[]
    scores={};seed_scores={}
    for row in selected.itertuples():
        data=cases[row.case]
        record=m.joblib.load(m.OUT/'cache'/m.signature(data['development'])/'refit'/(row.candidate+'.joblib'))
        market,representation=row.case.split('/')
        y=np.log(np.linalg.norm(cases[market+'/raw']['test'],axis=1))
        samples=[sample_log_magnitude(record,20000,seed) for seed in [701,702,703]]
        point=empirical_crps(np.concatenate(samples),y)
        individual=np.array([empirical_crps(z,y) for z in samples])
        label='GMM_'+row.matrix_toggle if row.family=='GMM' else row.family
        scores[market,representation,label]=point;seed_scores[market,representation,label]=individual
        daily.append(pd.DataFrame(dict(date=data['dates'][data['b']:],case=row.case,model=label,candidate=row.candidate,crps_log_magnitude=point)))
        summary.append(dict(case=row.case,model=label,candidate=row.candidate,crps=float(point.mean()),samples=60000,
                            seed_min=float(individual.mean(axis=1).min()),seed_max=float(individual.mean(axis=1).max())))
    for market in ['equities','fx']:
        for model in ['GMM_diag','GMM_full','Unlinked','Linked']:
            difference=scores[market,'raw',model]-scores[market,'umap',model]
            seed_difference=(seed_scores[market,'raw',model]-seed_scores[market,'umap',model]).mean(axis=1)
            for block in [5,20,60]:
                contrasts.append(dict(market=market,model=model,block=block,
                    positive_means='UMAP has lower log-magnitude CRPS',mc_seed_min=float(seed_difference.min()),mc_seed_max=float(seed_difference.max()),
                    **m.s.block_interval(difference,block,1500)))
    pd.concat(daily,ignore_index=True).to_csv(folder/'common_magnitude_daily.csv',index=False)
    summary=pd.DataFrame(summary);contrasts=pd.DataFrame(contrasts)
    summary.to_csv(folder/'common_magnitude_scores.csv',index=False);contrasts.to_csv(folder/'common_magnitude_gains.csv',index=False)
    m.s.write_json(folder/'common_magnitude_protocol.json',dict(target='log norm of original return vector',score='CRPS, lower better',
        seeds=[701,702,703],samples_per_seed=20000,model_reselection=False,report_source_sha256=m.s.sha(__file__),scope='Common retained marginal; not recovery of original directional distribution.',
        bootstrap_repetitions=1500,blocks=[5,20,60],source='https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf'))
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for ax,(market,g) in zip(axes,contrasts[contrasts.block.eq(20)].groupby('market')):
        for j,(_,r) in enumerate(g.iterrows()):
            ax.plot([r.lower,r.upper],[j,j],color='#2563eb');ax.scatter(r['mean'],j,color='#2563eb')
        ax.set_yticks(range(len(g)),g.model);ax.axvline(0,color='black',lw=1);ax.set_title(market)
        ax.set_xlabel('Raw CRPS − UMAP CRPS; positive favors UMAP');ax.grid(axis='x',alpha=.2)
    fig.suptitle('Comparable held-out log-magnitude calibration · fixed validation-selected models')
    fig.tight_layout();save(fig,folder,'04_common_magnitude_crps')
    return summary,contrasts


def magnitude_conclusions(gains):
    lines=[]
    for r in gains[gains.block.eq(20)].itertuples():
        verdict=('UMAP improves retained-magnitude calibration' if r.lower>0 and r.mc_seed_min>0 else
                 'raw features improve retained-magnitude calibration' if r.upper<0 and r.mc_seed_max<0 else
                 'the comparison is inconclusive after test and Monte Carlo sensitivity checks')
        lines.append(f'- **{r.market} / {r.model}:** {verdict}. Raw-minus-UMAP CRPS {r.mean:+.5f}, '
                     f'95% interval [{r.lower:+.5f}, {r.upper:+.5f}]; separate-seed estimates '
                     f'[{r.mc_seed_min:+.5f}, {r.mc_seed_max:+.5f}].')
    return '\n'.join(lines)
