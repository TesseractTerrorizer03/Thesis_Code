"""Uniform reporting and independent artifact checks for the sharing ablation."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import size_direction_experiment as b
import size_direction_sharing as s

ORDER=['GMM','Unlinked','Linked']+[s.family(mode,linked) for mode in s.MODES for linked in (False,True)]
LABELS=['GMM','Unlinked · shared','Linked · shared','Unlinked · separate variance','Linked · separate variance',
        'Unlinked · separate κ','Linked · separate κ','Unlinked · separate variance + κ','Linked · separate variance + κ']
COLORS=['#586b87','#bf7838','#287c91','#bf7838','#287c91','#bf7838','#287c91','#bf7838','#287c91']
VIEWS=['budget_low','budget_medium','budget_high','unrestricted','same_components']

def verify():
    results=pd.read_csv(s.OUT/'test_results.csv');datasets=b.load_data()
    manifest=json.loads((s.OUT/'evaluation.json').read_text())
    assert manifest['results_sha256']==b.digest_file(s.OUT/'test_results.csv')
    assert len(results)==90 and not results.duplicated(['dataset','view','family']).any()
    old=pd.read_csv(b.OUTPUT/'test_results.csv')
    trials=pd.read_csv(s.OUT/'validation_trials.csv')
    pairs=trials.pivot(index=['dataset','sharing','k'],columns='linked',values='train_mean_loglik')
    assert ((pairs[True]-pairs[False]).dropna()>=-1e-8).all()
    for name in datasets:
        original_trials=pd.read_csv(b.OUTPUT/f'{name}_validation_trials.csv')
        for linked in (False,True):
            target=original_trials.loc[(original_trials.k==1)&(original_trials.family==('Linked' if linked else 'Unlinked')),'validation_mean_loglik'].iloc[0]
            actual=trials.loc[(trials.dataset==name)&(trials.k==1)&(trials.linked==linked),'validation_mean_loglik']
            np.testing.assert_allclose(actual,target,atol=1e-7,rtol=0)

    for row in results.loc[(results.sharing=='shared')&(results.status=='selected')].itertuples():
        reference=old.loc[(old.dataset==row.dataset)&(old.view==row.view)&(old.family==row.family)].iloc[0]
        np.testing.assert_allclose(row.test_mean_loglik,reference.test_mean_loglik,atol=1e-8)
        assert row.candidate==reference.candidate and row.parameters==reference.parameters
    good=results.loc[results.test_mean_loglik.notna()]
    for (name,view),group in good.groupby(['dataset','view']):
        if view.startswith('budget'):assert group.parameters.max()/group.parameters.min()<=1.1+1e-12
        if view=='same_components':assert group.k.nunique()==1
    original_freeze=json.loads((b.OUTPUT/'frozen_selections.json').read_text())['freeze_sha256']
    checks=0
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        for name,data in datasets.items():
            daily=pd.read_csv(s.OUT/f'{name}_test_daily_loglik.csv',index_col=0)
            for row in good.loc[good.dataset==name].itertuples():
                np.testing.assert_allclose(daily[row.candidate].mean(),row.test_mean_loglik,atol=1e-8)
                np.testing.assert_allclose(row.aic,2*row.parameters-2*row.refit_total_loglik,rtol=1e-10)
                np.testing.assert_allclose(row.bic,np.log(row.refit_rows)*row.parameters-2*row.refit_total_loglik,rtol=1e-10)
                for length in (5,20,60):assert getattr(row,f'test_L{length}_low')<=getattr(row,f'test_L{length}_high')
                if row.family=='GMM':
                    model=joblib.load(b.OUTPUT/'refits'/name/original_freeze[:16]/(row.candidate+'.joblib'))['fit']
                    np.testing.assert_allclose([row.aic,row.bic],[model.aic(data.development),model.bic(data.development)],rtol=1e-10)
                checks+=1
            signature=s.hashlib.sha256((s.engine()+b.array_digest(data.development)).encode()).hexdigest()
            for ident in good.loc[(good.dataset==name)&(good.sharing!='shared'),'candidate'].unique():
                rec=joblib.load(s.OUT/'refits'/name/signature[:16]/(ident+'.joblib'));f=rec['fit']
                assert f['converged'] and not f['boundary'] and np.diff(f['history']).min()>=-1e-8
                np.testing.assert_allclose(np.linalg.norm(f['mu'],axis=1),1,atol=1e-10)
                if rec['sharing']=='variance_specific':np.testing.assert_allclose(f['kappa'],f['kappa'][0])
                if rec['sharing']=='kappa_specific':np.testing.assert_allclose(f['variance'],f['variance'][0])
                y,u=b.polar(data.test);logs=s.log_components(y,u,f)
                q=np.exp(logs-s.logsumexp(logs,axis=1,keepdims=True));np.testing.assert_allclose(q.sum(axis=1),1,atol=1e-10)
    outcome=dict(comparison_rows=90,scored_rows=checks,original_results_preserved=True,
        matched_budgets_checked=True,gmm_information_criteria_match_sklearn=True,
        posterior_and_em_checks_passed=True,results_sha256=manifest['results_sha256'])
    b.write_json(s.OUT/'verification.json',outcome)
    return outcome

def plots(results):
    figures=s.OUT/'figures';figures.mkdir(exist_ok=True)
    for view in VIEWS:
        fig,axes=plt.subplots(2,3,figsize=(18,10),layout='constrained')
        for row,name in enumerate(('equities','fx')):
            frame=results.loc[(results.dataset==name)&(results.view==view)].set_index('family').reindex(ORDER)
            y=np.arange(len(ORDER));valid=frame.test_mean_loglik.notna().to_numpy()
            for col,metric in enumerate(('test_mean_loglik','aic','bic')):
                ax=axes[row,col]
                values=frame[metric].to_numpy(dtype=float)
                if col==0:
                    lo=frame.test_L20_low.to_numpy(dtype=float);hi=frame.test_L20_high.to_numpy(dtype=float)
                    # Draw percentile endpoints directly; bootstrap intervals need not contain the estimate.
                    for i in np.flatnonzero(valid):
                        ax.plot([lo[i],hi[i]],[y[i],y[i]],color=COLORS[i],lw=2)
                        ax.scatter(values[i],y[i],color=COLORS[i],s=35,zorder=3)
                    ax.set_xlabel('Test mean log density (nats/observation) · higher better')
                else:
                    values=values-np.nanmin(values)
                    ax.barh(y[valid],values[valid],color=np.array(COLORS)[valid],alpha=.85)
                    ax.set_xlabel(f'Δ{metric.upper()} from best in this panel · lower better')
                ax.set_yticks(y,LABELS if col==0 else ['']*len(y),fontsize=9);ax.set_ylim(len(ORDER)-.5,-.5);ax.grid(axis='x',alpha=.2)
                ax.set_title(f'{name.upper()} · '+('test 95% block interval' if col==0 else metric.upper()+' on refit data'))
                for i in np.flatnonzero(~valid):ax.text(.02,y[i],'unavailable',transform=ax.get_yaxis_transform(),fontsize=8)
        fig.suptitle(view.replace('_',' ').title()+(' · PRIMARY' if view=='budget_medium' else '')+'\nIdentical data, density measure and evaluation protocol within each dataset',fontsize=15)
        fig.savefig(figures/f'{view}_likelihood_aic_bic.png',dpi=160);fig.savefig(figures/f'{view}_likelihood_aic_bic.pdf');plt.show()
    fig,axes=plt.subplots(2,5,figsize=(23,9),layout='constrained')
    for row,name in enumerate(('equities','fx')):
        for col,view in enumerate(VIEWS):
            ax=axes[row,col];f=results.loc[(results.dataset==name)&(results.view==view)].set_index('family').reindex(ORDER)
            for i in range(len(ORDER)):
                v=f.iloc[i]
                if pd.notna(v.get('gain_vs_gmm')):
                    ax.plot([v.gain_vs_gmm_L20_low,v.gain_vs_gmm_L20_high],[i,i],color=COLORS[i],lw=2)
                    ax.scatter(v.gain_vs_gmm,i,color=COLORS[i],s=22)
            ax.axvline(0,color='black',lw=.8);ax.set_yticks(range(len(ORDER)),LABELS if col==0 else ['']*len(ORDER),fontsize=8)
            ax.set_ylim(len(ORDER)-.5,-.5);ax.set_title(name.upper()+' · '+view.replace('budget_',''));ax.grid(axis='x',alpha=.2)
            ax.set_xlabel('Gain vs GMM (nats/obs.)')
    fig.suptitle('Paired test comparisons · 95% circular block-bootstrap intervals · 20 sessions',fontsize=15)
    fig.savefig(figures/'paired_gains_vs_gmm.png',dpi=160);fig.savefig(figures/'paired_gains_vs_gmm.pdf');plt.show()


    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    link_families=['Linked']+[s.family(mode,True) for mode in s.MODES]
    for ax,name in zip(axes,('equities','fx')):
        f=results.loc[(results.dataset==name)&(results.view=='same_components')].set_index('family').reindex(link_families)
        for i in range(4):
            v=f.iloc[i]
            if pd.notna(v.get('gain_vs_own_unlinked')):
                ax.plot([v.gain_vs_own_unlinked_L20_low,v.gain_vs_own_unlinked_L20_high],[i,i],color='#287c91',lw=2)
                ax.scatter(v.gain_vs_own_unlinked,i,color='#287c91',s=40)
        ax.axvline(0,color='black',lw=.8)
        ax.set_yticks(range(4),['Both shared','Separate variance','Separate concentration','Both separate'])
        ax.set_ylim(3.5,-.5);ax.set_title(name.upper()+' · fixed component count')
        ax.set_xlabel('Linked minus matching unlinked (nats/obs.)');ax.grid(axis='x',alpha=.2)
    fig.suptitle('Does the link help under each sharing choice? · paired 95% intervals')
    fig.savefig(figures/'same_k_link_gain_by_sharing.png',dpi=160)
    fig.savefig(figures/'same_k_link_gain_by_sharing.pdf');plt.show()


def conclusions(results):
    paragraphs=[]
    for name in ('equities','fx'):
        primary=results.loc[(results.dataset==name)&(results.view=='budget_medium')&results.test_mean_loglik.notna()]
        winner=primary.loc[primary.test_mean_loglik.idxmax()]
        aic=primary.loc[primary.aic.idxmin()];bic=primary.loc[primary.bic.idxmin()]
        text=(f'**{name.upper()}.** The highest primary-budget test score is {winner.family} '
              f'(K={int(winner.k)}, {int(winner.parameters)} parameters): {winner.test_mean_loglik:.4f} nats/observation. '
              f'Its paired gain over GMM is {winner.gain_vs_gmm:+.4f}, with 95% interval '
              f'[{winner.gain_vs_gmm_L20_low:+.4f}, {winner.gain_vs_gmm_L20_high:+.4f}]. '
              f'On refit data, AIC favors {aic.family}; BIC favors {bic.family}. '
              'These criteria describe development fit and complexity; test likelihood evaluates the frozen models on later observations.')
        nested=results.loc[(results.dataset==name)&(results.view=='same_components')&(results.family==s.family('both_specific',True))]
        if len(nested) and pd.notna(nested.iloc[0].get('gain_vs_own_unlinked')):
            v=nested.iloc[0]
            text+=(f' With both spreads component-specific at K={int(v.k)}, the linked-minus-unlinked test gain is '
                   f'{v.gain_vs_own_unlinked:+.4f} nats/observation, paired 95% interval '
                   f'[{v.gain_vs_own_unlinked_L20_low:+.4f}, {v.gain_vs_own_unlinked_L20_high:+.4f}].')
        paragraphs.append(text)
    paragraphs.append('This is an exploratory follow-up using previously inspected test dates. Intervals condition on the fitted models, use 1,500 paired circular block replicates, and do not include refitting or selection uncertainty. The medium budget remains primary; other views are sensitivity comparisons. Confidence intervals are pointwise, not adjusted for all comparisons. AIC/BIC use nominal parameter counts and ordinary observation-count penalties; mixture singularities and serial dependence limit their classical interpretation.')
    text='\n\n'.join(paragraphs);(s.OUT/'conclusions.md').write_text(text+'\n');return text
