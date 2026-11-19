"""Read frozen pilot artifacts; produce tables and scientific figures."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kent_experiment as experiment

COLORS={'GMM':'#586579','vMF unlinked':'#c78b32','vMF linked':'#2477a4',
        'Kent unlinked':'#9b5b9c','Kent linked':'#187e65'}


def load(folder):
    folder=Path(folder)
    return dict(folder=folder,protocol=json.loads((folder/'protocol.json').read_text()),
        gate=json.loads((folder/'numerical_gate.json').read_text()),
        search=pd.read_csv(folder/'search_trials.csv'),selection=pd.read_csv(folder/'selections.csv'),
        results=pd.read_csv(folder/'test_results.csv'),contrasts=pd.read_csv(folder/'contrasts.csv'),
        starts=pd.read_csv(folder/'refit_starts.csv'),decomposition=pd.read_csv(folder/'score_decomposition.csv'))


def save(fig,report,name):
    path=report['folder']/'figures';path.mkdir(exist_ok=True)
    fig.savefig(path/f'{name}.png',dpi=160,bbox_inches='tight')
    fig.savefig(path/f'{name}.pdf',bbox_inches='tight')
    return fig


def numerical_diagnostics(report):
    kn=experiment.kn;rows=[]
    for k in (0.,.01,1.,10.,100.,1000.,10000.):
        result=kn.normalizer(k,np.zeros(8),force_numeric=True)
        reference=kn.normalizer(k,np.zeros(8))
        rows.append(dict(case='vMF limit',kappa=k,absolute_logz_error=abs(result.logz-reference.logz),
            max_gradient_error=max(abs(result.dkappa-reference.dkappa),np.max(abs(result.dlambda-reference.dlambda))),
            nodes=result.nodes,refinement_error=result.error))
    for k,lam in [(10.,np.linspace(-3,3,8)),(100.,np.linspace(-49,49,8))]:
        result=kn.normalizer(k,lam);reference,_=kn.reference_normalizer(k,lam)
        rows.append(dict(case='anisotropic contour reference',kappa=k,
            absolute_logz_error=abs(result.logz-reference),max_gradient_error=np.nan,
            nodes=result.nodes,refinement_error=result.error))
    frame=pd.DataFrame(rows);frame.to_csv(report['folder']/'normalization_diagnostics.csv',index=False)
    return frame


def same_component_table(report):
    sel=report['selection'];sel=sel[sel.view.str.startswith('same_K')]
    result=sel[['view','id']].merge(report['results'],on='id',how='left')
    columns=['k','family','covariance','parameters','status','test_mean_loglik',
             'test_lower_L20','test_upper_L20','aic','bic','accepted_starts','fit_seconds']
    complete=pd.MultiIndex.from_product([experiment.km.GRID,experiment.FAMILIES],names=['k','family'])
    result=result.set_index(['k','family']).reindex(complete).reset_index()
    result['status']=result.status.fillna('no admissible validation fit')
    for i,row in result[result.parameters.isna()&result.family.ne('GMM')].iterrows():
        result.loc[i,'parameters']=experiment.km.parameter_count(row.family,int(row.k))
    return result[columns].reset_index(drop=True)


def search_starts(report):
    rows=[]
    for path in (report['folder']/'search').glob('*.joblib'):
        if path.name.startswith('initializations'):continue
        rec=joblib.load(path);spec=rec['pilot_spec']
        for row in rec['diagnostics']:
            numeric=row.get('normalizer',{})
            inner=row.get('inner',[])
            rows.append(dict(id=spec['id'],family=spec['family'],k=spec['k'],
                start=row['start'],origin=row.get('origin',''),accepted=row['accepted'],
                converged=row.get('converged',row['status'] in ('converged','boundary')),
                status=row['status'],seconds=row.get('elapsed_seconds'),iterations=row['iterations'],
                boundary_reasons='; '.join(row.get('boundary_reasons',[])),
                max_ovalness=row.get('max_ovalness'),kappa=row.get('kappa'),
                normalizer_calls=numeric.get('calls',0),normalizer_seconds=numeric.get('seconds',0),
                normalizer_failures=numeric.get('failures',0),
                quadrature_nodes=numeric.get('nodes',0),
                shape_iterations=sum(v['shape']['nit'] for v in inner),
                shape_evaluations=sum(v['shape']['nfev'] for v in inner),
                frame_iterations=sum(f['nit'] for v in inner for f in v['frames']),
                frame_evaluations=sum(f['nfev'] for v in inner for f in v['frames'])))
    result=pd.DataFrame(rows);result.to_csv(report['folder']/'search_starts.csv',index=False)
    return result


def normalization_run_summary(report):
    rows=[]
    for path in sorted((report['folder']/'search').glob('kent_*.joblib')):
        rec=joblib.load(path)
        rows.append(dict(id=path.stem,**rec['normalizer_stats']))
    result=pd.DataFrame(rows)
    result.to_csv(report['folder']/'normalization_run_summary.csv',index=False)
    return result


def budget_table(report):
    sel=report['selection'];sel=sel[sel.view.str.startswith(('budget10','exact'))]
    return sel[['view','id']].merge(report['results'],on='id',how='left')[[
        'view','family','k','parameters','status','test_mean_loglik','aic','bic']]


def comparison_feasibility(report):
    """Structural budget matches, including candidates excluded before selection."""
    rows=[];trials=report['search']
    pairs=[('Kent linked','vMF linked'),('Kent unlinked','vMF unlinked'),
           ('Kent linked','GMM'),('Kent unlinked','GMM')]
    for left,right in pairs:
        a=trials[trials.family==left].drop_duplicates(['k','covariance','parameters'])
        c=trials[trials.family==right].drop_duplicates(['k','covariance','parameters'])
        found=False
        for x in a.itertuples():
            for y in c.itertuples():
                ratio=max(x.parameters,y.parameters)/min(x.parameters,y.parameters)
                if ratio<=1.1:
                    found=True
                    left_ok=(trials[(trials.family==left)&(trials.k==x.k)&(trials.parameters==x.parameters)].status=='ok').any()
                    right_ok=(trials[(trials.family==right)&(trials.k==y.k)&(trials.parameters==y.parameters)].status=='ok').any()
                    rows.append(dict(comparison=f'{left} versus {right}',left_k=x.k,right_k=y.k,
                        left_parameters=x.parameters,right_parameters=y.parameters,ratio=ratio,
                        status='admissible validation candidates' if left_ok and right_ok else 'structural match; inadmissible fit'))
        if not found:rows.append(dict(comparison=f'{left} versus {right}',status='no parameter match within 10% in declared grid'))
    result=pd.DataFrame(rows);result.to_csv(report['folder']/'budget_feasibility.csv',index=False)
    return result


def validation_figure(report):
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    data=report['search'];data=data[data.status=='ok']
    selected=[]
    for (family,k),group in data.groupby(['family','k']):
        selected.append(experiment.b.best_validation(group))
    data=pd.DataFrame(selected)
    for family in experiment.FAMILIES:
        group=data[data.family==family].sort_values('k')
        if group.empty:continue
        for ax,x in zip(axes,['k','parameters']):
            ax.plot(group[x],group.validation_mean_loglik,'o-',label=family,color=COLORS[family])
    axes[0].set(xlabel='Mixture components K',xticks=experiment.km.GRID,ylabel='Validation mean log density (nats/date)',
                title='Declared component grid; GMM specification selected within K')
    axes[1].set(xlabel='Nominal free parameters',ylabel='Validation mean log density (nats/date)',title='The same component count is not the same model size')
    axes[1].legend(fontsize=8)
    return save(fig,report,'01_validation_complexity')


def same_k_figure(report):
    contrasts=report['contrasts'];contrasts=contrasts[(contrasts.block==20)&contrasts.view.str.startswith('same_K')]
    comparisons=['Kent linked minus vMF linked','Kent unlinked minus vMF unlinked',
                 'Kent linked minus Kent unlinked','vMF linked minus vMF unlinked']
    fig,axes=plt.subplots(2,2,figsize=(11.5,7),layout='constrained')
    for ax,label in zip(axes.flat,comparisons):
        group=contrasts[contrasts.comparison==label].copy()
        group['k']=group.view.str.replace('same_K_','').astype(int)
        for r in group.itertuples():
            ax.plot([r.lower,r.upper],[r.k,r.k],color='#187e65',lw=2)
            ax.scatter(r.gain,r.k,color='#187e65')
        ax.axvline(0,color='gray',lw=1);ax.set(title=label,yticks=experiment.km.GRID,
            ylim=(.3,10.7),ylabel='Components in both models',xlabel='Test log-score gain (nats/date), 95% interval')
        absent=set(experiment.km.GRID)-set(group.k)
        for k in absent:ax.text(.02,k,'Excluded before test scoring',transform=ax.get_yaxis_transform(),fontsize=8)
    fig.suptitle('Same-component planned contrasts · paired 20-session blocks')
    return save(fig,report,'02_same_components_intervals')


def budget_figure(report):
    data=report['contrasts'];data=data[(data.block==20)&data.view.str.startswith(('exact','budget10'))]
    data=data.drop_duplicates(['candidate','reference'])
    labels=[f"{r.candidate.replace('_',' ')} − {r.reference.replace('_',' ')}" for r in data.itertuples()]
    fig,ax=plt.subplots(figsize=(11,max(3.2,len(data)*.33+1)),layout='constrained')
    for i,r in enumerate(data.itertuples()):
        ax.plot([r.lower,r.upper],[i,i],color='#2477a4',lw=2);ax.scatter(r.gain,i,color='#2477a4')
    ax.axvline(0,color='gray',lw=1);ax.set(yticks=np.arange(len(data)),yticklabels=labels,
        xlabel='Test log-score gain (nats/date), paired 95% interval',title='Feasible parameter-matched comparisons · exact or within 10%')
    ax.invert_yaxis();ax.tick_params(axis='y',labelsize=8)
    return save(fig,report,'03_parameter_matched_intervals')


def baseline_figure(report):
    data=report['contrasts'];data=data[(data.block==20)&data.view.str.startswith('same_K')&data.comparison.str.endswith('minus GMM')]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,link in zip(axes,['linked','unlinked']):
        for offset,family in [(-.12,f'vMF {link}'),(.12,f'Kent {link}')]:
            group=data[data.comparison==family+' minus GMM']
            for r in group.itertuples():
                k=int(r.view.split('_')[-1]);y=list(experiment.km.GRID).index(k)+offset
                ax.plot([r.lower,r.upper],[y,y],color=COLORS[family],lw=2)
                ax.scatter(r.gain,y,color=COLORS[family])
            ax.plot([],[],color=COLORS[family],label=family+(' (unavailable)' if group.empty else ''))
        ax.axvline(0,color='gray',lw=1);ax.set(yticks=range(4),yticklabels=experiment.km.GRID,
            ylabel='Same component count K',xlabel='Gain over GMM (nats/date), 95% interval',title=f'{link.title()} direction–magnitude models versus GMM')
        ax.legend(fontsize=8)
    return save(fig,report,'04_gmm_comparisons')


def runtime_figure(report):
    sel=report['selection'];sel=sel[sel.view.str.startswith('same_K')]
    starts=search_starts(report)
    # Include rejected Kent fits; GMM uses its selected specification at each K.
    selected_gmm=set(sel[sel.family=='GMM'].id)
    starts=starts[(starts.family!='GMM')|starts.id.isin(selected_gmm)]
    fig,axes=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
    for family in experiment.FAMILIES:
        group=starts[starts.family==family]
        if group.empty:continue
        med=group.groupby('k')[['seconds','iterations','accepted']].mean()
        median=group.groupby('k')[['seconds','iterations']].median()
        axes[0].plot(median.index,median.seconds,'o-',label=family,color=COLORS[family])
        axes[1].plot(median.index,median.iterations,'o-',color=COLORS[family])
        axes[2].plot(med.index,med.accepted,'o-',color=COLORS[family])
    axes[0].set(ylabel='Median seconds per start (log scale)',yscale='log',title='Fresh training fits, including rejected starts')
    axes[1].set(ylabel='Median iterations per start',title='Iterations have different costs')
    axes[2].set(ylabel='Accepted fraction of ten starts',ylim=(-.03,1.03),title='Convergence and boundary acceptance')
    for ax in axes:ax.set(xlabel='Mixture components K',xticks=experiment.km.GRID)
    axes[0].legend(fontsize=8)
    return save(fig,report,'05_runtime_acceptance')


def convergence_figure(report):
    fig,axes=plt.subplots(1,4,figsize=(15,4),layout='constrained')
    selected=report['selection'];selected=selected[selected.view.str.startswith('same_K')]
    for ax,k in zip(axes,experiment.km.GRID):
        ids=list(selected[selected.k==k].id)
        ids.extend(f'kent_{link}_k{k:02d}' for link in ('unlinked','linked') if f'kent_{link}_k{k:02d}' not in ids)
        for ident in ids:
            rec=joblib.load(report['folder']/'search'/f'{ident}.joblib');family=rec['pilot_spec']['family']
            chosen_start=rec.get('selected_start',rec.get('best_rejected_start'))
            if chosen_start is None:continue
            diagnostic=rec['diagnostics'][chosen_start]
            history=np.asarray(diagnostic.get('history',[]));times=np.asarray(diagnostic.get('times',[]))
            if len(history)!=len(times) or not len(history):continue
            # Within-fit gaps remove the constant radial Jacobian.
            gap=np.maximum(history[-1]-history,1e-10)
            rejected=not diagnostic['accepted']
            ax.plot(times,gap,color=COLORS[family],linestyle='--' if rejected else '-',
                    label=family+(' (rejected)' if rejected else ''))
        ax.set(title=f'K={k}',xlabel='Seconds within selected start',yscale='log',
               ylabel='Gap to final mean training log likelihood')
    axes[-1].legend(fontsize=7)
    return save(fig,report,'06_convergence')


def anisotropy_figure(report):
    fig,axes=plt.subplots(2,4,figsize=(15,7),layout='constrained');rows=[];im=None
    for ri,family in enumerate(['Kent unlinked','Kent linked']):
        for ci,k in enumerate(experiment.km.GRID):
            ax=axes[ri,ci];ident=family.lower().replace(' ','_')+f'_k{k:02d}'
            rec=joblib.load(report['folder']/'search'/f'{ident}.joblib')
            accepted=rec['fit'] is not None;fit=rec['fit'] if accepted else rec.get('best_rejected_fit')
            if fit is None:ax.text(.5,.5,'Numerical failure',ha='center');ax.set_axis_off();continue
            occupancy=fit['weights'];order=np.argsort(-occupancy)
            eigen=np.sort(fit['lambda'][order],axis=1);kappa=fit['kappa']
            ratio=2*eigen/kappa if kappa>0 else np.zeros_like(eigen)
            precision=kappa-2*eigen
            im=ax.imshow(ratio,aspect='auto',cmap='RdBu_r',vmin=-7,vmax=1)
            ax.set_title(f"{family}, K={k}\n{'Accepted' if accepted else 'Rejected; training diagnostic only'}",fontsize=9)
            ax.set(xlabel='Ordered tangent eigenvalues',yticks=range(len(order)),
                   yticklabels=[f'{j+1} · {occupancy[j]:.0%}' for j in order])
            ax.tick_params(axis='y',labelsize=7)
            spread=precision.max(1)/precision.min(1) if kappa>0 else np.ones(len(order))
            for j in range(len(order)):
                rows.append(dict(id=ident,accepted=accepted,component=int(order[j]),training_mass=float(occupancy[order[j]]),
                    kappa=kappa,max_ovalness=float(ratio[j].max()),local_variance_ratio=float(spread[j])))
    if im is not None:fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.6,label='2λ/κ; upper constraint ≈ 1')
    fig.suptitle('Full tangent anisotropy · rows labeled by training mixture mass\nRejected fits are not evaluated on the test sample',fontsize=12)
    pd.DataFrame(rows).to_csv(report['folder']/'anisotropy_summary.csv',index=False)
    return save(fig,report,'07_anisotropy')


def verify(report):
    results=report['results'];daily=pd.read_csv(report['folder']/'test_daily_loglik.csv',index_col=0)
    data=experiment.b.load_data()['fx'];checks=0
    original=data.frame.to_numpy()
    magnitude,direction=experiment.b.polar(original)
    np.testing.assert_allclose(np.exp(magnitude)[:,None]*direction,original,atol=1e-14,rtol=1e-12)
    assert experiment.b.array_digest(original)==report['protocol']['data_hash']
    assert list(daily.index)==list(data.frame.index[data.b:].strftime('%Y-%m-%d'))
    for row in results[results.status=='ok'].itertuples():
        rec=joblib.load(report['folder']/'refit'/f'{row.id}.joblib')
        test=experiment.score_record(data.test,rec);dev=experiment.score_record(data.development,rec)
        np.testing.assert_allclose(test,daily[row.id],atol=1e-10,rtol=0)
        np.testing.assert_allclose([test.mean(),2*row.parameters-2*dev.sum(),row.parameters*np.log(len(dev))-2*dev.sum()],
                                  [row.test_mean_loglik,row.aic,row.bic],atol=1e-8,rtol=0)
        if row.family.startswith('Kent'):experiment.km.constraints(rec['fit'])
        checks+=1
    assert set(report['search'].k)==set(experiment.km.GRID)
    assert len(report['search'])==64
    status=dict(verified_models=checks,search_configurations=64,ten_starts_per_configuration=True,
                score_and_information_criteria_verified=True,original_data_hash_verified=True,
                polar_reconstruction_verified=True,test_dates_verified=True)
    for stage in ('search','refit'):
        for p in (report['folder']/stage).glob('*.joblib'):
            if p.name.startswith('initializations'):continue
            rec=joblib.load(p);assert len(rec['diagnostics'])==10
            assert rec['input_hash']==experiment.b.array_digest(data.train if stage=='search' else data.development)
            if rec['pilot_spec']['family'].startswith('Kent'):
                for diagnostic in rec['diagnostics']:
                    if 'history' in diagnostic:
                        assert np.min(np.diff(diagnostic['history']))>=-1e-8
                    if diagnostic['accepted']:
                        assert diagnostic['converged'] and not diagnostic['boundary_reasons']
    experiment.save_json(report['folder']/'verification.json',status)
    return status


def conclusions(report):
    rows=[];contrasts=report['contrasts']
    kent=report['search'][report['search'].family.str.startswith('Kent')]
    if not (kent.status=='ok').any():
        rows.append('**No admissible generalized Kent fit survived the declared validation-stage rules.** '
                    'Consequently this pilot does not establish whether Kent improves held-out fit. '
                    'Boundary-rejected training fits are shown only as diagnostics; they are not scored on test dates.')
    starts=search_starts(report);ks=starts[starts.family.str.startswith('Kent')]
    rows.append(f"\nAcross {len(ks)} Kent starts, {int(ks.converged.sum())} met the convergence criterion, "
                f"{int(ks.boundary_reasons.str.contains('unimodality_margin',na=False).sum())} reached the shape margin, "
                f"and {int(ks.accepted.sum())} were admissible. "
                f"The recorded number of normalization failures was {int(ks.normalizer_failures.sum())}.")
    if not (kent.status=='ok').any():
        rows.append('\n**Why no Kent test scores?** The frozen protocol excluded fits with an active shape constraint. '
                    'This is an acceptance-policy decision, not evidence that those normalized densities are invalid. '
                    'It prevented the intended directional-family comparison. A revised analysis should retain '
                    'converged, numerically valid fits at the existing shape margin, flag that constraint, and still '
                    'exclude nonconvergence and genuine numerical failures. That revised analysis has not been run here.')
    for comparison in ['Kent linked minus vMF linked','Kent unlinked minus vMF unlinked']:
        group=contrasts[(contrasts.block==20)&contrasts.view.str.startswith('same_K')&(contrasts.comparison==comparison)]
        for r in group.itertuples():
            direction='positive' if r.lower>0 else 'negative' if r.upper<0 else 'includes zero'
            rows.append(f"- **{comparison}, {r.view.replace('same_K_','K=')}:** {r.gain:+.4f} nats/date "
                        f"(95% interval [{r.lower:+.4f}, {r.upper:+.4f}]; {direction}).")
    best=report['selection'];best=best[best.view=='best_in_grid'].merge(report['results'][['id','status','test_mean_loglik','aic','bic']],on='id')
    rows.append('\n**Validation-selected models within the declared grid:**')
    for r in best.itertuples():
        rows.append(f'- {r.family}: K={r.k}, {r.parameters} parameters; '+
                    (f'test log score {r.test_mean_loglik:.4f}.' if r.status=='ok' else 'refit unavailable.'))
    boundary=best[best.k==max(experiment.km.GRID)]
    if len(boundary):rows.append('\nA validation winner at K=10 is at the upper grid boundary, not evidence that 10 is optimal.')
    failures=report['search'][report['search'].status!='ok']
    rows.append(f'\n{len(failures)} of 64 search configurations were unavailable. Rejected starts and selected-refit failures remain in the ledgers.')
    rows.append('\nThe unlinked Kent–vMF parameter match is unavailable within 10%. The linked match uses Kent K=1 and vMF K=3 (55 versus 58 parameters), subject to admissible fits. These tests concern retrospective density fit, not recovery of true regimes or trading profitability.')
    text='\n'.join(rows);(report['folder']/'conclusions.md').write_text(text+'\n');return text
