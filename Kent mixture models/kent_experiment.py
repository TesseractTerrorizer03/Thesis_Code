"""Isolated, resumable FX pilot. Test scoring follows a frozen manifest only."""
from pathlib import Path
from dataclasses import asdict
from datetime import datetime,timezone
import argparse,hashlib,json,sys,time,subprocess,platform,copy
import joblib
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
import kent_models as km
import kent_numerics as kn

b=km.baseline
ROOT=Path(__file__).resolve().parent
FAMILIES=('GMM','vMF unlinked','vMF linked','Kent unlinked','Kent linked')


def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x))


def save_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,default=plain,indent=2,allow_nan=False)+'\n');temp.replace(path)


def fingerprint():
    digest=hashlib.sha256()
    for name in ('kent_numerics.py','kent_models.py','kent_experiment.py','test_kent.py'):
        digest.update((ROOT/name).read_bytes())
    digest.update(b.engine_fingerprint().encode())
    return digest.hexdigest()


def protocol(data):
    return dict(engine=fingerprint(),data_hash=b.array_digest(data.frame.to_numpy()),
        original_engine=b.engine_fingerprint(),grid=list(km.GRID),starts=b.STARTS,
        max_iter=b.MAX_ITER,tol=b.TOL,seed=b.SEED,variance_floor=b.VAR_FLOOR,
        kappa_cap=b.KAPPA_MAX,min_effective=b.MIN_EFFECTIVE,unimodality_margin=kn.MARGIN,
        logz_tolerance=kn.LOGZ_TOL,gradient_tolerance=kn.GRAD_TOL,max_quadrature_nodes=kn.MAX_NODES,
        shape_ftol=km.SHAPE_FTOL,orientation_gtol=km.ORIENTATION_GTOL,inner_maxiter=km.INNER_MAXITER,
        sharing='shared concentration and radial variance; component-specific full tangent shape',
        dimension=9,normalizer='Fourier inversion with continuous Euler damping; analytic zero-anisotropy limit',
        splits=b.split_table({'fx':data}).to_dict('records'),
        regularizations=b.REGULARIZATIONS,covariances=b.COVARIANCES,
        bootstrap_repeats=1500,bootstrap_blocks=[5,20,60],primary_block=20,
        date_end=b.END,test_previously_inspected=True,threads=1,
        python=sys.version,platform=platform.platform(),numpy=np.__version__,
        scipy=b.scipy.__version__,sklearn=b.sklearn.__version__)


def run_directory(data):
    p=protocol(data);ident=hashlib.sha256(json.dumps(p,sort_keys=True,default=plain).encode()).hexdigest()[:16]
    path=ROOT/'results'/ident;path.mkdir(parents=True,exist_ok=True)
    save_json(path/'protocol.json',p)
    return path


def run_gate(folder):
    target=folder/'numerical_gate.json'
    if target.exists():return json.loads(target.read_text())['passed']
    started=time.perf_counter()
    proc=subprocess.run([sys.executable,'-m','unittest','test_kent','-v'],cwd=ROOT,
                        capture_output=True,text=True)
    (folder/'numerical_gate.txt').write_text(proc.stdout+proc.stderr)
    passed=proc.returncode==0
    save_json(target,dict(passed=passed,exit_code=proc.returncode,elapsed_seconds=time.perf_counter()-started))
    print('Numerical gate:', 'PASSED' if passed else 'FAILED',flush=True)
    return passed


def specs():
    rows=[]
    for k in km.GRID:
        for cov in b.COVARIANCES:
            for reg in b.REGULARIZATIONS:
                rows.append(dict(id=b.Spec('GMM',k,cov,reg).key,family='GMM',k=k,covariance=cov,regularization=reg))
        for family in FAMILIES[1:]:
            rows.append(dict(id=family.lower().replace(' ','_')+f'_k{k:02d}',family=family,k=k,covariance='scalar',regularization=0.))
    for i,row in enumerate(rows):
        row.update(parameters=km.parameter_count(row['family'],row['k'],9,row['covariance']),configuration_order=i)
    return rows


def fit_baseline(x,spec,initial,warm=None):
    """Call unchanged estimator with non-invasive per-iteration timing hooks."""
    original_gaussian=b.GaussianMixture;original_fit=b.fit_cylinder
    original_r=b.cylinder_responsibilities
    observations=[]
    class TimedGaussian(original_gaussian):
        def fit(self,X,y=None):
            self.pilot_times_=[];self.pilot_history_=[];self._pilot_start=time.perf_counter()
            try:return super().fit(X,y)
            finally:observations.append(dict(elapsed_seconds=time.perf_counter()-self._pilot_start,
                          history=np.asarray(self.pilot_history_),times=np.asarray(self.pilot_times_)))
        def _m_step(self,X,log_resp,**kwargs):
            result=super()._m_step(X,log_resp,**kwargs)
            self.pilot_history_.append(float(self.score(X)))
            self.pilot_times_.append(time.perf_counter()-self._pilot_start)
            return result
    def timed_cylinder(*args,**kwargs):
        started=time.perf_counter();times=[]
        def timed_responsibility(*a,**kw):
            times.append(time.perf_counter()-started);return original_r(*a,**kw)
        b.cylinder_responsibilities=timed_responsibility
        result=None
        try:
            result=original_fit(*args,**kwargs)
            return result
        finally:
            b.cylinder_responsibilities=original_r
            elapsed=time.perf_counter()-started
            observations.append(dict(elapsed_seconds=elapsed,
                history=np.array([]) if result is None else result['history'],times=np.r_[times,elapsed]))
    try:
        b.GaussianMixture=TimedGaussian;b.fit_cylinder=timed_cylinder
        family='GMM' if spec['family']=='GMM' else ('Linked' if spec['family']=='vMF linked' else 'Unlinked')
        record=b.fit_candidate(x,b.Spec(family,spec['k'],spec['covariance'] if family=='GMM' else '',spec['regularization']),initial,warm)
    finally:b.GaussianMixture=original_gaussian;b.fit_cylinder=original_fit;b.cylinder_responsibilities=original_r
    for row,observed in zip(record['diagnostics'],observations):
        row.update(observed)
        row['origin']='unlinked vMF' if spec['family']=='vMF linked' and row['start']==9 and warm is not None else 'matched initialization'
    if record['fit'] is not None and spec['family']=='GMM':
        # Serialize the public sklearn class, with identical fitted state.
        fitted=record['fit'];clean=original_gaussian(**fitted.get_params());clean.__dict__.update(fitted.__dict__)
        record['fit']=clean
    return record


def score_record(x,record):
    return km.score(x,record['fit']) if record['pilot_spec']['family'].startswith('Kent') else b.score_model(x,record)


def fit_stage(folder,x,stage,requested=None,validation=None):
    cache=folder/stage;cache.mkdir(exist_ok=True)
    records={};rows=[]
    for k in km.GRID:
        candidates=[s for s in specs() if s['k']==k]
        needed=set(s['id'] for s in candidates) if requested is None else set(requested)&set(s['id'] for s in candidates)
        # Preserve warm-start conventions even when dependencies were not selected.
        if any(s['family'].startswith('Kent') and s['id'] in needed for s in candidates):
            needed.update(s['id'] for s in candidates if s['family']!='GMM')
        if f'vmf_linked_k{k:02d}' in needed:needed.add(f'vmf_unlinked_k{k:02d}')
        if not needed:continue
        initial_path=cache/f'initializations_k{k:02d}.joblib'
        if initial_path.exists():initial=joblib.load(initial_path)
        else:
            initial=b.make_initializations(x,k);joblib.dump(initial,initial_path)
        for spec in candidates:
            ident=spec['id']
            if ident not in needed:continue
            path=cache/f'{ident}.joblib';cached=path.exists()
            if cached:record=joblib.load(path)
            else:
                family=spec['family'];warm_cost=0.
                if family.startswith('Kent'):
                    linked=family=='Kent linked'
                    vmf=records[f"vmf_{'linked' if linked else 'unlinked'}_k{k:02d}"]
                    unlinked=records.get(f'kent_unlinked_k{k:02d}')
                    record=km.candidate(x,linked,initial[1],vmf=vmf['fit'],
                        unlinked=None if unlinked is None else unlinked['fit'])
                    warm_cost=vmf['elapsed_seconds']+(0 if unlinked is None else unlinked['elapsed_seconds'])
                else:
                    warm=records.get(f'vmf_unlinked_k{k:02d}') if family=='vMF linked' else None
                    record=fit_baseline(x,spec,initial,None if warm is None else warm['fit'])
                    warm_cost=0 if warm is None else warm['elapsed_seconds']
                record.update(pilot_spec=spec,stage=stage,input_hash=b.array_digest(x),
                              warm_preparation_seconds=warm_cost)
                # Verify nesting against feasible warm solutions; don't discard failures.
                if record['fit'] is not None and family.startswith('Kent'):
                    comparators=[vmf]
                    if unlinked is not None:comparators.append(unlinked)
                    record['nesting_gap']=min((record['train_mean_loglik']-c['train_mean_loglik']
                        for c in comparators if c['fit'] is not None),default=0.)
                    if record['nesting_gap'] < -1e-7:
                        record['selection_failure']='Best accepted fit falls below a nested warm solution'
                        record['rejected_fit']=record['fit'];record['fit']=None
                joblib.dump(record,path,compress=3)
            assert record['input_hash']==b.array_digest(x)
            records[ident]=record
            row=dict(spec,accepted_starts=sum(v['accepted'] for v in record['diagnostics']),
                     elapsed_seconds=record['elapsed_seconds'],
                     warm_preparation_seconds=record.get('warm_preparation_seconds',0),
                     status='ok' if record['fit'] is not None else 'unavailable',
                     train_mean_loglik=record.get('train_mean_loglik'),validation_mean_loglik=None)
            if validation is not None and record['fit'] is not None:
                row['validation_mean_loglik']=float(score_record(validation,record).mean())
            rows.append(row)
            pd.DataFrame(rows).to_csv(folder/f'{stage}_trials.csv',index=False)
            print(f"{stage}: {ident} {row['status']} {row['accepted_starts']}/10 "
                  f"validation={row['validation_mean_loglik']} {'cached' if cached else ''}",flush=True)
    return pd.DataFrame(rows),records


def select(trials):
    """Only validation scores and declared counts enter selection."""
    good=trials[trials.status=='ok'].copy();chosen=[]
    def winner(group):return b.best_validation(group)
    def add(view,group,lo=None,hi=None):
        for family in FAMILIES:
            candidates=group[group.family==family]
            if len(candidates):
                row=winner(candidates)
                chosen.append(dict(view=view,family=family,id=row.id,k=int(row.k),
                    parameters=int(row.parameters),budget_lower=lo,budget_upper=hi))
    for k in km.GRID:add(f'same_K_{k}',good[good.k==k])
    add('best_in_grid',good)
    seen=set()
    for p in sorted(trials.parameters.unique()):
        for exact in (True,False):
            hi=p if exact else int(np.floor(1.1*p))
            group=good[good.parameters.between(p,hi)]
            if group.family.nunique()<2:continue
            wins=tuple(winner(group[group.family==f]).id for f in FAMILIES if f in set(group.family))
            key=(exact,wins)
            if key in seen:continue
            seen.add(key)
            add(f"{'exact' if exact else 'budget10'}_{p}_{hi}",group,int(p),int(hi))
    return pd.DataFrame(chosen)


def freeze(folder,trials):
    selection=select(trials)
    payload=dict(engine=fingerprint(),test_used_for_selection=False,
                 selections=selection.where(pd.notnull(selection),None).to_dict('records'))
    # Pandas keeps NaN in float columns; serialize through its JSON conversion.
    payload['selections']=json.loads(selection.to_json(orient='records'))
    payload['sha256']=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    path=folder/'frozen_selection.json'
    if path.exists():assert json.loads(path.read_text())==payload,'Frozen selection changed'
    else:save_json(path,payload)
    selection.to_csv(folder/'selections.csv',index=False)
    return selection


def evaluation(folder,data,selection,records):
    freeze_info=json.loads((folder/'frozen_selection.json').read_text())
    digest=freeze_info.pop('sha256')
    assert hashlib.sha256(json.dumps(freeze_info,sort_keys=True).encode()).hexdigest()==digest
    chosen=list(dict.fromkeys(selection.id));available=[i for i in chosen if records[i]['fit'] is not None]
    daily=pd.DataFrame({ident:score_record(data.test,records[ident]) for ident in available},index=data.frame.index[data.b:])
    daily.index.name='date';daily.to_csv(folder/'test_daily_loglik.csv',float_format='%.14g')
    assert np.isfinite(daily.to_numpy()).all()
    boot={length:b.bootstrap_means(daily.to_numpy(),length) for length in (5,20,60)}
    rows=[];starts=[];decomposition=[]
    for ident in chosen:
        record=records[ident];spec=record['pilot_spec'];fit=record['fit']
        for start in record['diagnostics']:
            numeric=start.get('normalizer',{})
            starts.append(dict(id=ident,family=spec['family'],k=spec['k'],start=start['start'],
                origin=start.get('origin',''),accepted=start['accepted'],status=start['status'],
                seconds=start.get('elapsed_seconds'),iterations=start['iterations'],
                boundary_reasons='; '.join(start.get('boundary_reasons',[])),
                max_ovalness=start.get('max_ovalness'),kappa=start.get('kappa'),
                normalizer_calls=numeric.get('calls',0),normalizer_seconds=numeric.get('seconds',0)))
        row=dict(spec,status='ok' if fit is not None else 'unavailable',
            accepted_starts=sum(v['accepted'] for v in record['diagnostics']),
            fit_seconds=record['elapsed_seconds'],warm_preparation_seconds=record.get('warm_preparation_seconds',0))
        if fit is not None:
            train=score_record(data.development,record);p=spec['parameters'];index=daily.columns.get_loc(ident)
            row.update(test_mean_loglik=float(daily[ident].mean()),development_mean_loglik=float(train.mean()),
                aic=float(2*p-2*train.sum()),bic=float(p*np.log(len(train))-2*train.sum()))
            for length in boot:
                row[f'test_lower_L{length}'],row[f'test_upper_L{length}']=np.quantile(boot[length][:,index],[.025,.975])
            if spec['family']!='GMM':
                y,u=b.polar(data.test)
                if spec['family'].startswith('Kent'):terms=km.directional_terms(u,fit)
                else:terms=b.log_vmf_normalizer(fit['kappa'],9)+fit['kappa']*(u@fit['mu'].T)
                direction=logsumexp(np.log(fit['weights'])+terms,axis=1)
                conditional=daily[ident].to_numpy()+9*y-direction
                decomposition.append(dict(id=ident,family=spec['family'],k=spec['k'],
                    direction_log_score=float(direction.mean()),conditional_magnitude_log_score=float(conditional.mean())))
        rows.append(row)
    result=pd.DataFrame(rows);result.to_csv(folder/'test_results.csv',index=False)
    pd.DataFrame(starts).to_csv(folder/'refit_starts.csv',index=False)
    pd.DataFrame(decomposition).to_csv(folder/'score_decomposition.csv',index=False)
    contrasts=[]
    pairs=[('Kent linked','vMF linked'),('Kent unlinked','vMF unlinked'),
           ('Kent linked','Kent unlinked'),('vMF linked','vMF unlinked')]
    pairs += [(f,'GMM') for f in FAMILIES[1:]]
    for view,group in selection.groupby('view',sort=False):
        mapping=dict(zip(group.family,group.id))
        for candidate,reference in pairs:
            if candidate not in mapping or reference not in mapping:continue
            a,c=mapping[candidate],mapping[reference]
            if a not in available or c not in available:continue
            ai,ci=daily.columns.get_loc(a),daily.columns.get_loc(c)
            for length,draws in boot.items():
                lo,hi=np.quantile(draws[:,ai]-draws[:,ci],[.025,.975])
                contrasts.append(dict(view=view,candidate=a,reference=c,
                    comparison=f'{candidate} minus {reference}',block=length,
                    gain=float((daily[a]-daily[c]).mean()),lower=float(lo),upper=float(hi)))
    pd.DataFrame(contrasts).to_csv(folder/'contrasts.csv',index=False)
    save_json(folder/'completion.json',dict(freeze_sha256=digest,test_rows=len(data.test),
        models_selected=len(chosen),models_available=len(available),
        completed_utc=datetime.now(timezone.utc).isoformat()))
    return result


def run(stage='all'):
    with threadpool_limits(limits=1):
        data=b.load_data()['fx'];folder=run_directory(data)
        save_json(ROOT/'results'/'latest.json',dict(run=folder.name))
        print('Run:',folder,flush=True)
        if not run_gate(folder):return folder
        if stage=='gate':return folder
        trials,_=fit_stage(folder,data.train,'search',validation=data.validation)
        selection=freeze(folder,trials)
        if stage=='search':return folder
        _,records=fit_stage(folder,data.development,'refit',requested=set(selection.id))
        evaluation(folder,data,selection,records)
        return folder


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['all','gate','search'],default='all')
    args=parser.parse_args();run(args.stage)
