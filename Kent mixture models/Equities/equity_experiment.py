"""Original-space equity pilot: four starts, cached individually, no refits."""
from pathlib import Path
import sys,json,time,hashlib,copy,argparse
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
import kent_experiment as fx
km,b,kn=fx.km,fx.b,fx.kn
BASE=(1,3,5)
STARTS=4
FAMILIES=fx.FAMILIES

def specs(d):
    budgets=[km.parameter_count('GMM',k,d,'full') for k in BASE]
    grids={f:set(BASE) for f in FAMILIES[1:]}
    for f in grids:
        slope=km.parameter_count(f,2,d)-km.parameter_count(f,1,d)
        grids[f].update(max(1,int(np.floor((p-1)/slope+.5))) for p in budgets)
    # Preserve the original linked-vMF nested initialization without extra starts.
    grids['vMF unlinked'].update(grids['vMF linked'])
    rows=[]
    for k in BASE:
        for cov in b.COVARIANCES:
            for reg in b.REGULARIZATIONS:
                rows.append(dict(id=b.Spec('GMM',k,cov,reg).key,family='GMM',k=k,covariance=cov,regularization=reg))
    for f in FAMILIES[1:]:
        for k in sorted(grids[f]):
            rows.append(dict(id=f.lower().replace(' ','_')+f'_k{k:03d}',family=f,k=k,covariance='scalar',regularization=0.))
    for i,s in enumerate(rows):s.update(parameters=km.parameter_count(s['family'],s['k'],d,s['covariance']),configuration_order=i)
    return rows

def folder(data):
    files=[Path(__file__),ROOT/'test_equity.py',ROOT.parent/'kent_models.py',ROOT.parent/'kent_numerics.py',ROOT.parent/'kent_experiment.py',Path(b.__file__)]
    digest=hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()
    protocol=dict(engine=digest,data_hash=b.array_digest(data.frame.to_numpy()),starts=STARTS,
        specs=specs(data.frame.shape[1]),splits=b.split_table({'equities':data}).to_dict('records'),
        fitting_sample='train only; no development refits',shape_boundary='accept converged feasible fits; report both eigenvalue checks',
        bootstrap_blocks=[5,20,60],bootstrap_repetitions=1500,threads=1,max_iter=b.MAX_ITER,tol=b.TOL,
        numpy=np.__version__,scipy=b.scipy.__version__,sklearn=b.sklearn.__version__)
    ident=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()[:16]
    out=ROOT/'results'/ident;out.mkdir(parents=True,exist_ok=True)
    fx.save_json(out/'protocol.json',protocol);fx.save_json(ROOT/'results'/'latest.json',dict(run=ident))
    return out

def dump(path,obj):
    temp=path.with_suffix('.tmp');joblib.dump(obj,temp,compress=3);temp.replace(path)

def initializations(out,x,k):
    path=out/f'initializations_{k}.joblib'
    if path.exists():return joblib.load(path)
    old=b.STARTS;b.STARTS=STARTS
    try:result=b.make_initializations(x,k)
    finally:b.STARTS=old
    dump(path,result);return result

def fit_one(x,spec,initial,start,records):
    family,k=spec['family'],spec['k'];began=time.perf_counter()
    diagnostic=dict(start=start,status='error',accepted=False,iterations=0,origin='matched initialization')
    fit=None;kn.reset_stats()
    try:
        if family.startswith('Kent'):
            linked=family=='Kent linked';warm=None
            if start==STARTS-1:
                v=records[f"vmf_{'linked' if linked else 'unlinked'}_k{k:03d}"]['fit']
                if v is not None:warm=km.from_vmf(v);diagnostic['origin']='vMF zero anisotropy'
            elif start==STARTS-2 and linked:
                v=records[f'kent_unlinked_k{k:03d}']['fit']
                if v is not None:warm=copy.deepcopy(v);warm['coef'][:,1:]=0;diagnostic['origin']='unlinked Kent'
            fit=km.fit_kent(x,linked,responsibility=initial[1][start],warm=warm)
            forbidden=set(fit['boundary_reasons'])-{'unimodality_margin'}
            mass=km.posterior(x,fit).sum(0).min()
            accepted=fit['converged'] and not forbidden and mass>=b.MIN_EFFECTIVE
            diagnostic.update(status='accepted' if accepted else fit['status'],accepted=bool(accepted),
                converged=fit['converged'],iterations=fit['iterations'],boundary_reasons=fit['boundary_reasons'],
                history=fit['history'],times=fit['times'],min_effective=float(mass),normalizer=dict(kn.STATS))
        else:
            warm=None
            if family=='vMF linked' and start==STARTS-1:
                warm=records[f'vmf_unlinked_k{k:03d}']['fit']
                if warm is not None:diagnostic['origin']='unlinked vMF'
            old=b.STARTS;b.STARTS=1
            try:r=fx.fit_baseline(x,spec,([initial[0][start]],[initial[1][start]]),warm)
            finally:b.STARTS=old
            origin=diagnostic['origin'];diagnostic.update(r['diagnostics'][0]);diagnostic.update(start=start,origin=origin)
            fit=r['fit']
        record=dict(fit=fit,pilot_spec=spec,diagnostic=diagnostic)
        if not family.startswith('Kent'):
            record['spec']=dict(family='GMM' if family=='GMM' else ('Linked' if family=='vMF linked' else 'Unlinked'))
        if fit is not None:diagnostic['train_mean_loglik']=float(fx.score_record(x,record).mean())
    except (ValueError,FloatingPointError,AssertionError,np.linalg.LinAlgError) as err:
        diagnostic.update(status=f'{type(err).__name__}: {err}',accepted=False)
        record=dict(fit=fit,pilot_spec=spec,diagnostic=diagnostic)
    diagnostic['elapsed_seconds']=time.perf_counter()-began
    return record

def search(out,data):
    records={};rows=[];cache=out/'fits';cache.mkdir(exist_ok=True)
    for spec in specs(data.frame.shape[1]):
        ident=spec['id'];initial=initializations(out,data.train,spec['k']);starts=[]
        for start in range(STARTS):
            path=cache/f'{ident}_start{start}.joblib'
            if path.exists():rec=joblib.load(path)
            else:
                rec=fit_one(data.train,spec,initial,start,records);dump(path,rec)
            starts.append(rec)
            d=rec['diagnostic'];print(ident,start,d['status'],f"{d['elapsed_seconds']:.1f}s",flush=True)
        acceptable=[r for r in starts if r['diagnostic']['accepted']]
        selected=max(acceptable,key=lambda r:r['diagnostic']['train_mean_loglik']) if acceptable else dict(fit=None,pilot_spec=spec)
        records[ident]=selected;dump(out/f'{ident}.joblib',selected)
        row=dict(spec,status='ok' if acceptable else 'unavailable',accepted_starts=len(acceptable),
            elapsed_seconds=sum(r['diagnostic']['elapsed_seconds'] for r in starts),validation_mean_loglik=np.nan)
        if acceptable:row['validation_mean_loglik']=float(fx.score_record(data.validation,selected).mean())
        rows.append(row);pd.DataFrame(rows).to_csv(out/'search.csv',index=False)
    return pd.DataFrame(rows),records

def select(trials,d):
    rows=[];good=trials[trials.status=='ok']
    def add(view,group):
        for f,g in group.groupby('family'):
            r=b.best_validation(g);rows.append(dict(view=view,family=f,id=r.id,k=int(r.k),parameters=int(r.parameters)))
    for k in BASE:add(f'same_K_{k}',good[good.k==k])
    for k in BASE:
        p=km.parameter_count('GMM',k,d,'full')
        # Symmetric window narrow enough that any two members differ by <=10%.
        add(f'budget_full_GMM_K{k}',good[good.parameters.between(p/np.sqrt(1.1),p*np.sqrt(1.1))])
    add('best_in_grid',good)
    return pd.DataFrame(rows)

def evaluate(out,data,trials,records):
    selection=select(trials,data.frame.shape[1]);selection.to_csv(out/'selections.csv',index=False)
    fx.save_json(out/'frozen_selection.json',dict(test_used_for_selection=False,selections=selection.to_dict('records')))
    # Score all admissible configuration winners; test scores never choose candidates.
    good=trials[trials.status=='ok'];rows=[];scores={};constraint=[]
    for r in good.itertuples():
        rec=records[r.id];fit=rec['fit'];test=fx.score_record(data.test,rec);train=fx.score_record(data.train,rec)
        assert np.isfinite(test).all();scores[r.id]=test
        rows.append(dict(id=r.id,family=r.family,k=r.k,parameters=r.parameters,test_log_score=test.mean(),
            training_aic=2*r.parameters-2*train.sum(),training_bic=r.parameters*np.log(len(train))-2*train.sum()))
        if r.family.startswith('Kent'):
            for j,lam in enumerate(fit['lambda']):
                constraint.append(dict(id=r.id,component=j,max_ratio=2*lam.max()/fit['kappa'],
                    absolute_ratio=2*np.abs(lam).max()/fit['kappa'],positive_pass=bool(2*lam.max()<fit['kappa']),
                    absolute_pass=bool(2*np.abs(lam).max()<fit['kappa'])))
    result=pd.DataFrame(rows);daily=pd.DataFrame(scores,index=data.frame.index[data.b:]);daily.to_csv(out/'daily_scores.csv')
    contrasts=[]
    for block in (5,20,60):
        boot=b.bootstrap_means(daily.to_numpy(),block)
        if block==20:result['lower'],result['upper']=np.quantile(boot,[.025,.975],axis=0)
        for view,g in selection.groupby('view'):
            mapping=dict(zip(g.family,g.id))
            pairs=[('Kent linked','vMF linked'),('Kent unlinked','vMF unlinked'),('Kent linked','Kent unlinked'),('vMF linked','vMF unlinked')]+[(f,'GMM') for f in FAMILIES[1:]]
            for left,right in pairs:
                if left not in mapping or right not in mapping:continue
                a,c=mapping[left],mapping[right];i,j=daily.columns.get_loc(a),daily.columns.get_loc(c)
                lo,hi=np.quantile(boot[:,i]-boot[:,j],[.025,.975])
                contrasts.append(dict(view=view,comparison=f'{left} minus {right}',candidate=a,reference=c,block=block,
                    gain=float((daily[a]-daily[c]).mean()),lower=lo,upper=hi))
    result.to_csv(out/'scores.csv',index=False);pd.DataFrame(contrasts).to_csv(out/'contrasts.csv',index=False)
    pd.DataFrame(constraint,columns=['id','component','max_ratio','absolute_ratio','positive_pass','absolute_pass']).to_csv(out/'constraints.csv',index=False)
    fx.save_json(out/'completion.json',dict(configurations=len(trials),admissible=len(good),test_dates=len(data.test)))

def run():
    import subprocess
    data=b.load_data()['equities'];out=folder(data)
    if (out/'completion.json').exists():return out
    if not (out/'gate_passed.json').exists():
        proc=subprocess.run([sys.executable,'-m','unittest','test_equity','-v'],cwd=ROOT,capture_output=True,text=True)
        (out/'numerical_gate.txt').write_text(proc.stdout+proc.stderr)
        if proc.returncode:raise RuntimeError('Equity numerical gate failed; see numerical_gate.txt')
        fx.save_json(out/'gate_passed.json',dict(passed=True))
    with threadpool_limits(limits=1):
        trials,records=search(out,data);evaluate(out,data,trials,records)
    return out

if __name__=='__main__':
    import fcntl
    with (ROOT/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
