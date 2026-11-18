"""Follow-up variance/concentration ablation; original experiment remains frozen."""
from __future__ import annotations
import argparse, copy, hashlib, inspect, json, math, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.special import ive, gammaln, logsumexp
from threadpoolctl import threadpool_limits
import size_direction_experiment as base

OUT = base.ROOT/'outputs'/'size_direction_sharing'
MODES = ('variance_specific', 'kappa_specific', 'both_specific')

def count(k,d,linked,mode):
    return k*((2*d+1) if linked else (d+1))+1+(k-1)*(int(mode in ('variance_specific','both_specific'))+int(mode in ('kappa_specific','both_specific')))

def normalizer(kappa,d):
    kappa=np.asarray(kappa,dtype=float)
    z=np.empty_like(kappa); positive=kappa>0
    z[~positive]=gammaln(d/2)-np.log(2)-d/2*np.log(np.pi)
    v=d/2-1; q=kappa[positive]
    z[positive]=v*np.log(q)-d/2*np.log(2*np.pi)-np.log(ive(v,q))-q
    return z

def log_components(y,u,fit):
    mean=fit['coef'][:,0]+u@fit['coef'][:,1:].T
    var=np.asarray(fit['variance'])
    kap=np.asarray(fit['kappa'])
    return (np.log(fit['weights'])+normalizer(kap,u.shape[1])+ (u@fit['mu'].T)*kap
            -.5*(np.log(2*np.pi*var)+(y[:,None]-mean)**2/var))

def m_step(y,u,r,linked,mode):
    n,d=u.shape; mass=r.sum(axis=0); k=len(mass)
    if mass.min()<base.MIN_EFFECTIVE: raise FloatingPointError('Collapsed component')
    vec=r.T@u; lengths=np.linalg.norm(vec,axis=1)
    mu=np.zeros((k,d));mu[:,0]=1
    nz=lengths>1e-14;mu[nz]=vec[nz]/lengths[nz,None]
    a=np.column_stack([np.ones(n),u]);coef=np.zeros((k,d+1))
    for j in range(k):
        coef[j]=base.weighted_regression(a,y,r[:,j]) if linked else np.r_[r[:,j]@y/mass[j],np.zeros(d)]
    ss=(r*(y[:,None]-a@coef.T)**2).sum(axis=0)
    var=ss/mass if mode in ('variance_specific','both_specific') else np.repeat(ss.sum()/n,k)
    if mode in ('kappa_specific','both_specific'):
        kap=np.array([base.concentration_from_resultant(v/m,d) for v,m in zip(lengths,mass)])
    else: kap=np.repeat(base.concentration_from_resultant(lengths.sum()/n,d),k)
    return dict(weights=mass/n,mu=mu,coef=coef,variance=np.maximum(var,base.VAR_FLOOR),kappa=kap)

def fit(y,u,linked,mode,responsibilities=None,warm=None):
    f=copy.deepcopy(warm) if warm is not None else m_step(y,u,responsibilities,linked,mode)
    history=[float(logsumexp(log_components(y,u,f),axis=1).mean())]; converged=False
    for iteration in range(1,base.MAX_ITER+1):
        logs=log_components(y,u,f);r=np.exp(logs-logsumexp(logs,axis=1,keepdims=True))
        f=m_step(y,u,r,linked,mode)
        ll=float(logsumexp(log_components(y,u,f),axis=1).mean());gain=ll-history[-1]
        if not np.isfinite(ll) or gain < -1e-8: raise FloatingPointError('Nonfinite/decreasing EM likelihood')
        history.append(ll)
        if abs(gain)<base.TOL: converged=True;break
    boundary=bool(np.any(f['variance']<=base.VAR_FLOOR*(1+1e-6)) or np.any(f['kappa']>=base.KAPPA_MAX*(1-1e-6)))
    logs=log_components(y,u,f);mass=np.exp(logs-logsumexp(logs,axis=1,keepdims=True)).sum(axis=0)
    if mass.min()<base.MIN_EFFECTIVE: raise FloatingPointError('Collapsed final posterior')
    f.update(history=np.array(history),converged=converged,boundary=boundary,iterations=iteration)
    return f

def score(x,record):
    if 'sharing' not in record: return base.score_model(x,record)
    y,u=base.polar(x)
    return logsumexp(log_components(y,u,record['fit']),axis=1)-x.shape[1]*y

def candidate(x,k,linked,mode,initializations,warm=None):
    y,u=base.polar(x); diagnostics=[]; accepted=[];begin=time.monotonic()
    for start in range(base.STARTS):
        row=dict(start=start,accepted=False)
        try:
            use=linked and warm is not None and start==base.STARTS-1
            f=fit(y,u,linked,mode,responsibilities=None if use else initializations[start],warm=warm if use else None)
            ll=float(f['history'][-1]-x.shape[1]*y.mean())
            ok=f['converged'] and not f['boundary'] and np.isfinite(ll)
            row.update(accepted=bool(ok),iterations=f['iterations'],train_mean_loglik=ll,
                       status='converged' if ok else 'boundary' if f['boundary'] else 'nonconverged')
            if ok: accepted.append((ll,start,f))
        except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc: row['status']=str(exc)
        diagnostics.append(row)
    result=dict(sharing=mode,linked=linked,k=k,diagnostics=diagnostics,elapsed_seconds=time.monotonic()-begin,fit=None)
    if accepted:
        ll,start,f=max(accepted,key=lambda a:a[0]);result.update(fit=f,train_mean_loglik=ll,selected_start=start)
    return result

def engine():
    source='\n'.join(inspect.getsource(f) for f in (count,normalizer,log_components,m_step,fit,candidate,score))
    return hashlib.sha256((source+base.engine_fingerprint()).encode()).hexdigest()

def key(mode,linked,k): return f'{mode}_{"linked" if linked else "unlinked"}_k{k:02d}'
def family(mode,linked): return ('Linked' if linked else 'Unlinked')+' / '+mode

def original_cache(name,train,validation):
    signature=hashlib.sha256((base.engine_fingerprint()+base.array_digest(train,validation)).encode()).hexdigest()
    return base.OUTPUT/'search'/name/signature[:16]

def search_group(name,train,validation,k):
    signature=hashlib.sha256((engine()+base.array_digest(train,validation)).encode()).hexdigest()
    cache=OUT/'search'/name/signature[:16];cache.mkdir(parents=True,exist_ok=True)
    init_path=original_cache(name,train,validation)/f'initializations_k{k:02d}.joblib'
    with threadpool_limits(limits=1):
        initial=joblib.load(init_path)[1] if init_path.exists() else base.make_initializations(train,k)[1]
        rows=[]
        for mode in MODES:
            warm=None
            for linked in (False,True):
                ident=key(mode,linked,k);p=cache/(ident+'.joblib')
                if p.exists():
                    record=joblib.load(p);assert record['signature']==signature
                else:
                    record=candidate(train,k,linked,mode,initial,warm)
                    record['signature']=signature
                    record['validation_mean_loglik']=float(score(validation,record).mean()) if record['fit'] is not None else None
                    base.dump_joblib(p,record)
                    print(name,ident,sum(r['accepted'] for r in record['diagnostics']),f'{record["elapsed_seconds"]:.1f}s',flush=True)
                if not linked: warm=record['fit']
                rows.append(dict(dataset=name,family=family(mode,linked),sharing=mode,linked=linked,
                    candidate=ident,k=k,parameters=count(k,train.shape[1],linked,mode),
                    configuration_order=1000+MODES.index(mode)*40+int(linked)*20+k,
                    validation_mean_loglik=record['validation_mean_loglik'],train_mean_loglik=record.get('train_mean_loglik'),
                    accepted_starts=sum(r['accepted'] for r in record['diagnostics']),cache_path=str(p),signature=signature))
    return rows

def search_all(datasets,workers=6):
    OUT.mkdir(parents=True,exist_ok=True);rows=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(search_group,name,data.train,data.validation,k) for name,data in datasets.items() for k in range(1,21)]
        for future in as_completed(futures):
            rows.extend(future.result());print('New completed configurations:',len(rows),'/240',flush=True)
    frame=pd.DataFrame(rows).sort_values(['dataset','configuration_order'])
    frame.to_csv(OUT/'validation_trials.csv',index=False)
    return frame

def freeze(trials,datasets):
    original=pd.read_csv(base.OUTPUT/'selected_configurations.csv',keep_default_na=False)
    original['sharing']='shared';original['linked']=original.family=='Linked'
    chosen=[]
    for name,data in datasets.items():
        d=data.frame.shape[1];same_k=int(original.loc[(original.dataset==name)&(original.view=='same_components'),'k'].iloc[0])
        for view,k0 in [('budget_low',3),('budget_medium',6),('budget_high',9),('unrestricted',None),('same_components',None)]:
            for fam,group in trials.loc[trials.dataset==name].groupby('family',sort=False):
                lower=k0*(2*d+1)-1 if k0 else None;upper=math.floor(1.1*lower) if k0 else None
                subset=group if lower is None else group.loc[group.parameters.between(lower,upper)]
                if view=='same_components': subset=subset.loc[subset.k==same_k]
                if subset.validation_mean_loglik.notna().any():
                    row=base.best_validation(subset).to_dict();row.update(view=view,budget_lower=lower,budget_upper=upper,status='selected')
                else:
                    row=dict(dataset=name,family=fam,view=view,sharing=group.sharing.iloc[0],linked=bool(group.linked.iloc[0]),
                             status='no parameter count in band' if subset.empty else 'no admissible validation fit',
                             budget_lower=lower,budget_upper=upper)
                chosen.append(row)
    original['status']='selected'
    selected=pd.concat([original,pd.DataFrame(chosen)],ignore_index=True)
    # All choices are fixed before refitting or scoring. Original validation choices remain incumbents.
    payload=dict(engine=engine(),original_freeze=base.digest_file(base.OUTPUT/'frozen_selections.json'),
        selections=json.loads(selected.to_json(orient='records')),data_hashes={n:base.array_digest(d.frame.to_numpy()) for n,d in datasets.items()},
        protocol='Exploratory follow-up on previously inspected test periods; validation-only selection; original same-K anchor retained.',
        selection_metric='validation_mean_loglik',test_used_for_selection=False,
        splits=base.split_table(datasets).to_dict(orient='records'),starts=10,max_iter=500,tolerance=1e-6)
    payload['freeze_sha256']=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    base.write_json(OUT/'frozen_selections.json',payload);selected.to_csv(OUT/'selected_configurations.csv',index=False)
    return selected,payload

def refit_group(name,development,mode,k,linked_needed,signature):
    cache=OUT/'refits'/name/signature[:16];cache.mkdir(parents=True,exist_ok=True)
    records={};initial=None;warm=None
    with threadpool_limits(limits=1):
        for linked in (False,True) if linked_needed else (False,):
            ident=key(mode,linked,k);p=cache/(ident+'.joblib')
            if p.exists():
                record=joblib.load(p);assert record['signature']==signature
            else:
                if initial is None: initial=base.make_initializations(development,k)[1]
                record=candidate(development,k,linked,mode,initial,warm);record['signature']=signature
                base.dump_joblib(p,record);print('Refit',name,ident,sum(r['accepted'] for r in record['diagnostics']),flush=True)
            warm=record['fit'];records[ident]=record
    return name,records

def evaluate(datasets,selected,manifest,workers=6):
    assert manifest['engine']==engine() and not manifest['test_used_for_selection']
    payload=dict(manifest);digest=payload.pop('freeze_sha256')
    assert digest==hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    fitted={name:{} for name in datasets}
    original_freeze=json.loads((base.OUTPUT/'frozen_selections.json').read_text())['freeze_sha256']
    # Reuse original refits exactly, then complete all new development fits before accessing test.
    for name,data in datasets.items():
        assert base.array_digest(data.frame.to_numpy())==manifest['data_hashes'][name]
        old=selected.loc[(selected.dataset==name)&(selected.sharing=='shared')]
        for ident in old.candidate.unique():
            fitted[name][ident]=joblib.load(base.OUTPUT/'refits'/name/original_freeze[:16]/(ident+'.joblib'))
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[]
        new=selected.loc[(selected.sharing!='shared')&(selected.status=='selected')]
        for (name,mode,k),group in new.groupby(['dataset','sharing','k']):
            signature=hashlib.sha256((engine()+base.array_digest(datasets[name].development)).encode()).hexdigest()
            futures.append(pool.submit(refit_group,name,datasets[name].development,mode,int(k),bool(group.linked.any()),signature))
        for f in as_completed(futures):
            name,records=f.result();fitted[name].update(records)
    rows=[]
    for name,data in datasets.items():
        choices=selected.loc[(selected.dataset==name)&(selected.status=='selected')]
        keys=[ident for ident in choices.candidate.unique() if fitted[name][ident]['fit'] is not None]
        with threadpool_limits(limits=1):
            scores=np.column_stack([score(data.test,fitted[name][ident]) for ident in keys])
            dev=np.array([score(data.development,fitted[name][ident]).sum() for ident in keys])
        assert np.isfinite(scores).all() and np.isfinite(dev).all()
        ix={v:i for i,v in enumerate(keys)};boot={b:base.bootstrap_means(scores,b) for b in (5,20,60)}
        for choice in choices.to_dict(orient='records'):
            ident=choice['candidate'];row=dict(choice);row.update(test_rows=len(data.test),refit_rows=len(data.development),freeze_sha256=digest)
            if ident not in ix:
                row['status']='refit failed; no substitution';rows.append(row);continue
            i=ix[ident];p=int(choice['parameters']);rec=fitted[name][ident]
            row.update(test_mean_loglik=float(scores[:,i].mean()),refit_total_loglik=float(dev[i]),
                       aic=2*p-2*dev[i],bic=p*np.log(len(data.development))-2*dev[i],
                       accepted_refit_starts=sum(s['accepted'] for s in rec['diagnostics']))
            group=choices.loc[choices.view==choice['view']]
            refs={'gmm':group.loc[group.family=='GMM','candidate'].iloc[0],
                  'shared_unlinked':group.loc[group.family=='Unlinked','candidate'].iloc[0]}
            partner=group.loc[(group.sharing==choice['sharing'])&(~group.linked.astype(bool))&(group.family!='GMM')]
            if len(partner)==1:refs['own_unlinked']=partner.candidate.iloc[0]
            for b,means in boot.items():
                lo,hi=np.quantile(means[:,i],[.025,.975]);row[f'test_L{b}_low']=lo;row[f'test_L{b}_high']=hi
                for label,ref in refs.items():
                    if ref in ix:
                        j=ix[ref];row[f'gain_vs_{label}']=float((scores[:,i]-scores[:,j]).mean())
                        lo,hi=np.quantile(means[:,i]-means[:,j],[.025,.975])
                        row[f'gain_vs_{label}_L{b}_low']=lo;row[f'gain_vs_{label}_L{b}_high']=hi
            rows.append(row)
        pd.DataFrame(scores,index=data.frame.index[data.b:],columns=keys).to_csv(OUT/f'{name}_test_daily_loglik.csv',index_label='date')
    unavailable=selected.loc[selected.status!='selected'].to_dict(orient='records')
    results=pd.DataFrame(rows+unavailable);results.to_csv(OUT/'test_results.csv',index=False,float_format='%.12g')
    base.write_json(OUT/'evaluation.json',dict(freeze_sha256=digest,results_sha256=base.digest_file(OUT/'test_results.csv'),
        bootstrap_repetitions=1500,block_lengths=[5,20,60],criterion_sample='training plus validation',
        module_sha256=base.digest_file(__file__),rows=len(results)))
    return results

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['search','all']);parser.add_argument('--workers',type=int,default=6);args=parser.parse_args()
    data=base.load_data();trials=search_all(data,args.workers)
    if args.action=='all':
        selected,manifest=freeze(trials,data);evaluate(data,selected,manifest,args.workers)
if __name__=='__main__':main()
