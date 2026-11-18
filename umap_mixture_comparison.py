"""Replicate Size_Direction_Comparison on raw and radius-preserving UMAP data.

Diagonal/full are genuine alternatives for the Cartesian GMM. The reference
cylinders have scalar radial variance: both matrix toggles denote the same 1x1
covariance. They are fitted once and reported under both toggles, never counted
as independent evidence. AIC/BIC condition on the fixed encoder, within space.
"""
from pathlib import Path
import sys,json,hashlib,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'GMM'))
import size_direction_experiment as b
import size_direction_sharing as h
import spherical_umap as s
OUT=ROOT/'umap_mixture_results'
MODES=('shared','variance_specific','kappa_specific','both_specific')


def signature(x):return hashlib.sha256((b.engine_fingerprint()+h.engine()+b.array_digest(x)).encode()).hexdigest()[:20]
def key(family,k,cov='',reg=0.,mode='shared'):
    return b.Spec('GMM',k,cov,reg).key if family=='GMM' else f'{family.lower()}_{mode}_k{k:02d}'
def parameters(family,k,d,cov,mode):
    if family=='GMM':return b.parameter_count(b.Spec(family,k,cov),d)
    return b.parameter_count(b.Spec(family,k),d) if mode=='shared' else h.count(k,d,family=='Linked',mode)
def score(x,record):return h.score(x,record)


def matrices(x,market,family,covariance):
    """Actual per-component covariance matrices; cylinder scalar is 1x1."""
    if family=='GMM':
        return np.array([np.diag(c) for c in market.covariances_]) if covariance=='diag' else market.covariances_
    variance=np.atleast_1d(market['variance'])
    return variance[:,None,None]


def prepare_cases():
    original=b.load_data();markets,_=s.load_markets();cases={};encoder=[]
    config=dict(s.DEFAULT_CONFIG)
    for name,data in original.items():
        x=data.frame.to_numpy()
        cases[name+'/raw']=dict(market=name,representation='raw',train=data.train,validation=data.validation,
            development=data.development,test=data.test,dates=data.frame.index,a=data.a,b=data.b)
        settings=s.datasets(markets)[name+'/returns']
        stages={}
        for stage in ['validation','test']:
            p=s.prepare(settings,'cosine',stage)
            with threadpool_limits(limits=1):
                a,q,info=s.cached_fit(p,'Spherical UMAP',15,42,config)
            tr=x[:data.a] if stage=='validation' else x[:data.b]
            query=x[data.a:data.b] if stage=='validation' else x[data.b:]
            stages[stage]=(np.linalg.norm(tr,axis=1)[:,None]*s.to_xyz(a),
                           np.linalg.norm(query,axis=1)[:,None]*s.to_xyz(q))
            encoder.append(dict(market=name,stage=stage,neighbors=15,seed=42,cache_key=info['cache_key'],
                                train_rows=len(tr),query_rows=len(query)))
        cases[name+'/umap']=dict(market=name,representation='umap',train=stages['validation'][0],validation=stages['validation'][1],
            development=stages['test'][0],test=stages['test'][1],dates=data.frame.index,a=data.a,b=data.b)
        for stage,xnew in [('train',stages['validation'][0]),('validation',stages['validation'][1]),('development',stages['test'][0]),('test',stages['test'][1])]:
            ref=cases[name+'/raw'][stage]
            np.testing.assert_allclose(np.linalg.norm(xnew,axis=1),np.linalg.norm(ref,axis=1),rtol=3e-7)
    return cases,encoder


def legacy(name,x,phase,family,k,cov,reg,mode):
    """Reuse only records whose original input/engine signature matches exactly."""
    if phase=='search':
        data=b.load_data()[name]
        expected=hashlib.sha256((b.engine_fingerprint()+b.array_digest(data.train,data.validation)).encode()).hexdigest()
        if not np.array_equal(x,data.train):return None
        if mode=='shared':
            path=b.OUTPUT/'search'/name/expected[:16]/(b.Spec(family,k,cov,reg).key+'.joblib')
        else:
            expected=hashlib.sha256((h.engine()+b.array_digest(data.train,data.validation)).encode()).hexdigest()
            path=h.OUT/'search'/name/expected[:16]/(h.key(mode,family=='Linked',k)+'.joblib')
        if path.exists():
            record=joblib.load(path)
            recorded=record.get('search_signature',record.get('signature'))
            if recorded==expected:return record
    return None


def fit_group(case,x,k,phase='search',requested=None):
    cache=OUT/'cache'/signature(x)/phase
    cache.mkdir(parents=True,exist_ok=True)
    rows=[];initial=None;warm={}
    market=case.split('/')[0];raw=case.endswith('/raw')
    specifications=[('GMM',cov,reg,'shared') for cov in ['diag','full'] for reg in b.REGULARIZATIONS]
    specifications += [(family,'scalar',0.,mode) for mode in MODES for family in ['Unlinked','Linked']]
    for family,cov,reg,mode in specifications:
        ident=key(family,k,cov,reg,mode)
        if requested is not None and ident not in requested and not (family=='Unlinked' and key('Linked',k,cov,reg,mode) in requested):continue
        path=cache/(ident+'.joblib')
        if path.exists():record=joblib.load(path)
        else:
            record=legacy(market,x,phase,family,k,cov,reg,mode) if raw else None
            if record is None:
                if initial is None:initial=b.make_initializations(x,k)
                if family=='GMM' or mode=='shared':
                    record=b.fit_candidate(x,b.Spec(family,k,cov if family=='GMM' else '',reg),initial,warm.get(mode))
                else:record=h.candidate(x,k,family=='Linked',mode,initial[1],warm.get(mode))
            s.joblib.dump(record,path,compress=3)
        if family=='Unlinked':warm[mode]=record['fit']
        rows.append(dict(case=case,candidate=ident,family=family,k=k,covariance=cov,regularization=reg,sharing=mode,
            parameters=parameters(family,k,x.shape[1],cov,mode),record_path=str(path),
            accepted_starts=sum(r['accepted'] for r in record['diagnostics']),
            status='ok' if record['fit'] is not None else 'unavailable'))
    return rows


def search_worker(case,data,k):
    with threadpool_limits(limits=1):
        rows=fit_group(case,data['train'],k)
        for r in rows:
            if r['status']=='ok':
                rec=joblib.load(r['record_path']);p=r['parameters'];n=len(data['train'])
                train=score(data['train'],rec);validation=score(data['validation'],rec)
                r.update(validation_mean_loglik=float(validation.mean()),train_mean_loglik=float(train.mean()),
                         aic=float(2*p-2*train.sum()),bic=float(p*np.log(n)-2*train.sum()))
                if r['family']!='GMM':
                    kap=np.atleast_1d(rec['fit']['kappa']);r.update(kappa_min=float(kap.min()),kappa_max=float(kap.max()))
        return rows


def select(trials):
    chosen=[]
    for (case,family),g in trials.groupby(['case','family']):
        # Keep both actual Gaussian matrix alternatives visible. Cylinder covariance
        # labels are exact aliases, so selection is performed once for both.
        groups=g.groupby('covariance') if family=='GMM' else [('scalar',g)]
        for covariance,rows in groups:
            valid=rows[rows.status.eq('ok') & rows.validation_mean_loglik.notna()]
            if valid.empty:raise RuntimeError(f'No admissible validation model: {case}/{family}/{covariance}')
            top=valid.validation_mean_loglik.max()
            winner=valid[valid.validation_mean_loglik>=top-1e-6].sort_values(['parameters','k','sharing','regularization']).iloc[0].to_dict()
            winner['component_boundary']=bool(winner['k']==20)
            for toggle in [covariance] if family=='GMM' else ['diag','full']:
                chosen.append(dict(winner,matrix_toggle=toggle,matrix_equivalent=family!='GMM'))
    return pd.DataFrame(chosen)


def score_interval(values,block):
    """Adapt one model's daily scores to the reference multi-model bootstrap API."""
    means=b.bootstrap_means(np.asarray(values)[:,None],block)
    return np.quantile(means[:,0],[.025,.975])


def evaluate(cases,chosen,folder):
    results=[];daily=[];contrasts=[];kappas=[];covariances=[]
    for case,data in cases.items():
        choices=chosen[chosen.case.eq(case)]
        records={}
        with threadpool_limits(limits=1):
            for k,group in choices.groupby('k'):
                fitted=fit_group(case,data['development'],int(k),'refit',set(group.candidate))
                records.update({r['candidate']:joblib.load(r['record_path']) for r in fitted})
            scores={}
            for row in choices.to_dict('records'):
                rec=records[row['candidate']]
                if rec['fit'] is None:raise RuntimeError(f'Selected refit failed: {case}/{row["candidate"]}')
                p=row['parameters'];development=score(data['development'],rec);values=score(data['test'],rec)
                scores[row['candidate']]=values
                result=dict(row,test_mean_loglik=float(values.mean()),aic=float(2*p-2*development.sum()),
                            bic=float(p*np.log(len(development))-2*development.sum()),refit_rows=len(development))
                for block in [5,20,60]:
                    result[f'test_L{block}_low'],result[f'test_L{block}_high']=score_interval(values,block)
                results.append(result)
                for i,matrix in enumerate(matrices(data['development'],rec['fit'],row['family'],row['matrix_toggle'])):
                    covariances.append(dict(case=case,candidate=row['candidate'],toggle=row['matrix_toggle'],component=i,matrix=matrix.tolist()))
                if row['family']!='GMM':
                    for j,kappa in enumerate(np.atleast_1d(rec['fit']['kappa'])):
                        kappas.append(dict(case=case,candidate=row['candidate'],component=j,kappa=float(kappa),sharing=row['sharing']))
            normal=choices[choices.family.eq('GMM')].sort_values('validation_mean_loglik',ascending=False).iloc[0]
            reference=scores[normal.candidate]
            for candidate,values in scores.items():
                daily.append(pd.DataFrame(dict(date=data['dates'][data['b']:],case=case,candidate=candidate,loglik=values)))
                for block in [5,20,60]:
                    contrasts.append(dict(case=case,candidate=candidate,baseline=normal.candidate,block=block,
                        **s.block_interval(values-reference,block,1500)))
    frames=dict(test_results=pd.DataFrame(results),test_daily=pd.concat(daily,ignore_index=True),paired_gains=pd.DataFrame(contrasts),kappas=pd.DataFrame(kappas))
    for name,frame in frames.items():frame.to_csv(folder/(name+'.csv'),index=False)
    s.write_json(folder/'covariance_matrices.json',covariances)
    return frames


def run(workers=6):
    from datetime import datetime,timezone
    folder=OUT/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');folder.mkdir(parents=True)
    cases,encoders=prepare_cases()
    s.write_json(folder/'protocol.json',dict(source_notebook='GMM/Size_Direction_Comparison.ipynb',engine=b.engine_fingerprint(),sharing_engine=h.engine(),
        study_sha256=s.sha(__file__),components=list(range(1,21)),starts=b.STARTS,iterations=b.MAX_ITER,
        concentration_selection='Training MLE; validation selects K and shared/component-specific concentration and variance.',
        covariance='diag/full for every family; identical 1x1 covariances in original cylinders.',
        encoder=encoders,encoder_selection='Existing frozen cosine-return S2 map: 15 neighbors, seed42,300 epochs; original radius retained.',
        density_measure='Original Cartesian coordinates within each representation, with -d*log(radius) correction for cylinders.',
        information_criteria='Fitted-development-sample AIC/BIC; conditional on frozen encoder. Never compare absolute scores between representations or markets.',
        test_status='Retrospective; historical test period previously inspected.',versions=s.versions()))
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(search_worker,case,data,k):(case,k) for case,data in cases.items() for k in range(1,21)}
        for future in as_completed(futures):
            case,k=futures[future];rows.extend(future.result());print(case,k,'complete',flush=True)
            pd.DataFrame(rows).to_csv(folder/'validation_checkpoint.csv',index=False)
    trials=pd.DataFrame(rows).sort_values(['case','family','k','sharing','covariance','regularization']);trials.to_csv(folder/'validation_trials.csv',index=False)
    selected=select(trials);selected.to_csv(folder/'frozen_selections.csv',index=False)
    s.write_json(folder/'freeze.json',dict(test_used_for_selection=False,selections=json.loads(selected.to_json(orient='records')),
        validation_sha256=s.sha(folder/'validation_trials.csv')))
    frames=evaluate(cases,selected,folder)
    s.write_json(folder/'status.json',dict(search='complete',evaluation='complete',notebook='pending'))
    print('COMPLETE',folder,flush=True)
    return folder

if __name__=='__main__':run()
