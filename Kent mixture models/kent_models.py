"""Joint generalized-Kent/Gaussian GEM, original shared kappa and variance.

All densities use sphere surface area times log-radius until score() applies
the Cartesian Jacobian. All timings are measured, not estimated from caches.
"""
from pathlib import Path
import sys, copy, time
import numpy as np
from scipy.linalg import null_space, expm, expm_frechet, helmert
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.special import logsumexp
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0,str(PROJECT/'GMM'))
import size_direction_experiment as baseline
import kent_numerics as numerical

GRID = (1,3,6,10)
SHAPE_FTOL = 1e-10
ORIENTATION_GTOL = 1e-7
INNER_MAXITER = 100


def parameter_count(family,k,d=9,covariance=''):
    if family == 'GMM':
        return baseline.parameter_count(baseline.Spec('GMM',k,covariance),d)
    if family.startswith('vMF'):
        return baseline.parameter_count(baseline.Spec('Linked' if family.endswith('linked') and not family.endswith('unlinked') else 'Unlinked',k),d)
    # d(d-1)/2 frame + d-2 eigenvalues per leaf; kappa and sigma shared.
    directional = d*(d-1)//2+d-2
    radial = d+1 if family == 'Kent linked' else 1
    return k*(directional+radial+1)+1


def frame_from_mean(mu,second=None):
    tangent = null_space(mu[None,:])
    if second is not None:
        _,vectors=np.linalg.eigh(tangent.T@second@tangent)
        tangent=tangent@vectors
    return np.column_stack([mu,tangent])


def from_vmf(fit,second=None):
    out = copy.deepcopy(fit)
    out['frame']=np.array([frame_from_mean(mu,None if second is None else second[j])
                           for j,mu in enumerate(fit['mu'])])
    out['lambda']=np.zeros((len(fit['mu']),fit['mu'].shape[1]-1))
    out.pop('history',None)
    refresh(out)
    return out


def refresh(fit,tol=numerical.LOGZ_TOL):
    fit['logz']=np.array([numerical.normalizer(fit['kappa'],lam,tol=tol).logz for lam in fit['lambda']])


def directional_terms(u,fit):
    coordinates=np.einsum('nd,kde->nke',u,fit['frame'],optimize=True)
    return (fit['kappa']*coordinates[:,:,0]+
            np.einsum('nkj,kj->nk',coordinates[:,:,1:]**2,fit['lambda'])-fit['logz'])


def components(y,u,fit):
    means=fit['coef'][:,0]+u@fit['coef'][:,1:].T
    return (np.log(fit['weights'])+directional_terms(u,fit)+
            norm.logpdf(y[:,None],means,np.sqrt(fit['variance'])))


def e_step(y,u,fit):
    terms=components(y,u,fit)
    ll=logsumexp(terms,axis=1)
    return np.exp(terms-ll[:,None]),ll


def score(x,fit):
    y,u=baseline.polar(x)
    return e_step(y,u,fit)[1]-x.shape[1]*y


def posterior(x,fit):
    y,u=baseline.polar(x)
    return e_step(y,u,fit)[0]


def orientation_objective(v,frame,mean,second,kappa,lam,gradient=True):
    d=len(mean); indices=np.triu_indices(d,1)
    skew=np.zeros((d,d));skew[indices]=v;skew-=skew.T
    rotation=expm(skew)
    candidate=frame@rotation
    eigen=np.r_[0.,lam]
    q=kappa*(candidate[:,0]@mean)+np.sum(eigen*np.diag(candidate.T@second@candidate))
    if not gradient:return -q
    ambient=2*(second@candidate)*eigen
    ambient[:,0]+=kappa*mean
    adjoint=expm_frechet(skew.T,frame.T@ambient,compute_expm=False)
    grad=(adjoint-adjoint.T)[indices]
    return -float(q),-grad


def optimize_frame(frame,mean,second,kappa,lam):
    d=len(mean);zero=np.zeros(d*(d-1)//2)
    initial=orientation_objective(zero,frame,mean,second,kappa,lam)[0]
    result=minimize(orientation_objective,zero,args=(frame,mean,second,kappa,lam),
                    jac=True,method='BFGS',options={'gtol':ORIENTATION_GTOL,'maxiter':INNER_MAXITER})
    if not np.isfinite(result.fun) or result.fun>initial+1e-11:
        return frame,dict(nit=result.nit,nfev=result.nfev,success=False,improved=False,
                          gradient=float(np.linalg.norm(result.jac)),message=str(result.message))
    skew=np.zeros((d,d));skew[np.triu_indices(d,1)]=result.x;skew-=skew.T
    return frame@expm(skew),dict(nit=result.nit,nfev=result.nfev,success=bool(result.success),
        improved=result.fun<initial-1e-12,gradient=float(np.linalg.norm(result.jac)),message=str(result.message))


def shape_objective(v,mean_projection,second_projection,masses,trace_basis):
    kappa=v[0];k=len(masses)
    lam=v[1:].reshape(k,-1)@trace_basis
    value=0.; gradient=np.zeros_like(v)
    for j in range(k):
        # SLSQP probes outside the linear constraint; the integral is still valid.
        z=numerical.normalizer(kappa,lam[j],constrained=False)
        value+=masses[j]*(z.logz-kappa*mean_projection[j]-lam[j]@second_projection[j])
        gradient[0]+=masses[j]*(z.dkappa-mean_projection[j])
        gradient[1+j*len(trace_basis):1+(j+1)*len(trace_basis)]=(
            masses[j]*(trace_basis@(z.dlambda-second_projection[j])))
    return float(value),gradient


def optimize_shape(fit,means,seconds,masses):
    k,d=fit['frame'].shape[:2]
    basis=helmert(d-1,full=False)
    mean_projection=np.einsum('ki,ki->k',fit['frame'][:,:,0],means)
    second_projection=np.array([np.diag(f.T@s@f)[1:] for f,s in zip(fit['frame'],seconds)])
    v=np.r_[fit['kappa'],(fit['lambda']@basis.T).ravel()]
    linear=np.zeros((k*(d-1),len(v)))
    linear[:,0]=(1-numerical.MARGIN)
    for j in range(k):linear[j*(d-1):(j+1)*(d-1),1+j*(d-2):1+(j+1)*(d-2)]=-2*basis.T
    args=(mean_projection,second_projection,masses,basis)
    before=shape_objective(v,*args)[0]
    bounds=Bounds(np.r_[0.,np.full(len(v)-1,-np.inf)],np.r_[baseline.KAPPA_MAX,np.full(len(v)-1,np.inf)])
    result=minimize(shape_objective,v,args=args,jac=True,method='SLSQP',bounds=bounds,
        constraints=LinearConstraint(linear,0,np.inf),
        options={'ftol':SHAPE_FTOL,'maxiter':INNER_MAXITER})
    feasible=np.min(linear@result.x)>=-1e-8
    improved=np.isfinite(result.fun) and result.fun<=before+1e-11 and feasible
    if improved:
        fit['kappa']=float(result.x[0]);fit['lambda']=result.x[1:].reshape(k,d-2)@basis
    return dict(nit=int(result.nit),nfev=int(result.nfev),success=bool(result.success),
                improved=bool(improved and result.fun<before-1e-12),
                message=str(result.message),change=float(before-result.fun))


def constraints(fit):
    frames=fit['frame'];d=frames.shape[1]
    np.testing.assert_allclose(np.einsum('kji,kjl->kil',frames,frames),
                               np.broadcast_to(np.eye(d),frames.shape),atol=2e-9)
    for lam in fit['lambda']:numerical.validate(fit['kappa'],lam)
    return True


def fit_kent(x,linked,responsibility=None,warm=None,max_iter=baseline.MAX_ITER):
    y,u=baseline.polar(x);n,d=u.shape
    if warm is None:
        vfit=baseline.cylinder_m_step(y,u,responsibility,linked)
        second=np.einsum('nk,ni,nj->kij',responsibility,u,u)/responsibility.sum(0)[:,None,None]
        fit=from_vmf(vfit,second)
    else:fit=copy.deepcopy(warm)
    if not linked:fit['coef'][:,1:]=0
    refresh(fit)
    r,ll=e_step(y,u,fit)
    history=[float(ll.mean())];times=[0.];inner=[]
    began=time.perf_counter();converged=False;status='nonconverged'
    for iteration in range(1,max_iter+1):
        old=copy.deepcopy(fit)
        mass=r.sum(axis=0)
        if mass.min()<baseline.MIN_EFFECTIVE:raise FloatingPointError('Collapsed Kent component')
        means=r.T@u/mass[:,None]
        seconds=np.einsum('nk,ni,nj->kij',r,u,u,optimize=True)/mass[:,None,None]
        fit['weights']=mass/n
        design=np.column_stack([np.ones(n),u])
        for j,w in enumerate(r.T):
            if linked:fit['coef'][j]=baseline.weighted_regression(design,y,w)
            else:fit['coef'][j,0]=w@y/mass[j]
        residual=y[:,None]-design@fit['coef'].T
        fit['variance']=max(float(np.sum(r*residual**2)/n),baseline.VAR_FLOOR)
        frame_diagnostics=[]
        for j in range(len(mass)):
            fit['frame'][j],diagnostic=optimize_frame(fit['frame'][j],means[j],seconds[j],fit['kappa'],fit['lambda'][j])
            frame_diagnostics.append(diagnostic)
        shape=optimize_shape(fit,means,seconds,mass/n)
        refresh(fit)
        r_new,ll_new=e_step(y,u,fit)
        gain=float(ll_new.mean())-history[-1]
        if gain < -1e-8:
            refresh(fit,tol=1e-12);refresh(old,tol=1e-12)
            old_ll=e_step(y,u,old)[1].mean();r_new,ll_new=e_step(y,u,fit)
            gain=float(ll_new.mean()-old_ll)
            if gain < -1e-8:raise FloatingPointError(f'Decreasing GEM likelihood {gain}')
        if not np.isfinite(gain):raise FloatingPointError('Nonfinite GEM likelihood')
        history.append(float(ll_new.mean()));times.append(time.perf_counter()-began)
        inner.append(dict(iteration=iteration,shape=shape,frames=frame_diagnostics))
        r=r_new
        if abs(gain)<baseline.TOL:
            # Failed internal solvers are not silently called outer convergence.
            bad_frame=any(not v['success'] and v['gradient']>1e-5 for v in frame_diagnostics)
            if shape['success'] and not bad_frame:
                converged=True;status='converged';break
            if not shape['improved'] and not any(v['improved'] for v in frame_diagnostics):
                status='optimizer_stalled';break
    constraints(fit)
    boundary=(fit['variance']<=baseline.VAR_FLOOR*(1+1e-6) or
              fit['kappa']>=baseline.KAPPA_MAX*(1-1e-6) or
              (fit['kappa']>0 and 2*fit['lambda'].max()>=(1-numerical.MARGIN-1e-7)*fit['kappa']))
    reasons=[]
    if fit['variance']<=baseline.VAR_FLOOR*(1+1e-6):reasons.append('radial_variance_floor')
    if fit['kappa']>=baseline.KAPPA_MAX*(1-1e-6):reasons.append('concentration_cap')
    if fit['kappa']>0 and 2*fit['lambda'].max()>=(1-numerical.MARGIN-1e-7)*fit['kappa']:reasons.append('unimodality_margin')
    fit.update(history=np.array(history),times=np.array(times),inner=inner,iterations=iteration,
               boundary_reasons=reasons,
               converged=converged,boundary=bool(boundary),status='boundary' if boundary else status)
    return fit


def candidate(x,linked,initializations,vmf=None,unlinked=None,progress=print):
    numerical.reset_stats();began=time.perf_counter()
    rows=[];accepted=[];rejected=[];k=initializations[0].shape[1]
    for start in range(baseline.STARTS):
        warm=None;origin='matched initialization'
        if start==8 and linked and unlinked is not None:
            warm=copy.deepcopy(unlinked);warm['coef'][:,1:]=0;origin='unlinked Kent'
        if start==9 and vmf is not None:
            warm=from_vmf(vmf);origin='vMF zero anisotropy'
        started=time.perf_counter();before_stats=dict(numerical.STATS)
        row=dict(start=start,origin=origin,accepted=False,status='error',iterations=0)
        try:
            fit=fit_kent(x,linked,responsibility=initializations[start] if warm is None else None,warm=warm)
            row.update(status=fit['status'],iterations=fit['iterations'],
                       train_mean_loglik=float(score(x,fit).mean()),
                       accepted=bool(fit['converged'] and not fit['boundary']),
                       history=fit['history'],times=fit['times'],inner=fit['inner'],
                       converged=fit['converged'],boundary_reasons=fit['boundary_reasons'],
                       kappa=fit['kappa'],variance=fit['variance'],
                       max_ovalness=0. if fit['kappa']==0 else float(2*fit['lambda'].max()/fit['kappa']))
            if row['accepted']:accepted.append((row['train_mean_loglik'],fit,start))
            else:rejected.append((row['train_mean_loglik'],fit,start))
        except (ValueError,FloatingPointError,np.linalg.LinAlgError,AssertionError) as err:
            row['status']=f'{type(err).__name__}: {err}'
        row['elapsed_seconds']=time.perf_counter()-started
        row['normalizer']={key:numerical.STATS[key]-before_stats[key] for key in ('calls','seconds','nodes','failures')}
        rows.append(row)
        progress(f"Kent {'linked' if linked else 'unlinked'} K={k} start={start}: {row['status']} "
                 f"({row['elapsed_seconds']:.1f}s, {row['iterations']} iterations)",flush=True)
    record=dict(fit=None,diagnostics=rows,elapsed_seconds=time.perf_counter()-began,
                normalizer_stats=dict(numerical.STATS))
    if accepted:
        ll,fit,start=max(accepted,key=lambda v:v[0])
        record.update(fit=fit,train_mean_loglik=ll,selected_start=start)
    if rejected:
        ll,fit,start=max(rejected,key=lambda v:v[0])
        record.update(best_rejected_fit=fit,best_rejected_start=start,
                      best_rejected_train_loglik=ll)
    return record
