"""Numerical acceptance gate; no market-model selection occurs here."""
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.linalg import helmert
from scipy.optimize._numdiff import approx_derivative
from scipy.special import i0e, logsumexp
from scipy.stats import vonmises_fisher
from threadpoolctl import threadpool_limits
import kent_numerics as num
import kent_models as model


class KentChecks(unittest.TestCase):
    def test_numeric_vmf_limit_and_derivatives(self):
        for k in (0.,.01,1.,10.,100.,1000.,10000.):
            actual=num.normalizer(k,np.zeros(8),force_numeric=True)
            reference=num.normalizer(k,np.zeros(8))
            self.assertLess(abs(actual.logz-reference.logz),1e-8)
            self.assertLess(abs(actual.dkappa-reference.dkappa),1e-9)
            np.testing.assert_allclose(actual.dlambda,reference.dlambda,atol=1e-9)
            self.assertLess(actual.error,1e-10)

    def test_normalization_independent_s8(self):
        for k,l in [(1,np.linspace(-.3,.3,8)),(10,np.linspace(-3,3,8)),
                    (100,np.linspace(-49,49,8)),(30,np.r_[-14.99998,np.zeros(6),14.99998])]:
            value=num.normalizer(k,l)
            reference,_=num.reference_normalizer(k,l)
            self.assertLess(abs(value.logz-reference),2e-8)

    def test_s2_direct_integral(self):
        for k,b in [(0.,0.),(1.,.4),(10.,3.),(100.,49.)]:
            actual=num.normalizer(k,np.array([b,-b]),force_numeric=True)
            integral=quad(lambda z:np.exp(k*(z-1)+b*(1-z*z))*i0e(b*(1-z*z)),
                          -1,1,epsabs=1e-12,epsrel=1e-12)[0]
            reference=np.log(2*np.pi*integral)+k
            self.assertLess(abs(actual.logz-reference),1e-8)

    def test_normalizer_gradient(self):
        for k in (2.,20.,200.):
            l=np.linspace(-.3*k,.3*k,8);v=np.r_[k,l]
            ref=approx_derivative(lambda p:num.normalizer(p[0],p[1:],constrained=False).logz,v).ravel()
            actual=num.normalizer(k,l)
            np.testing.assert_allclose(np.r_[actual.dkappa,actual.dlambda],ref,atol=2e-7,rtol=2e-6)

    def test_orientation_and_shared_shape_gradients(self):
        rng=np.random.default_rng(76);u=rng.normal(size=(400,9));u/=np.linalg.norm(u,axis=1)[:,None]
        frame=model.frame_from_mean(np.eye(9)[0]);mean=u.mean(0);second=u.T@u/len(u)
        v=rng.normal(0,.1,36);args=(frame,mean,second,10.,np.linspace(-3,3,8))
        _,actual=model.orientation_objective(v,*args)
        ref=approx_derivative(lambda p:model.orientation_objective(p,*args)[0],v).ravel()
        np.testing.assert_allclose(actual,ref,atol=1e-7)
        basis=helmert(8,full=False)
        args=(np.array([.4,.6]),np.ones((2,8))*.08,np.array([.3,.7]),basis)
        v=np.r_[10.,np.linspace(-1,1,14)]
        _,actual=model.shape_objective(v,*args)
        ref=approx_derivative(lambda p:model.shape_objective(p,*args)[0],v).ravel()
        np.testing.assert_allclose(actual,ref,atol=1e-7)

    def test_rotation_permutation_and_repeated_axes(self):
        rng=np.random.default_rng(81);u=rng.normal(size=(100,9));u/=np.linalg.norm(u,axis=1)[:,None]
        frame=np.linalg.qr(rng.normal(size=(9,9)))[0];rot=np.linalg.qr(rng.normal(size=(9,9)))[0]
        fit=dict(frame=frame[None],kappa=12.,**{'lambda':np.array([[-2,-2,0,0,0,0,2,2.]])})
        model.refresh(fit);original=model.directional_terms(u,fit)
        fit['frame']=(rot@frame)[None]
        np.testing.assert_allclose(original,model.directional_terms(u@rot.T,fit),atol=1e-11)
        perm=np.r_[0,1+rng.permutation(8)]
        fit['frame']=frame[:,perm][None];fit['lambda']=fit['lambda'][:,perm[1:]-1]
        model.refresh(fit)
        np.testing.assert_allclose(original,model.directional_terms(u,fit),atol=1e-11)
        model.constraints(fit)

    def test_vmf_joint_limit_jacobian_counts(self):
        rng=np.random.default_rng(55);x=rng.normal(size=(150,9))*.01;y,u=model.baseline.polar(x)
        vmf=model.baseline.cylinder_m_step(y,u,np.ones((len(x),1)),True)
        kent=model.from_vmf(vmf)
        np.testing.assert_allclose(model.components(y,u,kent),model.baseline.cylinder_component_logpdf(y,u,vmf),atol=1e-10)
        np.testing.assert_allclose(model.score(x,kent),logsumexp(model.components(y,u,kent),axis=1)-9*y,atol=1e-12)
        np.testing.assert_allclose(np.exp(y[:,None])*u,x,atol=1e-14)
        # Numerical Jacobian in local tangent coordinates.
        tangent=model.null_space(u[0][None]);z=np.r_[y[0],np.zeros(8)]
        def inverse(v):
            direction=u[0]+tangent@v[1:];return np.exp(v[0])*direction/np.linalg.norm(direction)
        jac=approx_derivative(inverse,z)
        self.assertAlmostEqual(np.linalg.slogdet(jac)[1],9*y[0],places=6)
        for k in model.GRID:
            for family,p in [('vMF unlinked',10*k+1),('vMF linked',19*k+1),
                             ('Kent unlinked',45*k+1),('Kent linked',54*k+1)]:
                self.assertEqual(model.parameter_count(family,k),p)

    def test_invalid_and_failed_quadrature(self):
        for k,l in [(-1,np.zeros(8)),(1,np.ones(8)),(1,np.r_[-1,np.zeros(6),1])]:
            with self.assertRaises(ValueError):num.normalizer(k,l)
        previous=num.MAX_NODES
        try:
            num.MAX_NODES=128
            with self.assertRaises(FloatingPointError):num.normalizer(10,np.linspace(-2,2,8))
        finally:num.MAX_NODES=previous
        with self.assertRaises(ValueError):model.baseline.polar(np.zeros((2,9)))

    def test_synthetic_isotropic_and_anisotropic_fit(self):
        rng=np.random.default_rng(203)
        with threadpool_limits(limits=1):
            for anisotropic in (False,True):
                proposal=vonmises_fisher(np.eye(9)[0],8.)
                lam=np.linspace(-3,3,8) if anisotropic else np.zeros(8)
                samples=[]
                while sum(len(v) for v in samples)<1000:
                    u=proposal.rvs(3000,random_state=rng)
                    use=np.log(rng.uniform(size=len(u)))<u[:,1:]**2@lam-lam.max()
                    samples.append(u[use])
                u=np.concatenate(samples)[:1000];y=-4+.5*u[:,1]+rng.normal(0,.3,len(u))
                x=np.exp(y[:,None])*u
                vmf=model.baseline.fit_cylinder(y,u,True,responsibilities=np.ones((len(x),1)))
                fit=model.fit_kent(x,True,warm=model.from_vmf(vmf),max_iter=80)
                self.assertTrue(fit['converged'],fit['status'])
                self.assertFalse(fit['boundary'])
                self.assertGreaterEqual(np.diff(fit['history']).min(),-1e-8)
                self.assertGreaterEqual(fit['history'][-1],vmf['history'][-1]-1e-8)
                np.testing.assert_allclose(model.posterior(x,fit).sum(1),1,atol=1e-12)
                if anisotropic:self.assertGreater(fit['history'][-1]-vmf['history'][-1],.03)

    def test_collapse_and_nonconvergence_are_explicit(self):
        rng=np.random.default_rng(32);x=rng.normal(size=(100,9))*.01
        r=np.zeros((len(x),2));r[:,0]=1
        with self.assertRaises(FloatingPointError):model.fit_kent(x,False,responsibility=r)
        fit=model.fit_kent(x,False,responsibility=np.ones((len(x),1)),max_iter=1)
        self.assertFalse(fit['converged'])


if __name__=='__main__':unittest.main(verbosity=2)
