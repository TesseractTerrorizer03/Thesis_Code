import unittest,inspect
import numpy as np
from scipy.stats import vonmises_fisher
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
import size_direction_sharing as s
import size_direction_experiment as b

class SharingChecks(unittest.TestCase):
    def test_counts(self):
        for d in (9,50):
            for k in (1,6,20):
                for linked in (False,True):
                    old=b.parameter_count(b.Spec('Linked' if linked else 'Unlinked',k),d)
                    self.assertEqual(s.count(k,d,linked,'shared'),old)
                    for mode in s.MODES:
                        self.assertEqual(s.count(k,d,linked,mode),old+(k-1)*(2 if mode=='both_specific' else 1))
    def test_normalizers(self):
        for d in (9,50):
            kap=np.array([0.,.001,2.,100.,9999.])
            np.testing.assert_allclose(s.normalizer(kap,d),[b.log_vmf_normalizer(k,d) for k in kap],atol=1e-10)
    def test_shared_equivalence_and_k1(self):
        rng=np.random.default_rng(23);x=rng.normal(size=(400,9))*.01;y,u=b.polar(x)
        r=rng.dirichlet([2,2,2],len(x))
        with threadpool_limits(limits=1):
            for linked in (False,True):
                old=b.cylinder_m_step(y,u,r,linked);new=s.m_step(y,u,r,linked,'shared')
                np.testing.assert_allclose(s.log_components(y,u,new),b.cylinder_component_logpdf(y,u,old),atol=1e-10)
                f=s.m_step(y,u,np.ones((len(x),1)),linked,'shared')
                for mode in s.MODES:
                    g=s.m_step(y,u,np.ones((len(x),1)),linked,mode)
                    np.testing.assert_allclose(s.log_components(y,u,f),s.log_components(y,u,g),atol=1e-10)
    def test_synthetic_recovery_and_monotonicity(self):
        rng=np.random.default_rng(410);n=5000;d=9;labels=rng.integers(2,size=n)
        mu=np.vstack([np.eye(d)[0],-np.eye(d)[0]]);u=np.zeros((n,d))
        for k in range(2):u[labels==k]=vonmises_fisher(mu[k],[8,35][k]).rvs(size=(labels==k).sum(),random_state=rng)
        beta=np.zeros((2,d));beta[:,1]=[.6,-.5]
        y=np.array([-3.,-4.])[labels]+(u*beta[labels]).sum(axis=1)+rng.normal(size=n)*np.array([.18,.5])[labels]
        with threadpool_limits(limits=1):
            for mode in ('shared',)+s.MODES:
                un=s.fit(y,u,False,mode,responsibilities=np.eye(2)[labels])
                fit=s.fit(y,u,True,mode,warm=un)
                self.assertTrue(fit['converged']);self.assertFalse(fit['boundary'])
                self.assertGreaterEqual(np.diff(fit['history']).min(),-1e-8)
                self.assertGreaterEqual(fit['history'][-1],un['history'][-1]-1e-8)
                logs=s.log_components(y,u,fit);q=np.exp(logs-logsumexp(logs,axis=1,keepdims=True))
                np.testing.assert_allclose(q.sum(axis=1),1,atol=1e-12)
                if mode=='both_specific':
                    np.testing.assert_allclose(fit['variance'],np.array([.18,.5])**2,rtol=.12)
                    np.testing.assert_allclose(fit['kappa'],[8,35],rtol=.12)
                    pred=fit['coef'][labels,0]+(u*fit['coef'][labels,1:]).sum(axis=1)
                    truth=np.array([-3.,-4.])[labels]+(u*beta[labels]).sum(axis=1)
                    self.assertLess(np.sqrt(np.mean((pred-truth)**2)),.05)
    def test_selection_api_excludes_test(self):
        self.assertNotIn('test',inspect.signature(s.search_group).parameters)
        self.assertNotIn('test',inspect.signature(s.candidate).parameters)
        self.assertNotIn('test',inspect.getsource(s.freeze).split('def freeze',1)[1].split('payload=',1)[0])
if __name__=='__main__':unittest.main()
