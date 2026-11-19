import unittest
import numpy as np
from threadpoolctl import threadpool_limits
import equity_experiment as e

class EquityChecks(unittest.TestCase):
    def test_normalization_and_gradient_s49(self):
        with threadpool_limits(limits=1):
            for k in (0.,1.,50.,1000.,10000.):
                a=e.kn.normalizer(k,np.zeros(49),force_numeric=True);b=e.kn.normalizer(k,np.zeros(49))
                np.testing.assert_allclose([a.logz,a.dkappa],[b.logz,b.dkappa],atol=2e-8,rtol=0)
            k=50.;lam=np.linspace(-15,15,49)
            a=e.kn.normalizer(k,lam);ref,_=e.kn.reference_normalizer(k,lam)
            self.assertLess(abs(a.logz-ref),2e-8)
            h=1e-4
            for j in (0,24,48):
                delta=np.eye(49)[j]*h
                derivative=(e.kn.normalizer(k,lam+delta,constrained=False).logz-e.kn.normalizer(k,lam-delta,constrained=False).logz)/(2*h)
                self.assertAlmostEqual(derivative,a.dlambda[j],places=7)

    def test_orientation_and_joint_limit(self):
        with threadpool_limits(limits=1):
            rng=np.random.default_rng(41);x=rng.normal(size=(200,50))*.01;y,u=e.b.polar(x)
            f=e.b.cylinder_m_step(y,u,np.ones((len(x),1)),True);kent=e.km.from_vmf(f)
            np.testing.assert_allclose(e.km.components(y,u,kent),e.b.cylinder_component_logpdf(y,u,f),atol=1e-10)
            np.testing.assert_allclose(np.exp(y[:,None])*u,x,atol=1e-14)
            args=(np.eye(50),u.mean(0),u.T@u/len(u),50.,np.linspace(-15,15,49))
            v=np.zeros(1225);_,gradient=e.km.orientation_objective(v,*args)
            for j in (0,48,500,1224):
                delta=np.eye(1,1225,j)[0]*1e-5
                ref=(e.km.orientation_objective(v+delta,*args)[0]-e.km.orientation_objective(v-delta,*args)[0])/2e-5
                self.assertAlmostEqual(ref,gradient[j],places=7)
            fit=e.km.fit_kent(x,True,responsibility=np.ones((len(x),1)),max_iter=2)
            self.assertTrue(np.all(np.diff(fit['history'])>=-1e-8))
            np.testing.assert_allclose(e.km.posterior(x,fit).sum(1),1,atol=1e-12)

    def test_counts_and_budgets(self):
        import pandas as pd
        rows=pd.DataFrame(e.specs(50));self.assertEqual(set(rows[rows.family=='GMM'].k),{1,3,5})
        for family,slope in [('vMF unlinked',51),('vMF linked',101),('Kent unlinked',1275),('Kent linked',1325)]:
            self.assertEqual(e.km.parameter_count(family,5,50),5*slope+1)
            for k in e.BASE:
                p=e.km.parameter_count('GMM',k,50,'full')
                self.assertTrue(rows[(rows.family==family)&rows.parameters.between(p/np.sqrt(1.1),p*np.sqrt(1.1))].shape[0]>0)
        self.assertEqual(e.STARTS,4)

if __name__=='__main__':unittest.main()
