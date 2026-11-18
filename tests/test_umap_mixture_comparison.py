from pathlib import Path
import sys,unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import umap_mixture_comparison as m
import umap_mixture_report as report

class ComparisonTests(unittest.TestCase):
    def test_reference_bootstrap_adapter_accepts_scalar_daily_scores(self):
        low,high=m.score_interval(np.full(80,.12),20)
        self.assertAlmostEqual(low,.12);self.assertAlmostEqual(high,.12)

    def test_crps_point_mass_and_gaussian_reference(self):
        np.testing.assert_allclose(report.empirical_crps(np.zeros(5),[1.,-2.]),[1.,2.])
        rng=np.random.default_rng(701)
        expected=(np.sqrt(2)-1)/np.sqrt(np.pi)
        self.assertAlmostEqual(report.empirical_crps(rng.normal(size=200000),[0.])[0],expected,delta=.004)

    def test_gaussian_magnitude_sampling_matches_log_chi_mean(self):
        from types import SimpleNamespace
        from scipy.special import digamma
        fit=SimpleNamespace(weights_=np.ones(1),means_=np.zeros((1,3)),covariances_=np.ones((1,3)),covariance_type='diag')
        draws=report.sample_log_magnitude({'spec':{'family':'GMM'},'fit':fit},50000,711)
        expected=.5*(digamma(1.5)+np.log(2))
        self.assertAlmostEqual(draws.mean(),expected,delta=.006)

    def test_linked_magnitude_sampling_matches_directional_expectation(self):
        fit=dict(weights=np.ones(1),mu=np.array([[1.,0.,0.]]),variance=.04,kappa=10.,coef=np.array([[-3.,.3,0.,0.]]))
        draws=report.sample_log_magnitude({'spec':{'family':'Linked'},'fit':fit},50000,713)
        expected=-3+.3*(1/np.tanh(10.)-.1)
        self.assertAlmostEqual(draws.mean(),expected,delta=.006)

    def test_complete_search_group_ignores_test_values(self):
        rng=np.random.default_rng(940)
        x=rng.normal(size=(500,3))*.01
        data=dict(train=x[:300],validation=x[300:400],test=x[400:])
        with patch.object(m.b,'STARTS',1),threadpool_limits(limits=1):
            first=m.search_worker('synthetic/umap',data,1)
            data['test']=np.full_like(data['test'],np.nan)
            second=m.search_worker('synthetic/umap',data,1)
        self.assertEqual(len(first),14)
        self.assertTrue(all(r['status']=='ok' for r in first))
        np.testing.assert_array_equal([r['validation_mean_loglik'] for r in first],
                                      [r['validation_mean_loglik'] for r in second])

    def test_scalar_matrix_toggles_are_equivalent(self):
        fit={'variance':np.array([.2,.4])}
        a=m.matrices(None,fit,'Linked','diag');b=m.matrices(None,fit,'Linked','full')
        np.testing.assert_array_equal(a,b);self.assertEqual(a.shape,(2,1,1))
    def test_counts_match_reference(self):
        for d in [3,9,50]:
            for family in ['GMM','Unlinked','Linked']:
                for mode in m.MODES:
                    p=m.parameters(family,4,d,'diag',mode)
                    self.assertGreater(p,0)
                    if family!='GMM':self.assertEqual(p,m.parameters(family,4,d,'full',mode))
    def test_selection_uses_validation_and_deduplicates_covariance_aliases(self):
        rows=[]
        for family in ['GMM','Unlinked','Linked']:
            for k in [1,2]:
                for covariance in ['diag','full'] if family=='GMM' else ['scalar']:
                    rows.append(dict(case='fx/raw',family=family,k=k,covariance=covariance,status='ok',
                        validation_mean_loglik=k,test_mean_loglik=-k,parameters=20*k,sharing='shared',regularization=0.,candidate=f'{family}{k}{covariance}'))
        chosen=m.select(pd.DataFrame(rows))
        self.assertEqual(len(chosen),6);self.assertTrue(chosen.k.eq(2).all())
        for family in ['Unlinked','Linked']:
            self.assertEqual(chosen[chosen.family.eq(family)].candidate.nunique(),1)
    def test_latent_cylinder_density_and_posteriors_are_finite(self):
        rng=np.random.default_rng(45);x=rng.normal(size=(400,3))*.01
        y,u=m.b.polar(x)
        with threadpool_limits(limits=1):
            fit=m.b.fit_cylinder(y,u,False,responsibilities=np.ones((400,1)))
        record={'spec':{'family':'Unlinked'},'fit':fit}
        self.assertTrue(np.isfinite(m.score(x,record)).all())
        np.testing.assert_allclose(m.b.cylinder_responsibilities(y,u,fit).sum(axis=1),1.)

if __name__=='__main__':unittest.main()
