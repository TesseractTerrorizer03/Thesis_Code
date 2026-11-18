"""Scientific and protocol checks for the 2000-onward experiment."""
import inspect
import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from scipy.linalg import null_space
from scipy.stats import vonmises_fisher, multivariate_normal
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
import size_direction_experiment as e


class ExperimentChecks(unittest.TestCase):
    def test_data_splits_include_crisis_in_training_and_no_scaling(self):
        data = e.load_data()
        self.assertEqual(data['equities'].frame.shape[1], 50)
        self.assertEqual(data['fx'].frame.shape[1], 9)
        for market in data.values():
            self.assertEqual(market.a, int(.6*len(market.frame)))
            self.assertEqual(market.b, int(.8*len(market.frame)))
            self.assertIn(pd.Timestamp('2008-09-15'), market.frame.index[:market.a])
            self.assertLess(market.frame.index[market.a-1], market.frame.index[market.a])
        prices = pd.read_csv(e.DATA/'equity_adjusted_prices.csv', index_col=0, parse_dates=True)
        expected = np.log(prices).diff().loc[e.START:e.END, e.STOCKS]
        pd.testing.assert_frame_equal(data['equities'].frame, expected)

    def test_polar_and_jacobian(self):
        rng = np.random.default_rng(830)
        for d in (9, 50):
            x = rng.normal(size=(30, d))*.01
            y, u = e.polar(x)
            np.testing.assert_allclose(np.exp(y[:, None])*u, x, atol=1e-14)
            np.testing.assert_allclose(np.linalg.norm(u, axis=1), 1, atol=1e-14)
            direction = u[0]
            tangent = null_space(direction[None])
            z = np.r_[y[0], np.zeros(d-1)]
            def inverse(v):
                w = direction+tangent @ v[1:]
                return np.exp(v[0])*w/np.linalg.norm(w)
            h = 1e-5
            jac = np.column_stack([(inverse(z+h*v)-inverse(z-h*v))/(2*h) for v in np.eye(d)])
            self.assertAlmostEqual(np.linalg.slogdet(jac)[1], d*y[0], places=6)
        with self.assertRaises(ValueError):
            e.polar(np.zeros((2, 9)))

    def test_vmf_density_and_concentration(self):
        for d in (9, 50):
            axis = np.eye(d)[0]
            for kappa in (.01, 3, 100):
                expected = vonmises_fisher(axis, kappa).logpdf(np.eye(d))
                actual = e.log_vmf_normalizer(kappa, d)+kappa*(np.eye(d) @ axis)
                np.testing.assert_allclose(actual, expected, atol=1e-10)
                nu = d/2-1
                r = e.ive(nu+1, kappa)/e.ive(nu, kappa)
                self.assertAlmostEqual(e.concentration_from_resultant(r, d), kappa, places=6)
        self.assertEqual(e.concentration_from_resultant(0, 50), 0)

    def test_synthetic_link_and_null(self):
        rng = np.random.default_rng(703)
        u = vonmises_fisher(np.eye(9)[0], 3).rvs(size=3200, random_state=rng)
        noise = rng.normal(0, .25, len(u))
        beta = np.array([.8, -.4, .3, 0, 0, 0, 0, 0, 0])
        with threadpool_limits(limits=1):
            for signal in (False, True):
                y = -3+noise+(u @ beta if signal else 0)
                product = e.fit_cylinder(y[:2000], u[:2000], False,
                                         responsibilities=np.ones((2000, 1)), min_effective=1)
                linked = e.fit_cylinder(y[:2000], u[:2000], True, warm=product, min_effective=1)
                for fit in (product, linked):
                    self.assertTrue(fit['converged'])
                    self.assertFalse(fit['boundary'])
                    self.assertGreaterEqual(np.diff(fit['history']).min(), -1e-8)
                    np.testing.assert_allclose(e.cylinder_responsibilities(y[2000:],u[2000:],fit).sum(axis=1),1)
                np.testing.assert_array_equal(product['coef'][:, 1:], 0)
                np.testing.assert_allclose(linked['coef'][0,1:], beta if signal else np.zeros(9), atol=.07)
                gain = (logsumexp(e.cylinder_component_logpdf(y[2000:],u[2000:],linked),axis=1)
                        -logsumexp(e.cylinder_component_logpdf(y[2000:],u[2000:],product),axis=1)).mean()
                self.assertGreater(gain,.2) if signal else self.assertLess(abs(gain),.02)

    def test_weighted_regression_matches_reference_and_rank_deficient_fallback(self):
        rng=np.random.default_rng(302)
        a=np.column_stack([np.ones(800),rng.normal(size=(800,50))])
        y=rng.normal(size=800)
        weights=rng.uniform(.01,1,size=800)
        beta,method=e.weighted_regression(a,y,weights,return_method=True)
        root=np.sqrt(weights)
        reference=np.linalg.lstsq(a*root[:,None],y*root,rcond=None)[0]
        self.assertEqual(method,'cholesky')
        np.testing.assert_allclose(beta,reference,atol=1e-11,rtol=1e-10)
        a[:,-1]=a[:,-2]
        beta,method=e.weighted_regression(a,y,weights,return_method=True)
        reference=np.linalg.lstsq(a*root[:,None],y*root,rcond=None)[0]
        self.assertEqual(method,'svd')
        np.testing.assert_allclose(beta,reference,atol=1e-12)

    def test_two_component_joint_mixture_recovers_a_real_link(self):
        rng=np.random.default_rng(905)
        n,d=2400,9
        labels=rng.choice(2,size=n,p=[.55,.45])
        axes=np.vstack([np.eye(d)[0],-np.eye(d)[0]])
        u=np.empty((n,d))
        for k in range(2):
            use=labels==k
            u[use]=vonmises_fisher(axes[k],8.).rvs(size=use.sum(),random_state=rng)
        coefficients=np.array([[.9,-.4,.3,0,0,0,0,0,0],[-.7,.5,0,0,0,0,0,0,0]])
        y=np.array([-3.,-4.])[labels]+np.sum(coefficients[labels]*u,axis=1)+rng.normal(0,.2,n)
        with threadpool_limits(limits=1):
            product=e.fit_cylinder(y[:1600],u[:1600],False,
                responsibilities=np.eye(2)[labels[:1600]])
            linked=e.fit_cylinder(y[:1600],u[:1600],True,warm=product)
        self.assertTrue(product['converged'] and linked['converged'])
        self.assertGreaterEqual(linked['history'][-1],product['history'][-1])
        q=e.cylinder_responsibilities(y[1600:],u[1600:],linked)
        np.testing.assert_allclose(q.sum(axis=1),1,atol=1e-12)
        self.assertGreater(np.mean(q.argmax(axis=1)==labels[1600:]),.95)
        gain=(logsumexp(e.cylinder_component_logpdf(y[1600:],u[1600:],linked),axis=1)
              -logsumexp(e.cylinder_component_logpdf(y[1600:],u[1600:],product),axis=1)).mean()
        self.assertGreater(gain,.15)
        np.testing.assert_allclose(linked['coef'][:,1:],coefficients,atol=.12)

    def test_parameter_counts_and_budget_selection(self):
        self.assertEqual(e.parameter_count(e.Spec('GMM',3,'full',1e-8),8),134)
        self.assertEqual(e.parameter_count(e.Spec('Unlinked',3),8),28)
        self.assertEqual(e.parameter_count(e.Spec('Linked',3),8),52)
        for d in (9,50):
            rows=[]
            for order,spec in enumerate(e.all_specs()):
                rows.append(dict(dataset='synthetic',candidate=spec.key,**e.asdict(spec),
                    parameters=e.parameter_count(spec,d),configuration_order=order,
                    validation_mean_loglik=-abs(spec.k-6),search_signature='test'))
            selected=e.select_configurations(pd.DataFrame(rows),d)
            self.assertEqual(len(selected),15)
            for _,group in selected.loc[selected.view.str.startswith('budget')].groupby('view'):
                self.assertLessEqual(group.parameters.max()/group.parameters.min(),1.10)
            self.assertEqual(selected.loc[selected.view=='same_components','k'].nunique(),1)
            # A failed entire GMM K must not prevent feasible lower-K comparison.
            trials=pd.DataFrame(rows)
            trials.loc[(trials.family=='GMM') & (trials.k==6),'validation_mean_loglik']=np.nan
            feasible=e.select_configurations(trials,d)
            self.assertNotEqual(int(feasible.loc[feasible.view=='same_components','k'].iloc[0]),6)

    def test_selection_is_validation_only_and_ties(self):
        self.assertEqual(list(inspect.signature(e.run_search).parameters),
                          ['name','train','validation','output','progress','components'])
        self.assertNotIn('test',inspect.signature(e.select_configurations).parameters)
        candidates=pd.DataFrame([dict(validation_mean_loglik=1.,parameters=20,k=3,configuration_order=0),
                                 dict(validation_mean_loglik=1.-5e-7,parameters=10,k=2,configuration_order=1),
                                 dict(validation_mean_loglik=0.,parameters=1,k=1,configuration_order=2)])
        self.assertEqual(e.best_validation(candidates).parameters,10)

    def test_paired_bootstrap_and_gaussian_density(self):
        scores=np.column_stack([np.arange(71),np.arange(71)+.4])
        boot=e.bootstrap_means(scores,20,repetitions=100)
        np.testing.assert_allclose(boot[:,1]-boot[:,0],.4,atol=1e-12)
        rng=np.random.default_rng(24)
        x=rng.normal(size=(160,9))*.01
        spec=e.Spec('GMM',1,'full',1e-8)
        with threadpool_limits(limits=1):
            record=e.fit_candidate(x,spec,e.make_initializations(x,1))
        self.assertIsNotNone(record['fit'])
        model=record['fit']
        reference=multivariate_normal.logpdf(x,mean=model.means_[0],cov=model.covariances_[0])
        np.testing.assert_allclose(e.score_model(x,record),reference,atol=1e-9)

    def test_parallel_refit_preserves_deduplicated_architectures(self):
        rng=np.random.default_rng(83)
        x=rng.normal(size=(300,9))*.01
        specs=[e.Spec('GMM',1,'diag',1e-8),e.Spec('GMM',2,'diag',1e-8)]
        selected=pd.DataFrame([dict(**e.asdict(spec),candidate=spec.key) for spec in specs])
        with tempfile.TemporaryDirectory(prefix='sd_refit_test_') as folder:
            records=e.refit_selected('synthetic',x,selected,'test_freeze',Path(folder))
            self.assertEqual(set(records),{spec.key for spec in specs})
            for record in records.values():
                self.assertIsNotNone(record['fit'])
                self.assertTrue(np.isfinite(e.score_model(x,record)).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
