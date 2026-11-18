"""Scientific invariants and an actual frozen-transform UMAP integration check."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import tempfile
import sys
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.linalg import helmert
from sklearn.metrics import pairwise_distances
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import spherical_umap as s


def fixture(n=800):
    rng=np.random.default_rng(508)
    index=pd.bdate_range('2000-01-03',periods=n)
    raw=pd.DataFrame(rng.normal(0,.01,(n,5)),index=index,columns=list('ABCDE'))
    return dict(raw=raw,assets=raw.copy(),basis=None,
                validation_start=index[int(.6*n)],test_start=index[int(.8*n)])


class SphericalUMAPTests(unittest.TestCase):
    def test_concurrent_cache_writes_publish_only_complete_arrays(self):
        s.CACHE.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=s.CACHE) as directory:
            path=Path(directory)/'shared.joblib'
            s.joblib.dump(np.zeros(250000),path,compress=0)
            def writer(label):
                for _ in range(6):s.joblib.dump(np.full(250000,label,dtype=float),path,compress=0)
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures=[pool.submit(writer,label) for label in [1,2,3]]
                while not all(f.done() for f in futures):
                    array=s.joblib.load(path)
                    self.assertEqual(array.shape,(250000,))
                    self.assertTrue(np.all(array==array[0]))
                for future in futures:future.result()
            self.assertFalse(list(Path(directory).glob('*.tmp')))

    def test_spherical_distance_mapping_gradient_and_wrapping(self):
        rng=np.random.default_rng(85)
        for _ in range(60):
            a,b=rng.uniform(-8,8,(2,2))
            distance,gradient=s.sphere_metric(a,b)
            xyz=s.to_xyz(np.array([a,b]))
            expected=s.angular_distances(xyz[:1],xyz[1:])[0]
            self.assertAlmostEqual(distance,expected,places=10)
            epsilon=1e-6
            finite=np.array([(s.sphere_metric(a+epsilon*np.eye(2)[j],b)[0]-s.sphere_metric(a-epsilon*np.eye(2)[j],b)[0])/(2*epsilon) for j in range(2)])
            np.testing.assert_allclose(gradient,finite,atol=2e-5,rtol=2e-5)
            wrapped=a+2*np.pi*np.array([2,-3])
            self.assertAlmostEqual(s.sphere_metric(wrapped,b)[0],distance,places=10)
        for b,d in [(np.array([0.,0.]),0.),(np.array([0.,np.pi]),np.pi),(np.array([np.pi/2,0.]),np.pi/2)]:
            result,gradient=s.sphere_metric(np.array([0.,0.]),b)
            self.assertAlmostEqual(result,d,places=7)
            self.assertTrue(np.isfinite(gradient).all())

    def test_causal_features_and_exact_formulas(self):
        market=fixture()
        raw=market['raw'];original=s.trend_features(raw)
        changed=raw.copy();changed.iloc[600:]*=20
        pd.testing.assert_frame_equal(original.iloc[:600],s.trend_features(changed).iloc[:600])
        pd.testing.assert_frame_equal(original.iloc[:600],s.trend_features(raw.iloc[:600]))
        np.testing.assert_allclose(original['momentum_63::A'].iloc[62:],raw.A.rolling(63).sum().iloc[62:])
        prices=np.exp(raw.A.cumsum())
        np.testing.assert_allclose(original['gap_200::A'].iloc[199:],(prices/prices.rolling(200).mean()-1).iloc[199:])
        self.assertEqual(original.iloc[:199]['gap_200::A'].notna().sum(),0)
        self.assertTrue(np.isfinite(original.iloc[199:]).all().all())

    def test_fx_numeraire_invariance_and_reconstruction(self):
        rng=np.random.default_rng(81)
        loglevels=np.cumsum(rng.normal(0,.01,(700,10)),axis=0)
        index=pd.bdate_range('2000-01-03',periods=700)
        # Foreign units per EUR = exp(EUR log value - foreign log value).
        quotes=pd.DataFrame(np.exp(loglevels[:,0,None]-loglevels[:,1:]),index=index,columns=s.CURRENCIES[1:])
        contrast,strength,basis=s.fx_contrasts(quotes)
        raw=np.diff(loglevels,axis=0)
        expected=raw-raw.mean(axis=1,keepdims=True)
        np.testing.assert_allclose(strength.iloc[1:],expected,atol=1e-15)
        np.testing.assert_allclose(contrast.iloc[1:].to_numpy()@basis,expected,atol=1e-15)
        rebased=raw-raw[:,3,None]
        np.testing.assert_allclose((rebased-rebased.mean(axis=1,keepdims=True))@basis.T,contrast.iloc[1:],atol=1e-15)
        shifted=pd.DataFrame(rebased-rebased.mean(axis=1,keepdims=True),index=index[1:],columns=s.CURRENCIES)
        pd.testing.assert_frame_equal(s.trend_features(strength.iloc[1:]),s.trend_features(shifted),atol=1e-12,rtol=1e-10)

    def test_split_boundaries_scaling_and_future_isolation(self):
        market=fixture();data=s.datasets({'equities':market})['equities/trend']
        prepared=s.prepare(data,'euclidean')
        self.assertEqual(prepared['train_dates'][0],market['raw'].index[199])
        self.assertEqual(prepared['query_dates'][0],market['validation_start'])
        self.assertTrue((prepared['query_dates']<market['test_start']).all())
        np.testing.assert_allclose(prepared['scaler'].mean_,data['frame'].loc[data['frame'].index<market['validation_start']].mean())
        changed=dict(data,frame=data['frame'].copy())
        changed['frame'].loc[changed['frame'].index>=market['test_start']]*=1000
        again=s.prepare(changed,'euclidean')
        np.testing.assert_array_equal(again['x'],prepared['x']);np.testing.assert_array_equal(again['y'],prepared['y'])
        changed['frame'].loc[changed['frame'].index>=market['validation_start']]*=5
        np.testing.assert_array_equal(s.prepare(changed,'euclidean')['x'],prepared['x'])

    def test_zero_direction_rejected_euclidean_valid(self):
        data=s.datasets({'equities':fixture()})['equities/returns']
        data['frame'].iloc[10]=0
        with self.assertRaisesRegex(ValueError,'zero-vector'):s.prepare(data,'cosine')
        result=s.prepare(data,'euclidean')
        self.assertEqual(result['train_norm'][10],0.)

    def test_inductive_neighbor_scores_and_trustworthiness(self):
        data=s.datasets({'equities':fixture()})['equities/returns']
        prepared=s.prepare(data,'euclidean');config=dict(s.DEFAULT_CONFIG,pair_samples=100)
        ref=s.reference(prepared,config)
        scores,summary=s.score_embedding(prepared,prepared['x'],prepared['y'],'PCA3',ref,config)
        np.testing.assert_allclose(scores,1.)
        self.assertAlmostEqual(summary['distance_rank_correlation'],1.)
        self.assertEqual(ref['ranks'].shape,(160,480))

    def test_phase_preserves_covariance_and_spectrum(self):
        raw=fixture()['raw']
        generated,error=s.phase_surrogate(raw,'phase',np.random.default_rng(42))
        np.testing.assert_allclose(generated.mean(),raw.mean(),atol=1e-14)
        np.testing.assert_allclose(np.cov(generated.T),np.cov(raw.T),atol=1e-14)
        self.assertLess(error,1e-6)
        envelope,error=s.phase_surrogate(raw,'volatility_envelope',np.random.default_rng(42))
        self.assertTrue(np.isfinite(envelope).all().all());self.assertGreater(error,0.)

    def test_surrogate_segment_isolation_and_feature_rebuilding(self):
        market=fixture();config=dict(s.DEFAULT_CONFIG)
        surrogate,_=s.surrogate_market(market,'phase',0,config)
        altered=dict(market,raw=market['raw'].copy())
        altered['raw'].loc[altered['raw'].index>=market['validation_start']]*=100
        again,_=s.surrogate_market(altered,'phase',0,config)
        pd.testing.assert_frame_equal(surrogate['raw'].loc[:market['validation_start']].iloc[:-1],again['raw'].loc[:market['validation_start']].iloc[:-1])
        self.assertLess(surrogate['raw'].index[-1],market['test_start'])
        observed=s.datasets({'equities':surrogate})['equities/trend']['frame']
        pd.testing.assert_frame_equal(observed,s.trend_features(surrogate['assets']).iloc[199:])

    def test_fx_surrogates_reconstruct_currency_contrasts(self):
        market=fixture();rng=np.random.default_rng(5)
        raw=pd.DataFrame(rng.normal(0,.01,(800,9)),index=market['raw'].index)
        basis=helmert(10,full=False)
        market.update(raw=raw,basis=basis,assets=pd.DataFrame(raw.to_numpy()@basis,index=raw.index,columns=s.CURRENCIES))
        surrogate,_=s.surrogate_market(market,'phase',0,s.DEFAULT_CONFIG)
        np.testing.assert_allclose(surrogate['assets'].sum(axis=1),0.,atol=1e-15)
        np.testing.assert_allclose(surrogate['assets'].to_numpy()@basis.T,surrogate['raw'],atol=1e-15)

    def test_actual_umap_frozen_transform_and_unit_sphere(self):
        config=dict(s.DEFAULT_CONFIG,neighbors=[15],epochs=30)
        data=s.datasets({'equities':fixture(600)})['equities/returns']
        prepared=s.prepare(data,'cosine')
        with threadpool_limits(limits=1):
            training,query,reducer=s.fit_embedding(prepared,'Spherical UMAP',15,42,config)
            before=reducer.embedding_.copy()
            transformed=reducer.transform(prepared['y'][:10])
        np.testing.assert_array_equal(before,reducer.embedding_)
        np.testing.assert_allclose(np.linalg.norm(s.to_xyz(query),axis=1),1.,atol=1e-6)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertEqual(training.shape,(360,2));self.assertEqual(query.shape,(120,2))

    def test_paired_block_bootstrap_and_finite_input(self):
        result=s.block_interval(np.ones(100)*.025,20,100)
        self.assertAlmostEqual(result['lower'],.025);self.assertAlmostEqual(result['upper'],.025)
        with self.assertRaises(ValueError):s.block_interval([np.nan],20,100)

    def test_complete_surrogate_search_integration(self):
        config=dict(s.DEFAULT_CONFIG,neighbors=[15],epochs=30,pair_samples=100)
        market=fixture()
        for family in config['surrogate_families']:
            row=s._control_worker('equities',market,family,0,config)
            self.assertEqual(row['status'],'ok',row.get('error'))
            self.assertEqual(len(row['trials']),8)
            self.assertEqual({r['track'] for r in row['trials']},{'equities/returns','equities/trend'})
            self.assertEqual({r['metric'] for r in row['trials']},{'cosine','euclidean'})
            self.assertTrue(np.isfinite(row['maximum_gain']))

    def test_surrogate_unresolved_counts_as_exceedance(self):
        comparisons=pd.DataFrame([dict(track='equities/returns',metric='cosine',stage=stage,
                    baseline='Planar UMAP',score='recall_15',block=20,mean=.1,lower=.05,upper=.15)
                    for stage in ['validation','retrospective_test']])
        controls=[dict(market='equities',family=family,repeat=i,status='unresolved' if i==0 else 'ok',maximum_gain=None if i==0 else 0.)
                  for family in s.DEFAULT_CONFIG['surrogate_families'] for i in range(19)]
        cards=s.scorecards({'comparisons':comparisons},controls,s.DEFAULT_CONFIG)
        self.assertEqual(cards.iloc[0].tail_phase,.1)
        self.assertEqual(cards.iloc[0].conclusion,'Spherical improvement')


if __name__=='__main__':unittest.main()
