"""End-to-end geometry controls; synthetic inputs never consume market holdout."""
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment import (make_candidates, chronological_splits, screen_candidates,
                        calibrate_search, attach_search_calibration,
                        compare_ellipsoid, final_evaluation)


def small_config():
    return dict(features=list('abc'), seed_subsets={'surface':list('abc')},
                subset_pool=list('abc'), search_mode='seeds', subset_sizes=[3],
                pca_dimensions=[None], max_candidates=10, scaling='none',
                n_splits=2, validation_size=100, gap=0, min_train_size=200,
                fit_max_points=300, sphere_max_nfev=150, ellipsoid_max_nfev=100,
                max_radius_rms=2., max_center_rms=1.5, max_resultant=.6,
                min_angular_span=.05, max_relative_shell_rmse=.15,
                ellipsoid_shortlist=1, ellipsoid_margin=.1,
                null_families=['phase', 'volatility_phase'], n_surrogates=19,
                volatility_window=20, alpha=.05, bootstrap_block=10,
                n_bootstrap=200, seed=42, progress_every=20)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = small_config()
        self.rng = np.random.default_rng(91)
        u = self.rng.normal(size=(1000, 3))
        u /= np.linalg.norm(u, axis=1)[:, None]
        self.unit = u
        self.index = pd.bdate_range('2000-01-01', periods=len(u))

    def frame(self, values):
        return pd.DataFrame(values, index=self.index, columns=list('abc'))

    def test_spherical_discovery_and_frozen_final_confirmation(self):
        frame = self.frame(self.unit*(2+self.rng.normal(0, .01, len(self.unit)))[:, None])
        development, holdout = frame.iloc[:800], frame.iloc[800:]
        candidates = make_candidates(self.config)
        splits, _ = chronological_splits(development, self.config)
        with threadpool_limits(limits=1):
            ranking, _, _ = screen_candidates(development, splits, candidates, self.config)
            nulls = calibrate_search(development, splits, candidates, self.config, progress=lambda _:None)
            ranking = attach_search_calibration(ranking, nulls)
            ranking, selected, _, _ = compare_ellipsoid(development, splits, candidates, ranking,
                                                        self.config, progress=lambda _:None)
            self.assertIsNotNone(selected)
            scores, summary = final_evaluation(development, holdout, candidates[0], self.config)
        self.assertTrue(summary['passes_confirmation'])
        self.assertEqual(len(scores), len(holdout))

    def test_planar_data_do_not_pass_sphere_fit_gate(self):
        x = self.rng.normal(size=(1000, 3))
        x[:, 2] *= .005
        development = self.frame(x).iloc[:800]
        candidates = make_candidates(self.config)
        splits, _ = chronological_splits(development, self.config)
        with threadpool_limits(limits=1):
            ranking, _, _ = screen_candidates(development, splits, candidates, self.config)
        self.assertFalse(bool(ranking.fit_ok.iloc[0]))

    def test_high_dimensional_cloud_does_not_pass_surrogate_calibration(self):
        # A full-dimensional cloud can look radially concentrated in high dimension.
        # It must not be called spherical just because its sphere/plane score is high.
        columns = [f'x{i}' for i in range(20)]
        frame = pd.DataFrame(self.rng.normal(size=(1000, 20)), index=self.index, columns=columns)
        config = dict(self.config, features=columns, subset_pool=columns,
                      seed_subsets={'cloud':columns})
        development = frame.iloc[:800]
        candidates = make_candidates(config)
        splits, _ = chronological_splits(development, config)
        with threadpool_limits(limits=1):
            ranking, _, _ = screen_candidates(development, splits, candidates, config)
            nulls = calibrate_search(development, splits, candidates, config, progress=lambda _:None)
            ranking = attach_search_calibration(ranking, nulls)
        self.assertFalse(bool(ranking[['p_phase', 'p_volatility_phase']].le(config['alpha']).all(axis=1).iloc[0]))

    def test_elongated_surface_is_not_accepted_as_sphere(self):
        frame = self.frame(self.unit*np.array([1., 1.7, 2.6]) + self.rng.normal(0, .005, (1000, 3)))
        development = frame.iloc[:800]
        candidates = make_candidates(self.config)
        splits, _ = chronological_splits(development, self.config)
        with threadpool_limits(limits=1):
            ranking, _, _ = screen_candidates(development, splits, candidates, self.config)
            # Grant the surrogate gate to isolate geometric discrimination.
            ranking['p_phase'] = .01
            ranking['p_volatility_phase'] = .01
            ranking, selected, _, _ = compare_ellipsoid(development, splits, candidates, ranking,
                                                        self.config, progress=lambda _:None)
        self.assertIsNone(selected)
        self.assertFalse(bool(ranking.candidate_supported.iloc[0]))


if __name__ == '__main__':
    unittest.main()
