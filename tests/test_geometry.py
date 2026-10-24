import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geometry import Coordinates, Sphere, Hyperplane, Ellipsoid, ellipsoid_signed_distance
from experiment import phase_surrogate, chronological_splits, make_candidates


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(18)

    def test_sphere_recovers_shifted_noisy_surface(self):
        u = self.rng.normal(size=(1500, 4))
        u /= np.linalg.norm(u, axis=1)[:, None]
        center = np.array([1., -2., .5, .2])
        x = center + (3+self.rng.normal(0, .015, len(u)))[:, None]*u
        fitted = Sphere().fit(x[:1000])
        self.assertTrue(fitted.converged_)
        self.assertFalse(fitted.boundary_)
        np.testing.assert_allclose(fitted.center_, center, atol=.01)
        self.assertAlmostEqual(fitted.radius_, 3., places=2)
        self.assertLess(np.mean(fitted.signed_distance(x[1000:])**2), .001)
        self.assertGreater(np.mean(Hyperplane().fit(x[:1000]).signed_distance(x[1000:])**2), 1.)

    def test_center_norm_is_enforced_during_fitting(self):
        # Rotate a near-planar cloud so its preferred displacement has several
        # nonzero coordinates. The old coordinate-wise box exceeded this limit.
        x = self.rng.normal(size=(600, 3))
        x[:, 2] *= .005
        rotation, _ = np.linalg.qr(self.rng.normal(size=(3, 3)))
        x = x @ rotation
        fitted = Sphere(max_center_rms=.3).fit(x)
        rms = np.sqrt(np.mean(np.sum((x-x.mean(axis=0))**2, axis=1)))
        displacement = np.linalg.norm(fitted.center_-x.mean(axis=0))/rms
        self.assertTrue(fitted.converged_)
        self.assertLessEqual(displacement, .3+1e-10)
        self.assertAlmostEqual(displacement, fitted.center_rms_, places=10)
        self.assertGreater(displacement, .99*.3)
        self.assertTrue(fitted.boundary_)  # Genuine contact is still excluded.

    def test_center_constraint_is_rotation_invariant(self):
        x = self.rng.normal(size=(600, 3))
        x[:, 2] *= .005
        rotation, _ = np.linalg.qr(self.rng.normal(size=(3, 3)))
        original = Sphere(max_center_rms=.4).fit(x)
        rotated = Sphere(max_center_rms=.4).fit(x @ rotation)
        self.assertTrue(original.converged_ and rotated.converged_)
        self.assertLessEqual(original.center_rms_, .4+1e-10)
        self.assertLessEqual(rotated.center_rms_, .4+1e-10)
        self.assertAlmostEqual(np.mean(original.signed_distance(x)**2),
                               np.mean(rotated.signed_distance(x @ rotation)**2), places=7)
        self.assertEqual(original.boundary_, rotated.boundary_)

    def test_ellipsoid_distance_sphere_limit_and_singular_interior(self):
        x = self.rng.normal(size=(100, 5))
        expected = np.linalg.norm(x, axis=1)-2
        np.testing.assert_allclose(ellipsoid_signed_distance(x, np.full(5, 2.)), expected, atol=1e-9)
        self.assertAlmostEqual(ellipsoid_signed_distance(np.zeros((1, 3)), np.array([1., 2., 3.]))[0], -1.)
        self.assertAlmostEqual(ellipsoid_signed_distance(np.array([[0., 0., 4.]]), np.array([1., 2., 3.]))[0], 1.)

    def test_ellipse_distances_against_independent_angle_search(self):
        axes = np.array([1., 2.5])
        x = np.vstack([self.rng.normal(size=(15, 2))*2, [0., 0.], [0., 1.], [1., 0.]])
        distances = ellipsoid_signed_distance(x, axes)
        for point, distance in zip(x, distances):
            def loss(t):
                surface = axes*np.array([np.cos(t), np.sin(t)])
                return np.sum((surface-point)**2)
            grid = np.linspace(-np.pi, np.pi, 2001)
            t0 = min(grid, key=loss)
            result = minimize_scalar(loss, bounds=(t0-.02, t0+.02), method='bounded')
            self.assertAlmostEqual(abs(distance), np.sqrt(result.fun), places=5)

    def test_ellipsoid_fits_elongated_surface(self):
        u = self.rng.normal(size=(600, 3))
        u /= np.linalg.norm(u, axis=1)[:, None]
        x = u*np.array([1., 2., 3.]) + self.rng.normal(0, .01, size=u.shape)
        sphere = Sphere().fit(x)
        ellipsoid = Ellipsoid(max_nfev=150).fit(x, sphere)
        self.assertTrue(ellipsoid.converged_)
        self.assertLess(np.mean(ellipsoid.signed_distance(x)**2), .01*np.mean(sphere.signed_distance(x)**2))

    def test_preprocessing_is_frozen_and_not_unit_normalized(self):
        x = self.rng.normal(size=(500, 5))
        transform = Coordinates(dimension=3).fit(x)
        center = transform.mean_.copy()
        scores = transform.transform(x*4)
        np.testing.assert_array_equal(center, transform.mean_)
        self.assertEqual(scores.shape, (500, 3))
        self.assertGreater(np.std(np.linalg.norm(scores, axis=1)), .2)

    def test_phase_surrogate_preserves_cross_spectra(self):
        frame = pd.DataFrame(self.rng.normal(size=(256, 4)))
        surrogate = phase_surrogate(frame, self.rng)
        a = np.fft.rfft(frame-frame.mean(), axis=0)
        b = np.fft.rfft(surrogate-surrogate.mean(), axis=0)
        for i in range(4):
            for j in range(4):
                np.testing.assert_allclose(a[:, i]*a[:, j].conj(), b[:, i]*b[:, j].conj(), atol=1e-9)

    def test_splits_and_automatic_neighbors(self):
        frame = pd.DataFrame(index=pd.bdate_range('2000-01-01', periods=1000))
        config = dict(n_splits=3, validation_size=100, gap=5, min_train_size=100,
                      features=list('abcde'), subset_pool=list('abcde'), seed_subsets={'one':list('abc')},
                      search_mode='seed_neighbors', subset_sizes=[3], pca_dimensions=[None, 2], max_candidates=30)
        splits, _ = chronological_splits(frame, config)
        for tr, va in splits:
            self.assertEqual(va[0]-tr[-1], 6)
        candidates = make_candidates(config)
        self.assertIn(('a', 'b', 'c', 'd'), [c.columns for c in candidates])
        self.assertEqual(len(set(c.name for c in candidates)), len(candidates))


if __name__ == '__main__':
    unittest.main()
