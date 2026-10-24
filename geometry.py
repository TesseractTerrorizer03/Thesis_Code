"""Geometry only: train-fitted coordinates and distances to codimension-one surfaces.

No clustering, mixture likelihoods, or unit-normalization of input observations.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares, minimize


@dataclass
class Coordinates:
    scaling: str = 'standard'
    dimension: int | None = None

    def fit(self, x):
        x = np.asarray(x, float)
        self.mean_ = x.mean(axis=0)
        if self.scaling == 'standard':
            self.scale_ = x.std(axis=0)
        elif self.scaling == 'none':
            self.scale_ = np.ones(x.shape[1])
        else:
            raise ValueError('scaling must be standard or none')
        if np.any(self.scale_ <= 1e-12):
            raise ValueError('Constant/nearly constant training feature.')
        z = (x - self.mean_) / self.scale_
        q = self.dimension if self.dimension is not None else x.shape[1]
        if not 2 <= q <= x.shape[1]:
            raise ValueError('Use at least 2 coordinates, no more than the subset size.')
        _, singular, vt = np.linalg.svd(z, full_matrices=False)
        self.rotation_ = vt[:q].T if q < x.shape[1] else np.eye(q)
        self.retained_variance_ = float((singular[:q]**2).sum() / (singular**2).sum())
        return self

    def transform(self, x):
        return ((np.asarray(x, float) - self.mean_) / self.scale_) @ self.rotation_


class Hyperplane:
    """Least-squares affine hyperplane: same intrinsic dimension as the sphere."""
    def fit(self, x):
        self.center_ = np.mean(x, axis=0)
        _, _, vt = np.linalg.svd(x - self.center_, full_matrices=False)
        self.normal_ = vt[-1]
        return self

    def signed_distance(self, x):
        return (x - self.center_) @ self.normal_


class Sphere:
    """Minimize true squared orthogonal distances, without normalizing points.

    Radius/center limits exclude the arbitrarily-large-sphere plane limit.
    Fits reaching those limits are reported and cannot support discovery.
    The center is constrained by its Euclidean norm during optimization, using
    the same limit as the acceptance check (not separate coordinate bounds).
    ``max_nfev`` is retained for notebook compatibility; for SLSQP it sets the
    iteration budget, while the ellipsoid solver still uses function evaluations.
    """
    def __init__(self, max_radius_rms=2., max_center_rms=1.5, max_nfev=150):
        self.max_radius_rms = max_radius_rms
        self.max_center_rms = max_center_rms
        self.max_nfev = max_nfev

    def fit(self, x):
        x = np.asarray(x, float)
        origin = x.mean(axis=0)
        centered = x - origin
        rms = np.sqrt(np.mean(np.sum(centered**2, axis=1)))
        if rms <= 1e-12:
            raise ValueError('Degenerate training geometry.')
        if self.max_center_rms <= 0 or self.max_radius_rms <= .05:
            raise ValueError('Center limit must be positive and radius limit must exceed .05.')
        p = x.shape[1]
        # Optimize dimensionless PARAMETERS for numerical stability. Observations
        # are only rescaled by one common training RMS, never row-normalized.
        scaled = centered / rms
        min_radius, max_radius = .05, self.max_radius_rms
        def residual(params):
            return np.linalg.norm(scaled - params[:-1], axis=1) - params[-1]
        def objective(params):
            errors = residual(params)
            return np.mean(errors**2)
        def gradient(params):
            delta = scaled - params[:-1]
            norm = np.maximum(np.linalg.norm(delta, axis=1), 1e-15)
            errors = norm - params[-1]
            return np.r_[-2*np.mean(errors[:, None]*delta/norm[:, None], axis=0), -2*errors.mean()]
        # This ball constraint is rotation invariant and exactly matches the
        # total-displacement quantity subsequently reported as center_rms_.
        constraint = dict(type='ineq',
                          fun=lambda params: self.max_center_rms**2 - params[:-1] @ params[:-1],
                          jac=lambda params: np.r_[-2*params[:-1], 0.])
        bounds = [(None, None)]*p + [(min_radius, max_radius)]
        # Algebraic fitting supplies an initialization only; final loss is geometric.
        algebraic = np.linalg.lstsq(np.column_stack([2*scaled, np.ones(len(x))]),
                                   (scaled**2).sum(axis=1), rcond=None)[0]
        starts = [np.r_[np.zeros(p), np.linalg.norm(scaled, axis=1).mean()],
                  np.r_[algebraic[:-1], np.sqrt(max(algebraic[-1] + (algebraic[:-1]**2).sum(), .01))]]
        fits = []
        for start in starts:
            start = start.copy()
            length = np.linalg.norm(start[:-1])
            if length > self.max_center_rms:
                start[:-1] *= self.max_center_rms / length
            start[-1] = np.clip(start[-1], min_radius, max_radius)
            fit = minimize(objective, start, jac=gradient, method='SLSQP',
                           bounds=bounds, constraints=[constraint],
                           options=dict(maxiter=self.max_nfev, ftol=1e-10))
            feasible = (np.isfinite(fit.x).all() and np.isfinite(fit.fun)
                        and np.linalg.norm(fit.x[:-1]) <= self.max_center_rms + 1e-7
                        and min_radius-1e-7 <= fit.x[-1] <= max_radius+1e-7)
            if feasible:
                # Remove only solver roundoff at the constraint, not an infeasible
                # optimum. Recompute loss for the tiny numerical adjustment.
                length = np.linalg.norm(fit.x[:-1])
                if length > self.max_center_rms:
                    fit.x[:-1] *= self.max_center_rms/length
                fit.x[-1] = np.clip(fit.x[-1], min_radius, max_radius)
                fit.fun = objective(fit.x)
                fits.append(fit)
        if not fits:
            raise FloatingPointError('No feasible sphere fit within the center/radius limits.')
        successful = [fit for fit in fits if fit.success]
        fit = min(successful or fits, key=lambda f: f.fun)
        self.center_, self.radius_ = origin + rms*fit.x[:-1], float(rms*fit.x[-1])
        self.converged_ = bool(fit.success)
        self.radius_rms_ = float(fit.x[-1])
        self.center_rms_ = float(np.linalg.norm(fit.x[:-1]))
        # Keep the existing 1% near-upper-bound exclusion as an intentional
        # diagnostic buffer. Lower-radius contact is checked numerically.
        self.boundary_ = bool(self.radius_rms_ >= .99*self.max_radius_rms
                              or self.center_rms_ >= .99*self.max_center_rms
                              or self.radius_rms_ <= min_radius + 1e-7)
        return self

    def signed_distance(self, x):
        return np.linalg.norm(x - self.center_, axis=1) - self.radius_

    def diagnostics(self, x):
        delta = x - self.center_
        norms = np.linalg.norm(delta, axis=1)
        if np.any(norms <= 1e-12):
            return dict(resultant=1., angular_span=0., radial_cv=np.inf,
                        relative_shell_rmse=np.inf,
                        radius_ratio=float(norms.mean()/self.radius_))
        # Used only to inspect coverage AFTER fitting; not as geometric evidence.
        u = delta / norms[:, None]
        return dict(resultant=float(np.linalg.norm(u.mean(axis=0))),
                    angular_span=float(x.shape[1] * np.linalg.eigvalsh(u.T @ u / len(u))[0]),
                    radial_cv=float(norms.std() / norms.mean()),
                    relative_shell_rmse=float(np.sqrt(np.mean((norms-self.radius_)**2))/self.radius_),
                    radius_ratio=float(norms.mean() / self.radius_))


def ellipsoid_signed_distance(x, axes, iterations=48):
    """True Euclidean distance to axis-aligned hyperellipsoid, inside AND outside.

    Solve the closest-point Lagrange multiplier by batched bisection. Handle the
    interior singular case when the point has zero shortest-axis coordinates.
    Input is already expressed in the ellipsoid's centered orthonormal frame.
    """
    x, axes = np.asarray(x, float), np.asarray(axes, float)
    if np.any(axes <= 0):
        raise ValueError('Ellipsoid axes must be positive.')
    a2 = axes**2
    smallest = a2.min()
    shortest = np.isclose(a2, smallest, rtol=1e-10, atol=1e-14)
    level = ((x / axes)**2).sum(axis=1)
    outside = level >= 1.
    low = np.where(outside, 0., -smallest + 1e-13*smallest)
    high = np.where(outside, axes.max()*np.linalg.norm(x, axis=1) + smallest, 0.)
    def equation(lam):
        return ((axes * x / (a2 + lam[:, None]))**2).sum(axis=1) - 1.
    singular = (~outside) & (equation(low) <= 0)
    for _ in range(iterations):
        middle = (low + high) / 2
        positive = equation(middle) > 0
        low = np.where(positive, middle, low)
        high = np.where(positive, high, middle)
    lam = (low + high) / 2
    closest = a2*x / (a2 + lam[:, None])
    if singular.any():
        xi = x[singular]
        yi = np.zeros_like(xi)
        other = ~shortest
        yi[:, other] = a2[other] * xi[:, other] / (a2[other] - smallest)
        remaining = np.maximum(0., 1 - ((yi[:, other] / axes[other])**2).sum(axis=1))
        yi[:, np.flatnonzero(shortest)[0]] = np.sqrt(smallest * remaining)
        closest[singular] = yi
    distance = np.linalg.norm(closest - x, axis=1)
    return np.where(outside, distance, -distance)


class Ellipsoid:
    """Flexible control with training-PCA orientation, optimized center and axes.

    This includes every sphere, but is not an unrestricted orientation search.
    Both fitting and evaluation use true orthogonal distances (not algebraic
    or radial ellipsoid residuals).
    """
    def __init__(self, max_nfev=100):
        self.max_nfev = max_nfev

    def fit(self, x, sphere):
        self.origin_ = x.mean(axis=0)
        _, _, vt = np.linalg.svd(x - self.origin_, full_matrices=False)
        self.rotation_ = vt.T
        z = (x - self.origin_) @ self.rotation_
        rms = np.sqrt(np.mean((z*z).sum(axis=1)))
        q = z.shape[1]
        lo = np.r_[np.full(q, -1.5*rms), np.full(q, np.log(.08*rms))]
        hi = np.r_[np.full(q, 1.5*rms), np.full(q, np.log(3*rms))]
        initial = np.r_[(sphere.center_ - self.origin_) @ self.rotation_,
                        np.full(q, np.log(sphere.radius_))]
        initial = np.clip(initial, lo+1e-8, hi-1e-8)
        fit = least_squares(lambda p: ellipsoid_signed_distance(z-p[:q], np.exp(p[q:])),
                            initial, bounds=(lo, hi), max_nfev=self.max_nfev,
                            ftol=1e-6, xtol=1e-6, gtol=1e-6)
        self.center_rotated_, self.axes_ = fit.x[:q], np.exp(fit.x[q:])
        self.converged_ = bool(fit.success)
        self.boundary_ = bool(np.any(fit.active_mask))
        self.axis_ratio_ = float(self.axes_.max()/self.axes_.min())
        return self

    def signed_distance(self, x):
        z = (x-self.origin_) @ self.rotation_ - self.center_rotated_
        return ellipsoid_signed_distance(z, self.axes_)
