"""Generalized Kent normalization on S^(d-1), with surface-area measure.

Fourier/Laplace inversion with continuous Euler damping, following Chen &
Tanaka (2021), https://arxiv.org/abs/2004.14660. We derive the contour sign
directly from Laplace inversion (the manuscript has inconsistent signs).
No saddlepoint density approximation is used: the saddle only locates a
numerically convenient contour to the right of every quadratic eigenvalue.
"""
from dataclasses import dataclass
from functools import lru_cache
import time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import erfc, gammaln, ive

MARGIN = 1e-6
LOGZ_TOL = 1e-10
GRAD_TOL = 2e-10
MAX_NODES = 131072


@dataclass
class NormalizerResult:
    logz: float
    dkappa: float
    dlambda: np.ndarray
    nodes: int
    error: float
    gradient_error: float


STATS = dict(calls=0, seconds=0., nodes=0, max_error=0., failures=0)


def reset_stats():
    STATS.update(calls=0, seconds=0., nodes=0, max_error=0., failures=0)


def validate(kappa, lam, constrained=True):
    lam = np.asarray(lam, float)
    if lam.ndim != 1 or len(lam) < 2 or not np.isfinite(lam).all() or not np.isfinite(kappa):
        raise ValueError('Finite concentration and at least two tangent eigenvalues required')
    if kappa < 0:
        raise ValueError('Negative concentration')
    if constrained:
        if abs(lam.sum()) > 1e-8 * max(1., np.max(abs(lam))):
            raise ValueError('Tangent quadratic matrix must have zero trace')
        if 2 * lam.max() > (1-MARGIN)*kappa + 1e-8:
            raise ValueError('Unimodality margin violated')
    return lam


def contour(kappa, lam):
    a = np.r_[0., lam]
    low = max(0., a.max())
    def derivative(c):
        return 1 - .5 * np.sum(1/(c-a)) - kappa*kappa/(4*c*c)
    upper = low + max(10., kappa + len(a))
    c = brentq(derivative, low + 1e-9, upper, xtol=1e-12)
    logh0 = -.5*np.log(c-a).sum() + kappa*kappa/(4*c)
    return a, c, logh0


@lru_cache(maxsize=48)
def grid(n, strip):
    # omega_d=1/2, omega_u=1; theorem 2 of Chen & Tanaka.
    h = np.sqrt(12*np.pi*strip/n)
    t = np.arange(n+1)*h
    p, q = np.sqrt(2*n*h), np.sqrt(n*h/8)
    weight = .5*erfc(t/p-q)
    weight[1:] *= 2
    return h, t, weight


def _integrand(t, kappa, a, c):
    z = c + 1j*t
    denom = z[:, None]-a
    # Center both large real and phase terms before exponentiation.
    exponent = (1j*t - .5*np.log(denom/(c-a)).sum(axis=1)
                + kappa*kappa/4*(1/z-1/c))
    value = np.exp(exponent)
    derivative = np.column_stack([kappa/(2*z), .5/denom[:, 1:]])
    return value, derivative


def normalizer(kappa, lam, tol=LOGZ_TOL, force_numeric=False, constrained=True):
    """Return log Z and derivatives w.r.t. kappa and all tangent eigenvalues.

Eigenvalue derivatives are unconstrained partials; apply the trace-zero
coordinate chain rule in the optimizer. Agreement under refinement is a
numerical diagnostic, not a rigorous error bound.
"""
    started = time.perf_counter()
    STATS['calls'] += 1
    try:
        lam = validate(float(kappa), lam, constrained)
        d = len(lam)+1
        if not force_numeric and np.all(lam == 0):
            if kappa == 0:
                result = NormalizerResult(np.log(2)+d/2*np.log(np.pi)-gammaln(d/2),
                                          0., np.full(d-1, 1/d), 0, 0., 0.)
            else:
                nu = d/2-1
                ratio = ive(nu+1, kappa)/ive(nu, kappa)
                logz = d/2*np.log(2*np.pi)+np.log(ive(nu,kappa))+kappa-nu*np.log(kappa)
                result = NormalizerResult(float(logz),float(ratio),np.full(d-1,ratio/kappa),0,0.,0.)
            return result
        a, c, logh0 = contour(kappa,lam)
        strip = min(1., .8*np.min(c-a))
        previous = None
        n = 128
        while n <= MAX_NODES:
            h,t,w = grid(n,strip)
            value,derivative = _integrand(t,kappa,a,c)
            integral = h*np.sum(w*value.real)
            if integral > 0 and np.isfinite(integral):
                logz = (d/2-1)*np.log(np.pi)+c+logh0+np.log(integral)
                gradient = h*np.sum(w[:,None]*(value[:,None]*derivative).real,axis=0)/integral
                if previous is not None:
                    error = abs(logz-previous[0])
                    ge = float(np.max(abs(gradient-previous[1])))
                    if error < tol and ge < min(GRAD_TOL, 2*tol):
                        STATS['nodes'] += 2*n+1
                        STATS['max_error'] = max(STATS['max_error'],error)
                        return NormalizerResult(float(logz),float(gradient[0]),gradient[1:],
                                                2*n+1,float(error),ge)
                previous = logz,gradient
            n *= 2
        raise FloatingPointError('Fourier normalizer refinement did not converge')
    except Exception:
        STATS['failures'] += 1
        raise
    finally:
        STATS['seconds'] += time.perf_counter()-started


def reference_normalizer(kappa, lam):
    """Independent adaptive, undamped contour integration (validation only)."""
    lam = validate(kappa,lam)
    a,c,logh0=contour(kappa,lam)
    def integrand(t):
        return _integrand(np.array([t]),kappa,a,c)[0][0].real
    # Separate periods to avoid cancellation across a huge adaptive interval.
    total = 0.
    error = 0.
    stop = 256.
    while stop <= 32768:
        start = 0. if stop == 256 else stop/2
        chunk = 0.
        for lo in np.arange(start,stop,8*np.pi):
            val,err=quad(integrand,lo,min(lo+8*np.pi,stop),epsabs=1e-12,epsrel=1e-11,limit=150)
            chunk += val
            error += err
        total += chunk
        if stop > 256 and abs(chunk) < 1e-10*max(abs(total),1e-30):
            return float((len(a)/2-1)*np.log(np.pi)+c+logh0+np.log(2*total)), error
        stop *= 2
    raise FloatingPointError('Independent contour tail did not settle')
