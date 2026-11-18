"""Reproducible equity/FX size-direction experiments; notebook companion.

No feature standardization is performed. Selection APIs accept only training
and validation arrays. Final testing requires a frozen selection manifest.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import inspect
import io
import json
import math
from pathlib import Path
import ssl
import tempfile
import time
import urllib.request
import warnings
import zipfile

import certifi
import joblib
import numpy as np
import pandas as pd
import scipy
from scipy.linalg import helmert, cho_factor, cho_solve
from scipy.linalg.lapack import dpocon
from scipy.optimize import brentq
from scipy.special import gammaln, ive, logsumexp
from scipy.stats import norm, vonmises_fisher
import sklearn
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "size_direction_2000"
OUTPUT = ROOT / "outputs" / "size_direction_2000"
GROUPS = {
    "Technology": "MSFT AAPL NVDA ORCL IBM CSCO INTC QCOM TXN ADI".split(),
    "Finance": "JPM BAC WFC C GS MS PNC STT AXP SCHW".split(),
    "Healthcare": "JNJ PFE MRK ABT AMGN GILD BMY MDT UNH TMO".split(),
    "Consumer businesses": "PG KO PEP CL KMB WMT COST HD LOW NKE".split(),
    "Industrials": "CAT DE HON GE UNP UPS FDX EMR ETN ITW".split(),
}
STOCKS = [ticker for tickers in GROUPS.values() for ticker in tickers]
CURRENCIES = ["EUR", "USD", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD", "NOK", "SEK"]
START, END = "2000-01-01", "2026-10-30"
SEED, STARTS, MAX_ITER, TOL = 42, 10, 500, 1e-6
VAR_FLOOR, KAPPA_MAX, MIN_EFFECTIVE = 1e-6, 1e4, 5.0
REGULARIZATIONS = (1e-8, 1e-7, 1e-6)
COVARIANCES = ("spherical", "diag", "tied", "full")
FAMILIES = ("GMM", "Unlinked", "Linked")


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def acquire_data(refresh=False):
    """Download only missing caches; never disable certificate validation."""
    DATA.mkdir(parents=True, exist_ok=True)
    equity_path, fx_path = DATA / "equity_adjusted_prices.csv", DATA / "ecb_eur_quotes.csv"
    if equity_path.exists() and not refresh:
        _verify_cache(equity_path)
    else:
        import yfinance as yf
        # Keep provider authentication/time-zone caches separate from price data.
        yf.set_tz_cache_location(str(Path(tempfile.gettempdir()) / "size_direction_provider_cache"))
        downloaded = yf.download(
            STOCKS, start="1999-12-27", end="2026-10-31", interval="1d",
            auto_adjust=True, back_adjust=False, actions=False, repair=False,
            keepna=True, rounding=False, progress=False, threads=False, ignore_tz=True,
        )
        prices = downloaded["Close"].reindex(columns=STOCKS)
        prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
        prices.index.name = "date"
        if prices.empty or prices.isna().all().any():
            raise RuntimeError("Equity acquisition incomplete; no tickers may be silently replaced.")
        _validate_prices(prices, STOCKS)
        prices.to_csv(equity_path, float_format="%.17g")
        _record_cache(equity_path, prices, {
            "provider": "Yahoo Finance via yfinance", "yfinance_version": yf.__version__,
            "auto_adjust": True, "back_adjust": False, "repair": False,
            "requested_start": "1999-12-27", "end_exclusive": "2026-10-31",
            "universe_substitution": "PNC replaces BK: early-2000 public coverage verified before fitting.",
            "limitations": "Fixed surviving-stock cohort, current-vintage adjusted histories; not point-in-time index constituents. Corporate restructurings can change economic exposures.",
        })
        print(f"Cached equity prices: {prices.shape}", flush=True)
    if fx_path.exists() and not refresh:
        _verify_cache(fx_path)
    else:
        url = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip"
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, context=context, timeout=60) as response:
            payload = response.read()
        archive = zipfile.ZipFile(io.BytesIO(payload))
        quotes = pd.read_csv(archive.open(archive.namelist()[0]), na_values=["N/A"])
        quotes = quotes.rename(columns={"Date": "date"}).set_index("date")
        quotes.index = pd.to_datetime(quotes.index)
        quotes = quotes.sort_index().loc["1999-12-27":END, CURRENCIES[1:]].astype(float)
        _validate_prices(quotes, CURRENCIES[1:])
        quotes.to_csv(fx_path, float_format="%.17g")
        _record_cache(fx_path, quotes, {
            "provider": "European Central Bank", "source_url": url,
            "download_sha256": hashlib.sha256(payload).hexdigest(),
            "quote_convention": "Units of foreign currency per EUR",
            "limitations": "Information-only daily reference fixings, not executable quotes or an arbitrage backtest.",
        })
        print(f"Cached ECB prices: {quotes.shape}", flush=True)
    return equity_path, fx_path


def _validate_prices(frame, expected_columns):
    assert list(frame.columns) == list(expected_columns)
    assert frame.index.is_monotonic_increasing and frame.index.is_unique
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"Missing/nonfinite prices; investigate rather than fill/drop: {frame.isna().sum().to_dict()}")
    assert (frame.to_numpy() > 0).all()
    assert frame.index.min() < pd.Timestamp(START), "A preceding 1999 price is required."
    assert frame.index.max() == pd.Timestamp(END), "Requested end date not covered."


def _record_cache(path, prices, metadata):
    metadata.update(retrieved_utc=datetime.now(timezone.utc).isoformat(),
                    file_sha256=digest_file(path), columns=list(prices.columns),
                    rows=len(prices), first_date=str(prices.index[0].date()),
                    last_date=str(prices.index[-1].date()))
    write_json(path.with_suffix(".provenance.json"), metadata)


def _verify_cache(path):
    provenance = json.loads(path.with_suffix(".provenance.json").read_text())
    if digest_file(path) != provenance["file_sha256"]:
        raise ValueError(f"Cache hash mismatch: {path}. Do not silently refresh research inputs.")


@dataclass
class MarketData:
    name: str
    frame: pd.DataFrame
    a: int
    b: int
    basis: np.ndarray | None = None

    @property
    def train(self):
        return self.frame.iloc[:self.a].to_numpy(copy=True)

    @property
    def validation(self):
        return self.frame.iloc[self.a:self.b].to_numpy(copy=True)

    @property
    def development(self):
        return self.frame.iloc[:self.b].to_numpy(copy=True)

    @property
    def test(self):
        return self.frame.iloc[self.b:].to_numpy(copy=True)


def load_data():
    """Read frozen caches. No downloading, centering or volatility scaling."""
    for filename in ("equity_adjusted_prices.csv", "ecb_eur_quotes.csv"):
        _verify_cache(DATA / filename)
    prices = pd.read_csv(DATA / "equity_adjusted_prices.csv", index_col=0, parse_dates=True)
    quotes = pd.read_csv(DATA / "ecb_eur_quotes.csv", index_col=0, parse_dates=True)
    _validate_prices(prices, STOCKS)
    _validate_prices(quotes, CURRENCIES[1:])
    equity = np.log(prices).diff().loc[START:END, STOCKS]
    foreign_strength = -np.log(quotes).diff().loc[START:END]
    strength = np.column_stack([np.zeros(len(foreign_strength)), foreign_strength.to_numpy()])
    centered = strength - strength.mean(axis=1, keepdims=True)
    basis = helmert(len(CURRENCIES), full=False)  # rows form an orthonormal basis of 1-perp
    contrast = centered @ basis.T
    fx = pd.DataFrame(contrast, index=foreign_strength.index,
                      columns=[f"Helmert {i+1}" for i in range(len(CURRENCIES)-1)])
    np.testing.assert_allclose(basis @ basis.T, np.eye(9), atol=1e-14)
    np.testing.assert_allclose(contrast @ basis, centered, atol=1e-14)
    for i, j, k in [(0, 1, 2), (1, 3, 5), (4, 6, 9)]:
        np.testing.assert_allclose((centered[:, i]-centered[:, j]) +
                                  (centered[:, j]-centered[:, k]),
                                  centered[:, i]-centered[:, k], atol=1e-14)
    # Changing the original numeraire adds a common level; contrasts must be unchanged.
    rebased = strength - strength[:, 3, None]
    np.testing.assert_allclose((rebased-rebased.mean(axis=1, keepdims=True)) @ basis.T,
                               contrast, atol=1e-14)
    datasets = {}
    for name, frame, transform in [("equities", equity, None), ("fx", fx, basis)]:
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError(f"Invalid {name} returns; no forward fill or silent row deletion.")
        n = len(frame)
        a, b = int(0.6*n), int(0.8*n)
        assert frame.index.is_unique and frame.index.is_monotonic_increasing
        assert frame.index[0].year == 2000 and frame.index[-1] == pd.Timestamp(END)
        assert frame.index[a-1] > pd.Timestamp("2009-12-31")
        assert pd.Timestamp("2008-09-15") in frame.index[:a]
        assert set((2007, 2008, 2009)).issubset(set(frame.index[:a].year))
        assert np.linalg.matrix_rank(frame.iloc[:a].to_numpy()) == frame.shape[1]
        polar(frame.to_numpy())
        datasets[name] = MarketData(name, frame, a, b, transform)
    return datasets


def split_table(datasets):
    rows = []
    for name, data in datasets.items():
        for split, lo, hi in [("train", 0, data.a), ("validation", data.a, data.b),
                              ("test", data.b, len(data.frame))]:
            rows.append(dict(dataset=name, split=split, rows=hi-lo,
                             dimensions=data.frame.shape[1],
                             start=str(data.frame.index[lo].date()),
                             end=str(data.frame.index[hi-1].date())))
    return pd.DataFrame(rows)


def polar(x):
    x = np.asarray(x, dtype=float)
    radii = np.linalg.norm(x, axis=1)
    if not np.isfinite(x).all() or np.any(radii <= 0):
        raise ValueError("Nonfinite or zero return vector; direction is undefined.")
    y, u = np.log(radii), x / radii[:, None]
    np.testing.assert_allclose(np.exp(y[:, None])*u, x, atol=1e-14)
    return y, u


@dataclass(frozen=True)
class Spec:
    family: str
    k: int
    covariance: str = ""
    regularization: float = 0.0

    @property
    def key(self):
        return (f"gmm_k{self.k:02d}_{self.covariance}_{self.regularization:g}"
                if self.family == "GMM" else f"{self.family.lower()}_k{self.k:02d}")


def parameter_count(spec, d):
    k = spec.k
    if spec.family == "Unlinked":
        return k*(d+1)+1
    if spec.family == "Linked":
        return k*(2*d+1)+1
    covariance = {"spherical": k, "diag": k*d,
                  "tied": d*(d+1)//2, "full": k*d*(d+1)//2}[spec.covariance]
    return k-1+k*d+covariance


def all_specs():
    return [Spec("GMM", k, covariance, regularization)
            for k in range(1, 21) for covariance in COVARIANCES
            for regularization in REGULARIZATIONS] + [
                Spec(family, k) for k in range(1, 21) for family in ("Unlinked", "Linked")]


def log_vmf_normalizer(kappa, d):
    if kappa == 0:
        return gammaln(d/2)-np.log(2)-d/2*np.log(np.pi)
    # Delegate the normalized distribution to SciPy; orthogonal vectors remove
    # its linear directional term. This includes scaled-Bessel stabilization.
    a, b = np.zeros(d), np.zeros(d)
    a[0], b[1] = 1, 1
    return float(vonmises_fisher(a, kappa).logpdf(b))


def concentration_from_resultant(r, d):
    if r <= 1e-10:
        return 0.0
    nu = d/2-1
    def ratio(k):
        return ive(nu+1, k)/ive(nu, k)
    if r >= ratio(KAPPA_MAX):
        return KAPPA_MAX
    return float(brentq(lambda k: ratio(k)-r, 1e-8, KAPPA_MAX, xtol=1e-10))


def cylinder_component_logpdf(y, u, fit):
    means = fit["coef"][:, 0][None, :] + u @ fit["coef"][:, 1:].T
    return (np.log(fit["weights"])[None, :]
            + log_vmf_normalizer(fit["kappa"], u.shape[1])
            + fit["kappa"]*(u @ fit["mu"].T)
            + norm.logpdf(y[:, None], loc=means, scale=np.sqrt(fit["variance"])))


def cylinder_responsibilities(y, u, fit):
    logs = cylinder_component_logpdf(y, u, fit)
    return np.exp(logs-logsumexp(logs, axis=1, keepdims=True))


def weighted_regression(design, y, weights, return_method=False):
    """Exact unpenalized WLS; fast well-conditioned solve, reference SVD fallback.

    A reciprocal-condition estimate protects the normal-equation solution.
    No ridge or coefficient threshold is introduced. Degenerate and poorly
    conditioned problems retain numpy's rank-revealing least-squares behavior.
    """
    weighted_transpose = design.T*weights
    gram = weighted_transpose @ design
    target = weighted_transpose @ y
    try:
        factor, lower = cho_factor(gram, lower=True, check_finite=False)
        reciprocal_condition, info = dpocon(factor, np.linalg.norm(gram, 1), uplo="L")
        if info == 0 and reciprocal_condition >= 1e-7:
            coef = cho_solve((factor, lower), target, check_finite=False)
            if np.isfinite(coef).all():
                return (coef, "cholesky") if return_method else coef
    except np.linalg.LinAlgError:
        pass
    root = np.sqrt(weights)
    coef = np.linalg.lstsq(design*root[:, None], y*root, rcond=None)[0]
    return (coef, "svd") if return_method else coef


def cylinder_m_step(y, u, r, linked, min_effective=MIN_EFFECTIVE):
    n, d = u.shape
    effective = r.sum(axis=0)
    if (effective < min_effective).any():
        raise FloatingPointError("Collapsed component: effective membership below threshold")
    k = len(effective)
    resultant_vectors = r.T @ u
    lengths = np.linalg.norm(resultant_vectors, axis=1)
    mu = np.zeros((k, d))
    mu[:, 0] = 1
    nz = lengths > 1e-14
    mu[nz] = resultant_vectors[nz]/lengths[nz, None]
    coef = np.zeros((k, d+1))
    design = np.column_stack([np.ones(n), u])
    for j in range(k):
        if linked:
            coef[j] = weighted_regression(design, y, r[:, j])
        else:
            coef[j, 0] = r[:, j] @ y/effective[j]
    residual = y[:, None]-design @ coef.T
    variance = max(float(np.sum(r*residual**2)/n), VAR_FLOOR)
    return dict(weights=effective/n, mu=mu, coef=coef, variance=variance,
                kappa=concentration_from_resultant(lengths.sum()/n, d))


def fit_cylinder(y, u, linked, responsibilities=None, warm=None,
                 max_iter=MAX_ITER, tol=TOL, min_effective=MIN_EFFECTIVE):
    import copy
    fit = (copy.deepcopy(warm) if warm is not None else
           cylinder_m_step(y, u, responsibilities, linked, min_effective))
    if not linked:
        fit["coef"][:, 1:] = 0
    history = [float(logsumexp(cylinder_component_logpdf(y, u, fit), axis=1).mean())]
    converged = False
    for iteration in range(1, max_iter+1):
        r = cylinder_responsibilities(y, u, fit)
        fit = cylinder_m_step(y, u, r, linked, min_effective)
        likelihood = float(logsumexp(cylinder_component_logpdf(y, u, fit), axis=1).mean())
        improvement = likelihood-history[-1]
        if not np.isfinite(likelihood) or improvement < -1e-8:
            raise FloatingPointError(f"Nonfinite/decreasing EM likelihood: gain={improvement}")
        history.append(likelihood)
        if abs(improvement) < tol:
            converged = True
            break
    boundary = fit["variance"] <= VAR_FLOOR*(1+1e-6) or fit["kappa"] >= KAPPA_MAX*(1-1e-6)
    fit.update(history=np.asarray(history), converged=converged,
               boundary=bool(boundary), iterations=iteration)
    return fit


def make_initializations(x, k):
    """Reuse deterministic K-means starts across covariance/reg candidates."""
    y, u = polar(x)
    gaussian, cylindrical = [], []
    joint = np.column_stack([y, u])
    for start in range(STARTS):
        if k == 1:
            labels_g = labels_c = np.zeros(len(x), dtype=int)
        else:
            labels_g = KMeans(n_clusters=k, n_init=1, random_state=SEED+start).fit_predict(x)
            labels_c = KMeans(n_clusters=k, n_init=1, random_state=SEED+start).fit_predict(joint)
        gaussian.append(labels_g)
        cylindrical.append(np.eye(k)[labels_c])
    return gaussian, cylindrical


def gaussian_initial_parameters(x, labels, spec):
    """Public GMM initialization arguments, equivalent to hard responsibilities."""
    n, d = x.shape
    counts = np.bincount(labels, minlength=spec.k)
    if (counts == 0).any():
        raise FloatingPointError("Empty initialization group")
    weights = counts/n
    means = np.array([x[labels == j].mean(axis=0) for j in range(spec.k)])
    centered = [x[labels == j]-means[j] for j in range(spec.k)]
    reg = spec.regularization
    if spec.covariance in ("diag", "spherical"):
        covariances = np.array([(block**2).mean(axis=0)+reg for block in centered])
        if spec.covariance == "spherical":
            covariances = covariances.mean(axis=1)
        precision = 1/covariances
    elif spec.covariance == "tied":
        covariance = sum(block.T @ block for block in centered)/n+reg*np.eye(d)
        precision = np.linalg.inv(covariance)
    else:
        covariance = [block.T @ block/len(block)+reg*np.eye(d) for block in centered]
        precision = np.array([np.linalg.inv(cov) for cov in covariance])
    return weights, means, precision


def fit_candidate(x, spec, initializations, warm=None):
    """Ten starts; rejected starts never participate in selecting a fit."""
    y, u = polar(x)
    gaussian, cylindrical = initializations
    diagnostics, acceptable = [], []
    begun = time.monotonic()
    for start in range(STARTS):
        row = dict(start=start, accepted=False, status="error", iterations=0,
                   train_mean_loglik=None, warning="")
        try:
            if spec.family == "GMM":
                weights, means, precision = gaussian_initial_parameters(x, gaussian[start], spec)
                model = GaussianMixture(
                    n_components=spec.k, covariance_type=spec.covariance,
                    reg_covar=spec.regularization, tol=TOL, max_iter=MAX_ITER,
                    n_init=1, random_state=SEED+start, weights_init=weights,
                    means_init=means, precisions_init=precision)
                with warnings.catch_warnings(record=True) as messages:
                    warnings.simplefilter("always")
                    model.fit(x)
                row["warning"] = "; ".join(str(m.message) for m in messages)
                converged, boundary = model.converged_, False
                row["iterations"] = int(model.n_iter_)
                minimum = model.predict_proba(x).sum(axis=0).min()
                if minimum < MIN_EFFECTIVE:
                    raise FloatingPointError("Collapsed GMM component")
                # sklearn's ridge-constrained covariance updates need not maximize
                # unpenalized likelihood exactly; preserve its diagnostics rather
                # than incorrectly asserting strict monotonicity for ridge GMMs.
                ll = float(model.score(x))
            else:
                use_warm = spec.family == "Linked" and warm is not None and start == STARTS-1
                model = fit_cylinder(y, u, spec.family == "Linked",
                                     responsibilities=None if use_warm else cylindrical[start],
                                     warm=warm if use_warm else None)
                converged, boundary = model["converged"], model["boundary"]
                row["iterations"] = int(model["iterations"])
                ll = float(model["history"][-1]-x.shape[1]*y.mean())
            row.update(status=("nonconverged" if not converged else "boundary" if boundary else "converged"),
                       accepted=bool(converged and not boundary and np.isfinite(ll)),
                       train_mean_loglik=ll)
            if row["accepted"]:
                acceptable.append((ll, model, start))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            row["status"] = f"{type(error).__name__}: {error}"
        diagnostics.append(row)
    if not acceptable:
        return dict(spec=asdict(spec), fit=None, diagnostics=diagnostics,
                    elapsed_seconds=time.monotonic()-begun)
    ll, model, best_start = max(acceptable, key=lambda item: item[0])
    if spec.family == "Linked" and warm is not None:
        warm_ll = float(warm["history"][-1]-x.shape[1]*y.mean())
        if ll < warm_ll-1e-8:
            raise FloatingPointError("Selected linked fit violated zero-link nesting")
    return dict(spec=asdict(spec), fit=model, diagnostics=diagnostics,
                selected_start=best_start, train_mean_loglik=ll,
                elapsed_seconds=time.monotonic()-begun)


def score_model(x, record):
    if record["spec"]["family"] == "GMM":
        return record["fit"].score_samples(x)
    y, u = polar(x)
    return logsumexp(cylinder_component_logpdf(y, u, record["fit"]), axis=1)-x.shape[1]*y


def posterior_model(x, record):
    if record["spec"]["family"] == "GMM":
        return record["fit"].predict_proba(x)
    y, u = polar(x)
    return cylinder_responsibilities(y, u, record["fit"])


def engine_fingerprint():
    functions = [Spec, parameter_count, all_specs, polar, log_vmf_normalizer,
                 concentration_from_resultant, cylinder_component_logpdf,
                 cylinder_responsibilities, weighted_regression, cylinder_m_step, fit_cylinder,
                 make_initializations, gaussian_initial_parameters, fit_candidate, score_model]
    source = "\n".join(inspect.getsource(function) for function in functions)
    settings = dict(seed=SEED, starts=STARTS, max_iter=MAX_ITER, tol=TOL,
                    variance_floor=VAR_FLOOR, concentration_max=KAPPA_MAX,
                    minimum_effective=MIN_EFFECTIVE, regularizations=REGULARIZATIONS,
                    covariances=COVARIANCES, numpy=np.__version__, scipy=scipy.__version__,
                    sklearn=sklearn.__version__)
    return hashlib.sha256((source+json.dumps(settings, sort_keys=True)).encode()).hexdigest()


def array_digest(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array, dtype="<f8")
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def dump_joblib(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    joblib.dump(value, temporary, compress=3)
    temporary.replace(path)


def run_search(name, train, validation, output=OUTPUT, progress=print, components=range(1, 21)):
    """Intentionally has no test argument: only train and validation are visible."""
    assert train.shape[1] == validation.shape[1]
    signature = hashlib.sha256((engine_fingerprint()+array_digest(train, validation)).encode()).hexdigest()
    cache = output / "search" / name / signature[:16]
    cache.mkdir(parents=True, exist_ok=True)
    write_json(cache / "protocol.json", dict(signature=signature, engine=engine_fingerprint(),
               dataset=name, training_rows=len(train), validation_rows=len(validation),
               dimensions=train.shape[1], seed=SEED, starts=STARTS, max_iter=MAX_ITER, tol=TOL,
               input_hash=array_digest(train, validation), test_data_available=False))
    components = tuple(components)
    assert components and set(components).issubset(set(range(1, 21)))
    rows = []
    order = {spec.key: index for index, spec in enumerate(all_specs())}
    with threadpool_limits(limits=1):
        for k in components:
            specs = [Spec("GMM", k, cov, reg) for cov in COVARIANCES for reg in REGULARIZATIONS]
            specs += [Spec("Unlinked", k), Spec("Linked", k)]
            initializations = None
            unlinked = None
            for spec in specs:
                path = cache / (spec.key+".joblib")
                cached = path.exists()
                if cached:
                    record = joblib.load(path)
                    assert record["search_signature"] == signature
                else:
                    if initializations is None:
                        init_path = cache / f"initializations_k{k:02d}.joblib"
                        if init_path.exists():
                            initializations = joblib.load(init_path)
                        else:
                            initializations = make_initializations(train, k)
                            dump_joblib(init_path, initializations)
                    record = fit_candidate(train, spec, initializations, warm=unlinked)
                    record["search_signature"] = signature
                    if record["fit"] is not None:
                        values = score_model(validation, record)
                        if not np.isfinite(values).all():
                            raise FloatingPointError("Nonfinite validation likelihood")
                        record["validation_mean_loglik"] = float(values.mean())
                    else:
                        record["validation_mean_loglik"] = None
                    dump_joblib(path, record)
                if spec.family == "Unlinked":
                    unlinked = record["fit"]
                accepted = sum(start["accepted"] for start in record["diagnostics"])
                row = dict(dataset=name, candidate=spec.key, **asdict(spec),
                    parameters=parameter_count(spec, train.shape[1]),
                    configuration_order=order[spec.key], accepted_starts=accepted,
                    rejected_starts=STARTS-accepted,
                    train_mean_loglik=record.get("train_mean_loglik"),
                    validation_mean_loglik=record["validation_mean_loglik"],
                    elapsed_seconds=record["elapsed_seconds"], search_signature=signature)
                rows.append(row)
                if not cached:
                    progress(f'{name}: {spec.key}: {accepted}/{STARTS} accepted; '
                             f'validation={row["validation_mean_loglik"]}; '
                             f'{record["elapsed_seconds"]:.1f}s', flush=True)
                pd.DataFrame(rows).to_csv(cache / f"trials_k{min(components):02d}_{max(components):02d}.csv", index=False)
            progress(f"{name}: completed K={k}/20 ({len(rows)}/280 configurations)", flush=True)
    frame = pd.DataFrame(rows).sort_values("configuration_order").reset_index(drop=True)
    assert len(frame) == 14*len(components)
    if len(components) == 20:
        frame.to_csv(output / f"{name}_validation_trials.csv", index=False)
    return frame, cache


def best_validation(rows):
    good = rows.loc[rows.validation_mean_loglik.notna()].copy()
    if good.empty:
        raise RuntimeError("No admissible validation fit in this comparison; do not widen the budget after testing.")
    maximum = good.validation_mean_loglik.max()
    tied = good.loc[good.validation_mean_loglik >= maximum-1e-6]
    return tied.sort_values(["parameters", "k", "configuration_order"]).iloc[0]


def select_configurations(trials, d):
    """Validation-only selection; compare three families in every view."""
    chosen = []
    for title, k0 in [("budget_low", 3), ("budget_medium", 6), ("budget_high", 9),
                      ("unrestricted", None)]:
        lower = k0*(2*d+1)-1 if k0 else None
        upper = math.floor(1.10*lower) if lower else None
        subset = trials if lower is None else trials.loc[trials.parameters.between(lower, upper)]
        for family in FAMILIES:
            winner = best_validation(subset.loc[subset.family == family])
            chosen.append(_selection_row(winner, title, lower, upper))
    # Isolate the link: choose one K jointly for the nested pair, not using GMM/test.
    pair = trials.loc[trials.family.isin(["Unlinked", "Linked"])].pivot(
        index="k", columns="family", values="validation_mean_loglik").dropna()
    # The same-component view contains all three families; a K with no
    # admissible GMM cannot form this comparison. This is a feasibility gate,
    # not an opportunity to use GMM/test scores to rank cylindrical pairs.
    admissible_gmm_k = trials.loc[(trials.family == "GMM") &
                                  trials.validation_mean_loglik.notna(), "k"]
    pair = pair.loc[pair.index.isin(admissible_gmm_k)]
    if pair.empty:
        raise RuntimeError("No jointly admissible same-component comparison")
    pair["average"] = pair[["Unlinked", "Linked"]].mean(axis=1)
    k = int(pair.loc[pair.average >= pair.average.max()-1e-6].index.min())
    for family in FAMILIES:
        winner = best_validation(trials.loc[(trials.family == family) & (trials.k == k)])
        chosen.append(_selection_row(winner, "same_components", None, None))
    selected = pd.DataFrame(chosen)
    for view in ("budget_low", "budget_medium", "budget_high"):
        values = selected.loc[selected.view == view, "parameters"]
        assert values.max()/values.min() <= 1.10+1e-12
    assert selected.loc[selected.view == "same_components", "k"].nunique() == 1
    return selected


def _selection_row(winner, view, lower, upper):
    return dict(dataset=winner.dataset, view=view, family=winner.family,
                candidate=winner.candidate, k=int(winner.k),
                covariance=str(winner.covariance), regularization=float(winner.regularization),
                parameters=int(winner.parameters),
                validation_mean_loglik=float(winner.validation_mean_loglik),
                budget_lower=lower, budget_upper=upper,
                search_signature=winner.search_signature)


def freeze_selections(searches, datasets, output=OUTPUT):
    """Freeze architecture, input hashes, split dates and selection provenance."""
    frames = [select_configurations(trials, datasets[name].frame.shape[1])
              for name, (trials, _) in searches.items()]
    selected = pd.concat(frames, ignore_index=True)
    path = output / "frozen_selections.json"
    selection_records = json.loads(selected.to_json(orient="records"))
    payload = dict(engine=engine_fingerprint(), selections=selection_records,
                   splits=split_table(datasets).to_dict(orient="records"),
                   data_hashes={name: array_digest(data.frame.to_numpy()) for name, data in datasets.items()},
                   primary_comparison="budget_medium", selection_metric="validation_mean_loglik",
                   stock_universe=STOCKS, currency_order=CURRENCIES,
                   no_feature_standardization=True,
                   test_used_for_selection=False)
    serialized = json.dumps(payload, sort_keys=True)
    payload["freeze_sha256"] = hashlib.sha256(serialized.encode()).hexdigest()
    # Identical reruns reuse the same freeze. Changed selections require explicit final reevaluation.
    write_json(path, payload)
    selected.to_csv(output / "selected_configurations.csv", index=False)
    split_table(datasets).to_csv(output / "splits.csv", index=False)
    return selected, payload


def refit_selected(name, development, selected, freeze_hash, output=OUTPUT, parallel=True):
    """No test data accepted; deduplicate architectures selected in multiple views."""
    # A component-count group owns its initializations and nested warm start.
    # Processes isolate sklearn warning capture and BLAS thread limits.
    groups = [group.drop_duplicates("candidate") for _, group in selected.groupby("k", sort=True)]
    if parallel and len(groups) > 1:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        fitted = {}
        with ProcessPoolExecutor(max_workers=min(4, len(groups)),
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(_refit_worker, name, development, group, freeze_hash, output)
                       for group in groups]
            for future in futures:
                fitted.update(future.result())
        return fitted
    cache = output / "refits" / name / freeze_hash[:16]
    cache.mkdir(parents=True, exist_ok=True)
    fitted, initializations = {}, {}
    signature = hashlib.sha256((engine_fingerprint()+array_digest(development)).encode()).hexdigest()
    def refit(spec):
        if spec.key in fitted:
            return fitted[spec.key]
        path = cache / (spec.key+".joblib")
        if path.exists():
            record = joblib.load(path)
            assert record["refit_signature"] == signature
        else:
            warm = None
            if spec.family == "Linked":
                warm_record = refit(Spec("Unlinked", spec.k))
                warm = warm_record["fit"]
            if spec.k not in initializations:
                initializations[spec.k] = make_initializations(development, spec.k)
            record = fit_candidate(development, spec, initializations[spec.k], warm)
            record["refit_signature"] = signature
            dump_joblib(path, record)
            print(f'{name}: refit {spec.key}: {sum(s["accepted"] for s in record["diagnostics"])}/10 accepted', flush=True)
        if record["fit"] is None:
            raise RuntimeError(f"Selected refit failed: {name}/{spec.key}; do not substitute a test-selected model")
        fitted[spec.key] = record
        return record
    with threadpool_limits(limits=1):
        for row in selected.itertuples():
            refit(Spec(row.family, int(row.k), row.covariance, row.regularization))
    return fitted


def _refit_worker(name, development, selected, freeze_hash, output):
    return refit_selected(name, development, selected, freeze_hash, output, parallel=False)


def bootstrap_means(scores, length, repetitions=1500, seed=SEED):
    """Paired circular moving blocks, batched to avoid large temporary tensors."""
    n = len(scores)
    rng = np.random.default_rng(seed)
    means = np.empty((repetitions, scores.shape[1]))
    blocks = math.ceil(n/length)
    for lo in range(0, repetitions, 50):
        hi = min(repetitions, lo+50)
        starts = rng.integers(0, n, size=(hi-lo, blocks))
        indices = ((starts[:, :, None]+np.arange(length)) % n).reshape(hi-lo, -1)[:, :n]
        means[lo:hi] = scores[indices].mean(axis=1)
    return means


def final_evaluation(datasets, searches, output=OUTPUT):
    freeze = json.loads((output / "frozen_selections.json").read_text())
    freeze_hash = freeze.pop("freeze_sha256")
    assert hashlib.sha256(json.dumps(freeze, sort_keys=True).encode()).hexdigest() == freeze_hash
    assert freeze["engine"] == engine_fingerprint() and not freeze["test_used_for_selection"]
    selected = pd.DataFrame(freeze["selections"])
    # Finish ALL train+validation refits before any final-test likelihood is computed.
    all_fitted = {}
    for name, data in datasets.items():
        assert array_digest(data.frame.to_numpy()) == freeze["data_hashes"][name]
        all_fitted[name] = refit_selected(name, data.development,
                                         selected.loc[selected.dataset == name], freeze_hash, output)
    rows, scores_export = [], []
    with threadpool_limits(limits=1):
        for name, data in datasets.items():
            choices = selected.loc[selected.dataset == name]
            fitted = all_fitted[name]
            keys = list(dict.fromkeys(choices.candidate))
            scores = np.column_stack([score_model(data.test, fitted[key]) for key in keys])
            assert np.isfinite(scores).all()
            key_index = {key: i for i, key in enumerate(keys)}
            bootstrap = {length: bootstrap_means(scores, length) for length in (5, 20, 60)}
            for view, group in choices.groupby("view", sort=False):
                baseline = group.set_index("family").candidate
                gmm_index, unlinked_index = key_index[baseline["GMM"]], key_index[baseline["Unlinked"]]
                for choice in group.itertuples():
                    index = key_index[choice.candidate]
                    fit = fitted[choice.candidate]
                    values = scores[:, index]
                    posterior = posterior_model(data.test, fit)
                    np.testing.assert_allclose(posterior.sum(axis=1), 1, atol=1e-10)
                    row = dict(dataset=name, view=view, primary=(view == "budget_medium"),
                        family=choice.family, candidate=choice.candidate, components=int(choice.k),
                        covariance=choice.covariance, regularization=choice.regularization,
                        parameters=int(choice.parameters), dimensions=data.frame.shape[1],
                        train_rows=data.a, validation_rows=data.b-data.a,
                        refit_rows=data.b, test_rows=len(data.test),
                        validation_mean_loglik=choice.validation_mean_loglik,
                        refit_mean_loglik=float(fit["train_mean_loglik"]),
                        test_mean_loglik=float(values.mean()),
                        gain_vs_gmm=float((values-scores[:, gmm_index]).mean()),
                        gain_vs_unlinked=float((values-scores[:, unlinked_index]).mean()),
                        accepted_refit_starts=sum(s["accepted"] for s in fit["diagnostics"]),
                        rejected_refit_starts=sum(not s["accepted"] for s in fit["diagnostics"]),
                        freeze_sha256=freeze_hash)
                    for length, means in bootstrap.items():
                        for label, baseline_index in [("gmm", gmm_index), ("unlinked", unlinked_index)]:
                            lower, upper = np.quantile(means[:, index]-means[:, baseline_index], [0.025, 0.975])
                            row[f"gain_vs_{label}_L{length}_low"] = float(lower)
                            row[f"gain_vs_{label}_L{length}_high"] = float(upper)
                    rows.append(row)
            export = pd.DataFrame(scores, index=data.frame.index[data.b:], columns=keys)
            export.index.name = "date"
            export.to_csv(output / f"{name}_test_daily_loglik.csv", float_format="%.12g")
    results = pd.DataFrame(rows)
    assert len(results) == 30
    results.to_csv(output / "test_results.csv", index=False, float_format="%.12g")
    roundtrip = pd.read_csv(output / "test_results.csv", keep_default_na=False)
    np.testing.assert_allclose(roundtrip.test_mean_loglik, results.test_mean_loglik, rtol=1e-10)
    write_json(output / "final_evaluation.json", dict(freeze_sha256=freeze_hash,
               test_results_sha256=digest_file(output / "test_results.csv"),
               completed_utc=datetime.now(timezone.utc).isoformat(),
               module_sha256=digest_file(__file__), bootstrap_repetitions=1500,
               block_lengths=[5, 20, 60], results_rows=len(results)))
    return results, all_fitted


def run_full_search(names=("equities", "fx"), components=range(1, 21)):
    datasets = load_data()
    searches = {}
    for name in names:
        data = datasets[name]
        searches[name] = run_search(name, data.train, data.validation, components=components)
    return datasets, searches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["acquire", "search", "finalize", "all"])
    parser.add_argument("--dataset", choices=["equities", "fx", "both"], default="both")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--k-start", type=int, default=1)
    parser.add_argument("--k-end", type=int, default=20)
    args = parser.parse_args()
    if args.action in ("acquire", "all"):
        acquire_data(args.refresh)
    if args.action in ("search", "finalize", "all"):
        names = ("equities", "fx") if args.dataset == "both" else (args.dataset,)
        if not 1 <= args.k_start <= args.k_end <= 20:
            raise ValueError("Component block must lie within 1..20")
        if args.action in ("finalize", "all") and (args.k_start != 1 or args.k_end != 20):
            raise ValueError("Finalization requires the complete component grid")
        datasets, searches = run_full_search(names, range(args.k_start, args.k_end+1))
        if args.action in ("finalize", "all"):
            if len(searches) != 2:
                raise ValueError("Freeze and test the two complete searches together")
            selected, manifest = freeze_selections(searches, datasets)
            print(selected.to_string(index=False), flush=True)
            final_evaluation(datasets, searches)


if __name__ == "__main__":
    main()
