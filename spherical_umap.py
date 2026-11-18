"""Chronological spherical UMAP experiment. No forecasts or supervised embeddings.

A row is a market date. Validation/test points are transformed into a map fitted
only on earlier observations. Spherical output is a modeling constraint, not a
natural-geometry test. The notebook is the user-facing experiment interface.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import inspect
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import warnings

import joblib as _joblib
import numpy as np
import pandas as pd
from scipy.linalg import helmert, orthogonal_procrustes
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from numba import njit
from umap import UMAP
from umap.umap_ import nearest_neighbors


def _atomic_joblib_dump(value, filename, **kwargs):
    """Publish complete files when identical runs overlap; preserve fit identities."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    os.close(descriptor)
    try:
        _joblib.dump(value, temporary, **kwargs)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return [str(path)]


# Cache transport is separate from numerically unchanged fitting functions.
joblib = SimpleNamespace(load=_joblib.load, dump=_atomic_joblib_dump)


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / '.cache' / 'spherical_umap'
DATA = ROOT / 'GMM' / 'data' / 'size_direction_2000'
CURRENCIES = ['EUR', 'USD', 'JPY', 'GBP', 'CHF', 'CAD', 'AUD', 'NZD', 'NOK', 'SEK']
GROUPS = {
    'Technology': 'MSFT AAPL NVDA ORCL IBM CSCO INTC QCOM TXN ADI'.split(),
    'Finance': 'JPM BAC WFC C GS MS PNC STT AXP SCHW'.split(),
    'Healthcare': 'JNJ PFE MRK ABT AMGN GILD BMY MDT UNH TMO'.split(),
    'Consumer': 'PG KO PEP CL KMB WMT COST HD LOW NKE'.split(),
    'Industrials': 'CAT DE HON GE UNP UPS FDX EMR ETN ITW'.split(),
}
DEFAULT_CONFIG = dict(neighbors=[15, 50], min_dist=.1, epochs=300,
                      selection_seed=42, stability_seeds=[42, 43, 44],
                      metric_ks=[5, 15, 50], primary_k=15, pair_samples=20000,
                      bootstrap_repeats=1500, bootstrap_blocks=[5, 20, 60],
                      surrogate_repeats=19, surrogate_families=['phase', 'volatility_envelope'],
                      volatility_window=20, noninferiority_margin=.02,
                      workers=4, synthetic_rows=900)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def versions():
    import platform
    return dict(python=platform.python_version(), **{
        p: version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn',
                                'umap-learn', 'numba', 'pynndescent', 'nbclient', 'matplotlib']})


@njit(cache=True)
def sphere_metric(a, b):
    """Unit-sphere geodesic distance and gradient in periodic latitude/longitude.

    xyz = (cos(lat) cos(lon), cos(lat) sin(lon), sin(lat)). Both angles may
    leave their conventional chart: the Cartesian map is still well defined.
    atan2 gives accurate distances near coincidence/antipodes; a small gradient
    denominator floor handles the nondifferentiable endpoints. Unlike the stock
    haversine gradient, this formula uses the same latitude convention everywhere.
    """
    ca, sa = np.cos(a[0]), np.sin(a[0])
    cb, sb = np.cos(b[0]), np.sin(b[0])
    dl = a[1] - b[1]
    dot = min(1., max(-1., sa*sb + ca*cb*np.cos(dl)))
    cross = np.sqrt(max(0., 1.-dot*dot))
    distance = np.arctan2(cross, dot)
    denominator = max(cross, 1e-10)
    gradient = np.array([-(ca*sb-sa*cb*np.cos(dl))/denominator,
                         ca*cb*np.sin(dl)/denominator])
    return distance, gradient


def to_xyz(angles):
    a = np.asarray(angles)
    return np.column_stack((np.cos(a[:, 0])*np.cos(a[:, 1]),
                            np.cos(a[:, 0])*np.sin(a[:, 1]), np.sin(a[:, 0])))


def angular_distances(a, b):
    """Stable angular distances for corresponding unit-vector rows."""
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=1),
                      np.clip(np.sum(a*b, axis=1), -1., 1.))


def fx_contrasts(quotes):
    """Daily log strength vs equal-weight geometric currency basket, then Helmert.

    Quotes are foreign units per EUR. Negative changes mean foreign appreciation
    against EUR. Adding EUR=0 and demeaning removes the arbitrary numeraire.
    """
    foreign = -np.log(quotes).diff()
    strength = pd.concat([pd.Series(0., index=quotes.index, name='EUR'), foreign], axis=1)
    strength = strength.sub(strength.mean(axis=1), axis=0)
    strength.iloc[0] = np.nan
    basis = helmert(len(strength.columns), full=False)
    contrasts = pd.DataFrame(strength.to_numpy() @ basis.T, index=quotes.index,
                            columns=[f'Helmert {i+1}' for i in range(len(basis))])
    return contrasts, strength, basis


def load_markets():
    """Read checked immutable caches. No automatic acquisition or refresh."""
    frames, inputs = {}, {}
    for name, filename in [('equities', 'equity_adjusted_prices.csv'), ('fx', 'ecb_eur_quotes.csv')]:
        path = DATA / filename
        provenance = json.loads(path.with_suffix('.provenance.json').read_text())
        if sha(path) != provenance['file_sha256']:
            raise ValueError(f'Input checksum mismatch: {path}')
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        if (not frame.index.is_unique or not frame.index.is_monotonic_increasing
                or not np.isfinite(frame.to_numpy()).all() or (frame <= 0).any().any()):
            raise ValueError(f'Invalid price cache: {path}')
        if name == 'equities':
            expected = sum(GROUPS.values(), [])
            if list(frame.columns) != expected:
                raise ValueError('Equity universe/order changed.')
            returns = np.log(frame).diff()
            asset_returns = returns
            basis = None
        else:
            if list(frame.columns) != CURRENCIES[1:]:
                raise ValueError('Currency universe/order changed.')
            returns, asset_returns, basis = fx_contrasts(frame)
        returns = returns.loc['2000-01-01':'2026-10-30'].copy()
        asset_returns = asset_returns.reindex(returns.index).copy()
        if not np.isfinite(returns.to_numpy()).all():
            raise ValueError('Missing daily returns; no filling or silent deletion.')
        if returns.index[-1] != pd.Timestamp('2026-10-30'):
            raise ValueError('Unexpected final date.')
        a, b = int(.6*len(returns)), int(.8*len(returns))
        frames[name] = dict(raw=returns, assets=asset_returns, basis=basis,
                            validation_start=returns.index[a], test_start=returns.index[b])
        inputs[name] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path), provenance=provenance)
    return frames, inputs


def trend_features(asset_returns):
    """Per asset: trailing log momentum 21/63/126, price/MA200-1, vol21.

    FX asset prices are cumulative basket-relative log strengths, exponentiated;
    no directional meaning is assigned to positive currency-basket volatility.
    All descriptors are invariant to a constant multiplicative price level.
    """
    prices = np.exp(asset_returns.cumsum())
    pieces = {}
    for h in (21, 63, 126):
        pieces[f'momentum_{h}'] = asset_returns.rolling(h, min_periods=h).sum()
    pieces['gap_200'] = prices/prices.rolling(200, min_periods=200).mean()-1
    pieces['volatility_21'] = asset_returns.rolling(21, min_periods=21).std()*np.sqrt(252)
    return pd.concat(pieces, axis=1).set_axis(
        [f'{descriptor}::{asset}' for descriptor, frame in pieces.items() for asset in frame.columns], axis=1)


def feature_catalog(markets):
    rows = []
    definitions = {'momentum_21': 'sum of trailing 21 daily log returns',
                   'momentum_63': 'sum of trailing 63 daily log returns',
                   'momentum_126': 'sum of trailing 126 daily log returns',
                   'gap_200': 'exp(cumulative log return) / trailing 200-session mean price - 1',
                   'volatility_21': 'sample std of trailing 21 daily log returns times sqrt(252)'}
    for market, data in markets.items():
        for c in data['raw']:
            rows.append(dict(market=market, family='returns', feature=c, definition='daily log return' if market=='equities' else 'orthonormal contrast of basket-relative currency log strengths'))
        for c in trend_features(data['assets']):
            descriptor, asset = c.split('::')
            rows.append(dict(market=market, family='trend', feature=c, definition=f'{asset}: {definitions[descriptor]}'))
    return pd.DataFrame(rows)


def datasets(markets):
    result = {}
    for name, market in markets.items():
        for family in ('returns', 'trend'):
            frame = market['raw'].copy() if family == 'returns' else trend_features(market['assets'])
            # Only the declared rolling warmup may be removed.
            if family == 'trend':
                frame = frame.iloc[199:]
            if not np.isfinite(frame.to_numpy()).all():
                raise ValueError(f'Nonfinite {name}/{family} after declared warmup.')
            result[f'{name}/{family}'] = dict(market=name, family=family, frame=frame,
                                             validation_start=market['validation_start'],
                                             test_start=market['test_start'])
    return result


def prepare(data, metric, stage='validation'):
    """Train-only column scaling, optional directional normalization; fixed dates.

    Zero direction vectors are rejected with their dates. They have no direction;
    dropping them would change the experiment's dates and cosine neighborhoods.
    Euclidean zero vectors remain valid.
    """
    frame = data['frame']
    boundary = data['validation_start'] if stage == 'validation' else data['test_start']
    training = frame.loc[frame.index < boundary]
    query = frame.loc[(frame.index >= boundary) & (frame.index < data['test_start'])] if stage == 'validation' else frame.loc[frame.index >= boundary]
    if len(training) < 100 or len(query) < 60:
        raise ValueError('Insufficient chronological observations.')
    scaler = StandardScaler().fit(training) if data['family'] == 'trend' else None
    x = scaler.transform(training) if scaler is not None else training.to_numpy(copy=True)
    y = scaler.transform(query) if scaler is not None else query.to_numpy(copy=True)
    norms = (np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1))
    if metric == 'cosine':
        for z, norm, index in [(x, norms[0], training.index), (y, norms[1], query.index)]:
            bad = norm <= 1e-12
            if bad.any():
                raise ValueError(f'Undefined zero-vector directions: {list(index[bad][:5])}')
            z /= norm[:, None]
    elif metric != 'euclidean':
        raise ValueError('Input metric must be cosine or euclidean.')
    return dict(x=np.asarray(x, dtype=np.float32), y=np.asarray(y, dtype=np.float32),
                train_dates=training.index, query_dates=query.index, scaler=scaler,
                train_norm=norms[0], query_norm=norms[1], metric=metric)


def _engine_id():
    functions = [sphere_metric.py_func, to_xyz, prepare, fit_embedding, graph_for, _fit_options]
    text = '\n'.join(inspect.getsource(f) for f in functions)
    return hashlib.sha256((text+json.dumps(versions(), sort_keys=True)).encode()).hexdigest()[:16]


def _key(prepared, settings):
    h = hashlib.sha256()
    for a in (prepared['x'], prepared['y']):
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    h.update(json.dumps(settings, sort_keys=True).encode())
    h.update(_engine_id().encode())
    return h.hexdigest()


def graph_for(prepared, config):
    """Reuse one training-only approximate neighbor graph across output geometries.

    The search index is required for UMAP's frozen transform. Layout seed changes
    hold this graph fixed so the stability diagnostic measures layout variation.
    """
    k = max(config['neighbors'])+1
    key = _key(dict(x=prepared['x'], y=np.empty((0, 0), np.float32)),
               dict(graph_k=k, metric=prepared['metric'], seed=config['selection_seed']))
    path = CACHE / 'graphs' / f'{key}.joblib'
    if path.exists():
        return joblib.load(path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        graph = nearest_neighbors(prepared['x'], k, prepared['metric'], {},
                                  prepared['metric']=='cosine', np.random.RandomState(config['selection_seed']),
                                  low_memory=True, use_pynndescent=True, n_jobs=1, verbose=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(graph, path, compress=0)
    return graph


def _fit_options(method, neighbors, seed, config):
    return dict(method=method, neighbors=int(neighbors), seed=int(seed),
                min_dist=config['min_dist'], epochs=config['epochs'],
                graph_k=max(config['neighbors'])+1, graph_seed=config['selection_seed'])


def fit_embedding(prepared, method, neighbors, seed, config, graph=None):
    """Fit on x, transform y without updating training embedding. Return reducer."""
    x, y = prepared['x'], prepared['y']
    if method.startswith('PCA'):
        reducer = PCA(n_components=int(method[-1]), svd_solver='full')
        a = reducer.fit_transform(x)
        b = reducer.transform(y)
    else:
        if graph is None:
            graph = graph_for(prepared, config)
        init = 'random'
        output = 'euclidean'
        if method == 'Spherical UMAP':
            rng = np.random.default_rng(seed)
            init = np.column_stack([np.arcsin(rng.uniform(-1, 1, len(x))),
                                    rng.uniform(-np.pi, np.pi, len(x))]).astype(np.float32)
            output = sphere_metric
        elif method != 'Planar UMAP':
            raise ValueError(f'Unknown method: {method}')
        reducer = UMAP(n_components=2, n_neighbors=neighbors, min_dist=config['min_dist'],
                       metric=prepared['metric'], output_metric=output, n_epochs=config['epochs'],
                       init=init, random_state=seed, transform_seed=seed, n_jobs=1,
                       precomputed_knn=graph, low_memory=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            a = reducer.fit_transform(x)
            original = reducer.embedding_.copy()
            b = reducer.transform(y)
            if not np.array_equal(original, reducer.embedding_):
                raise RuntimeError('Frozen transform changed the training map.')
        reducer.experiment_warnings_ = [str(w.message) for w in caught]
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError('Nonfinite embedding; fit cannot be scored.')
    return a, b, reducer


def cached_fit(prepared, method, neighbors, seed, config, graph=None):
    settings = _fit_options(method, neighbors, seed, config)
    key = _key(prepared, settings)
    path = CACHE / 'fits' / f'{key}.npz'
    meta = path.with_suffix('.json')
    if path.exists() and meta.exists():
        with np.load(path) as f:
            return f['training'], f['query'], json.loads(meta.read_text())
    start = time.perf_counter()
    try:
        a, b, reducer = fit_embedding(prepared, method, neighbors, seed, config, graph)
    except Exception as exc:
        write_json(CACHE/'failures'/f'{key}.json',dict(settings=settings,cache_key=key,
                   error=f'{type(exc).__name__}: {exc}',seconds=time.perf_counter()-start))
        raise
    info = dict(settings=settings, cache_key=key, seconds=time.perf_counter()-start,
                warnings=getattr(reducer, 'experiment_warnings_', []), frozen_transform=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + f'.{os.getpid()}.tmp.npz')
    np.savez_compressed(tmp, training=a, query=b)
    tmp.replace(path)
    write_json(meta, info)
    return a, b, info


def reference(prepared, config, full=True):
    """Original-space neighbors of each held-out query among training dates."""
    n = len(prepared['x'])
    maximum = min(max(config['metric_ks']), n-1)
    d = pairwise_distances(prepared['y'], prepared['x'], metric=prepared['metric'], n_jobs=1)
    order = np.argsort(d, axis=1, kind='stable')
    result = dict(neighbors=order[:, :maximum], n_train=n)
    if full:
        ranks = np.empty(order.shape, dtype=np.int32)
        np.put_along_axis(ranks, order, np.arange(1, n+1)[None, :], axis=1)
        result['ranks'] = ranks
    return result


def embedding_neighbors(training, query, method, k):
    if method == 'Spherical UMAP':
        training, query = to_xyz(training), to_xyz(query)
    # Chord distance and geodesic distance have identical neighbor order on S2.
    return NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=1).fit(training).kneighbors(query, return_distance=False)


def score_embedding(prepared, training, query, method, ref, config, full=True):
    ks = config['metric_ks'] if full else [config['primary_k']]
    neighbors = embedding_neighbors(training, query, method, max(ks))
    result = pd.DataFrame(index=prepared['query_dates'])
    for k in ks:
        a, b = ref['neighbors'][:, :k], neighbors[:, :k]
        result[f'recall_{k}'] = np.array([len(np.intersect1d(i, j, assume_unique=True))/k for i,j in zip(a,b)])
        if full:
            ranks = np.take_along_axis(ref['ranks'], b, axis=1)
            penalty = np.maximum(ranks-k, 0).sum(axis=1)
            # Inductive trustworthiness: N training candidates, ranks 1..N.
            result[f'trustworthiness_{k}'] = 1.-2.*penalty/(k*(2*ref['n_train']-3*k+1))
    summary = {c: float(result[c].mean()) for c in result}
    if full:
        rng = np.random.default_rng(175)
        i = rng.integers(0, len(query), config['pair_samples'])
        j = rng.integers(0, len(query)-1, config['pair_samples']); j += j >= i
        if prepared['metric'] == 'cosine':
            original = 1.-np.sum(prepared['y'][i]*prepared['y'][j], axis=1)
        else:
            original = np.linalg.norm(prepared['y'][i]-prepared['y'][j], axis=1)
        embedded = angular_distances(to_xyz(query[i]), to_xyz(query[j])) if method=='Spherical UMAP' else np.linalg.norm(query[i]-query[j], axis=1)
        correlation = spearmanr(original, embedded).statistic
        summary['distance_rank_correlation'] = float(correlation) if np.isfinite(correlation) else None
    return result, summary


def select_track(data, metric, config, full=True):
    """Selection sees training and validation only. No test arrays are returned."""
    prepared = prepare(data, metric)
    ref = reference(prepared, config, full=full)
    graph = graph_for(prepared, config)
    trials, selected, embeddings, daily, fits, failures = [], {}, {}, {}, [], []
    for method in ['Spherical UMAP', 'Planar UMAP']:
        eligible = []
        for neighbors in config['neighbors']:
            try:
                a,b,info = cached_fit(prepared, method, neighbors, config['selection_seed'], config, graph)
                scores, summary = score_embedding(prepared,a,b,method,ref,config,full=full)
                trials.append(dict(method=method, neighbors=neighbors, status='ok', **summary))
                eligible.append((summary[f'recall_{config["primary_k"]}'], -neighbors, a,b,scores,info,summary))
            except Exception as exc:
                failure = dict(method=method, neighbors=neighbors, status='failed', error=f'{type(exc).__name__}: {exc}')
                trials.append(failure); failures.append(failure)
        # A failed grid member makes selection unresolved, rather than shrinking
        # the declared search opportunistically (also applies to controls).
        if len(eligible) != len(config['neighbors']):
            raise RuntimeError(f'Unresolved {method} selection: {failures}')
        best = max(eligible, key=lambda x:(x[0], x[1]))
        selected[method] = int(-best[1])
        embeddings[method] = (best[2],best[3])
        daily[method] = best[4]
        fits.append(best[5])
    if full:
        for method in ['PCA2', 'PCA3']:
            a,b,info = cached_fit(prepared, method, 0, config['selection_seed'], config)
            daily[method], summary = score_embedding(prepared,a,b,method,ref,config)
            trials.append(dict(method=method, neighbors=0, status='ok', **summary))
            embeddings[method] = (a,b); fits.append(info)
    gain = float((daily['Spherical UMAP'][f'recall_{config["primary_k"]}']-
                  daily['Planar UMAP'][f'recall_{config["primary_k"]}']).mean())
    return dict(selected=selected, trials=trials, embeddings=embeddings, daily=daily,
                fits=fits, failures=failures, gain=gain, prepared=prepared)


def block_interval(values, block, repeats, seed=42):
    """Paired circular block bootstrap, conditional on fixed fitted models."""
    x = np.asarray(values, float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError('Bootstrap requires finite nonempty scores.')
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(x), (repeats, int(np.ceil(len(x)/block))))
    index = (starts[:,:,None]+np.arange(block)[None,None,:]) % len(x)
    means = x[index.reshape(repeats, -1)[:,:len(x)]].mean(axis=1)
    lo,hi = np.quantile(means,[.025,.975])
    return dict(mean=float(x.mean()), lower=float(lo), upper=float(hi))


def comparisons(daily, config, stage, track, metric):
    rows = []
    for baseline in ['Planar UMAP', 'PCA2', 'PCA3']:
        for column in [c for c in daily['Spherical UMAP'] if c.startswith('recall_') or c.startswith('trustworthiness_')]:
            diff = daily['Spherical UMAP'][column]-daily[baseline][column]
            for block in config['bootstrap_blocks']:
                rows.append(dict(track=track, metric=metric, stage=stage, baseline=baseline,
                                 score=column, block=block,
                                 **block_interval(diff,block,config['bootstrap_repeats'])))
    return rows


def stability(data, metric, result, config):
    prepared = result['prepared']
    graph = graph_for(prepared,config)
    rows = []
    k = config['primary_k']
    ref=reference(prepared,config,full=False)
    for method in ['Spherical UMAP','Planar UMAP']:
        a0,b0 = result['embeddings'][method]
        n0 = embedding_neighbors(a0,b0,method,k)
        for seed in config['stability_seeds']:
            a,b,info = cached_fit(prepared,method,result['selected'][method],seed,config,graph)
            n = embedding_neighbors(a,b,method,k)
            recall=float(np.mean([len(np.intersect1d(i,j,assume_unique=True))/k
                                  for i,j in zip(ref['neighbors'][:,:k],n)]))
            overlap = np.mean([len(np.intersect1d(i,j,assume_unique=True))/k for i,j in zip(n0,n)])
            aligned = None
            if method == 'Spherical UMAP':
                # Rotation/reflection fitted on training anchors only.
                rotation,_ = orthogonal_procrustes(to_xyz(a),to_xyz(a0))
                aligned = float(angular_distances(to_xyz(b)@rotation,to_xyz(b0)).mean())
            rows.append(dict(method=method,seed=seed,neighbor_overlap=float(overlap),
                             aligned_angular_error=aligned,recall_15=recall,seconds=info['seconds']))
    return rows


def phase_surrogate(raw, family, rng, window=20):
    """Common Fourier phases preserve cross-spectra; envelope variant is approximate.

    Call independently on training and validation segments: validation values must
    not enter the surrogate training transform through the Fourier operation.
    """
    x = raw.to_numpy(float)
    mean = x.mean(axis=0)
    z = x-mean
    envelope = np.ones(len(z))
    if family == 'volatility_envelope':
        sd = z.std(axis=0)
        if (sd <= 1e-12).any():
            raise ValueError('Constant raw input.')
        energy = pd.Series(np.mean((z/sd)**2,axis=1)).rolling(window,min_periods=1).mean()
        envelope = np.sqrt(np.maximum(energy.to_numpy(),1e-6))
        z = z/envelope[:,None]
    elif family != 'phase':
        raise ValueError('Unknown surrogate family.')
    spectrum = np.fft.rfft(z,axis=0)
    phase = rng.uniform(0,2*np.pi,len(spectrum)); phase[0]=0.
    if len(z)%2 == 0: phase[-1]=0.
    generated = np.fft.irfft(spectrum*np.exp(1j*phase[:,None]),n=len(z),axis=0)*envelope[:,None]+mean
    source = np.fft.rfft(x-mean,axis=0)
    target = np.fft.rfft(generated-generated.mean(axis=0),axis=0)
    aa = np.sum(abs(source)**2,axis=1); bb = np.sum(abs(target)**2,axis=1)
    inner = np.sum(source*np.conj(target),axis=1)
    error = np.sqrt(np.maximum(aa*aa+bb*bb-2*abs(inner)**2,0.).sum()/max((aa*aa).sum(),1e-30))
    return pd.DataFrame(generated,index=raw.index,columns=raw.columns),float(error)


def surrogate_market(market, family, repeat, config):
    raw = market['raw'].loc[market['raw'].index < market['test_start']]
    boundary = market['validation_start']
    rng = np.random.default_rng(94000+repeat+1000*config['surrogate_families'].index(family))
    parts, errors = [],[]
    for segment in (raw.loc[raw.index<boundary],raw.loc[raw.index>=boundary]):
        out,error = phase_surrogate(segment,family,rng,config['volatility_window'])
        parts.append(out); errors.append(error)
    generated = pd.concat(parts)
    if market['basis'] is None:
        assets = generated.copy()
    else:
        assets = pd.DataFrame(generated.to_numpy()@market['basis'],index=generated.index,columns=CURRENCIES)
    return dict(raw=generated,assets=assets,basis=market['basis'],
                validation_start=boundary,test_start=market['test_start']),errors


def _control_worker(name, market, family, repeat, config):
    identity = dict(market=name,family=family,repeat=repeat)
    # Job cache also covers scoring and feature-building code, not just UMAP fits.
    key = hashlib.sha256((sha(__file__)+json.dumps(config,sort_keys=True)+json.dumps(identity,sort_keys=True)+
                          hashlib.sha256(market['raw'].to_numpy().tobytes()).hexdigest()).encode()).hexdigest()
    path = CACHE/'controls'/f'{key}.json'
    if path.exists():
        return json.loads(path.read_text())
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        try:
            surrogate,errors = surrogate_market(market,family,repeat,config)
            if family=='phase' and max(errors)>1e-6:
                raise RuntimeError(f'Shared-phase cross-spectrum preservation failed: {errors}')
            tracks = datasets({name:surrogate})
            trials=[]; gains=[]
            for track,data in tracks.items():
                for metric in ['cosine','euclidean']:
                    result=select_track(data,metric,config,full=False)
                    gains.append(result['gain'])
                    for trial in result['trials']:
                        trials.append(dict(track=track,metric=metric,**trial))
            row=dict(**identity,status='ok',maximum_gain=max(gains),spectral_error_train=errors[0],
                     spectral_error_validation=errors[1],seconds=time.perf_counter()-start,trials=trials)
        except Exception as exc:
            # +infinity is represented by null/status, so numerical failures
            # count as an exceedance for every observed statistic.
            row=dict(**identity,status='unresolved',maximum_gain=None,error=f'{type(exc).__name__}: {exc}',
                     seconds=time.perf_counter()-start,trials=[])
    write_json(path,row)
    return row


def run_controls(markets,config,run_dir):
    jobs=[(name,market,family,r,config) for name,market in markets.items()
          for family in config['surrogate_families'] for r in range(config['surrogate_repeats'])]
    results=[]
    with ProcessPoolExecutor(max_workers=config['workers']) as pool:
        futures={pool.submit(_control_worker,*args):args for args in jobs}
        for future in as_completed(futures):
            row=future.result();results.append(row)
            print(f"Control {len(results)}/{len(jobs)}: {row['market']} {row['family']} {row['repeat']} {row['status']}",flush=True)
            write_json(Path(run_dir)/'control_checkpoint.json',results)
    return sorted(results,key=lambda r:(r['market'],r['family'],r['repeat']))


def descriptors(markets):
    result={}
    for market,data in markets.items():
        r=data['assets']
        out=pd.DataFrame(index=r.index)
        out['magnitude']=np.linalg.norm(data['raw'].to_numpy(),axis=1)
        out['date']=np.arange(len(r),dtype=float)
        if market=='equities':
            out['equity_direction']=r.mean(axis=1)
            out['breadth']=(r>0).mean(axis=1)
            group={name:r[columns].mean(axis=1) for name,columns in GROUPS.items()}
            for name,series in group.items():out[f'{name}_vs_market']=series-r.mean(axis=1)
        else:
            for currency in CURRENCIES:out[f'{currency}_strength_21']=r[currency].rolling(21,min_periods=21).sum()
            out['currency_dispersion']=r.std(axis=1)
        result[market]=out
    return result


def interpretation_metrics(prepared,training,query,method,description,config):
    """Descriptive held-out local coherence, never a tuning objective.

    Neighbor means use training descriptor values only. 'Explained fraction'
    is 1-MSE/variance about the training mean and can be negative. Descriptors
    derived from inputs are not independent validation evidence.
    """
    k=config['primary_k']
    neighbors=embedding_neighbors(training,query,method,k)
    train=description.reindex(prepared['train_dates'])
    validation=description.reindex(prepared['query_dates'])
    rows=[]
    for column in train:
        if column=='date':continue
        x=train[column].to_numpy();y=validation[column].to_numpy()
        prediction=np.nanmean(x[neighbors],axis=1)
        valid=np.isfinite(prediction)&np.isfinite(y)
        denominator=np.mean((y[valid]-np.nanmean(x))**2)
        rows.append(dict(method=method,descriptor=column,
                         local_explained_fraction=float(1-np.mean((prediction[valid]-y[valid])**2)/max(denominator,1e-20))))
    return rows


def synthetic_examples(config):
    rng=np.random.default_rng(208)
    n=config['synthetic_rows']
    sphere=rng.normal(size=(n,3));sphere/=np.linalg.norm(sphere,axis=1)[:,None]
    plane=np.column_stack([rng.uniform(-1,1,(n,2)),np.zeros(n)])
    cloud=rng.normal(size=(n,3))
    rotation,_=np.linalg.qr(rng.normal(size=(8,3)))
    rows=[];examples={}
    for name,latent in [('Spherical surface',sphere),('Plane',plane),('Gaussian cloud',cloud)]:
        x=latent@rotation.T+rng.normal(0,.01,(n,8))
        index=pd.bdate_range('2000-01-03',periods=n)
        data=dict(frame=pd.DataFrame(x,index=index),family='returns',validation_start=index[int(.6*n)],test_start=index[int(.8*n)])
        result=select_track(data,'euclidean',config,full=True)
        for method in result['daily']:
            rows.append(dict(source=name,method=method,recall_15=float(result['daily'][method].recall_15.mean()),
                             output_radius_error=float(abs(np.linalg.norm(to_xyz(result['embeddings'][method][1]),axis=1)-1).max()) if method=='Spherical UMAP' else None))
        xyz=to_xyz(result['embeddings']['Spherical UMAP'][1])
        examples[name]=dict(result=result,color=latent[int(.6*n):int(.8*n),2])
    return pd.DataFrame(rows),examples


def run_observed(markets,config,run_dir):
    tracks=datasets(markets);desc=descriptors(markets)
    results={};trials=[];selected={};fit_log=[];metrics=[];daily_rows=[];contrasts=[];stabilities=[];periods=[];profiles=[];norm_rows=[]
    # Freeze all validation choices before accessing any retrospective test scores.
    for track,data in tracks.items():
        for metric in ['cosine','euclidean']:
            print(f'Validation: {track} / {metric}',flush=True)
            with threadpool_limits(limits=1):
                result=select_track(data,metric,config)
                result['stability']=stability(data,metric,result,config)
            key=f'{track}/{metric}';results[key]=result;selected[key]=result['selected']
            for row in result['trials']:trials.append(dict(track=track,metric=metric,**row))
    write_json(Path(run_dir)/'frozen_selections.json',dict(config=config,selections=selected,
               note='All validation choices frozen before retrospective test scoring; historical test period previously examined.'))
    for track,data in tracks.items():
        for metric in ['cosine','euclidean']:
            key=f'{track}/{metric}';result=results[key]
            for row in result['stability']:stabilities.append(dict(track=track,metric=metric,**row))
            for stage in ['validation','retrospective_test']:
                with threadpool_limits(limits=1):
                    if stage=='validation':
                        prepared=result['prepared'];daily=result['daily'];embeddings=result['embeddings'];infos=result['fits']
                    else:
                        prepared=prepare(data,metric,stage='test');ref=reference(prepared,config)
                        graph=graph_for(prepared,config);daily={};embeddings={};infos=[]
                        for method in ['Spherical UMAP','Planar UMAP','PCA2','PCA3']:
                            neighbors=result['selected'].get(method,0)
                            a,b,info=cached_fit(prepared,method,neighbors,config['selection_seed'],config,graph)
                            daily[method],_=score_embedding(prepared,a,b,method,ref,config)
                            embeddings[method]=(a,b);infos.append(info)
                        result['test']=dict(prepared=prepared,daily=daily,embeddings=embeddings)
                    contrasts.extend(comparisons(daily,config,stage,track,metric))
                    report_reference=reference(prepared,config)
                    for info in infos:fit_log.append(dict(track=track,metric=metric,stage=stage,**info))
                    for method,scores in daily.items():
                        _,summary=score_embedding(prepared,*embeddings[method],method,report_reference,config)
                        metrics.append(dict(track=track,metric=metric,stage=stage,method=method,**summary))
                        dates=scores.reset_index(names='date');dates['track']=track;dates['metric']=metric;dates['stage']=stage;dates['method']=method
                        daily_rows.append(dates)
                        profiles.extend(dict(track=track,metric=metric,stage=stage,**r) for r in interpretation_metrics(prepared,*embeddings[method],method,desc[data['market']],config))
                        # Three consecutive validation periods, conditional on the same frozen map.
                        if stage=='validation':
                            for part,positions in enumerate(np.array_split(np.arange(len(scores)),3),1):
                                periods.append(dict(track=track,metric=metric,method=method,period=part,
                                                    start=str(scores.index[positions[0]].date()),end=str(scores.index[positions[-1]].date()),
                                                    recall_15=float(scores.iloc[positions].recall_15.mean())))
                    for kind,dates,norm in [('training',prepared['train_dates'],prepared['train_norm']),('query',prepared['query_dates'],prepared['query_norm'])]:
                        norm_rows.append(pd.DataFrame(dict(date=dates,norm=norm,role=kind,track=track,metric=metric,stage=stage)))
    tables=dict(validation_trials=pd.DataFrame(trials),metrics=pd.DataFrame(metrics),daily_scores=pd.concat(daily_rows,ignore_index=True),
                comparisons=pd.DataFrame(contrasts),seed_stability=pd.DataFrame(stabilities),validation_periods=pd.DataFrame(periods),
                descriptor_profiles=pd.DataFrame(profiles),input_norms=pd.concat(norm_rows,ignore_index=True))
    for name,frame in tables.items():frame.to_csv(Path(run_dir)/f'{name}.csv',index=False)
    write_json(Path(run_dir)/'fit_log.json',fit_log)
    return results,tables


def scorecards(tables,controls,config):
    rows=[];nulls=pd.DataFrame([{k:v for k,v in r.items() if k!='trials'} for r in controls])
    for (track,metric),g in tables['comparisons'].query("stage=='retrospective_test' and baseline=='Planar UMAP' and score=='recall_15' and block==20").groupby(['track','metric']):
        row=g.iloc[0]
        if row.lower>0:conclusion='Spherical improvement'
        elif row.lower >= -config['noninferiority_margin']:conclusion='Comparable within 2-point margin'
        elif row.upper < -config['noninferiority_margin']:conclusion='Spherical degradation'
        else:conclusion='Inconclusive comparison'
        validation=tables['comparisons'].query("stage=='validation' and baseline=='Planar UMAP' and score=='recall_15' and block==20")
        observed=float(validation[(validation.track==track)&(validation.metric==metric)]['mean'].iloc[0])
        item=dict(track=track,metric=metric,test_gain=float(row['mean']),lower=float(row.lower),upper=float(row.upper),
                  validation_gain=observed,conclusion=conclusion)
        market=track.split('/')[0]
        for family in config['surrogate_families']:
            group=nulls[(nulls.market==market)&(nulls.family==family)]
            exceed=(group.status.ne('ok')|group.maximum_gain.ge(observed)).sum()
            item[f'tail_{family}']=float((1+exceed)/(1+len(group)))
            item[f'unresolved_{family}']=int(group.status.ne('ok').sum())
        rows.append(item)
    return pd.DataFrame(rows)


def run_experiment(config=None,run_dir=None):
    config={**DEFAULT_CONFIG,**(config or {})}
    markets,inputs=load_markets()
    if run_dir is None:
        run_dir=ROOT/'spherical_umap_results'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run_dir=Path(run_dir);run_dir.mkdir(parents=True,exist_ok=True)
    protocol=dict(config=config,inputs=inputs,code_sha256=sha(__file__),versions=versions(),
                  question='Does spherical UMAP preserve meaningful, stable market structure?',
                  interpretation='Retrospective representation validation; no natural-sphere or forecast claim.',
                  query_neighbors='Held-out query neighbors among training dates; no query-to-query graph tuning.',
                  uncertainty='Pointwise paired circular block bootstrap conditional on fitted models; validation intervals post-selection.')
    write_json(run_dir/'protocol.json',protocol)
    splits=[]
    for track,data in datasets(markets).items():
        f=data['frame']
        for label,mask in [('train',f.index<data['validation_start']),('validation',(f.index>=data['validation_start'])&(f.index<data['test_start'])),('retrospective_test',f.index>=data['test_start'])]:
            index=f.index[mask];splits.append(dict(track=track,split=label,n=len(index),start=str(index[0].date()),end=str(index[-1].date())))
    pd.DataFrame(splits).to_csv(run_dir/'splits.csv',index=False)
    feature_catalog(markets).to_csv(run_dir/'feature_catalog.csv',index=False)
    results,tables=run_observed(markets,config,run_dir)
    write_json(run_dir/'status.json',dict(observed='complete',controls='running',synthetic='pending'))
    controls=run_controls(markets,config,run_dir)
    write_json(run_dir/'controls.json',controls)
    flat=[dict(**{k:v for k,v in row.items() if k!='trials'}) for row in controls]
    pd.DataFrame(flat).to_csv(run_dir/'control_maxima.csv',index=False)
    pd.DataFrame([dict(market=row['market'],family=row['family'],repeat=row['repeat'],**trial) for row in controls for trial in row['trials']]).to_csv(run_dir/'control_trials.csv',index=False)
    cards=scorecards(tables,controls,config);cards.to_csv(run_dir/'scorecards.csv',index=False)
    print('Synthetic examples',flush=True)
    with threadpool_limits(limits=1):synthetic,examples=synthetic_examples(config)
    synthetic.to_csv(run_dir/'synthetic_metrics.csv',index=False)
    # Data-only results bundle: no fitted-model pickle is needed to view the report.
    bundle=dict(run_dir=str(run_dir),config=config,results=results,tables=tables,scorecards=cards,
                controls=controls,synthetic=synthetic,examples=examples,markets=markets,descriptors=descriptors(markets))
    # StandardScaler objects in prepared state contain training-only statistics.
    joblib.dump(bundle,run_dir/'report_bundle.joblib',compress=3)
    write_json(run_dir/'status.json',dict(observed='complete',controls='complete',synthetic='complete',notebook='pending'))
    print(f'Experiment complete: {run_dir}',flush=True)
    return bundle


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--workers',type=int,default=DEFAULT_CONFIG['workers'])
    args=parser.parse_args()
    run_experiment(dict(workers=args.workers),args.run_dir)
