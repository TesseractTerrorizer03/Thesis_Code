"""Chronological subset search, geometry controls, and surrogate calibration."""
from dataclasses import dataclass
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from geometry import Coordinates, Hyperplane, Sphere, Ellipsoid


@dataclass(frozen=True)
class Candidate:
    columns: tuple
    dimension: int | None = None

    @property
    def name(self):
        return ','.join(self.columns) + (' | raw' if self.dimension is None else f' | PCA={self.dimension}')


def make_candidates(config):
    features = config['features']
    if len(set(features)) != len(features):
        raise ValueError('Duplicate feature names.')
    pool = config['subset_pool']
    if not set(pool).issubset(features):
        raise ValueError('Subset pool contains unknown columns.')
    seeds = [tuple(cols) for cols in config['seed_subsets'].values()]
    mode = config['search_mode']
    subsets = set()
    if mode in ('seeds', 'seed_neighbors'):
        for seed in seeds:
            subsets.add(frozenset(seed))
            if mode == 'seed_neighbors':
                subsets.update(frozenset(set(seed) | {f}) for f in pool if f not in seed)
                subsets.update(frozenset(set(seed)-{f}) for f in seed if len(seed) > 2)
    elif mode == 'exhaustive':
        for size in config['subset_sizes']:
            subsets.update(frozenset(c) for c in combinations(pool, size))
    else:
        raise ValueError('Unknown search mode.')
    candidates = []
    for subset in sorted(subsets, key=lambda s: (len(s), tuple(sorted(s)))):
        if not subset.issubset(features) or len(subset) < 2:
            raise ValueError('Subsets must contain >=2 known features.')
        ordered = tuple(f for f in features if f in subset)
        for q in config['pca_dimensions']:
            if q is not None and (q < 2 or q >= len(ordered)):
                continue  # Full dimension is already represented by the raw candidate.
            candidates.append(Candidate(ordered, q))
    if not candidates or len(candidates) > config['max_candidates']:
        raise ValueError(f'{len(candidates)} candidates; revise the pool/grid or max_candidates explicitly.')
    return candidates


def chronological_splits(development, config):
    splitter = TimeSeriesSplit(n_splits=config['n_splits'], test_size=config['validation_size'], gap=config['gap'])
    splits = list(splitter.split(development))
    if min(len(tr) for tr, va in splits) < config['min_train_size']:
        raise ValueError('Insufficient initial training window.')
    rows = []
    for fold, (tr, va) in enumerate(splits, 1):
        assert tr[-1] < va[0]
        rows.append(dict(fold=fold, train_n=len(tr), train_start=development.index[tr[0]],
                         train_end=development.index[tr[-1]], validation_start=development.index[va[0]],
                         validation_end=development.index[va[-1]]))
    return splits, pd.DataFrame(rows)


def fit_surfaces(train, candidate, config, with_ellipsoid=False, seed=42):
    x = train[list(candidate.columns)].to_numpy()
    coordinates = Coordinates(config['scaling'], candidate.dimension).fit(x)
    z = coordinates.transform(x)
    # Fixed-size random training subsample limits nonlinear fitting cost.
    # Neither this sample nor preprocessing contains validation observations.
    count = min(len(z), config['fit_max_points'])
    indices = np.sort(np.random.default_rng(seed).choice(len(z), count, replace=False))
    fit_z = z[indices]
    sphere = Sphere(config['max_radius_rms'], config['max_center_rms'], config['sphere_max_nfev']).fit(fit_z)
    plane = Hyperplane().fit(fit_z)
    energy = float(np.mean(np.sum((z-z.mean(axis=0))**2, axis=1)))
    if energy <= 1e-12:
        raise ValueError('Degenerate training geometry.')
    ellipsoid = Ellipsoid(config['ellipsoid_max_nfev']).fit(fit_z, sphere) if with_ellipsoid else None
    return dict(coordinates=coordinates, sphere=sphere, plane=plane, ellipsoid=ellipsoid,
                energy=energy, candidate=candidate)


def score_surfaces(fitted, frame):
    candidate = fitted['candidate']
    z = fitted['coordinates'].transform(frame[list(candidate.columns)].to_numpy())
    out = pd.DataFrame(index=frame.index)
    for name in ('sphere', 'plane', 'ellipsoid'):
        if fitted[name] is not None:
            out[name] = fitted[name].signed_distance(z)**2 / fitted['energy']
    if not np.isfinite(out.to_numpy()).all():
        raise FloatingPointError('Nonfinite geometric distance.')
    diag = fitted['sphere'].diagnostics(z)
    diag.update(converged=fitted['sphere'].converged_, boundary=fitted['sphere'].boundary_,
                radius_rms=fitted['sphere'].radius_rms_, center_rms=fitted['sphere'].center_rms_,
                retained_variance=fitted['coordinates'].retained_variance_)
    if fitted['ellipsoid'] is not None:
        ell = fitted['ellipsoid']
        diag.update(ellipsoid_converged=ell.converged_, ellipsoid_boundary=ell.boundary_,
                    ellipsoid_axis_ratio=ell.axis_ratio_)
    return out, diag


def search_statistic(scores):
    sphere, plane = scores['sphere'].mean(), scores['plane'].mean()
    if plane <= 1e-12:
        return -np.inf  # An already-flat representation cannot establish curvature here.
    return float(np.log(plane / max(sphere, 1e-15)))


def screen_candidates(development, splits, candidates, config, keep_scores=True):
    rows, diagnostics, saved = [], [], []
    for candidate in candidates:
        parts, diags = [], []
        for fold, (tr, va) in enumerate(splits, 1):
            try:
                fitted = fit_surfaces(development.iloc[tr], candidate, config, seed=config['seed']+fold)
                scores, diag = score_surfaces(fitted, development.iloc[va])
                scores['fold'] = fold
                scores['candidate'] = candidate.name
                parts.append(scores)
                diag.update(candidate=candidate.name, fold=fold, error='')
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                diag = dict(candidate=candidate.name, fold=fold, converged=False, boundary=True, error=str(exc))
            diags.append(diag)
        diagnostics.extend(diags)
        complete = len(parts) == len(splits)
        if complete:
            scores = pd.concat(parts)
            eligible = all(d['converged'] and not d['boundary'] for d in diags)
            coverage = all(d['resultant'] <= config['max_resultant']
                           and d['angular_span'] >= config['min_angular_span'] for d in diags)
            rows.append(dict(candidate=candidate.name, feature_count=len(candidate.columns),
                             coordinate_dimension=candidate.dimension or len(candidate.columns),
                             representation='raw' if candidate.dimension is None else 'PCA',
                             statistic=search_statistic(scores), sphere_loss=scores.sphere.mean(),
                             plane_loss=scores.plane.mean(), fit_ok=eligible, coverage_ok=coverage,
                             thickness_ok=all(d['relative_shell_rmse'] <= config['max_relative_shell_rmse'] for d in diags),
                             max_relative_shell_rmse=max(d['relative_shell_rmse'] for d in diags),
                             max_radial_cv=max(d['radial_cv'] for d in diags),
                             min_retained_variance=min(d['retained_variance'] for d in diags)))
            if keep_scores:
                saved.append(scores)
        else:
            rows.append(dict(candidate=candidate.name, statistic=-np.inf, fit_ok=False, coverage_ok=False, thickness_ok=False))
    ranking = pd.DataFrame(rows).set_index('candidate').sort_values('statistic', ascending=False)
    scores = pd.concat(saved) if saved else pd.DataFrame()
    return ranking, pd.DataFrame(diagnostics), scores


def phase_surrogate(frame, rng, volatility=False, volatility_window=20):
    """Shared frequency phases preserve multivariate cross-spectra exactly.

    The volatility variant phases residuals after removing a common trailing RMS
    envelope and then restores that envelope. It approximates a different null;
    it does NOT preserve raw cross-spectra or marginal distributions exactly.
    Surrogates are conditional controls, not universal nonspherical processes.
    """
    x = frame.to_numpy(dtype=float)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    if np.any(scale < 1e-12):
        raise ValueError('Constant feature in surrogate source.')
    z = (x-mean)/scale
    envelope = np.ones(len(z))
    if volatility:
        power = pd.Series(np.mean(z*z, axis=1)).rolling(volatility_window, min_periods=1).mean().to_numpy()
        envelope = np.sqrt(np.maximum(power, 1e-6))
        z = z / envelope[:, None]
    spectrum = np.fft.rfft(z, axis=0)
    phase = rng.uniform(0, 2*np.pi, len(spectrum))
    phase[0] = 0.
    if len(z) % 2 == 0:
        phase[-1] = 0.  # Nyquist component must remain real.
    generated = np.fft.irfft(spectrum*np.exp(1j*phase[:, None]), n=len(z), axis=0)
    generated *= envelope[:, None]
    if volatility:
        generated = (generated-generated.mean(axis=0))/generated.std(axis=0)
    return pd.DataFrame(generated*scale+mean, index=frame.index, columns=frame.columns)


def calibrate_search(development, splits, candidates, config, progress=print):
    """Rerun the WHOLE declared subset/PCA search for every surrogate.

    Max statistics calibrate the candidate search, not only its observed winner.
    Filtering controls by coverage is deliberately avoided (conservative gate).
    """
    if config['n_surrogates'] < 1 or not config['null_families']:
        raise ValueError('Declare at least one surrogate family and repetition.')
    rows = []
    for family_number, family in enumerate(config['null_families']):
        if family not in ('phase', 'volatility_phase'):
            raise ValueError('Unknown surrogate family.')
        rng = np.random.default_rng(config['seed'] + 1000*(family_number+1))
        for repeat in range(config['n_surrogates']):
            surrogate = phase_surrogate(development, rng, family == 'volatility_phase', config['volatility_window'])
            ranking, diag, _ = screen_candidates(surrogate, splits, candidates, config, keep_scores=False)
            # Failed or boundary fits never silently shrink the calibrated search.
            converged = diag.groupby('candidate').converged.all()
            # A constrained boundary solution is ineligible in BOTH real and null searches.
            eligible = ranking.loc[ranking.fit_ok, 'statistic']
            if not converged.all():
                maximum = np.inf  # Numerically unresolved null search: conservative.
            else:
                maximum = eligible.max() if len(eligible) else -np.inf
            rows.append(dict(family=family, repeat=repeat+1, maximum_statistic=maximum,
                             eligible_candidates=len(eligible), unresolved=int((~converged).sum())))
            if (repeat+1) % config['progress_every'] == 0 or repeat+1 == config['n_surrogates']:
                progress(f'{family}: {repeat+1}/{config["n_surrogates"]} surrogate searches')
    return pd.DataFrame(rows)


def attach_search_calibration(ranking, nulls):
    ranking = ranking.copy()
    for family, rows in nulls.groupby('family'):
        maxima = rows.maximum_statistic.to_numpy()
        ranking[f'p_{family}'] = [(1 + np.sum(maxima >= row.statistic)) / (len(maxima)+1)
                                  if row.fit_ok else np.nan for row in ranking.itertuples()]
    return ranking


def bootstrap_difference(scores, comparator, block_length=20, n_boot=500, seed=42):
    """Paired moving-block resampling within folds; conditional approximate CI."""
    rng = np.random.default_rng(seed)
    differences = []
    for _, part in scores.groupby('fold', sort=True):
        differences.append((part.sphere-part[comparator]).to_numpy())
    draws = np.empty(n_boot)
    for repeat in range(n_boot):
        total, count = 0., 0
        for values in differences:
            n = len(values)
            b = min(block_length, n)
            starts = rng.integers(0, n, int(np.ceil(n/b)))
            indices = ((starts[:, None]+np.arange(b)) % n).ravel()[:n]
            total += values[indices].sum()
            count += n
        draws[repeat] = total/count
    values = np.concatenate(differences)
    return dict(mean=float(values.mean()), lower=float(np.quantile(draws, .025)),
                upper=float(np.quantile(draws, .975)), upper_one_sided=float(np.quantile(draws, .95)))


def compare_ellipsoid(development, splits, candidates, ranking, config, progress=print):
    """Fit expensive ellipsoid controls only for a predeclared top-L shortlist."""
    shortlist = ranking[ranking.fit_ok & ranking.coverage_ok].head(config['ellipsoid_shortlist'])
    mapping = {c.name: c for c in candidates}
    rows, saved, diagnostics = [], [], []
    for name in shortlist.index:
        candidate = mapping[name]
        parts, diags = [], []
        for fold, (tr, va) in enumerate(splits, 1):
            fitted = fit_surfaces(development.iloc[tr], candidate, config, with_ellipsoid=True,
                                  seed=config['seed']+fold)
            scores, diag = score_surfaces(fitted, development.iloc[va])
            scores['fold'], scores['candidate'] = fold, name
            diag.update(candidate=name, fold=fold)
            parts.append(scores)
            diags.append(diag)
        scores = pd.concat(parts)
        plane_ci = bootstrap_difference(scores, 'plane', config['bootstrap_block'], config['n_bootstrap'], config['seed'])
        ellipse_ci = bootstrap_difference(scores, 'ellipsoid', config['bootstrap_block'], config['n_bootstrap'], config['seed'])
        tolerance = config['ellipsoid_margin']*scores.plane.mean()
        rows.append(dict(candidate=name, sphere_minus_plane_upper=plane_ci['upper_one_sided'],
                         sphere_minus_ellipsoid=ellipse_ci['mean'],
                         sphere_minus_ellipsoid_upper=ellipse_ci['upper_one_sided'],
                         ellipsoid_tolerance=tolerance,
                         ellipsoid_noninferior=ellipse_ci['upper_one_sided'] <= tolerance,
                         ellipsoid_fit_ok=all(d['ellipsoid_converged'] and not d['ellipsoid_boundary'] for d in diags)))
        saved.append(scores)
        diagnostics.extend(diags)
        progress(f'Ellipsoid control complete: {name}')
    comparisons = pd.DataFrame(rows).set_index('candidate') if rows else pd.DataFrame()
    if not comparisons.empty:
        ranking = ranking.join(comparisons)
    else:
        for column in ('sphere_minus_plane_upper', 'ellipsoid_noninferior', 'ellipsoid_fit_ok'):
            ranking[column] = np.nan
    p_columns = [c for c in ranking.columns if c.startswith('p_')]
    calibrated = ranking[p_columns].le(config['alpha']).all(axis=1) if p_columns else pd.Series(False, index=ranking.index)
    ranking['candidate_supported'] = (ranking.fit_ok & ranking.coverage_ok & ranking.thickness_ok & calibrated
                                      & ranking.ellipsoid_fit_ok.eq(True)
                                      & ranking.ellipsoid_noninferior.eq(True)
                                      & ranking.sphere_minus_plane_upper.lt(0))
    supported = ranking[ranking.candidate_supported]
    selected = supported.index[0] if len(supported) else None
    return ranking, selected, (pd.concat(saved) if saved else pd.DataFrame()), pd.DataFrame(diagnostics)


def final_evaluation(development, holdout, candidate, config):
    """Single frozen candidate. Final surrogates never refit a geometry."""
    train = development.iloc[:-config['gap']] if config['gap'] else development
    fitted = fit_surfaces(train, candidate, config, with_ellipsoid=True, seed=config['seed'])
    scores, diagnostics = score_surfaces(fitted, holdout)
    scores['fold'] = 1
    diagnostics['statistic'] = search_statistic(scores)
    for i, family in enumerate(config['null_families']):
        rng = np.random.default_rng(config['seed']+9000+1000*i)
        statistics = []
        for _ in range(config['n_surrogates']):
            surrogate = phase_surrogate(holdout, rng, family == 'volatility_phase', config['volatility_window'])
            surrogate_scores, _ = score_surfaces(fitted, surrogate)
            statistics.append(search_statistic(surrogate_scores))
        diagnostics[f'p_{family}'] = float((1+np.sum(np.array(statistics) >= diagnostics['statistic']))/(len(statistics)+1))
    diagnostics['sphere_minus_plane_upper'] = bootstrap_difference(scores, 'plane', config['bootstrap_block'], config['n_bootstrap'], config['seed'])['upper_one_sided']
    diagnostics['sphere_minus_ellipsoid_upper'] = bootstrap_difference(scores, 'ellipsoid', config['bootstrap_block'], config['n_bootstrap'], config['seed'])['upper_one_sided']
    diagnostics['ellipsoid_tolerance'] = config['ellipsoid_margin']*scores.plane.mean()
    diagnostics['passes_confirmation'] = bool(
        diagnostics['converged'] and not diagnostics['boundary']
        and diagnostics['ellipsoid_converged'] and not diagnostics['ellipsoid_boundary']
        and diagnostics['resultant'] <= config['max_resultant']
        and diagnostics['angular_span'] >= config['min_angular_span']
        and diagnostics['relative_shell_rmse'] <= config['max_relative_shell_rmse']
        and diagnostics['sphere_minus_plane_upper'] < 0
        and diagnostics['sphere_minus_ellipsoid_upper'] <= diagnostics['ellipsoid_tolerance']
        and all(diagnostics[f'p_{f}'] <= config['alpha'] for f in config['null_families']))
    return scores, diagnostics
