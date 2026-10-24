"""Geometry first: unnormalized point clouds, surface errors, and raw-data controls."""
from dataclasses import dataclass
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import least_squares
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import adjusted_rand_score, silhouette_score
from geometry import Coordinates, Sphere, Hyperplane, ellipsoid_signed_distance
from trend_features import build_trend_features


@dataclass(frozen=True)
class Candidate:
    id: str
    columns: tuple
    dimension: int | None
    description: str


def candidates_from_groups(groups, config):
    subsets = {}
    order = list(dict.fromkeys(c for cols in groups.values() for c in cols))
    for name, cols in groups.items():
        subsets.setdefault(frozenset(cols), name)
        if config['search_mode'] == 'seed_deletions':
            for col in cols:
                if len(cols)-1 >= config['min_subset_size']:
                    subsets.setdefault(frozenset(set(cols)-{col}), f'{name} without {col}')
        elif config['search_mode'] != 'seeds':
            raise ValueError('Use seeds or seed_deletions.')
    out = []
    for subset, description in subsets.items():
        columns = tuple(c for c in order if c in subset)
        for q in config['pca_dimensions']:
            if q is not None and (q < 2 or q >= len(columns)):
                continue
            out.append(Candidate(f'C{len(out)+1:03d}', columns, q,
                                 description+(' / original coordinates' if q is None else f' / PCA {q}')))
    if not out or len(out) > config['max_candidates']:
        raise ValueError('Revise the candidate budget or search configuration.')
    return out


class FreeEllipsoid:
    """Learn center, semiaxes, AND orientation; evaluate true orthogonal distance.

    Training-PCA axes initialize the rotation but do not fix it. The finite center
    box and axis limits include every admissible sphere; contact makes the
    comparison unresolved. Rotation coordinates at +/- pi are equivalent angles.
    """
    def __init__(self, max_nfev=150):
        self.max_nfev = max_nfev

    def fit(self, x, sphere):
        self.origin_ = x.mean(axis=0)
        self.rms_ = np.sqrt(np.mean(np.sum((x-self.origin_)**2, axis=1)))
        z = (x-self.origin_)/self.rms_
        _, _, vt = np.linalg.svd(z, full_matrices=False)
        self.base_ = vt.T
        self.q_ = z.shape[1]
        self.pairs_ = list(combinations(range(self.q_), 2))
        q, m = self.q_, len(self.pairs_)
        lo = np.r_[np.full(q, -1.5), np.full(q, np.log(.05)), np.full(m, -np.pi)]
        hi = np.r_[np.full(q, 1.5), np.full(q, np.log(3.)), np.full(m, np.pi)]
        center = (sphere.center_-self.origin_)/self.rms_
        common = sphere.radius_/self.rms_
        estimated = np.clip(np.sqrt(q)*np.std(z@self.base_, axis=0), .06, 2.9)
        starts = [np.r_[center, np.full(q, np.log(common)), np.zeros(m)],
                  np.r_[center, np.log(estimated), np.zeros(m)]]
        fits = []
        for initial in starts:
            initial = np.clip(initial, lo+1e-8, hi-1e-8)
            fit = least_squares(lambda p: self._distance(z, p), initial, bounds=(lo, hi),
                                max_nfev=self.max_nfev, ftol=1e-7, xtol=1e-7, gtol=1e-7)
            if np.isfinite(fit.fun).all():
                fits.append(fit)
        if not fits:
            raise FloatingPointError('No finite ellipsoid solution.')
        successful = [f for f in fits if f.success]
        fit = min(successful or fits, key=lambda f: np.mean(f.fun**2))
        self.params_ = fit.x
        self.converged_ = bool(fit.success)
        self.boundary_ = bool(np.any(fit.active_mask[:2*q]))
        self.axis_ratio_ = float(np.exp(fit.x[q:2*q]).max()/np.exp(fit.x[q:2*q]).min())
        return self

    def _distance(self, z, params):
        q = self.q_
        skew = np.zeros((q, q))
        for value, (i, j) in zip(params[2*q:], self.pairs_):
            skew[i, j], skew[j, i] = value, -value
        rotation = self.base_@expm(skew)
        return ellipsoid_signed_distance((z-params[:q])@rotation, np.exp(params[q:2*q]))

    def signed_distance(self, x):
        return self.rms_*self._distance((x-self.origin_)/self.rms_, self.params_)


def fit_surfaces(train, candidate, config, ellipsoid=False, seed=None):
    transform = Coordinates(config['scaling'], candidate.dimension).fit(train[list(candidate.columns)])
    z = transform.transform(train[list(candidate.columns)])
    rng = np.random.default_rng(config['seed'] if seed is None else seed)
    index = np.sort(rng.choice(len(z), min(config['fit_max_points'], len(z)), replace=False))
    sample = z[index]
    sphere = Sphere(config['max_radius_rms'], config['max_center_rms'], config['sphere_iterations']).fit(sample)
    plane = Hyperplane().fit(sample)
    fitted = dict(candidate=candidate, transform=transform, sphere=sphere, plane=plane,
                  energy=np.mean(np.sum((z-z.mean(axis=0))**2, axis=1)), ellipsoid=None)
    if ellipsoid:
        fitted['ellipsoid'] = FreeEllipsoid(config['ellipsoid_evaluations']).fit(sample, sphere)
    return fitted


def evaluate(fitted, frame, fold=1):
    candidate = fitted['candidate']
    z = fitted['transform'].transform(frame[list(candidate.columns)])
    out = pd.DataFrame(index=frame.index)
    out.index.name = 'date'
    for name in ['sphere', 'plane', 'ellipsoid']:
        if fitted[name] is not None:
            out[name] = fitted[name].signed_distance(z)**2/fitted['energy']
    out['relative_shell_mse'] = fitted['sphere'].signed_distance(z)**2/fitted['sphere'].radius_**2
    out['fold'], out['candidate'] = fold, candidate.id
    diag = fitted['sphere'].diagnostics(z)
    diag.update(candidate=candidate.id, fold=fold, converged=fitted['sphere'].converged_,
                boundary=fitted['sphere'].boundary_, center_rms=fitted['sphere'].center_rms_,
                radius_rms=fitted['sphere'].radius_rms_, retained_variance=fitted['transform'].retained_variance_)
    if fitted['ellipsoid'] is not None:
        e = fitted['ellipsoid']
        diag.update(ellipsoid_converged=e.converged_, ellipsoid_boundary=e.boundary_, axis_ratio=e.axis_ratio_)
    if not np.isfinite(out.select_dtypes('number').to_numpy()).all():
        raise FloatingPointError('Nonfinite geometric score.')
    return out, diag


def statistic(scores):
    return float(np.log(scores.plane.mean()/max(scores.sphere.mean(), 1e-15))) if scores.plane.mean() > 1e-12 else -np.inf


def uncertainty(scores, config):
    """Conditional circular-block uncertainty; never cross a fold boundary."""
    columns = ['sphere', 'plane', 'relative_shell_mse']+(['ellipsoid'] if 'ellipsoid' in scores else [])
    rng = np.random.default_rng(config['seed'])
    boot = np.zeros((config['n_bootstrap'], len(columns)))
    count = 0
    for _, group in scores.groupby('fold'):
        x = group[columns].to_numpy()
        n = len(x)
        b = min(config['bootstrap_block'], n)
        starts = rng.integers(0, n, (config['n_bootstrap'], int(np.ceil(n/b))))
        indices = ((starts[:, :, None]+np.arange(b)) % n).reshape(config['n_bootstrap'], -1)[:, :n]
        boot += x[indices].sum(axis=1)
        count += n
    boot /= count
    result = dict(shell_rmse=float(np.sqrt(scores.relative_shell_mse.mean())),
                  shell_rmse_upper=float(np.sqrt(np.quantile(boot[:, 2], .95))),
                  sphere_minus_plane_upper=float(np.quantile(boot[:, 0]-boot[:, 1], .95)))
    if 'ellipsoid' in scores:
        result['sphere_minus_ellipsoid_upper'] = float(np.quantile(boot[:, 0]-boot[:, 3], .95))
    return result


def chronological_splits(frame, config):
    splits = list(TimeSeriesSplit(n_splits=config['n_splits'], test_size=config['validation_size'],
                                 gap=config['gap']).split(frame))
    if min(len(tr) for tr, va in splits) < config['min_train_size']:
        raise ValueError('Insufficient training data.')
    table = pd.DataFrame([dict(fold=i, training_rows=len(tr), train_start=frame.index[tr[0]],
                               train_end=frame.index[tr[-1]], validation_start=frame.index[va[0]],
                               validation_end=frame.index[va[-1]]) for i, (tr, va) in enumerate(splits, 1)])
    return splits, table


def screen(frame, splits, candidates, config, intervals=True):
    rows, all_scores, diagnostics = [], [], []
    for c in candidates:
        parts, diags = [], []
        for fold, (tr, va) in enumerate(splits, 1):
            try:
                fitted = fit_surfaces(frame.iloc[tr], c, config, seed=config['seed']+fold)
                scores, diag = evaluate(fitted, frame.iloc[va], fold)
                diag['error'] = ''
                parts.append(scores)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                diag = dict(candidate=c.id, fold=fold, converged=False, boundary=True, error=str(exc))
            diags.append(diag)
        diagnostics.extend(diags)
        fit_ok = len(parts) == len(splits) and all(d['converged'] and not d['boundary'] for d in diags)
        row = dict(candidate=c.id, description=c.description, features=list(c.columns),
                   coordinates=c.dimension or len(c.columns), fit_ok=fit_ok, basic_geometry=False, statistic=-np.inf)
        if len(parts) == len(splits):
            scores = pd.concat(parts)
            row.update(statistic=statistic(scores), sphere_loss=scores.sphere.mean(), plane_loss=scores.plane.mean(),
                       coverage_ok=all(d['resultant'] <= config['max_resultant'] and d['angular_span'] >= config['min_angular_span'] for d in diags),
                       retained_variance=min(d['retained_variance'] for d in diags))
            if intervals:
                row.update(uncertainty(scores, config))
                row['basic_geometry'] = bool(fit_ok and row['coverage_ok']
                    and row['shell_rmse_upper'] <= config['max_shell_fraction'] and row['sphere_minus_plane_upper'] < 0)
                all_scores.append(scores)
        rows.append(row)
    ranking = pd.DataFrame(rows).set_index('candidate').sort_values('statistic', ascending=False)
    return ranking, pd.DataFrame(diagnostics), pd.concat(all_scores) if all_scores else pd.DataFrame()


def ellipsoid_checks(frame, splits, candidates, ranking, config):
    mapping = {c.id:c for c in candidates}
    rows, all_scores, diagnostics = [], [], []
    for name in ranking[ranking.basic_geometry].head(config['ellipsoid_shortlist']).index:
        parts, diags = [], []
        for fold, (tr, va) in enumerate(splits, 1):
            fitted = fit_surfaces(frame.iloc[tr], mapping[name], config, ellipsoid=True, seed=config['seed']+fold)
            scores, diag = evaluate(fitted, frame.iloc[va], fold)
            parts.append(scores);diags.append(diag)
        scores = pd.concat(parts)
        ci = uncertainty(scores, config)
        tolerance = config['ellipsoid_margin']*scores.plane.mean()
        valid = all(d['ellipsoid_converged'] and not d['ellipsoid_boundary'] for d in diags)
        rows.append(dict(candidate=name, ellipsoid_ok=valid,
                         sphere_minus_ellipsoid_upper=ci['sphere_minus_ellipsoid_upper'],
                         ellipsoid_tolerance=tolerance,
                         ellipsoid_pass=valid and ci['sphere_minus_ellipsoid_upper'] <= tolerance))
        all_scores.append(scores);diagnostics.extend(diags)
    ranking = ranking.copy()
    if rows:
        ranking = ranking.join(pd.DataFrame(rows).set_index('candidate'))
    else:
        ranking['ellipsoid_pass'] = False
    ranking['pre_control_support'] = ranking.basic_geometry & ranking.ellipsoid_pass.eq(True)
    return ranking, pd.concat(all_scores) if all_scores else pd.DataFrame(), pd.DataFrame(diagnostics)


def cross_spectral_error(a, b):
    """Relative Frobenius error of all cross-periodograms, without a 3-D tensor."""
    an = np.sum(np.abs(a)**2, axis=1)
    bn = np.sum(np.abs(b)**2, axis=1)
    inner = np.sum(a*np.conj(b), axis=1)
    numerator = np.maximum(an**2+bn**2-2*np.abs(inner)**2, 0.).sum()
    return float(np.sqrt(numerator/max(np.sum(an**2), 1e-30)))


def raw_surrogate(raw, family, rng, config):
    """Randomize DAILY inputs, then callers rebuild all rolling descriptors.

    Shared-phase controls preserve cross-spectra exactly. The iterative rank
    control preserves each input's values exactly and approximates cross-spectra.
    Its residual error is measured and must pass the declared quality limit.
    """
    x = raw.to_numpy(float)
    mean, sd = x.mean(axis=0), x.std(axis=0)
    if np.any(sd <= 1e-12):
        raise ValueError('A raw input is constant.')
    z = (x-mean)/sd
    original = np.fft.rfft(z, axis=0)
    phase = rng.uniform(-np.pi, np.pi, len(original));phase[0] = 0.
    if len(z) % 2 == 0:phase[-1] = 0.
    generated = np.fft.irfft(original*np.exp(1j*phase[:, None]), n=len(z), axis=0)
    iterations = 0
    if family == 'marginal_spectrum':
        ordered = np.sort(z, axis=0)
        best, best_error = None, np.inf
        for iterations in range(1, config['surrogate_iterations']+1):
            ranks = np.argsort(np.argsort(generated, axis=0, kind='stable'), axis=0, kind='stable')
            ranked = np.take_along_axis(ordered, ranks, axis=0)
            spectrum = np.fft.rfft(ranked, axis=0)
            error = cross_spectral_error(original, spectrum)
            if error < best_error:
                best, best_error = ranked.copy(), error
            if error <= config['surrogate_spectral_tolerance']:
                break
            phase = np.angle(np.sum(spectrum*np.conj(original), axis=1));phase[0] = 0.
            if len(z) % 2 == 0:phase[-1] = 0. if np.cos(phase[-1]) >= 0 else np.pi
            generated = np.fft.irfft(original*np.exp(1j*phase[:, None]), n=len(z), axis=0)
        generated = best
    elif family != 'phase':
        raise ValueError('Unknown control family.')
    error = cross_spectral_error(original, np.fft.rfft(generated, axis=0))
    quality = error <= config['surrogate_spectral_tolerance']
    return pd.DataFrame(generated*sd+mean, index=raw.index, columns=raw.columns), dict(
        spectral_error=error, quality_ok=quality, iterations=iterations)


def calibrate(raw_development, frame, splits, candidates, config, progress=print):
    rows = []
    for i, family in enumerate(config['null_families']):
        rng = np.random.default_rng(config['seed']+1000*(i+1))
        for b in range(config['n_surrogates']):
            raw_control, quality = raw_surrogate(raw_development, family, rng, config)
            features = build_trend_features(raw_control).reindex(frame.index)
            ranking, diagnostics, _ = screen(features, splits, candidates, config, intervals=False)
            unresolved = int((~diagnostics.groupby('candidate').converged.all()).sum())
            eligible = ranking.loc[ranking.fit_ok, 'statistic']
            maximum = (eligible.max() if len(eligible) else -np.inf)
            if unresolved or not quality['quality_ok']:
                maximum = np.inf  # Unresolved controls cannot make rejection easier.
            rows.append(dict(family=family, repeat=b+1, maximum_statistic=maximum,
                             eligible_candidates=len(eligible), unresolved=unresolved, **quality))
            if (b+1) % 5 == 0 or b+1 == config['n_surrogates']:
                progress(f'{family}: {b+1}/{config["n_surrogates"]} complete-search controls')
    return pd.DataFrame(rows)


def select_geometry(ranking, nulls, config):
    out = ranking.copy()
    for family in config['null_families']:
        if nulls.empty:
            out[f'control_p_{family}'] = np.nan
        else:
            maxima = nulls.loc[nulls.family.eq(family), 'maximum_statistic'].to_numpy()
            out[f'control_p_{family}'] = [(1+np.sum(maxima >= t))/(len(maxima)+1) if ok else np.nan
                                         for t, ok in zip(out.statistic, out.pre_control_support)]
    p = [f'control_p_{f}' for f in config['null_families']]
    out['geometry_supported'] = out.pre_control_support & out[p].le(config['alpha']).all(axis=1)
    supported = out[out.geometry_supported]
    return out, supported.index[0] if len(supported) else None


def confirm_geometry(raw_development, raw_final, train_frame, final_frame, candidate, config):
    """Frozen candidate and surfaces; final controls rebuild indicators with past history."""
    fitted = fit_surfaces(train_frame, candidate, config, ellipsoid=True)
    scores, diag = evaluate(fitted, final_frame)
    summary = dict(**diag, **uncertainty(scores, config), statistic=statistic(scores))
    tolerance = config['ellipsoid_margin']*scores.plane.mean()
    basic = (diag['converged'] and not diag['boundary'] and diag['ellipsoid_converged']
             and not diag['ellipsoid_boundary'] and diag['resultant'] <= config['max_resultant']
             and diag['angular_span'] >= config['min_angular_span']
             and summary['shell_rmse_upper'] <= config['max_shell_fraction']
             and summary['sphere_minus_plane_upper'] < 0
             and summary['sphere_minus_ellipsoid_upper'] <= tolerance)
    rows = []
    for i, family in enumerate(config['null_families']):
        p = np.nan
        if basic:
            rng = np.random.default_rng(config['seed']+9000+1000*i)
            control_statistics = []
            for b in range(config['n_surrogates']):
                control_tail, quality = raw_surrogate(raw_final, family, rng, config)
                control_frame = build_trend_features(pd.concat([raw_development, control_tail])).reindex(final_frame.index)
                control_scores, _ = evaluate(fitted, control_frame)
                value = statistic(control_scores) if quality['quality_ok'] else np.inf
                control_statistics.append(value)
                rows.append(dict(family=family, repeat=b+1, statistic=value, **quality))
            p = (1+np.sum(np.array(control_statistics) >= summary['statistic']))/(config['n_surrogates']+1)
        summary[f'control_p_{family}'] = float(p)
    summary['geometry_confirmed'] = bool(basic and all(summary[f'control_p_{f}'] <= config['alpha'] for f in config['null_families']))
    return summary, scores, pd.DataFrame(rows)


def _spherical_centers(u, k, seed, iterations=100):
    rng = np.random.default_rng(seed)
    centers = u[rng.choice(len(u), k, replace=False)].copy()
    for _ in range(iterations):
        labels = (u@centers.T).argmax(axis=1)
        if len(np.unique(labels)) < k:raise ValueError('Empty angular region.')
        updated = np.array([u[labels == j].mean(axis=0) for j in range(k)])
        norm = np.linalg.norm(updated, axis=1)
        if np.any(norm < 1e-12):raise ValueError('Undefined region center.')
        updated /= norm[:, None]
        if np.max(np.abs(updated-centers)) < 1e-7:break
        centers = updated
    return updated, (u@updated.T).argmax(axis=1)


def identify_regions(frame, candidate, config):
    """Only call AFTER geometry selection; this is descriptive, not predictive."""
    fitted = fit_surfaces(frame, candidate, config)
    s = fitted['sphere']
    if not s.converged_ or s.boundary_:
        return dict(status='Full-development sphere refit is not admissible.'), pd.DataFrame(), pd.DataFrame()
    z = fitted['transform'].transform(frame[list(candidate.columns)])-s.center_
    norm = np.linalg.norm(z, axis=1)
    if np.any(norm <= 1e-12):raise ValueError('Undefined direction at fitted center.')
    u = z/norm[:, None]
    k = config['regions']
    solutions = []
    for seed in range(config['seed'], config['seed']+5):
        try:
            centers, labels = _spherical_centers(u, k, seed)
            solutions.append((np.max(u@centers.T, axis=1).sum(), centers, labels))
        except ValueError:pass
    if not solutions:return dict(status='No stable angular initialization.'), pd.DataFrame(), pd.DataFrame()
    _, centers, labels = max(solutions, key=lambda x:x[0])
    rng = np.random.default_rng(config['seed'])
    stability = []
    for b in range(config['region_bootstraps']):
        n = len(u);block = min(config['bootstrap_block'], n)
        starts = rng.integers(0, n, int(np.ceil(n/block)))
        index = ((starts[:, None]+np.arange(block)) % n).ravel()[:n]
        try:
            other, _ = _spherical_centers(u[index], k, config['seed']+b)
            stability.append(adjusted_rand_score(labels, (u@other.T).argmax(axis=1)))
        except ValueError:stability.append(0.)
    sample = rng.choice(len(u), min(800, len(u)), replace=False)
    distances = np.arccos(np.clip(u[sample]@u[sample].T, -1, 1));np.fill_diagonal(distances, 0.)
    silhouette = float(silhouette_score(distances, labels[sample], metric='precomputed'))
    shares = pd.Series(labels).value_counts(normalize=True)
    stable = np.median(stability) >= config['min_region_stability'] and silhouette >= config['min_region_silhouette'] and shares.min() >= .05
    assignments = pd.DataFrame(dict(region=labels), index=frame.index)
    profiles = frame.assign(region=labels).groupby('region').median()
    profiles.insert(0, 'observations', pd.Series(labels).value_counts().sort_index())
    profiles.insert(1, 'description', ['bullish-like' if row.momentum_63 > 0 and row.gap_200 > 0 else
                                      'bearish-like' if row.momentum_63 < 0 and row.gap_200 < 0 else 'mixed'
                                      for _, row in profiles.iterrows()])
    report = dict(status='Stable exploratory partition' if stable else 'No stable discrete partition established',
                  regions_stable=bool(stable), median_stability=float(np.median(stability)), angular_silhouette=silhouette)
    return report, assignments, profiles
