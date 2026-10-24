"""Nested chronological prediction tests for alternative behavior representations."""
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss, accuracy_score, balanced_accuracy_score
from behavior_features import purged_splits, REPRESENTATIONS

CLASSES = np.array([0, 1, 2])


def model_specs(config, family):
    if family == 'linear':
        return [dict(C=float(c)) for c in config['logistic_C']]
    if family == 'nonlinear':
        return [dict(max_leaf_nodes=int(n)) for n in config['tree_leaves']]
    raise ValueError('Unknown prediction family.')


def make_model(family, params, config):
    if family == 'linear':
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=params['C'], max_iter=config['logistic_max_iter'], solver='lbfgs'))
    return HistGradientBoostingClassifier(
        max_leaf_nodes=params['max_leaf_nodes'], max_iter=config['tree_iterations'],
        min_samples_leaf=config['tree_min_leaf'], learning_rate=.05,
        l2_regularization=1., early_stopping=False, random_state=config['seed'])


def fit_probabilities(train_x, train_y, test_x, family, params, config):
    if not np.array_equal(np.unique(train_y), CLASSES):
        raise ValueError('All three outcome classes are required in each training fold.')
    model = make_model(family, params, config)
    # Nonconvergence must not silently receive a favorable validation score.
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(train_x, train_y)
    probability = model.predict_proba(test_x)
    if not np.array_equal(model.classes_, CLASSES):
        raise ValueError('Unexpected probability column order.')
    if not np.isfinite(probability).all():
        raise FloatingPointError('Nonfinite prediction.')
    return probability


def choose_params(features, targets, index, columns, family, config):
    """Tune ONLY within the current training window, with another purged split."""
    splits, boundaries = purged_splits(index, targets, config['inner_splits'],
                                       config['inner_validation_size'], config['inner_min_train'])
    rows = []
    for params in model_specs(config, family):
        losses = []
        for train, valid in splits:
            tr, va = index[train], index[valid]
            p = fit_probabilities(features.loc[tr, columns], targets.loc[tr, 'outcome'].astype(int),
                                  features.loc[va, columns], family, params, config)
            actual = targets.loc[va, 'outcome'].to_numpy(dtype=int)
            losses.extend(-np.log(np.clip(p[np.arange(len(p)), actual], 1e-12, 1)))
        rows.append(dict(params=params, inner_log_loss=float(np.mean(losses))))
    best = min(rows, key=lambda r: r['inner_log_loss'])
    return best['params'], rows, boundaries


def prediction_rows(index, actual, probability, fold, representation, family):
    frame = pd.DataFrame(probability, index=index, columns=['p_bearish', 'p_neutral', 'p_bullish'])
    frame.index.name = 'date'
    frame['outcome'] = np.asarray(actual, int)
    frame['fold'], frame['representation'], frame['family'] = fold, representation, family
    frame['log_loss'] = -np.log(np.clip(probability[np.arange(len(frame)), frame.outcome], 1e-12, 1))
    return frame.reset_index()


def prior_probabilities(outcomes, n):
    counts = np.bincount(np.asarray(outcomes, int), minlength=3)
    return np.tile((counts+1)/(counts.sum()+3), (n, 1))


def nested_validation(features, targets, index, designs, splits, config, progress=print):
    predictions, tuning, inner_dates = [], [], []
    for fold, (train, valid) in enumerate(splits, 1):
        tr, va = index[train], index[valid]
        y_train, y_valid = targets.loc[tr, 'outcome'].astype(int), targets.loc[va, 'outcome'].astype(int)
        predictions.append(prediction_rows(va, y_valid, prior_probabilities(y_train, len(va)),
                                           fold, 'class_frequency', 'reference'))
        for representation in REPRESENTATIONS:
            for family in config['families']:
                params, trials, boundaries = choose_params(features, targets, tr,
                                                           designs[representation], family, config)
                p = fit_probabilities(features.loc[tr, designs[representation]], y_train,
                                      features.loc[va, designs[representation]], family, params, config)
                predictions.append(prediction_rows(va, y_valid, p, fold, representation, family))
                tuning.extend(dict(fold=fold, representation=representation, family=family,
                                   params=json.dumps(t['params']), inner_log_loss=t['inner_log_loss'],
                                   selected=t['params'] == params) for t in trials)
                boundaries = boundaries.assign(outer_fold=fold, representation=representation, family=family)
                inner_dates.append(boundaries)
        progress(f'Completed validation window {fold}/{len(splits)}')
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(tuning), pd.concat(inner_dates, ignore_index=True)


def summarize_predictions(predictions):
    rows = []
    for (family, representation), part in predictions.groupby(['family', 'representation']):
        p = part[['p_bearish', 'p_neutral', 'p_bullish']].to_numpy()
        y = part.outcome.to_numpy(dtype=int)
        one_hot = np.eye(3)[y]
        rows.append(dict(family=family, representation=representation, observations=len(y),
                         log_loss=log_loss(y, p, labels=CLASSES),
                         brier_score=np.mean(np.sum((p-one_hot)**2, axis=1)),
                         accuracy=accuracy_score(y, p.argmax(axis=1)),
                         balanced_accuracy=balanced_accuracy_score(y, p.argmax(axis=1))))
    return pd.DataFrame(rows).sort_values(['family', 'log_loss']).reset_index(drop=True)


def paired_tests(predictions, config):
    """Two predeclared tests. Positive gain means directional encoding is better.

    Moving-block bootstrap within each outer validation fold. Inference is
    approximate, conditional on fitted models, and not a universal geometry test.
    Holm correction accounts for the two predeclared comparator tests.
    """
    part = predictions[predictions.family.eq(config['primary_family'])]
    wide = part.pivot(index=['fold', 'date'], columns='representation', values='log_loss').sort_index()
    comparators = ['context', 'raw_context']
    required = comparators+['direction_context']
    if wide[required].isna().any().any():
        raise ValueError('Paired comparisons require identical prediction dates.')
    difference = wide[comparators].subtract(wide.direction_context, axis=0)
    observed = difference.mean().to_numpy()
    rng = np.random.default_rng(config['seed'])
    draws = np.empty((config['n_bootstrap'], len(comparators)))
    pieces = [g.to_numpy() for _, g in difference.groupby(level='fold')]
    for repeat in range(config['n_bootstrap']):
        total = np.zeros(len(comparators))
        count = 0
        for values in pieces:
            n = len(values)
            block = min(config['bootstrap_block'], n)
            starts = rng.integers(0, n, int(np.ceil(n/block)))
            indices = ((starts[:, None]+np.arange(block)) % n).ravel()[:n]
            total += values[indices].sum(axis=0)
            count += n
        draws[repeat] = total/count
    # Center bootstrap scores under the zero-gain hypothesis.
    p_values = (1+np.sum((draws-observed) >= observed, axis=0))/(config['n_bootstrap']+1)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values, dtype=float)
    running = 0.
    for position, i in enumerate(order):
        running = max(running, (len(order)-position)*p_values[i])
        adjusted[i] = min(1., running)
    lower, upper = np.quantile(draws, [.025, .975], axis=0)
    return pd.DataFrame(dict(comparator=comparators, mean_log_loss_gain=observed,
                             lower_95=lower, upper_95=upper, p_one_sided=p_values,
                             p_holm=adjusted,
                             supports_gain=(observed > 0) & (lower > 0) & (adjusted <= config['alpha'])))


def final_validation(features, targets, development, holdout_index, designs, config):
    """Fixed representations/family; tune only on pre-holdout development data."""
    if not len(holdout_index):
        raise ValueError('No mature final outcomes.')
    if not targets.loc[development, 'label_end'].lt(holdout_index[0]).all():
        raise ValueError('A training label crosses the final boundary.')
    actual = targets.loc[holdout_index, 'outcome'].astype(int)
    rows = [prediction_rows(holdout_index, actual,
                            prior_probabilities(targets.loc[development, 'outcome'], len(actual)),
                            1, 'class_frequency', 'reference')]
    choices = []
    for representation in REPRESENTATIONS:
        family = config['primary_family']
        params, _, _ = choose_params(features, targets, development, designs[representation], family, config)
        probability = fit_probabilities(features.loc[development, designs[representation]],
                                         targets.loc[development, 'outcome'].astype(int),
                                         features.loc[holdout_index, designs[representation]], family, params, config)
        rows.append(prediction_rows(holdout_index, actual, probability, 1, representation, family))
        choices.append(dict(representation=representation, family=family, params=json.dumps(params)))
    predictions = pd.concat(rows, ignore_index=True)
    return predictions, pd.DataFrame(choices), paired_tests(predictions, config)
