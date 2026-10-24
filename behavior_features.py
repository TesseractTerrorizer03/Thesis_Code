"""Causal behavioral features, separately constructed forward labels, and splits."""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

CLASS_NAMES = {0: 'bearish', 1: 'neutral', 2: 'bullish'}
SIGNALS = ['US_equity', 'Developed_equity', 'Emerging_equity', 'Credit_improvement', 'Volatility_improvement']
REPRESENTATIONS = ['context', 'raw_context', 'direction_context', 'direction_only']


def validate_frame(frame, name):
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans:
        raise ValueError(f'{name}: expected valid dates.')
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError(f'{name}: dates must be unique and increasing.')
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f'{name}: resolve missing/nonfinite source values explicitly.')


def build_features(market, prices, levels, config):
    """Every predictor at t uses only inputs dated at or before t.

    Daily-change volatility is lagged one observation. Features use trailing
    windows, never centered windows or backward filling. The economic origin
    of the signal vector is zero movement, not its full-sample mean.
    """
    for frame, name in [(market, 'market'), (prices, 'prices'), (levels, 'levels')]:
        validate_frame(frame, name)
    if (market[['SPY', 'EFA', 'EEM']] <= -1).any().any() or (prices.SPY <= 0).any():
        raise ValueError('Invalid returns/prices for log transforms.')
    h, vol_window = config['lookback'], config['vol_window']
    if h < 1 or vol_window < 2:
        raise ValueError('Invalid lookback or volatility window.')
    changes = pd.DataFrame(index=market.index)
    for source, name in [('SPY', 'US_equity'), ('EFA', 'Developed_equity'), ('EEM', 'Emerging_equity')]:
        changes[name] = np.log1p(market[source])
    changes['Credit_improvement'] = -market['d_Credit_Spread']
    changes['Volatility_improvement'] = -market['d_log_VIX']
    past_scale = changes.rolling(vol_window, min_periods=vol_window).std().shift(1)
    past_scale = past_scale.where(past_scale > config['min_daily_scale'])
    signals = changes.rolling(h, min_periods=h).sum() / (past_scale*np.sqrt(h))
    intensity = np.sqrt((signals**2).sum(axis=1, min_count=len(SIGNALS)))
    reliable = intensity > config['min_direction_norm']
    direction = signals.div(intensity.where(reliable), axis=0)
    # A near-zero vector has no meaningful direction; expose an explicit flag.
    direction.loc[intensity.notna() & ~reliable] = 0.
    aligned_levels = levels.reindex(market.index, method='ffill')
    if (aligned_levels.VIX_level.dropna() <= 0).any():
        raise ValueError('VIX levels must be positive.')
    px = prices.SPY
    drawdown = px/px.rolling(config['drawdown_window'], min_periods=config['drawdown_window']).max()-1
    context = pd.DataFrame(index=market.index)
    context['log_intensity'] = np.log1p(intensity)
    context['direction_reliable'] = reliable.astype(float)
    context['realized_volatility'] = changes.US_equity.rolling(config['realized_vol_window'],
                                                              min_periods=config['realized_vol_window']).std()
    context['log_VIX_level'] = np.log(aligned_levels.VIX_level)
    context['credit_spread_level'] = aligned_levels.Credit_spread_level
    context['market_drawdown'] = drawdown.reindex(market.index)
    return pd.concat([signals.add_prefix('signal_'), direction.add_prefix('direction_'), context], axis=1)


def build_targets(market, prices, config):
    """Labels depend on FUTURE prices, and never enter the predictor columns.

    Horizons use actual price sessions. Store the precise date on which each
    label becomes observable, so purging works even when feature dates have gaps.
    """
    horizon = config['forecast_horizon']
    if horizon < 1 or config['label_threshold'] <= 0:
        raise ValueError('Invalid forecast horizon or outcome threshold.')
    px = prices.SPY
    future_return = np.log(px.shift(-horizon)/px).reindex(market.index)
    maturity = pd.Series(prices.index, index=prices.index).shift(-horizon).reindex(market.index)
    past_vol = np.log1p(market.SPY).rolling(config['vol_window'], min_periods=config['vol_window']).std().shift(1)
    threshold = config['label_threshold'] * past_vol * np.sqrt(horizon)
    valid = future_return.notna() & maturity.notna() & (threshold > config['min_daily_scale'])
    y = pd.Series(np.nan, index=market.index, name='outcome')
    y.loc[valid] = 1
    y.loc[valid & (future_return < -threshold)] = 0
    y.loc[valid & (future_return > threshold)] = 2
    return pd.DataFrame(dict(outcome=y, label_end=maturity, future_log_return=future_return,
                             bullish_threshold=threshold))


def design_columns(features):
    """Explicit predictor whitelist prevents accidental use of future labels."""
    signals = [f'signal_{s}' for s in SIGNALS]
    directions = [f'direction_{s}' for s in SIGNALS]
    context = ['log_intensity', 'direction_reliable', 'realized_volatility',
               'log_VIX_level', 'credit_spread_level', 'market_drawdown']
    out = dict(context=context, raw_context=signals+context,
               direction_context=directions+context,
               direction_only=directions+['direction_reliable'])
    if not set(sum(out.values(), [])).issubset(features.columns):
        raise ValueError('Missing engineered feature columns.')
    return out


def development_index(features, targets, holdout_start):
    usable = np.isfinite(features.to_numpy()).all(axis=1) & targets.outcome.notna()
    # Even validation outcomes must finish BEFORE the reserved final period.
    return features.index[usable & (features.index < holdout_start) & (targets.label_end < holdout_start)]


def purged_splits(index, targets, n_splits, test_size, min_train_size):
    """Expanding splits; every training label finishes before validation starts."""
    cv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    splits, rows = [], []
    for fold, (nominal_train, valid) in enumerate(cv.split(index), 1):
        start = index[valid[0]]
        available = (targets.loc[index[nominal_train], 'label_end'] < start).to_numpy()
        train = nominal_train[available]
        if len(train) < min_train_size:
            raise ValueError('Insufficient purged training history; reduce split sizes or add data.')
        last_label = targets.loc[index[train], 'label_end'].max()
        assert last_label < start
        splits.append((train, valid))
        rows.append(dict(fold=fold, training_rows=len(train), purged_rows=len(nominal_train)-len(train),
                         train_start=index[train[0]], train_end=index[train[-1]],
                         last_training_label_end=last_label,
                         validation_start=start, validation_end=index[valid[-1]], validation_rows=len(valid)))
    return splits, pd.DataFrame(rows)
