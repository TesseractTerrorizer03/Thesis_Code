"""Research-motivated trend descriptors. No targets or row normalization."""
import numpy as np
import pandas as pd

SECTORS = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']
SOURCE_TREND = 'https://fairmodel.econ.yale.edu/ec439/mosk.pdf'
SOURCE_TECHNICAL = 'https://www.fidelity.com/learning-center/trading-investing/technical-analysis/what-is-technical-analysis'
SOURCE_BREADTH = 'https://www.fidelity.com/learning-center/trading-investing/advance-decline'
SOURCE_RISK = 'https://www.federalreserve.gov/econres/notes/feds-notes/a-new-index-to-measure-us-financial-conditions-20230630.html'
SOURCE_VIX = 'https://www.cboe.com/tradable-products/vix/'


def raw_inputs(market, auxiliary):
    """Daily inputs randomized BEFORE rolling indicators are rebuilt for controls."""
    if not market.index.is_unique or not market.index.is_monotonic_increasing:
        raise ValueError('Market dates must be unique and increasing.')
    if not auxiliary.index.is_unique or not auxiliary.index.is_monotonic_increasing:
        raise ValueError('Auxiliary dates must be unique and increasing.')
    out = pd.DataFrame(index=market.index)
    for ticker in ['SPY', 'EFA', 'EEM', 'DBC']:
        if (market[ticker] <= -1).any():
            raise ValueError('Invalid simple return.')
        out[f'return_{ticker}'] = np.log1p(market[ticker])
    out['credit_change'] = market.d_Credit_Spread
    out['vix_log_change'] = market.d_log_VIX
    if (auxiliary <= 0).any().any():
        raise ValueError('Prices/volume must be positive.')
    sector_returns = np.log(auxiliary[SECTORS]).diff().reindex(market.index)
    for ticker in SECTORS:
        out[f'return_{ticker}'] = sector_returns[ticker]
    out['log_volume'] = np.log(auxiliary.SPY_volume.reindex(market.index))
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError('Incomplete raw overlap; inspect dates rather than silently filling prices.')
    return out


def build_trend_features(raw):
    """All windows trail their date. Normalization is not imposed on feature rows.

    Reconstructing price paths from returns uses an arbitrary common starting
    price; every price feature below is invariant to that starting level.
    """
    if not np.isfinite(raw.to_numpy()).all():
        raise ValueError('Raw inputs must be finite.')
    returns = raw[[f'return_{t}' for t in ['SPY', 'EFA', 'EEM', 'DBC']+SECTORS]].copy()
    returns.columns = [c.removeprefix('return_') for c in returns]
    log_prices = returns.cumsum()
    prices = np.exp(log_prices)
    spy = prices.SPY
    out = pd.DataFrame(index=raw.index)
    for h in [21, 63, 126]:
        out[f'momentum_{h}'] = returns.SPY.rolling(h, min_periods=h).sum()
    ma50, ma200 = spy.rolling(50, min_periods=50).mean(), spy.rolling(200, min_periods=200).mean()
    out['gap_50'] = spy/ma50-1
    out['gap_200'] = spy/ma200-1
    out['ma_slope_200'] = np.log(ma200/ma200.shift(21))
    out['drawdown_252'] = spy/spy.rolling(252, min_periods=252).max()-1
    out['realized_vol_21'] = returns.SPY.rolling(21, min_periods=21).std()*np.sqrt(252)
    for ticker in ['EFA', 'EEM', 'DBC']:
        out[f'{ticker.lower()}_momentum_63'] = returns[ticker].rolling(63, min_periods=63).sum()
    out['credit_improvement_63'] = -raw.credit_change.rolling(63, min_periods=63).sum()
    out['vix_improvement_21'] = -raw.vix_log_change.rolling(21, min_periods=21).sum()
    for h in [50, 200]:
        average = prices[SECTORS].rolling(h, min_periods=h).mean()
        above = prices[SECTORS].gt(average).astype(float).where(average.notna())
        out[f'sector_above_{h}'] = above.mean(axis=1).where(above.notna().all(axis=1))
    sector_trend = returns[SECTORS].rolling(63, min_periods=63).sum()
    out['sector_positive_63'] = sector_trend.gt(0).astype(float).mean(axis=1).where(sector_trend.notna().all(axis=1))
    out['sector_mean_63'] = sector_trend.mean(axis=1)
    out['sector_dispersion_63'] = sector_trend.std(axis=1)
    # Constant rescaling of volume changes neither descriptor and avoids large units.
    volume = np.exp(raw.log_volume-raw.log_volume.iloc[0])
    out['signed_volume_21'] = (np.sign(returns.SPY)*volume).rolling(21, min_periods=21).sum()/volume.rolling(21, min_periods=21).sum()
    out['relative_volume_21'] = np.log(volume.rolling(21, min_periods=21).mean()/volume.rolling(63, min_periods=63).mean().shift(1))
    return out


def feature_catalog():
    rows = []
    def add(name, formula, meaning, source):
        rows.append(dict(feature=name, definition=formula, interpretation=meaning, source=source))
    for h in [21, 63, 126]:
        add(f'momentum_{h}', f'Sum of trailing {h} SPY log returns', 'Positive: upward price trend', SOURCE_TREND)
    for h in [50, 200]:
        add(f'gap_{h}', f'SPY / trailing {h}-observation simple average - 1', 'Positive: price above its average', SOURCE_TECHNICAL)
    add('ma_slope_200', 'log(current 200-observation average / its value 21 observations earlier)', 'Positive: long average rising', SOURCE_TECHNICAL)
    add('drawdown_252', 'SPY / trailing 252-observation high - 1', 'Closer to zero: nearer a trailing high', SOURCE_TECHNICAL)
    add('realized_vol_21', 'Trailing 21-observation return standard deviation times sqrt(252)', 'Movement intensity; not direction by itself', SOURCE_VIX)
    for ticker in ['EFA', 'EEM', 'DBC']:
        add(f'{ticker.lower()}_momentum_63', f'Trailing 63 {ticker} log returns', 'Cross-market context; commodities are not universally bullish', SOURCE_TREND)
    add('credit_improvement_63', 'Negative trailing 63 credit-spread changes', 'Positive: spread tightening; confirmation/context', SOURCE_RISK)
    add('vix_improvement_21', 'Negative trailing 21 changes in log VIX', 'Positive: implied volatility declining; not a direction label', SOURCE_VIX)
    for h in [50, 200]:
        add(f'sector_above_{h}', f'Fraction of nine sector ETFs above their trailing {h}-observation averages', 'Participation proxy; not constituent-stock breadth', SOURCE_BREADTH)
    add('sector_positive_63', 'Fraction of nine sectors with positive trailing 63-observation return', 'Participation proxy', SOURCE_BREADTH)
    add('sector_mean_63', 'Equal-weight mean of sector 63-observation log returns', 'Direction across sectors', SOURCE_BREADTH)
    add('sector_dispersion_63', 'Cross-sector standard deviation of 63-observation returns', 'Disagreement; not bullish/bearish by itself', SOURCE_BREADTH)
    add('signed_volume_21', 'Sum(sign(SPY daily return) * volume) / sum(volume), trailing 21', 'Positive: more SPY volume on up days; not exchange-wide advancing volume', SOURCE_TECHNICAL)
    add('relative_volume_21', 'log(21-observation mean volume / lagged 63-observation mean volume)', 'Activity intensity; not direction by itself', SOURCE_TECHNICAL)
    return pd.DataFrame(rows)


def default_groups():
    return dict(
        trend_horizons=['momentum_21', 'momentum_63', 'momentum_126'],
        trend_position=['momentum_63', 'gap_50', 'gap_200', 'ma_slope_200'],
        trend_participation=['momentum_63', 'sector_above_50', 'sector_above_200', 'sector_positive_63'],
        risk_confirmation=['momentum_63', 'credit_improvement_63', 'vix_improvement_21', 'realized_vol_21'],
        cross_market=['momentum_63', 'efa_momentum_63', 'eem_momentum_63', 'dbc_momentum_63'],
        volume_confirmation=['momentum_63', 'signed_volume_21', 'relative_volume_21', 'drawdown_252'],
        broad_state=['momentum_63', 'gap_200', 'sector_above_200', 'credit_improvement_63', 'vix_improvement_21', 'signed_volume_21'])
