"""Behavior features and forward-label validation; no market holdout is scored."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from behavior_features import (SIGNALS, build_features, build_targets, design_columns,
                               development_index, purged_splits)
from behavior_validation import nested_validation, final_validation, paired_tests


def config():
    return dict(lookback=5, vol_window=30, realized_vol_window=10, drawdown_window=30,
                min_daily_scale=1e-10, min_direction_norm=1e-8,
                forecast_horizon=5, label_threshold=.5,
                inner_splits=2, inner_validation_size=32, inner_min_train=100,
                logistic_C=[1.], logistic_max_iter=2000, tree_leaves=[3],
                tree_iterations=20, tree_min_leaf=10, families=['linear', 'nonlinear'],
                primary_family='nonlinear', seed=42, bootstrap_block=10,
                n_bootstrap=100, alpha=.05)


def synthetic_inputs(n=1100):
    rng = np.random.default_rng(83)
    dates = pd.bdate_range('2000-01-01', periods=n)
    shock = rng.normal(0, .01, (n, 3))
    market = pd.DataFrame(np.expm1(shock), index=dates, columns=['SPY', 'EFA', 'EEM'])
    market['d_Credit_Spread'] = rng.normal(0, .015, n)
    market['d_log_VIX'] = rng.normal(0, .03, n)
    prices = pd.DataFrame({'SPY':100*np.exp(np.cumsum(shock[:, 0]))}, index=dates)
    levels = pd.DataFrame({'VIX_level':20*np.exp(np.cumsum(market.d_log_VIX)/5),
                           'Credit_spread_level':2+market.d_Credit_Spread.cumsum()}, index=dates)
    return market, prices, levels


class BehaviorTests(unittest.TestCase):
    def test_features_are_causal(self):
        market, prices, levels = synthetic_inputs()
        original = build_features(market, prices, levels, config())
        changed_market, changed_prices, changed_levels = market.copy(), prices.copy(), levels.copy()
        cut = 700
        changed_market.iloc[cut:] *= 1.4
        changed_prices.iloc[cut:] *= 2
        changed_levels.iloc[cut:] *= 1.7
        changed = build_features(changed_market, changed_prices, changed_levels, config())
        pd.testing.assert_frame_equal(original.iloc[:cut], changed.iloc[:cut])
        prefix = build_features(market.iloc[:cut], prices.iloc[:cut], levels.iloc[:cut], config())
        pd.testing.assert_frame_equal(original.iloc[:cut], prefix)

    def test_direction_and_intensity_preserve_the_signal_vector(self):
        market, prices, levels = synthetic_inputs()
        f = build_features(market, prices, levels, config()).dropna()
        u = f[[f'direction_{s}' for s in SIGNALS]].to_numpy()
        signal = f[[f'signal_{s}' for s in SIGNALS]].to_numpy()
        np.testing.assert_allclose(np.linalg.norm(u, axis=1), 1., atol=1e-12)
        np.testing.assert_allclose(u*np.expm1(f.log_intensity.to_numpy())[:, None], signal, atol=1e-12)
        designs = design_columns(f)
        self.assertEqual(len(designs['raw_context']), len(designs['direction_context']))
        self.assertFalse(any('future' in c or 'outcome' in c for cols in designs.values() for c in cols))

    def test_zero_direction_is_explicitly_flagged(self):
        market, prices, levels = synthetic_inputs()
        market.iloc[-5:] = 0.
        f = build_features(market, prices, levels, config())
        self.assertEqual(f.direction_reliable.iloc[-1], 0.)
        self.assertEqual(f.log_intensity.iloc[-1], 0.)
        np.testing.assert_array_equal(f[[f'direction_{s}' for s in SIGNALS]].iloc[-1], 0.)

    def test_targets_use_actual_future_price_dates_and_leave_unknowns_missing(self):
        market, prices, _ = synthetic_inputs()
        market = market.iloc[::2]  # Missing feature dates must not shorten the price horizon.
        c = config()
        t = build_targets(market, prices, c)
        origin = market.index[100]
        future = prices.index[prices.index.get_loc(origin)+c['forecast_horizon']]
        self.assertEqual(t.loc[origin, 'label_end'], future)
        self.assertAlmostEqual(t.loc[origin, 'future_log_return'], np.log(prices.loc[future, 'SPY']/prices.loc[origin, 'SPY']))
        late = t.index > prices.index[-1-c['forecast_horizon']]
        self.assertTrue(t.loc[late, 'outcome'].isna().all())

    def test_purging_and_final_boundary_use_label_maturity(self):
        market, prices, levels = synthetic_inputs()
        f = build_features(market, prices, levels, config())
        targets = build_targets(market, prices, config())
        boundary = market.index[-150]
        development = development_index(f, targets, boundary)
        self.assertTrue((targets.loc[development, 'label_end'] < boundary).all())
        splits, audit = purged_splits(development, targets, 2, 64, 100)
        self.assertTrue((audit.purged_rows == config()['forecast_horizon']).all())
        for train, valid in splits:
            self.assertTrue((targets.loc[development[train], 'label_end'] < development[valid[0]]).all())

    def test_nested_and_final_validation_on_synthetic_data(self):
        c = config()
        market, prices, levels = synthetic_inputs()
        f = build_features(market, prices, levels, c)
        targets = build_targets(market, prices, c)
        boundary = market.index[-150]
        development = development_index(f, targets, boundary)
        designs = design_columns(f)
        splits, _ = purged_splits(development, targets, 2, 64, 100)
        with threadpool_limits(limits=1):
            predictions, tuning, inner_dates = nested_validation(f, targets, development, designs,
                                                                 splits, c, progress=lambda _:None)
            usable = (f.index >= boundary) & np.isfinite(f.to_numpy()).all(axis=1) & targets.outcome.notna()
            final, _, comparisons = final_validation(f, targets, development, f.index[usable], designs, c)
        self.assertEqual(len(predictions), 2*64*(4*2+1))
        np.testing.assert_allclose(predictions[['p_bearish', 'p_neutral', 'p_bullish']].sum(axis=1), 1.)
        self.assertTrue((inner_dates.last_training_label_end < inner_dates.validation_start).all())
        self.assertEqual(len(comparisons), 2)
        self.assertTrue(np.isfinite(final.log_loss).all())
        self.assertTrue(tuning.groupby(['fold', 'representation', 'family']).selected.sum().eq(1).all())

    def test_paired_tests_require_gain_over_both_comparators(self):
        rows = []
        for fold in [1, 2]:
            for date in pd.bdate_range(f'200{fold}-01-01', periods=80):
                for representation, loss in [('context', 1.2), ('raw_context', .7), ('direction_context', .8)]:
                    rows.append(dict(fold=fold, date=date, representation=representation,
                                     family='nonlinear', log_loss=loss))
        test = paired_tests(pd.DataFrame(rows), config()).set_index('comparator')
        self.assertTrue(test.loc['context', 'supports_gain'])
        self.assertFalse(test.loc['raw_context', 'supports_gain'])
        positive = pd.DataFrame(rows)
        positive.loc[positive.representation.eq('raw_context'), 'log_loss'] = 1.1
        self.assertTrue(paired_tests(positive, config()).supports_gain.all())


if __name__ == '__main__':
    unittest.main()
