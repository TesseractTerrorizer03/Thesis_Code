# Bullish/bearish directional experiment

Start with `Bullish_Bearish_Directional_Validation.ipynb`. Each code cell has a plain-language purpose above it. All ordinary edits belong in its first configuration cell. The previous spherical-surface notebook and its helper files are separate and unchanged by this experiment.

## What is being tested

Do the relative strengths of equity, credit, and volatility movements help forecast later market direction beyond their overall intensity and current conditions? Does expressing those relative strengths as direction help a forecasting method compared with supplying the original signals?

These are two different questions, and both must pass the notebook's predeclared primary comparisons. Neither establishes that observations naturally occupy a fixed-radius sphere. Dividing a vector by its length only constructs a directional representation. Keeping direction and intensity preserves the original information, apart from the explicit near-zero treatment; an improvement is therefore about representation and the fitted prediction method, not newly created information.

The first notebook searches for noisy spherical surfaces. This notebook tests the predictive usefulness of a directional/radial description. It does not use regime clustering or mixture models, and it does not assign every original market feature to an exclusive geometry category.

## Inputs and timing

Inputs are in `Thesis/data`:

- `market_features.csv`: cached daily returns and daily changes.
- `market_prices.csv`: cached SPY prices, used for outcomes and drawdown.
- `market_levels.csv`: cached VIX and credit-spread levels.
- `behavior_provenance.json`: sources/checksums of the two additional copies.

No downloads are performed. Source CSVs in RegimeDetection remain unchanged. Use the existing Python environment or the dependencies in `requirements.txt`.

A prediction origin is the end of date t. Predictors may include that day's movement; the outcome starts after its close. A practical trading study needs to audit actual release times and historical revision vintages. The CSVs alone do not establish those properties.

## Signals in plain language

Each signal sums movement over the trailing `lookback` observations and divides by the lagged daily standard deviation over `vol_window`, multiplied by sqrt(lookback). The lagged estimate ends before the current origin. This is a conventional variability scale, not an assumption that the signals are normally distributed or that serial correlation vanishes.

| Signal | Daily movement used | Positive value means |
|---|---|---|
| `US_equity` | log(1 + SPY return) | US equities have risen |
| `Developed_equity` | log(1 + EFA return) | Developed-market equities have risen |
| `Emerging_equity` | log(1 + EEM return) | Emerging-market equities have risen |
| `Credit_improvement` | −change in credit spread | Credit spreads have tightened |
| `Volatility_improvement` | −change in log VIX | VIX has declined |

The five values together form the original signal vector. Its length is intensity. Dividing by that length gives its direction, with bullish and bearish signs preserved. Intensity is stored as `log(1 + length)`. The origin remains zero movement; full-sample mean subtraction is not used to define direction.

Near-zero vectors receive a zero direction placeholder plus `direction_reliable=0`; they are not assigned a fabricated unit direction. Other vectors have reliability 1. Warmup rows and zero-volatility rows remain unavailable rather than being filled using later observations.

Context is: log intensity, direction reliability, trailing realized SPY volatility, log VIX level, credit-spread level, and SPY drawdown from its trailing high. All use information at or before t. No target or future-return column is allowed in predictor designs.

## Four input sets

| Internal name | What it contains | Why it is included |
|---|---|---|
| `context` | Intensity and current conditions | Tests whether apparent behavior is mostly intensity/context |
| `raw_context` | Original five signals plus identical context | Flexible comparison using the same underlying information |
| `direction_context` | Five direction coordinates plus identical intensity/context | Main representation being tested |
| `direction_only` | Direction coordinates and reliability | Secondary diagnostic of what magnitude/context contribute |

The current model feature counts are 6, 11, 11, and 6 respectively. `direction_only` is not used to select the primary conclusion. It is intentionally missing intensity/context, so its performance cannot establish geometry superiority on its own.

## Outcome definitions

For each origin, measure the SPY log price return over the next `forecast_horizon` actual price sessions. A return above `label_threshold × lagged daily SPY volatility × sqrt(horizon)` is bullish (2), below the negative threshold is bearish (0), and otherwise is neutral (1).

These are forward-direction outcomes over a chosen horizon, not retrospective labels of secular bull/bear markets. Neutral observations are retained. Unknown future prices yield missing outcomes, never neutral labels. `label_end` stores the exact future price date on which each outcome becomes observable, including when feature dates have gaps.

## Correct chronology and model selection

The final boundary is fixed from the last `holdout_size` rows of the original feature CSV, before warmup removal. With the current inputs it is 2023-08-25, matching the reserved period in the surface experiment.

Development uses only feature origins before that boundary whose complete outcomes also end before the boundary. This deliberately removes the last pre-test origins with outcomes extending into the reserved period. All features must be finite and the outcome known.

For each of three outer validation windows:

1. Take earlier training origins.
2. Remove any origin whose `label_end` is on or after validation start.
3. Inside this training set, construct two more chronological splits with the same maturity-based removal.
4. Choose model settings by average inner-validation log loss.
5. Fit those chosen settings on the outer training set and predict its later outer validation window.

Outer validation outcomes do not select that window's model settings. `splits.csv` and `inner_splits.csv` let you verify that every `last_training_label_end` precedes `validation_start`.

Warmup features may legitimately use earlier raw observations, including observations that are excluded as labeled training origins. What is prohibited is using unavailable outcomes or future predictors. Features for final-period dates can use earlier final-period prices when those dates arrive; the final prediction models themselves remain fixed.

## Prediction methods

- `linear`: standardized multinomial logistic regression, with regularization chosen on inner chronological splits. Its scaler is learned only from the fitting rows. This model-only scaling does not redefine how the directional vector was constructed.
- `nonlinear`: histogram gradient-boosted classification trees. A small leaf-count grid is tuned on inner splits; iterations and other settings are fixed. Random early-stopping validation is disabled.
- `reference`: training class frequencies with small additive smoothing; no feature inputs.

The same method grids and fitting schedule are used for each representation. Raw and directional-plus-context input sets have the same number of columns. A flexible raw-signal comparison helps check whether an advantage is limited to making a linear classifier's task easier. This does not prove one representation wins for every possible prediction method or capacity.

## Scores and hypothesis decisions

Lower log loss is the primary objective. Lower Brier score also indicates better probabilities; this notebook uses the sum of squared class-probability errors per observation, not that sum divided by three. Accuracy measures the correct predicted class; balanced accuracy averages recall equally across the three classes.

The predeclared primary method is `nonlinear`. Changing this after inspecting results would be another model-selection decision. Two paired improvements are tested:

1. `context log loss − direction_context log loss`.
2. `raw_context log loss − direction_context log loss`.

Positive values favor the directional representation. Circular moving blocks of predictions are resampled separately within each outer window; a block never crosses a model-refit boundary. The default block length 42 exceeds the 21-session overlapping outcome horizon. Check sensitivity to dependence assumptions, particularly if changing the horizon.

Intervals use bootstrap means. One-sided approximate tail values center the resampled differences under zero mean improvement. Holm adjustment covers the two predeclared tests. `supports_gain` requires positive observed gain, a positive lower 95% interval endpoint, and adjusted tail value at most alpha. Both comparisons must pass for the notebook to report evidence of preferring the directional encoding.

Inference is approximate and conditional on the fitted models. Resampling scores does not repeat training, account fully for overlapping training histories, or solve market nonstationarity. The two-test correction does not cover repeatedly trying lookbacks, horizons, outcomes, contexts, or model families and reporting the winner. Use development for that process and reserve the final period for frozen confirmation.

If there is a gain over context but not the raw signals, direction may contain useful information without showing that normalization is preferable. If a gain appears only for the linear method, a likely explanation is a convenient nonlinear feature transformation. If there is no gain, that does not rule out every directional feature construction.

Compare absolute scores against the training-frequency reference as well. Beating two weaker feature models is not enough to establish predictive usefulness beyond unconditional outcome frequencies.

## Final evaluation

`RUN_FINAL_TEST=False` leaves final forecast scores unevaluated. The raw input histories and feature/target tables may contain final-period rows, but they are not used for fitting, model selection, development outcomes, or forecast scoring.

When enabled after freezing the experiment, the notebook chooses settings on pre-test development only, fits the four fixed representations for the predeclared primary method, and evaluates them on mature final outcomes. It repeats the two fixed comparisons; it does not choose which representation to report from final scores. Origins with unavailable future outcomes are excluded. The probability models remain fixed throughout that period.

If test results influence a revised experiment, the period is no longer untouched. More outer chronological replications or fresh data are then needed for confirmation.

## Editing and saved records

Most changes belong in the notebook's first configuration cell. For new signals, change `build_features` in `behavior_features.py`, its explicit input-set definitions, and their tests. For another forecasting method, change the small factories in `behavior_validation.py`. Restart the kernel after helper edits.

Each timestamped `behavior_results` folder contains settings/code/input hashes, exact predictor columns, outer/inner splits, tuning trials, per-date probabilities, aggregate scores, paired comparisons, gains by window, and development feature/outcome tables. Final files appear only if final evaluation is enabled.

Suggested iteration log: hypothesis → one change → expected effect → split/fit checks → validation result → next decision. Start by changing one of lookback, forecast horizon, or volatility window. Outcome thresholds define different tasks, so absolute log losses across different outcome definitions are not direct model comparisons.

Implementation references: [time-series splits](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html), [histogram gradient boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html). Signal motivation: [time-series momentum research](https://fairmodel.econ.yale.edu/ec439/mosk.pdf); this motivates trend/variability features, not spherical geometry.

## Verification and initial development result

The notebook was executed on the cached market inputs. It uses 4,170 development origins and evaluates 756 later validation dates. Each outer training window removes 21 origins for immature labels, and all exported inner and outer label-end checks pass. Final market-data evaluation remains disabled.

For the predeclared nonlinear method, direction plus intensity/context has log loss 1.1037, compared with 1.1160 for intensity/context and 1.1184 for original signals/context. Gains of 0.0124 and 0.0148 are uncertain: both bootstrap intervals include zero, and neither adjusted test passes. Gains are positive in two windows and negative in one. The class-frequency reference has lower average log loss (1.0578), so these results do not establish useful forecasting or a directional advantage.

Seven new tests cover feature causality, direction/intensity reconstruction, near-zero vectors, price-session outcome dates, mature-label purging, nested/final synthetic validation, and the paired-comparison decision rule. The full Thesis suite has 20 passing tests. A separate full notebook run exercises every cell and the enabled final-evaluation branch on synthetic inputs without saving synthetic results as market evidence.
