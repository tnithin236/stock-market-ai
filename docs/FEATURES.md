# Feature reference

All features at row *t* use only data up to day *t*. Targets look one day ahead and are never used as inputs.

| Group | Feature(s) | Meaning | Why it may help |
|---|---|---|---|
| Raw | Open, High, Low, Close, Volume | Daily OHLCV (adjusted prices) | Base information; Close anchors price-level forecasts |
| Returns | return, log_return | Daily % change and log change | Stationary, comparable across time and stocks |
| Trend | sma_10/20/50, ema_10/20/50 | Simple / exponential moving averages | Smooth noise; show trend direction at several horizons |
| Trend (scale-free) | close_sma_N_ratio, close_ema_N_ratio | Close / average - 1 | Stationary distance from trend; stretched prices tend to revert or extend |
| Momentum | rsi_14 | Wilder RSI, 0-100 | Overbought/oversold pressure |
| Momentum | macd, macd_signal, macd_hist | EMA12 - EMA26, its EMA9, and the gap | Trend strength and turning points |
| Momentum | mom_5, mom_10 | Close / Close(n days ago) - 1 | Short and medium-term price momentum |
| Volatility | bb_mid/upper/lower | 20-day mean +/- 2 std | Dynamic price envelope |
| Volatility | bb_pctb, bb_width | Position inside bands; relative band width | Scale-free squeeze/expansion signals |
| Volatility | atr_14, atr_14_pct | Average True Range, and as % of price | Typical daily range including gaps; risk sizing |
| Volatility | vol_10, vol_20 | Rolling std of log returns | Volatility clustering: calm and turbulent regimes persist |
| Volume | volume_change, volume_ratio_20 | Day-over-day change; volume vs 20-day average | Unusual participation often accompanies large moves |
| Lags | close_lag_1/2/3/5, ret_lag_1/2/3/5 | Past prices and returns | Give tree models short-term memory |
| Targets | target_close, target_return, target_direction | Next-day close, return, up/down | What we predict (never inputs) |

Scale-dependent columns (price levels, raw volume, MACD, ATR) trend over time. Tree models cannot extrapolate beyond the training range, so later experiments compare `feature_columns(df)` against `feature_columns(df, scale_free_only=True)`.
