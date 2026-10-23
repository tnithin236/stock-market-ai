# Interview preparation - Stock Market Prediction & Analysis System

Numbers refer to the AAPL snapshot in the README (test period 2025-01-17 to 2026-10-13, 436 days). Practise answering
out loud in 60-90 seconds each. Be upfront that the project's finding is a *negative* result and that this is a strength.

## 1. The 60-second pitch

"I built an end-to-end system that forecasts next-day stock price and direction using baselines, XGBoost, an LSTM and a
Transformer. The emphasis was evaluating honestly: chronological splits, train-only scaling, confidence intervals,
significance tests and a cost-aware backtest. The result was that none of the models significantly beat naive baselines -
best test AUC was 0.546 with an interval that includes 0.5 - and after costs no strategy beat buy-and-hold. I also built a
FastAPI service, a Streamlit dashboard, Docker and CI, with about 130 tests. The main takeaway for me is how easy it is to
fool yourself with time-series ML and how to build safeguards against that."

## 2. Questions and model answers

### Data, leakage and validation

**Q1. Why not shuffle the data / use k-fold?**
Shuffling lets the model train on days after the ones it is tested on, so it can learn patterns from the future and
neighbouring days are highly correlated. A chronological split mimics deployment: train on the past, test on the future.
I used 70/15/15 with a one-row gap because the label of day t is day t+1's price.

**Q2. What is data leakage and where could it happen in your pipeline?**
Any information from the evaluation period influencing training. Places I guarded: (1) features - indicators use only past
windows, and a test recomputes them on truncated history and with altered future prices; (2) scaling - the scaler is fitted on
training rows only; (3) tuning - early stopping, feature choice, Optuna and calibration use validation only, test is touched
once; (4) backtest timing - signal at the close, trade at the next open; (5) a partial current-day bar, which the loader drops.

**Q3. How did you prevent look-ahead bias in the backtest?**
The model sees features at the close of day t and the trade executes at the open of t+1; the old position earns the overnight
gap and the new one the intraday move. A test shows changing a future position cannot change any earlier day's return.

**Q4. What is survivorship bias and does it affect you?**
Studying only assets that survived (AAPL did) overstates how well a strategy works because failures vanish from the data.
Yes, it affects me: I used one successful large-cap. A proper study would use a point-in-time universe including delisted stocks.

**Q5. What is stationarity and why does it matter?**
A stationary series has stable statistical properties over time. Prices are not (ADF p = 0.99); returns are (p ~ 0). Models
fed non-stationary levels can learn spurious trends, and trees cannot extrapolate beyond the training range. So I predict
returns and convert back to price, and I compare "all features" with scale-free features.

**Q6. What is the problem with R-squared on prices?**
Prices trend, so almost any forecast gets R-squared near 0.99. I judge models by RMSE relative to the previous-day forecast
and by out-of-sample R-squared on returns relative to a zero forecast.

### Modelling

**Q7. Why XGBoost, and how did you train it?**
Strong on tabular data, fast and robust with small samples. Early stopping on validation, shallow trees, subsampling,
class weighting for the mild imbalance, Optuna search on validation loss, and only accepting tuned parameters if they beat the
defaults by at least 0.1%.

**Q8. Explain how an LSTM works and why you used it.**
An LSTM keeps a cell state updated through input, forget and output gates, letting it carry information across time steps
while mitigating vanishing gradients. I fed it 30-day windows of scale-free features and read the last time step.

**Q9. Why might a Transformer capture longer-range dependencies better?**
Self-attention connects every pair of days directly, so the last day can weight any earlier day in one step instead of
through many recurrent updates, and it parallelises well. Positional encoding supplies order. But it only helps if distant
days carry signal, and with ~2,000 windows it is data-hungry, so it did not help (and was slightly worse).

**Q10. What are the signs of overfitting in your runs?**
Best validation epoch at 2-6 of 12-16 epochs, validation loss rising while training loss falls, and the ordering of
out-of-sample error (XGBoost 1.0019, LSTM 1.0029, Transformer 1.0052 vs naive) following model flexibility.

**Q11. How did you handle class imbalance?**
It is mild (53% up days). I compared accuracy against the always-UP baseline rather than 50%, used class weights during
training, and reported precision/recall/F1, balanced accuracy, MCC and ROC-AUC. Accuracy alone is misleading here.

**Q12. What is early stopping and LR scheduling?**
Early stopping halts training when validation loss stops improving and restores the best weights. `ReduceLROnPlateau`
halves the learning rate when validation loss plateaus, helping fine convergence.

**Q13. Why did the regression models just predict the average drift?**
Daily returns are mostly noise, so the best low-variance forecast is a small positive constant. An out-of-sample R-squared
of ~0.003 equals mean-squared-drift over return variance (about 0.001^2 / 0.019^2), which is what a constant forecast scores.

### Evaluation and statistics

**Q14. Why use confidence intervals and a significance test?**
With 436 test days, accuracy has a standard error of ~2.4 points, so 1-2 point differences are noise. I used Wilson intervals,
a one-sided binomial test against the majority baseline, and bootstrap intervals for AUC.

**Q15. What is ROC-AUC and why use it besides accuracy?**
The probability that a random up day gets a higher score than a random down day; threshold-free and insensitive to class
balance. 0.5 is chance. My best was 0.546 with a 95% interval [0.493, 0.598] - not distinguishable from chance.

**Q16. What is calibration and what bug did you hit?**
Calibration maps scores to probabilities. I used Platt scaling on validation. For models with negative-slope scores it flipped
their direction, turning an AUC of 0.48 into 0.52 - fitting noise from ~400 labels. I changed it to fall back to the
base rate if the slope is not positive.

**Q17. What is the multiple-comparisons problem?**
Try enough strategies and one will look good by luck. In a synthetic test one strategy had p = 0.015 with 3 trades. I
Bonferroni-corrected p-values by the number of strategies tried, and flagged that few trades make statistics unreliable.

**Q18. SHAP vs permutation importance vs gain?**
Gain counts split improvements in trees; SHAP attributes each prediction to features (exact for trees); permutation measures
performance drop when a feature is shuffled (model-agnostic, works for LSTM/Transformer). All describe what the model uses, not
causality; correlated features share credit and permutation creates unrealistic inputs. I also found that early stopping
leaves discarded trees whose gains must be excluded.

**Q19. Your LSTM's AUC was 0.438 (below chance). Should you flip the signal?**
No. Flipping after seeing test results is hindsight; validation did not indicate it (val AUC ~0.49). Below-chance results
usually signal non-stationarity - patterns from training reversed - or noise. The interval [0.383, 0.491] is also optimistic
because bootstrap assumes independent days.

### Backtesting and finance

**Q20. Explain Sharpe ratio and max drawdown.**
Sharpe = mean excess return / standard deviation, annualised by sqrt(252); return per unit of risk. Max drawdown = largest
peak-to-trough loss of the equity curve. Buy-and-hold had Sharpe 0.99 and drawdown -30%; the XGBoost classifier had 0.94 and
-16%, but it was only invested half the time.

**Q21. Why compare against random strategies with the same exposure?**
A strategy out of the market 50% of the time will lag a bull market and have smaller drawdowns by construction. Random
strategies with the same exposure show what luck alone produces. XGBoost beat 84% of them (p = 0.16), so not significant.

**Q22. How do transaction costs change the picture?**
They scale with turnover. The momentum rule made 110 trades and lost 10.95% of capital to costs: +19.7% at 0 bps but -30.8% at 25 bps.

**Q23. What is walk-forward validation and why is it better?**
Repeatedly train on a window, test on the next period, roll forward. It gives many out-of-sample estimates across regimes,
instead of one split. It is my first suggested improvement.

**Q24. Why can't ML easily predict markets?**
Low signal-to-noise, efficient-market competition that arbitrages away patterns, non-stationarity and regime changes, and
limited data. Edge, if any, needs better information (alternative data, speed) rather than bigger models.

### Engineering

**Q25. How is the system structured and why a service layer?**
Business logic lives in a framework-free service used by both FastAPI and Streamlit, so it is unit-tested without HTTP.
Training saves predictions to CSV, so the backtester doesn't import model code.

**Q26. How do you make predictions honest in the API?**
No "confidence" field; a raw score with `score_is_probability: false`; a calibrated probability with an explanation; the
held-out evidence and a significance flag; warnings and a disclaimer; stale-data warnings; strict ticker validation.

**Q27. How do you test ML code?**
Property tests for causality (truncated history gives identical features; altering future data does not change the past),
hand-computed expectations for the backtest engine, parity tests (inference equals the training pipeline), statistical helper
tests, and API/dashboard tests with synthetic data - no network.

**Q28. How did you ensure reproducibility?**
Frozen data end date (a refreshed download once shifted the test window by two days and made tables non-comparable), fixed
seeds, recorded test periods in every result file and a warning when they differ.

**Q29. How would you deploy and monitor this?**
Docker/Compose with CPU-only PyTorch, a non-root user, mounted artifacts and CI. For production: authentication, rate
limiting, a model registry (MLflow), scheduled retraining, drift monitoring, and alerting on data staleness.

**Q30. What would you do next, and what would change your conclusion?**
Walk-forward validation across many years and tickers, a point-in-time universe, richer data (news sentiment, fundamentals),
and portfolio-level evaluation. A significant edge that survives costs, multiple-testing correction and walk-forward testing
would change my conclusion.

## 3. Bugs and lessons (great "tell me about a challenge" stories)

1. **Partial bar leakage:** a download during market hours included an unfinished day; fixed in the loader.
2. **Data snapshot drift:** refreshing the cache shifted the test window and mixed results; now frozen and checked.
3. **Calibrator flipped model direction:** AUC "improved" from 0.48 to 0.52 via a negative slope; now guarded.
4. **Feature importance counted discarded trees:** SHAP and gain disagreed; fixed to use only the best iteration.
5. **Module named `*_test.py`:** pytest collected it as a test; renamed and set `testpaths`.

## 4. Technical concepts you should be able to explain

**Time series and finance:** stationarity (ADF test), returns vs log returns, volatility clustering, fat tails, autocorrelation
of returns vs squared returns, regimes, efficient market hypothesis, survivorship bias, look-ahead bias, transaction costs,
slippage, Sharpe, Sortino, drawdown, Calmar, buy-and-hold, long/flat vs long/short.

**Indicators:** SMA/EMA, RSI, MACD, Bollinger bands, ATR, momentum, volume ratios.

**ML fundamentals:** bias-variance, overfitting and regularisation, train/validation/test roles, data leakage, cross-validation
vs walk-forward, class imbalance, baselines, hyperparameter search (Bayesian/TPE with Optuna).

**Models:** gradient boosting and XGBoost parameters (depth, learning rate, subsample, colsample, min_child_weight, lambda,
early stopping), LSTM gates and vanishing gradients, Transformer self-attention (queries/keys/values), multi-head attention,
positional encoding, layer norm, residuals, pre-norm.

**Training:** AdamW and weight decay, learning-rate scheduling, gradient clipping, early stopping, checkpoints, Huber vs MSE
loss, BCE with logits and class weights, seeding and determinism.

**Evaluation:** MAE, RMSE, MAPE, R-squared and its pitfalls, precision, recall, F1, ROC-AUC, confusion matrix, balanced accuracy,
MCC, Brier score and log-loss, calibration (Platt/isotonic), confidence intervals (Wilson, bootstrap), hypothesis tests,
multiple comparisons (Bonferroni), null/permutation tests.

**Explainability:** SHAP/TreeSHAP, gain importance, permutation importance and their limits; attention is not explanation.

**Engineering/MLOps:** REST design and status codes, pydantic validation, background jobs, service-layer architecture,
pytest strategies for ML, Docker (images, layers, volumes, non-root, healthchecks), docker-compose, CI with GitHub Actions,
reproducibility, experiment tracking (MLflow), model registry, monitoring and drift.
