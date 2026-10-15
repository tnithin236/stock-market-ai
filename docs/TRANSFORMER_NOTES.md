# Transformer notes (for the README and interviews)

## Architecture used here
Daily feature vector -> linear embedding (d_model) -> + sinusoidal positional encoding -> N pre-norm
encoder blocks (multi-head self-attention + feed-forward) -> last day's representation -> LayerNorm ->
dropout -> linear head (one scaled return, or one logit).

## Why attention can capture longer-range dependencies
* An LSTM passes information through a single hidden state, one step at a time. Something from 40 days
  ago must survive 40 gated updates, and the gradient to learn it must flow back through all of them.
* Self-attention connects every pair of days in the window directly. The last day can weight any earlier
  day (for example a volatility spike three weeks ago) in one step, with learned weights per head.
* All days can be processed in parallel, which trains faster on GPUs.

## Why that does not help much here
* Attention is a better way to *retrieve* information, not a way to *create* it. If distant days carry no
  predictive information about tomorrow, there is nothing for attention to find.
* Transformers are data hungry. We have about 2,000 training windows, which favours small, strongly
  regularised models.
* Self-attention has no built-in sense of order, so positional encoding is required.

## Leakage check
Each window contains only days up to and including the prediction day. Attention inside a window is
bidirectional, but every token is already in the past relative to the target, so nothing from the
future is visible.

## Interpretation caveat
Attention weights are NOT explanations of the prediction. We use permutation importance instead and
report its limits (correlated features share credit; unrealistic shuffled inputs).
