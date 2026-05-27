# Multi-step prediction head — design note (future model version)

## Motivation

The current single-step prediction target (last timestep or one-step-ahead) can be **too local**: the model may predict a plausible next value even when **longer-range temporal structure** has been altered (e.g. pattern-shift attacks that preserve many local transitions).

## Proposed change (training / architecture)

**Input:** first \(T_{\text{in}} = 90\) timesteps of the window.  
**Target:** the **next** \(T_{\text{out}} = 10\) timesteps (the “future tail” of the same window, or the immediately following segment if using streaming data).

Instead of \(\hat{p} \approx x_{T}\) only, the head outputs:

\[
\hat{X}_{\text{future}} \in \mathbb{R}^{B \times T_{\text{out}} \times C}
\quad\text{matching}\quad
X_{\text{future}} \in \mathbb{R}^{B \times T_{\text{out}} \times C}
\]

## Loss

\[
L_{\text{pred,multi}} = \mathrm{MSE}\bigl( X_{\text{future}}, \hat{X}_{\text{future}} \bigr)
\]

(optionally weighted per channel / horizon).

## Why this enforces temporal pattern understanding

1. **Longer dependency path:** gradients must flow through representations that encode **90 steps** of context to predict **10 distinct** future values — harder to satisfy with permuted or reordered structure inside the conditioning span.

2. **Pattern-shift stress:** reordering segments often preserves **short** correlations but breaks **global** phase alignment; multi-step MSE is more likely to spike when the model’s internal dynamics no longer match the true evolution.

## Architecture adjustments (conceptual)

- Encoder consumes \(x_{1:90}\) (same backbone as today or slightly widened).
- Decoder / dense head maps latent state to **vector** of length \(10 \cdot C\), reshaped to \((10, C)\).
- Optional: causal transformer or additional GRU layer on the latent for autoregressive multi-step decoding.

## Inference score (future)

Keep reconstruction and gradient terms; replace single-step prediction error with multi-step error:

\[
\text{score} =
\text{recon\_error}
+ \text{multi\_step\_prediction\_error}
+ \text{gradient\_error}
\]

with the same **weighting policy** as today (fixed \(W_{\text{RECON}}, W_{\text{PRED}}, W_{\text{GRAD}}\) from `thresholds.json` or successor config — **not** modified in this research folder).

## Separation loss

\(L_{\text{sep}}\) remains **training-only** if present; **not** used in inference, consistent with current API semantics.

---

*Design only — no retraining or checkpoint export in this experiment.*
