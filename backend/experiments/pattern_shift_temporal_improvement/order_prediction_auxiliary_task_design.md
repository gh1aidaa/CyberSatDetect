# Order-prediction auxiliary task — design note (self-supervised, no attack labels)

## Idea

Add a lightweight **`order_head`** that consumes a **latent summary** \(z\) of the window (e.g. pooled encoder output) and predicts a binary label:

- **`correct_order`:** the window is presented in natural temporal order (all normal training windows).
- **`shifted_or_shuffled_order`:** the same values under a **pseudo** pattern-shift (random cyclic shift, block permutation, or time-reversal) applied **only** to unlabeled normal data during training.

## Training data (no real attack labels)

- **Positive (order correct):** original normal windows.
- **Negative (order wrong):** synthetically shuffled windows built from the **same** normal examples (self-supervised). No dependency on `attacked_v2` labels.

## Loss

\[
L_{\text{order}} = \mathrm{BCE}\bigl( y_{\text{order}}, \hat{y}_{\text{order}} \bigr)
\]

with \(y_{\text{order}} \in \{0,1\}\) indicating shuffled vs not.

## Suggested total loss (training only)

\[
L_{\text{total}} =
W_{\text{recon}} L_{\text{recon}}
+ W_{\text{pred}} L_{\text{pred}}
+ W_{\text{grad}} L_{\text{grad}}
+ W_{\text{sep}} L_{\text{sep}}
+ W_{\text{order}} L_{\text{order}}
\]

\(L_{\text{sep}}\) remains **training-only** if used; inference stays without separation, per current system rules.

## Why this helps temporal understanding

1. **Encourages latent sensitivity to ordering:** the encoder must retain cues that are destroyed by shuffling (phase progression, smooth physics), not only magnitude statistics.

2. **No need for attack-class labels:** negatives are **algorithmically generated** from normal data — pure self-supervision.

3. **Complements reconstruction:** reconstruction can be low when marginals match; order classification asks a **different** question: “is this trajectory **plausible as a time evolution**?”

## Inference options (future)

- **Option A (auxiliary only at train):** order head is a regularizer; inference score unchanged.
- **Option B (score add-on):** \(\text{score} \leftarrow \text{score} + W_{\text{order,inf}} \cdot \mathrm{BCE}(\text{order\_pred})\) or use logits as anomaly term — requires calibration study.

## Positioning

This is a **standard self-supervised auxiliary** pattern (similar in spirit to jigsaw, rotation, or contrastive ordering tasks). It extends the model’s inductive bias without claiming new attack labels.

---

*Design only — no training executed in this experiment folder.*
