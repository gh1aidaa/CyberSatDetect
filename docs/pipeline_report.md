1. Data Cleaning

This document describes the end-to-end data pipeline used in the **CyberSatDetect** project, from raw telemetry ingestion to strict model evaluation. The system is designed as an **Unsupervised / Self-Supervised Anomaly Detection** pipeline:

- The model is trained on **normal data only**.
- Any labels are used **only for evaluation** (measurement), not for training.

1.1 Input Formats

CyberSatDetect supports two telemetry input formats at inference time:

- **CSV**: multi-channel tabular telemetry where each numeric column is treated as an independent channel.
- **NPY**: NumPy arrays with multiple supported shapes (raw series or pre-windowed tensors).

1.2 Numeric Cleaning (Inference-Time)

For raw 1D series (per channel), the system applies a minimal but robust numeric cleaning procedure:

- Convert to **float32**
- Replace non-finite values (Inf/−Inf) with NaN
- Linear interpolation for NaN gaps
- Fill remaining non-finite values with 0.0

This prevents inference crashes and reduces score artifacts caused by non-finite values.

1.3 CSV Cleaning Rules

For CSV uploads, the pipeline enforces:

- Attempt parsing with default delimiter, then fallback to `;` if needed
- Drop rows that are fully empty
- Use numeric columns only
- If no numeric columns exist, attempt `to_numeric` conversion with coercion

If no numeric columns can be produced, the file is rejected.

1.4 NPY Shape Validation Rules

The inference pipeline accepts the following NPY shapes:

- **(N,)**: raw series for a single channel
- **(N, C)**: raw multi-channel matrix (C channels)
- **(B, T)**: pre-windowed single-channel windows
- **(B, T, 1)**: pre-windowed single-channel windows

Any other shape is rejected. If windows are provided, they must be consistent with:

- Window length \(T = 100\)
- Channels \(C = 1\)


2. Data Preparation

The preparation stage converts telemetry into a format suitable for the model and consistent across inputs.

2.1 Sliding Window Segmentation

For raw series inputs, the pipeline uses **sliding windows**:

- Window length: \(W = 100\)
- Stride: \(S = 50\) (50% overlap)

The number of windows is:

\[
N_{windows} = \left\lfloor \frac{L - W}{S} \right\rfloor + 1
\]

where \(L\) is the length of the series.

2.2 Window Tensor Shape

All model inputs are standardized to:

- \(X \in \mathbb{R}^{B \times 100 \times 1}\)

where \(B\) is the number of windows.

2.3 Output Artifacts Per Run (Inference)

For each uploaded run:

- The raw file is stored under `backend/app/data/uploads/<run_id>/raw.*`
- Per-channel results are written as CSV files under `.../<run_id>/results/`
- A JSON summary is written to `.../<run_id>/results/summary.json`


3. Data Structuring (Self-Supervised Framework)

CyberSatDetect uses a self-supervised structuring principle: learn a compact representation of **normal behavior**, then detect deviations using an anomaly score.

3.1 Continual Learning Buffer (Optional)

During inference, windows with sufficiently low scores can be stored in a **normal pool** as candidate “safe” windows for later continual learning. This module is designed to avoid training on suspicious data by selecting only windows that are well below the threshold.

3.2 Strict Evaluation Dataset (attacked_v2)

To evaluate the system correctly (without the common mistake “mark the entire attacked file as anomaly”), we regenerate a strict attacked dataset:

- Source: **normal test split only** from `data/reduced` using `backend/config/data_split.json`
- Output: `data/attacked_v2/*.npz`
- Each `.npz` stores:
  - `X`: attacked windows \((B,100,1)\)
  - `y_timestep`: timestep-level mask \((B,100)\)
  - `y_window`: window-level labels \((B,)\)
  - `attack_type`, `attack_start`, `attack_end`, `source_file`

3.3 Window-Level Labeling Rule

For each window:

- The window is labeled anomaly if \(\ge 10\%\) of its timesteps are attacked.

Formally:

\[
y_{window} = \mathbb{1}\left(\frac{1}{W}\sum_{t=1}^{W} y_{timestep}(t) \ge 0.10 \right)
\]

3.4 Attack Types (Evaluation-Only)

The attacked_v2 generator supports partial attacks inside windows:

- Freeze attack
- Spike attack
- Drift attack
- Pattern shift attack
- Noise attack
- Drop/zero masking attack
- Scale attack

Note: **attacked_v2 is never used for training** in the unsupervised regime; it is used only to measure performance.


4. Model Training (Baseline v1)

4.1 Training Regime

The baseline model (`best_model.keras`) is trained using normal windows only:

- Train split: `data/reduced` → `train` list in `data_split.json`
- Validation split: `data/reduced` → `validation` list in `data_split.json`
- Test split is reserved strictly for evaluation and is not used for training.

4.2 Model Inputs and Outputs

Input:

- \(X \in \mathbb{R}^{B \times 100 \times 1}\)

Outputs:

- Reconstruction head: \(\hat{X} \in \mathbb{R}^{B \times 100 \times 1}\)
- Prediction head: \(\hat{p} \in \mathbb{R}^{B \times 1 \times 1}\) (conceptually; stored as \((B,1)\) then reshaped)

4.3 Composite Loss (Training Only)

Training optimizes a composite loss:

\[
L_{total} = W_{recon}L_{recon} + W_{pred}L_{pred} + W_{grad}L_{grad} + W_{sep}L_{sep}
\]

Where:

- \(L_{recon} = MSE(X, \hat{X})\)
- \(L_{pred} = MSE(X_{last}, \hat{p})\)
- \(L_{grad} = MSE(\Delta X, \Delta \hat{X})\)
- \(L_{sep}\) is a margin-based separation loss that enforces pseudo-anomaly scores above normal scores.

4.4 Inference-Time Score (No Separation Term)

At inference/evaluation time (strict scoring), the anomaly score is computed using only:

\[
score = e_{recon} + e_{pred} + e_{grad}
\]

This ensures \(L_{sep}\) remains a training regularizer and does not alter runtime inference computations.


5. Strict Evaluation Results (Baseline v1)

This section reports strict, window-level results produced by:

- `backend/models/evaluate_model_strict_v2.py`
- Normal source: `data/reduced` (test split only)
- Attacked source: `data/attacked_v2` (window labels `y_window`)
- Thresholds p99/p995/p997/3sigma computed from **normal test only** (unsupervised thresholding rule)

5.1 Dataset Size (Strict v2)

From `backend/app/evaluation_strict_v2/evaluation_summary.json`:

- Normal:
  - Files used: 144
  - Total windows: 241,896
- Attacked_v2:
  - Files used: 144
  - Total windows: 241,896
  - Anomaly windows: 72,588
  - Normal windows inside attacked_v2: 169,308

5.2 Core Metrics (Best F1 Operating Point)

At the best-F1 threshold (analysis-only selection):

- Accuracy ≈ 93.60%
- Balanced Accuracy ≈ 86.40%
- ROC-AUC ≈ 0.889
- PR-AUC ≈ 0.744
- Precision ≈ 0.80
- Recall/TPR ≈ 0.76
- FAR/FPR ≈ 3.31%
- F1-score ≈ 0.78

5.3 Threshold Trade-Offs (Operational)

The evaluation produces multiple candidate thresholds:

- Statistical (computed from normal only): p99, p995, p997, 3sigma
- Analysis-only: best-F1, best-Youden-J, FAR-constrained thresholds

Key observation:

- Reducing FAR to \(\le 1\%\) generally reduces recall significantly, reflecting overlap between normal and attacked score distributions.

5.4 Final Verdict (Baseline)

The baseline model demonstrates meaningful separation (ROC-AUC ~ 0.889), but does not reach:

- ROC-AUC ≥ 0.95
- F1 ≥ 0.90
- Balanced Accuracy ≥ 95%

Therefore, the baseline is acceptable as a stable inference model, but further improvements require either:

- a stronger representation (training improvements), or
- improved signal normalization / feature learning, or
- a higher-quality normal dataset (less noise, more consistent operating regimes).


Appendix A: File Map (CyberSatDetect)

- Inference API: `backend/app/api.py`
- Baseline model: `backend/app/best_model.keras`
- Baseline thresholds: `backend/app/thresholds.json`
- Strict evaluation output: `backend/app/evaluation_strict_v2/`
- Strict attacked dataset generator: `backend/models/regenerate_attacked_dataset.py`
- Strict evaluator: `backend/models/evaluate_model_strict_v2.py`

