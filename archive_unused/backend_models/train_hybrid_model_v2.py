"""
CyberSatDetect - Hybrid LSTM/GRU Autoencoder (v2) Training Script
===============================================================

Goal:
Train an improved self-supervised/unsupervised anomaly detector using NORMAL data only.

STRICT RULES (per user request):
- Do NOT modify the old model files: backend/app/best_model.keras, backend/app/final_model.keras
- Do NOT modify the original training script: backend/models/train_hybrid_model.py
- Do NOT use attacked_v2 in training
- Do NOT use test split in training
- Do NOT modify data/reduced or data/attacked_v2
- Do NOT fake results; evaluation is done separately using evaluate_model_strict_v2.py

Training philosophy:
- Model learns the normal manifold from normal windows only.
- Pseudo-anomalies are generated on-the-fly from normal batches to shape the score distribution
  (self-supervised separation loss). No labels are used.

Outputs (v2 only, never overwrite v1):
- Model            : backend/app/best_model_v2.keras
- Thresholds       : backend/app/thresholds_v2.json        (computed from NORMAL validation only)
- Training history : backend/app/training_v2_history.json
- Metadata         : backend/app/training_v2_metadata.json

Final evaluation command (run separately):
python backend/models/evaluate_model_strict_v2.py ^
  --model backend/app/best_model_v2.keras ^
  --normal-dir data/reduced ^
  --attacked-dir data/attacked_v2 ^
  --split-json backend/config/data_split.json ^
  --output-dir backend/app/evaluation_v2_improved ^
  --window-size 100 ^
  --stride 50
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers


def _maybe_tqdm():
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm
    except Exception:
        return None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_split(split_json: Path) -> Dict[str, List[str]]:
    with split_json.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return {k: list(v) for k, v in obj.items()}


def load_npy_windows(path: Path, window_size: int = 100, channels: int = 1) -> np.ndarray:
    """
    Load a .npy file expected to contain windows:
      (B, T) or (B, T, C), where T=100, C=1 in this project.
    """
    X = np.load(path).astype(np.float32)
    if X.ndim == 2:
        X = X[..., None]
    if X.ndim != 3:
        raise ValueError(f"Unsupported shape for {path}: {X.shape}")
    if int(X.shape[1]) != int(window_size) or int(X.shape[2]) != int(channels):
        raise ValueError(f"Bad shape for {path.name}: expected (*,{window_size},{channels}) got {X.shape}")
    return X


def build_model_v2(T: int = 100, C: int = 1) -> Model:
    """
    Improved architecture (still same functional goal):
    - Conv1D feature extractor
    - BatchNorm + Dropout
    - Bidirectional LSTM + GRU encoder
    - Reconstruction decoder head
    - Forecasting prediction head (predict last/next step as a single step)

    Outputs:
      recon: (B,T,C)
      pred : (B,C)  -> reshaped to (B,1,C) in scoring (matches training formula)
    """
    inp = layers.Input(shape=(T, C), name="input")

    # ---- Feature extractor (local patterns) ----
    x = layers.Conv1D(64, 5, padding="causal", activation="relu", name="conv1")(inp)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Dropout(0.10, name="drop1")(x)

    x = layers.Conv1D(64, 3, padding="causal", activation="relu", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Dropout(0.10, name="drop2")(x)

    # ---- Sequence encoder (bidirectional context + compact latent) ----
    x = layers.Bidirectional(
        layers.LSTM(96, return_sequences=True, dropout=0.10, recurrent_dropout=0.0),
        name="bilstm_enc",
    )(x)
    x = layers.GRU(64, return_sequences=False, dropout=0.10, name="gru_enc")(x)

    x_latent = x
    x = layers.RepeatVector(T, name="repeat_vector")(x)

    # ---- Decoder (reconstruction) ----
    d = layers.GRU(64, return_sequences=True, dropout=0.10, name="gru_dec")(x)
    d = layers.Bidirectional(layers.LSTM(96, return_sequences=True, dropout=0.10), name="bilstm_dec")(d)

    recon = layers.TimeDistributed(layers.Dense(C, name="td_dense_recon"), name="reconstruction")(d)

    # ---- Prediction head (single-step forecast / last step target) ----
    pred_feat = layers.Dense(64, activation="relu", name="pred_fc1")(x_latent)
    pred_feat = layers.Dropout(0.10, name="pred_drop")(pred_feat)
    pred = layers.Dense(C, name="prediction")(pred_feat)  # (B,C)

    return Model(inputs=inp, outputs=[recon, pred], name="Hybrid_LSTM_GRU_AE_v2")


def batch_scores_tf(x_true, x_recon, x_pred, w_recon: float, w_pred: float, w_grad: float):
    e_recon = tf.reduce_mean(tf.square(x_true - x_recon), axis=[1, 2])
    pred_reshaped = tf.reshape(x_pred, [-1, 1, tf.shape(x_true)[2]])
    e_pred = tf.reduce_mean(tf.square(x_true[:, -1:, :] - pred_reshaped), axis=[1, 2])

    dx_true = x_true[:, 1:, :] - x_true[:, :-1, :]
    dx_recon = x_recon[:, 1:, :] - x_recon[:, :-1, :]
    e_grad = tf.reduce_mean(tf.square(dx_true - dx_recon), axis=[1, 2])

    return (w_recon * e_recon) + (w_pred * e_pred) + (w_grad * e_grad)


def mse(a, b):
    return tf.reduce_mean(tf.square(a - b))


def grad_mse(x_true, x_recon):
    dx_true = x_true[:, 1:, :] - x_true[:, :-1, :]
    dx_recon = x_recon[:, 1:, :] - x_recon[:, :-1, :]
    return tf.reduce_mean(tf.square(dx_true - dx_recon))

@tf.function
def train_step(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    batch: tf.Tensor,
    attack_cfg: AttackCfg,
    W_recon: tf.Tensor,
    W_pred: tf.Tensor,
    W_grad: tf.Tensor,
    W_sep: tf.Tensor,
    margin: tf.Tensor,
):
    """
    Single training step compiled with tf.function for speed.
    Returns (loss_total, L_recon, L_pred, L_grad, L_sep)
    """
    with tf.GradientTape() as tape:
        recon, pred = model(batch, training=True)

        L_recon = tf.reduce_mean(tf.square(batch - recon))
        pred_reshaped = tf.reshape(pred, [-1, 1, tf.shape(batch)[2]])
        L_pred = tf.reduce_mean(tf.square(batch[:, -1:, :] - pred_reshaped))
        L_grad = grad_mse(batch, recon)

        score_normal = batch_scores_tf(batch, recon, pred, W_recon, W_pred, W_grad)
        pseudo = build_pseudo_anomalies_v2(batch, attack_cfg)
        recon_p, pred_p = model(pseudo, training=True)
        score_pseudo = batch_scores_tf(pseudo, recon_p, pred_p, W_recon, W_pred, W_grad)
        L_sep = tf.reduce_mean(tf.nn.relu(margin - (score_pseudo - score_normal)))

        loss = (W_recon * L_recon) + (W_pred * L_pred) + (W_grad * L_grad) + (W_sep * L_sep)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, L_recon, L_pred, L_grad, L_sep


@tf.function
def train_step_microbatch(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    batch: tf.Tensor,
    attack_cfg: AttackCfg,
    W_recon: tf.Tensor,
    W_pred: tf.Tensor,
    W_grad: tf.Tensor,
    W_sep: tf.Tensor,
    margin: tf.Tensor,
    micro_batch_size: tf.Tensor,
):
    """
    Memory-safe training step using gradient accumulation over micro-batches.
    This avoids CPU OOM for large batch sizes.
    """
    batch = tf.convert_to_tensor(batch)
    B = tf.shape(batch)[0]
    mbs = tf.cast(micro_batch_size, tf.int32)
    mbs = tf.maximum(1, mbs)

    # Initialize accumulators
    acc_grads = [tf.zeros_like(v) for v in model.trainable_variables]
    tot_loss = tf.constant(0.0, dtype=tf.float32)
    tot_recon = tf.constant(0.0, dtype=tf.float32)
    tot_pred = tf.constant(0.0, dtype=tf.float32)
    tot_grad = tf.constant(0.0, dtype=tf.float32)
    tot_sep = tf.constant(0.0, dtype=tf.float32)
    n_steps = tf.constant(0.0, dtype=tf.float32)

    i = tf.constant(0, dtype=tf.int32)
    cond = lambda i, *_: i < B

    def body(i, acc_grads, tot_loss, tot_recon, tot_pred, tot_grad, tot_sep, n_steps):
        x = batch[i : tf.minimum(B, i + mbs)]
        with tf.GradientTape() as tape:
            recon, pred = model(x, training=True)

            L_recon = tf.reduce_mean(tf.square(x - recon))
            pred_reshaped = tf.reshape(pred, [-1, 1, tf.shape(x)[2]])
            L_pred = tf.reduce_mean(tf.square(x[:, -1:, :] - pred_reshaped))
            L_grad = grad_mse(x, recon)

            score_normal = batch_scores_tf(x, recon, pred, W_recon, W_pred, W_grad)
            pseudo = build_pseudo_anomalies_v2(x, attack_cfg)
            recon_p, pred_p = model(pseudo, training=True)
            score_pseudo = batch_scores_tf(pseudo, recon_p, pred_p, W_recon, W_pred, W_grad)
            L_sep = tf.reduce_mean(tf.nn.relu(margin - (score_pseudo - score_normal)))

            loss = (W_recon * L_recon) + (W_pred * L_pred) + (W_grad * L_grad) + (W_sep * L_sep)

        grads = tape.gradient(loss, model.trainable_variables)
        acc_grads = [ag + (g if g is not None else tf.zeros_like(v)) for ag, g, v in zip(acc_grads, grads, model.trainable_variables)]

        tot_loss += loss
        tot_recon += L_recon
        tot_pred += L_pred
        tot_grad += L_grad
        tot_sep += L_sep
        n_steps += 1.0
        return i + mbs, acc_grads, tot_loss, tot_recon, tot_pred, tot_grad, tot_sep, n_steps

    _, acc_grads, tot_loss, tot_recon, tot_pred, tot_grad, tot_sep, n_steps = tf.while_loop(
        cond,
        body,
        loop_vars=[i, acc_grads, tot_loss, tot_recon, tot_pred, tot_grad, tot_sep, n_steps],
        parallel_iterations=1,
    )

    # Average grads and apply
    inv = tf.maximum(n_steps, 1.0)
    acc_grads = [g / inv for g in acc_grads]
    optimizer.apply_gradients(zip(acc_grads, model.trainable_variables))

    return tot_loss / inv, tot_recon / inv, tot_pred / inv, tot_grad / inv, tot_sep / inv


@dataclass(frozen=True)
class AttackCfg:
    p_freeze: float = 0.15
    p_spike: float = 0.15
    p_drift: float = 0.15
    p_shift: float = 0.15
    p_noise: float = 0.15
    p_drop: float = 0.15
    p_scale: float = 0.10
    min_len: int = 10
    max_len: int = 40


def _rand_segment(T: int, min_len: int, max_len: int) -> Tuple[int, int]:
    L = tf.random.uniform([], minval=min_len, maxval=max_len + 1, dtype=tf.int32)
    start = tf.random.uniform([], minval=0, maxval=T - L + 1, dtype=tf.int32)
    end = start + L
    return start, end


def freeze_attack_tf(x: tf.Tensor, start: tf.Tensor, end: tf.Tensor, offset: float = 0.8) -> tf.Tensor:
    # freeze segment to baseline (at start) + offset
    baseline = x[:, start : start + 1, :]
    seg = tf.repeat(baseline + offset, repeats=(end - start), axis=1)
    return tf.concat([x[:, :start, :], seg, x[:, end:, :]], axis=1)


def spike_attack_tf(x: tf.Tensor, start: tf.Tensor, end: tf.Tensor, amp: float = 5.0) -> tf.Tensor:
    T = tf.shape(x)[1]
    spike_t = tf.random.uniform([], minval=start, maxval=end, dtype=tf.int32)
    noise = tf.random.normal(tf.shape(x[:, spike_t : spike_t + 1, :]), mean=amp, stddev=0.2 * amp)
    mask = tf.one_hot(spike_t, depth=T, dtype=x.dtype)  # (T,)
    mask = tf.reshape(mask, [1, T, 1])
    return x + mask * noise


def drift_attack_tf(x: tf.Tensor, start: tf.Tensor, end: tf.Tensor, strength: float = 2.5) -> tf.Tensor:
    L = end - start
    ramp = tf.linspace(0.0, strength, L)
    ramp = tf.reshape(ramp, [1, L, 1])
    seg = x[:, start:end, :] + ramp
    return tf.concat([x[:, :start, :], seg, x[:, end:, :]], axis=1)


def pattern_shift_tf(x: tf.Tensor, start: tf.Tensor, end: tf.Tensor, shift: int = 15) -> tf.Tensor:
    seg = x[:, start:end, :]
    seg2 = tf.roll(seg, shift=shift, axis=1)
    return tf.concat([x[:, :start, :], seg2, x[:, end:, :]], axis=1)


def noise_attack_tf(x: tf.Tensor, start: tf.Tensor, end: tf.Tensor, sigma: float = 0.8) -> tf.Tensor:
    noise = tf.random.normal(tf.shape(x[:, start:end, :]), mean=0.0, stddev=sigma)
    seg = x[:, start:end, :] + noise
    return tf.concat([x[:, :start, :], seg, x[:, end:, :]], axis=1)


def drop_attack_tf(x: tf.Tensor, start: tf.Tensor, end: tf.Tensor, drop_prob: float = 0.5) -> tf.Tensor:
    seg = x[:, start:end, :]
    mask = tf.cast(tf.random.uniform(tf.shape(seg)) > drop_prob, seg.dtype)
    seg2 = seg * mask
    return tf.concat([x[:, :start, :], seg2, x[:, end:, :]], axis=1)


def scale_attack_tf(x: tf.Tensor, start: tf.Tensor, end: tf.Tensor, scale_low: float = 0.2, scale_high: float = 2.5) -> tf.Tensor:
    scale = tf.random.uniform([], minval=scale_low, maxval=scale_high, dtype=x.dtype)
    seg = x[:, start:end, :] * scale
    return tf.concat([x[:, :start, :], seg, x[:, end:, :]], axis=1)


def build_pseudo_anomalies_v2(x: tf.Tensor, cfg: AttackCfg) -> tf.Tensor:
    """
    Mix multiple pseudo-attack families (self-supervised).
    """
    T = int(x.shape[1])
    start, end = _rand_segment(T, cfg.min_len, cfg.max_len)

    # Attack candidates
    a_freeze = freeze_attack_tf(x, start, end, offset=0.8)
    a_spike = spike_attack_tf(x, start, end, amp=5.0)
    a_drift = drift_attack_tf(x, start, end, strength=2.5)
    a_shift = pattern_shift_tf(x, start, end, shift=15)
    a_noise = noise_attack_tf(x, start, end, sigma=0.8)
    a_drop = drop_attack_tf(x, start, end, drop_prob=0.6)
    a_scale = scale_attack_tf(x, start, end, scale_low=0.3, scale_high=2.0)

    probs = tf.constant(
        [cfg.p_freeze, cfg.p_spike, cfg.p_drift, cfg.p_shift, cfg.p_noise, cfg.p_drop, cfg.p_scale],
        dtype=tf.float32,
    )
    probs = probs / tf.reduce_sum(probs)

    # Choose ONE pseudo-attack type per batch for stability and speed
    sel = tf.random.categorical(tf.math.log([probs]), 1)[0, 0]

    # switch-like selection
    def pick(i: int, tensor: tf.Tensor) -> tf.Tensor:
        return tf.cond(tf.equal(sel, i), lambda: tensor, lambda: tf.zeros_like(tensor))

    candidates = [a_freeze, a_spike, a_drift, a_shift, a_noise, a_drop, a_scale]
    out = tf.zeros_like(x)
    for i, t in enumerate(candidates):
        out = out + pick(i, t)
    return out


def compute_thresholds_from_scores(scores: np.ndarray) -> Dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    mean = float(scores.mean())
    std = float(scores.std())
    return {
        "mean": mean,
        "std": std,
        "thresholds": {
            "3sigma": float(mean + 3.0 * std),
            "p99": float(np.quantile(scores, 0.99)),
            "p995": float(np.quantile(scores, 0.995)),
            "p997": float(np.quantile(scores, 0.997)),
        },
    }


def compute_scores_np(model: Model, X: np.ndarray, w_recon: float, w_pred: float, w_grad: float, batch_size: int = 256) -> np.ndarray:
    recon, pred = model.predict(X, verbose=0, batch_size=batch_size)
    recon = np.asarray(recon, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    if pred.ndim == 2:
        pred = pred[:, None, :]
    else:
        pred = pred[:, :1, :]

    e_recon = np.mean((X - recon) ** 2, axis=(1, 2))
    e_pred = np.mean((X[:, -1:, :] - pred) ** 2, axis=(1, 2))
    dx_true = X[:, 1:, :] - X[:, :-1, :]
    dx_recon = recon[:, 1:, :] - recon[:, :-1, :]
    e_grad = np.mean((dx_true - dx_recon) ** 2, axis=(1, 2))
    return (w_recon * e_recon + w_pred * e_pred + w_grad * e_grad).astype(np.float32)


def _shuffle_inplace(xs: List[Path], seed: int):
    rnd = random.Random(seed)
    rnd.shuffle(xs)


def save_json(path: Path, obj: Dict):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Train Hybrid model v2 (self-supervised) on normal data only.")
    ap.add_argument("--data-dir", type=str, required=True, help="Path to data/reduced")
    ap.add_argument("--split-json", type=str, required=True, help="Path to backend/config/data_split.json")
    ap.add_argument("--output-model", type=str, required=True, help="Output model path (.keras)")
    ap.add_argument("--output-thresholds", type=str, required=True, help="Output thresholds json path")
    ap.add_argument("--window-size", type=int, default=100)
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument(
        "--micro-batch-size",
        type=int,
        default=8,
        help="Internal micro-batch size for gradient accumulation to avoid CPU OOM (effective batch is still --batch-size).",
    )
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--min-learning-rate", type=float, default=1e-5)
    ap.add_argument("--early-stop-patience", type=int, default=10, help="Early stopping patience on validation score")
    ap.add_argument("--reduce-on-plateau", type=int, default=5, help="Reduce LR after N non-improving val checks")
    ap.add_argument("--reduce-factor", type=float, default=0.5, help="LR *= factor on plateau")
    ap.add_argument("--val-files-per-epoch", type=int, default=0, help="If >0, sample this many val files per epoch")
    ap.add_argument(
        "--train-files-per-epoch",
        type=int,
        default=1,
        help=(
            "Number of train chunk files to process per epoch (chunk-epoch training). "
            "Default=1 makes training feasible on CPU while still using ONLY train split. "
            "Set to 0 to process ALL train files per epoch (can be extremely slow)."
        ),
    )

    # Composite loss weights (requested defaults)
    ap.add_argument("--W-recon", type=float, default=1.0)
    ap.add_argument("--W-pred", type=float, default=2.0)
    ap.add_argument("--W-grad", type=float, default=2.0)
    ap.add_argument("--W-sep", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=0.25, help="Separation margin delta (e.g., 0.15 or 0.25)")

    # Pseudo-anomaly configuration
    ap.add_argument("--min-attack-len", type=int, default=10)
    ap.add_argument("--max-attack-len", type=int, default=40)
    ap.add_argument("--p-freeze", type=float, default=0.15)
    ap.add_argument("--p-spike", type=float, default=0.15)
    ap.add_argument("--p-drift", type=float, default=0.15)
    ap.add_argument("--p-shift", type=float, default=0.15)
    ap.add_argument("--p-noise", type=float, default=0.15)
    ap.add_argument("--p-drop", type=float, default=0.15)
    ap.add_argument("--p-scale", type=float, default=0.10)

    # Output bookkeeping
    ap.add_argument("--history-path", type=str, default="backend/app/training_v2_history.json")
    ap.add_argument("--metadata-path", type=str, default="backend/app/training_v2_metadata.json")
    ap.add_argument("--checkpoint-path", type=str, default="backend/app/checkpoint_best_model_v2.keras")
    args = ap.parse_args()

    set_seed(int(args.seed))

    root = Path(__file__).resolve().parents[2]
    data_dir = (root / args.data_dir).resolve()
    split_json = (root / args.split_json).resolve()
    out_model = (root / args.output_model).resolve()
    out_thr = (root / args.output_thresholds).resolve()
    history_path = (root / args.history_path).resolve()
    metadata_path = (root / args.metadata_path).resolve()
    checkpoint_path = (root / args.checkpoint_path).resolve()

    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_thr.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("TRAIN v2 START")
    print(f"  data_dir      : {data_dir}")
    print(f"  split_json    : {split_json}")
    print(f"  output_model  : {out_model}")
    print(f"  output_thresh : {out_thr}")
    print(f"  window_size   : {int(args.window_size)}  channels: {int(args.channels)}")
    print(f"  batch_size    : {int(args.batch_size)}  epochs: {int(args.epochs)}  lr: {float(args.learning_rate)}")
    print(f"  micro_batch   : {int(args.micro_batch_size)} (gradient accumulation)")
    print(f"  loss weights  : W_recon={float(args.W_recon)} W_pred={float(args.W_pred)} W_grad={float(args.W_grad)} W_sep={float(args.W_sep)}")
    print(f"  margin(delta) : {float(args.margin)}")
    print("=" * 78)

    split = load_split(split_json)
    train_files = [data_dir / n for n in split.get("train", [])]
    val_files = [data_dir / n for n in split.get("validation", [])]
    if not train_files or not val_files:
        raise ValueError("train/validation split missing or empty in split-json")

    print(f"Loaded split: train_files={len(train_files)}  val_files={len(val_files)}")
    print(f"Training mode: train_files_per_epoch={int(args.train_files_per_epoch)} (0 means full pass)")
    print("Validating data shape from first train file ...")

    # Validate one file shape to avoid silent mismatch
    _ = load_npy_windows(train_files[0], window_size=int(args.window_size), channels=int(args.channels))

    print("Building model ...")
    model = build_model_v2(T=int(args.window_size), C=int(args.channels))
    print("Model built:", model.name, "input_shape=", model.input_shape)

    optimizer = tf.keras.optimizers.Adam(float(args.learning_rate))
    # Important for tf.function: ensure optimizer variables are created once.
    optimizer.build(model.trainable_variables)

    attack_cfg = AttackCfg(
        p_freeze=float(args.p_freeze),
        p_spike=float(args.p_spike),
        p_drift=float(args.p_drift),
        p_shift=float(args.p_shift),
        p_noise=float(args.p_noise),
        p_drop=float(args.p_drop),
        p_scale=float(args.p_scale),
        min_len=int(args.min_attack_len),
        max_len=int(args.max_attack_len),
    )

    history: Dict[str, List[float]] = {
        "epoch_train_loss": [],
        "epoch_val_score_mean": [],
        "epoch_lr": [],
    }

    best_val = float("inf")
    no_improve = 0

    tqdm = _maybe_tqdm()

    for epoch in range(int(args.epochs)):
        # Shuffle training files each epoch deterministically (seed + epoch)
        train_files_epoch = list(train_files)
        _shuffle_inplace(train_files_epoch, seed=int(args.seed) + epoch)
        if int(args.train_files_per_epoch) > 0:
            train_files_epoch = train_files_epoch[: int(args.train_files_per_epoch)]

        # ---- Train ----
        batch_total: List[float] = []
        batch_recon: List[float] = []
        batch_pred: List[float] = []
        batch_grad: List[float] = []
        batch_sep: List[float] = []

        # Logging only (does not affect training)
        log_every_files = 1
        log_every_batches = 20
        seen_batches = 0
        it_files = train_files_epoch
        if tqdm is not None:
            it_files = tqdm(train_files_epoch, desc=f"Epoch {epoch+1}/{int(args.epochs)} (train)", unit="file")

        for file_i, fp in enumerate(it_files, 1):
            if not fp.exists():
                # corrupted split reference; warn and continue
                print(f"[WARN] missing train file: {fp}")
                continue
            try:
                t0 = time.time()
                print(f"  [epoch {epoch+1:03d}] loading train file {file_i}/{len(train_files_epoch)}: {fp.name}")
                X = load_npy_windows(fp, window_size=int(args.window_size), channels=int(args.channels))
                print(f"  [epoch {epoch+1:03d}] loaded {fp.name} shape={X.shape} in {time.time()-t0:.1f}s")
            except Exception as e:
                print(f"[WARN] skipping bad train file {fp.name}: {type(e).__name__}: {e}")
                continue

            ds = tf.data.Dataset.from_tensor_slices(X)
            ds = ds.shuffle(20000).batch(int(args.batch_size))

            for batch in ds:
                # Auto-retry on CPU OOM by shrinking micro-batch size
                mb = int(args.micro_batch_size)
                while True:
                    try:
                        loss, L_recon, L_pred, L_grad, L_sep = train_step_microbatch(
                            model=model,
                            optimizer=optimizer,
                            batch=batch,
                            attack_cfg=attack_cfg,
                            W_recon=tf.constant(float(args.W_recon), dtype=tf.float32),
                            W_pred=tf.constant(float(args.W_pred), dtype=tf.float32),
                            W_grad=tf.constant(float(args.W_grad), dtype=tf.float32),
                            W_sep=tf.constant(float(args.W_sep), dtype=tf.float32),
                            margin=tf.constant(float(args.margin), dtype=tf.float32),
                            micro_batch_size=tf.constant(int(mb), dtype=tf.int32),
                        )
                        break
                    except tf.errors.ResourceExhaustedError as e:
                        if mb <= 1:
                            raise
                        new_mb = max(1, mb // 2)
                        print(f"[WARN] OOM at micro_batch_size={mb}. Retrying with micro_batch_size={new_mb}.")
                        mb = new_mb
                seen_batches += 1
                batch_total.append(float(loss))
                batch_recon.append(float(L_recon))
                batch_pred.append(float(L_pred))
                batch_grad.append(float(L_grad))
                batch_sep.append(float(L_sep))

                if (seen_batches % log_every_batches) == 0:
                    k = min(len(batch_total), log_every_batches)
                    print(
                        f"  [epoch {epoch+1:03d}] batches={seen_batches:>6d} "
                        f"loss={float(np.mean(batch_total[-k:])):.6f} "
                        f"(recon={float(np.mean(batch_recon[-k:])):.6f} "
                        f"pred={float(np.mean(batch_pred[-k:])):.6f} "
                        f"grad={float(np.mean(batch_grad[-k:])):.6f} "
                        f"sep={float(np.mean(batch_sep[-k:])):.6f})"
                    )

            if (file_i % log_every_files) == 0 and batch_total:
                print(
                    f"  [epoch {epoch+1:03d}] processed_files={file_i}/{len(train_files_epoch)} "
                    f"avg_loss_so_far={float(np.mean(batch_total)):.6f}"
                )

        train_loss = float(np.mean(batch_total)) if batch_total else float("nan")

        # ---- Validate (NORMAL only) ----
        val_list = list(val_files)
        if int(args.val_files_per_epoch) > 0:
            k = min(int(args.val_files_per_epoch), len(val_list))
            rnd = random.Random(int(args.seed) + 1000 + epoch)
            val_list = rnd.sample(val_list, k)

        val_scores = []
        for vfp in val_list:
            if not vfp.exists():
                print(f"[WARN] missing val file: {vfp}")
                continue
            try:
                Xv = load_npy_windows(vfp, window_size=int(args.window_size), channels=int(args.channels))
                s = compute_scores_np(model, Xv, float(args.W_recon), float(args.W_pred), float(args.W_grad), batch_size=int(args.batch_size))
                val_scores.append(s)
            except Exception as e:
                print(f"[WARN] skipping bad val file {vfp.name}: {type(e).__name__}: {e}")
                continue

        if not val_scores:
            raise RuntimeError("Validation produced no scores (all files failed).")
        val_scores = np.concatenate(val_scores).astype(np.float32)
        val_mean = float(val_scores.mean())

        lr_now = float(optimizer.learning_rate.numpy())
        history["epoch_train_loss"].append(train_loss)
        history["epoch_val_score_mean"].append(val_mean)
        history["epoch_lr"].append(lr_now)

        print(
            f"[epoch {epoch+1:03d}] train_loss={train_loss:.6f} "
            f"| val_score_mean={val_mean:.6f} | lr={lr_now:.6g}"
        )

        improved = val_mean < best_val
        if improved:
            best_val = val_mean
            no_improve = 0
            # Save best checkpoint (full model) for later restore
            model.save(checkpoint_path)
            print(f"  -> checkpoint saved: {checkpoint_path}")
        else:
            no_improve += 1
            print(f"  -> no improvement ({no_improve}/{int(args.early_stop_patience)})")

            # Reduce LR on plateau
            if no_improve >= int(args.reduce_on_plateau) and no_improve % int(args.reduce_on_plateau) == 0:
                new_lr = max(lr_now * float(args.reduce_factor), float(args.min_learning_rate))
                try:
                    optimizer.learning_rate.assign(new_lr)
                except Exception:
                    # Fallback (some TF versions use different attribute name)
                    optimizer.lr.assign(new_lr)  # type: ignore[attr-defined]
                print(f"  -> reduced lr to {new_lr:.6g}")

            # Early stopping
            if no_improve >= int(args.early_stop_patience):
                print("Early stopping.")
                break

    # Restore best checkpoint
    if checkpoint_path.exists():
        model = tf.keras.models.load_model(checkpoint_path, custom_objects={"tf": tf}, compile=False)

    # Compute thresholds from ALL VALIDATION files (normal only) - strict unsupervised rule
    all_val_scores = []
    for vfp in val_files:
        try:
            Xv = load_npy_windows(vfp, window_size=int(args.window_size), channels=int(args.channels))
            s = compute_scores_np(model, Xv, float(args.W_recon), float(args.W_pred), float(args.W_grad), batch_size=int(args.batch_size))
            all_val_scores.append(s)
        except Exception as e:
            print(f"[WARN] skipping bad val file (thresholding) {vfp.name}: {type(e).__name__}: {e}")
            continue

    if not all_val_scores:
        raise RuntimeError("Could not compute validation scores for thresholds.")
    all_val_scores = np.concatenate(all_val_scores).astype(np.float32)
    thr_obj = compute_thresholds_from_scores(all_val_scores)
    thr_obj["weights"] = {"W_recon": float(args.W_recon), "W_pred": float(args.W_pred), "W_grad": float(args.W_grad)}
    thr_obj["self_supervised"] = {
        "W_sep": float(args.W_sep),
        "margin": float(args.margin),
        "pseudo_anomalies": ["freeze", "spike", "drift", "pattern_shift", "noise", "drop", "scale"],
        "attack_cfg": asdict(attack_cfg),
    }
    thr_obj["data"] = {
        "data_dir": str(data_dir),
        "split_json": str(split_json),
        "used_splits": ["train", "validation"],
        "excluded_splits": ["test"],
        "window_size": int(args.window_size),
        "channels": int(args.channels),
    }

    # Save v2 artifacts
    model.save(out_model)
    save_json(out_thr, thr_obj)
    save_json(history_path, {"history": history, "best_val_score_mean": best_val})
    save_json(
        metadata_path,
        {
            "model_name": model.name,
            "output_model": str(out_model),
            "output_thresholds": str(out_thr),
            "checkpoint_path": str(checkpoint_path),
            "seed": int(args.seed),
            "batch_size": int(args.batch_size),
            "epochs_requested": int(args.epochs),
            "epochs_ran": len(history["epoch_train_loss"]),
            "learning_rate_initial": float(args.learning_rate),
            "learning_rate_min": float(args.min_learning_rate),
            "early_stop_patience": int(args.early_stop_patience),
            "best_val_score_mean": float(best_val),
            "loss_weights": {"W_recon": float(args.W_recon), "W_pred": float(args.W_pred), "W_grad": float(args.W_grad), "W_sep": float(args.W_sep)},
            "margin": float(args.margin),
            "attack_cfg": asdict(attack_cfg),
        },
    )

    print("\nTraining v2 complete.")
    print("Saved model:", out_model)
    print("Saved thresholds:", out_thr)
    print("Saved history:", history_path)
    print("Saved metadata:", metadata_path)
    print("\nRun strict evaluation (unchanged):")
    print(
        "python backend/models/evaluate_model_strict_v2.py "
        f"--model {out_model.as_posix()} --normal-dir data/reduced --attacked-dir data/attacked_v2 "
        f"--split-json backend/config/data_split.json --output-dir backend/app/evaluation_v2_improved "
        f"--window-size {int(args.window_size)} --stride 50"
    )


if __name__ == "__main__":
    import os

    os.chdir(Path(__file__).resolve().parents[2])
    main()

