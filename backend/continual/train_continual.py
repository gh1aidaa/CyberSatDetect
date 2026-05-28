"""
Fine-tune the hybrid autoencoder on approved / combined normal windows.
Used by the API (`fine_tune`) after `build_dataset` or chunk selection.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Tuple, Union

import numpy as np

from continual.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MIN_DATASET_SIZE_FOR_TRAIN,
    MODELS_DIR,
    N_FEATURES,
    WINDOW_LEN,
)


def _load_train_hybrid_module():
    backend_dir = Path(__file__).resolve().parent.parent
    path = backend_dir / "models" / "train_hybrid_model.py"
    spec = importlib.util.spec_from_file_location("train_hybrid_model", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load train_hybrid_model from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_hybrid_model"] = mod
    spec.loader.exec_module(mod)
    return mod


def _pick_base_model_path() -> Path:
    import os

    backend_dir = Path(__file__).resolve().parent.parent
    app_dir = backend_dir / "app"
    env = os.getenv("CSD_MODEL_PATH")
    if env:
        return Path(env)
for name in ("best_model_render.keras", "best_model_qc_filtered.keras", "best_model.keras", "final_model.keras"):        p = app_dir / name
        if p.exists():
            return p
    return app_dir / "best_model_qc_filtered.keras"


def _save_threshold_json(scores: np.ndarray, out_path: Path, thm) -> None:
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    data = {
        "mean": mean,
        "std": std,
        "thresholds": {
            "3sigma": mean + 3 * std,
            "p99": float(np.quantile(scores, 0.99)),
            "p995": float(np.quantile(scores, 0.995)),
            "p997": float(np.quantile(scores, 0.997)),
        },
        "weights": {
            "W_RECON": float(thm.W_RECON),
            "W_PRED": float(thm.W_PRED),
            "W_GRAD": float(thm.W_GRAD),
            "W_SEP": float(thm.W_SEP),
        },
        "self_supervised": {
            "pseudo_anomalies": ["freeze", "pattern_shift"],
            "anomaly_margin": float(thm.ANOMALY_MARGIN),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=4), encoding="utf-8")


def fine_tune(dataset_path: Union[str, Path]) -> Tuple[Path, Path, float]:
    """
    Load the production hybrid model, run a short self-supervised fine-tune
    on normal windows, write a new .keras + thresholds.json under ``MODELS_DIR``.

    Returns
    -------
    (model_path, threshold_path, accuracy_proxy)
        ``accuracy_proxy`` is an estimated true-negative rate on the held-out
        normal validation split using the train-set p99 score as threshold.
    """
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    X_anom = np.zeros((0, WINDOW_LEN, N_FEATURES), dtype=np.float32)
    if dataset_path.suffix.lower() == ".npz":
        z = np.load(str(dataset_path))
        if "X_normal" not in z:
            raise ValueError("combined_dataset.npz must include X_normal")
        X = z["X_normal"].astype(np.float32)
        if "X_anomaly" in z:
            X_anom = z["X_anomaly"].astype(np.float32)
    else:
        X = np.load(str(dataset_path)).astype(np.float32)

    if X.ndim == 2:
        X = X[..., None]
    if X.ndim != 3 or X.shape[1] != WINDOW_LEN or X.shape[2] != N_FEATURES:
        raise ValueError(f"Expected normal windows (N,{WINDOW_LEN},{N_FEATURES}), got {X.shape}")

    if X_anom.ndim == 2:
        X_anom = X_anom[..., None]
    if X_anom.ndim != 3 or X_anom.shape[1] != WINDOW_LEN or X_anom.shape[2] != N_FEATURES:
        # tolerate empty/invalid anomaly payload by disabling it
        X_anom = np.zeros((0, WINDOW_LEN, N_FEATURES), dtype=np.float32)

    n = X.shape[0]
    if n < MIN_DATASET_SIZE_FOR_TRAIN:
        raise ValueError(
            f"Need at least {MIN_DATASET_SIZE_FOR_TRAIN} windows for continual train, got {n}. "
            "Lower via CSD_CONTINUAL_MIN_WINDOWS for demo/testing."
        )

    thm = _load_train_hybrid_module()
    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    X = X[indices]
    split = max(1, int(0.9 * n))
    X_train, X_val = X[:split], X[split:]

    # shuffle anomalies (used only for separation, not for reconstruction supervision)
    if X_anom.shape[0] > 0:
        X_anom = X_anom[rng.permutation(X_anom.shape[0])]

    base_path = _pick_base_model_path()
    if not base_path.exists():
        raise FileNotFoundError(
            f"Base model not found: {base_path}. Set CSD_MODEL_PATH or add best_model.keras under app/."
        )

    tf.keras.utils.set_random_seed(42)
    try:
        import keras

        keras.config.enable_unsafe_deserialization()
    except Exception:
        pass

    model = load_model(
        str(base_path), compile=False, safe_mode=False, custom_objects={"tf": tf}
    )

    opt = tf.keras.optimizers.Adam(LEARNING_RATE)
    w_recon, w_pred, w_grad, w_sep = thm.W_RECON, thm.W_PRED, thm.W_GRAD, thm.W_SEP
    margin = float(thm.ANOMALY_MARGIN)

    @tf.function(reduce_retracing=True)
    def train_step(batch, anom_batch):
        with tf.GradientTape() as tape:
            recon, pred = model(batch, training=True)

            loss_recon = thm.mse(batch, recon)
            pred_reshaped = tf.reshape(pred, [-1, 1, 1])
            loss_pred = thm.mse(batch[:, -1:, :], pred_reshaped)
            loss_grad = thm.grad_mse(batch, recon)
            normal_scores = thm.batch_scores_tf(batch, recon, pred)

            # Separation: prefer REAL approved anomalies if present, else fallback to pseudo anomalies
            if anom_batch is not None:
                a_recon, a_pred = model(anom_batch, training=True)
                anom_scores = thm.batch_scores_tf(anom_batch, a_recon, a_pred)
                loss_sep = tf.reduce_mean(tf.nn.relu(margin - (anom_scores - normal_scores)))
            else:
                pseudo_batch = thm.build_pseudo_anomalies(batch)
                pseudo_recon, pseudo_pred = model(pseudo_batch, training=True)
                pseudo_scores = thm.batch_scores_tf(pseudo_batch, pseudo_recon, pseudo_pred)
                loss_sep = tf.reduce_mean(tf.nn.relu(margin - (pseudo_scores - normal_scores)))

            total_loss = (
                w_recon * loss_recon
                + w_pred * loss_pred
                + w_grad * loss_grad
                + w_sep * loss_sep
            )
        grads = tape.gradient(total_loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss

    shuffle_buf = min(20000, max(int(X_train.shape[0]), 1))
    ds = (
        tf.data.Dataset.from_tensor_slices(X_train)
        .shuffle(shuffle_buf)
        .batch(BATCH_SIZE)
        .repeat(EPOCHS)
    )

    anom_ds = None
    if X_anom.shape[0] > 0:
        anom_ds = tf.data.Dataset.from_tensor_slices(X_anom).shuffle(min(20000, int(X_anom.shape[0]))).batch(BATCH_SIZE).repeat()
        anom_iter = iter(anom_ds)
    else:
        anom_iter = None

    steps_per_epoch = max(1, (X_train.shape[0] + BATCH_SIZE - 1) // BATCH_SIZE)
    total_steps = steps_per_epoch * EPOCHS
    step = 0
    for batch in ds:
        anom_batch = next(anom_iter) if anom_iter is not None else None
        train_step(batch, anom_batch)
        step += 1
        if step >= total_steps:
            break

    # Threshold calibration MUST be on normal windows only
    train_scores = thm.compute_scores(model, X_train)
    thr_star = float(np.quantile(train_scores, 0.99))
    val_scores = thm.compute_scores(model, X_val)
    tnr = float(np.mean(val_scores < thr_star))
    accuracy = max(0.0, min(1.0, tnr))

    cal_scores = np.concatenate([train_scores, val_scores])
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    ver = f"continual_{stamp}_{uuid.uuid4().hex[:6]}"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_out = MODELS_DIR / f"{ver}.keras"
    thresh_out = MODELS_DIR / f"{ver}_thresholds.json"

    model.save(str(model_out))
    _save_threshold_json(cal_scores, thresh_out, thm)

    return model_out, thresh_out, accuracy
