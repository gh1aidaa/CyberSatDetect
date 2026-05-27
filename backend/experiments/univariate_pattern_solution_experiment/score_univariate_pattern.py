"""
Inference-time scoring for the univariate pattern experiment models.

score = w_recon * recon_err + w_pred * pred_err + w_grad * grad_err + w_order * order_anomaly
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def compute_window_scores(
    model: tf.keras.Model,
    X_raw: np.ndarray,
    x_aug: np.ndarray,
    *,
    recon_target: str,
    w_recon: float = 1.0,
    w_pred: float = 2.0,
    w_grad: float = 2.0,
    w_order: float = 0.25,
    batch_size: int = 256,
) -> np.ndarray:
    """
    X_raw: (N,100,1) original windows (for targets / gradient reference)
    x_aug: (N,100,8) augmented windows fed to the model
    """
    X_raw = np.asarray(X_raw, dtype=np.float32)
    x_aug = np.asarray(x_aug, dtype=np.float32)
    n = int(X_raw.shape[0])
    outs: list[np.ndarray] = []
    for s in range(0, n, int(batch_size)):
        e = min(n, s + int(batch_size))
        xa = x_aug[s:e]
        xr = X_raw[s:e]
        recon, pred10, order_p = model.predict(xa, verbose=0, batch_size=int(batch_size))
        recon = np.asarray(recon, dtype=np.float32)
        pred10 = np.asarray(pred10, dtype=np.float32)
        order_p = np.asarray(order_p, dtype=np.float32).reshape(-1)

        if recon_target == "original_only":
            y_recon = xr
        else:
            y_recon = xa

        e_recon = np.mean((y_recon - recon) ** 2, axis=(1, 2)).astype(np.float32)
        e_pred = np.mean((xr[:, -10:, :] - pred10) ** 2, axis=(1, 2)).astype(np.float32)

        sig = xr[:, :, 0]
        r0 = recon[:, :, 0] if recon.shape[-1] > 1 else recon[:, :, 0]
        dx_t = sig[:, 1:] - sig[:, :-1]
        dx_r = r0[:, 1:] - r0[:, :-1]
        e_grad = np.mean((dx_t - dx_r) ** 2, axis=1).astype(np.float32)

        order_anom = order_p.astype(np.float32)
        score = (
            float(w_recon) * e_recon
            + float(w_pred) * e_pred
            + float(w_grad) * e_grad
            + float(w_order) * order_anom
        )
        outs.append(score)
    return np.concatenate(outs, axis=0) if outs else np.zeros((0,), dtype=np.float32)


def compute_window_scores_decomposed(
    model: tf.keras.Model,
    X_raw: np.ndarray,
    x_aug: np.ndarray,
    *,
    recon_target: str,
    w_recon: float = 1.0,
    w_pred: float = 2.0,
    w_grad: float = 2.0,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Structural score = recon + pred + grad (no order term).
    order_p = sigmoid head output in [0,1] (higher ~ more pseudo-shift-like).
    """
    X_raw = np.asarray(X_raw, dtype=np.float32)
    x_aug = np.asarray(x_aug, dtype=np.float32)
    n = int(X_raw.shape[0])
    s_parts: list[np.ndarray] = []
    o_parts: list[np.ndarray] = []
    for s in range(0, n, int(batch_size)):
        e = min(n, s + int(batch_size))
        xa = x_aug[s:e]
        xr = X_raw[s:e]
        recon, pred10, order_p = model.predict(xa, verbose=0, batch_size=int(batch_size))
        recon = np.asarray(recon, dtype=np.float32)
        pred10 = np.asarray(pred10, dtype=np.float32)
        order_p = np.asarray(order_p, dtype=np.float32).reshape(-1)

        if recon_target == "original_only":
            y_recon = xr
        else:
            y_recon = xa

        e_recon = np.mean((y_recon - recon) ** 2, axis=(1, 2)).astype(np.float32)
        e_pred = np.mean((xr[:, -10:, :] - pred10) ** 2, axis=(1, 2)).astype(np.float32)

        sig = xr[:, :, 0]
        r0 = recon[:, :, 0] if recon.shape[-1] > 1 else recon[:, :, 0]
        dx_t = sig[:, 1:] - sig[:, :-1]
        dx_r = r0[:, 1:] - r0[:, :-1]
        e_grad = np.mean((dx_t - dx_r) ** 2, axis=1).astype(np.float32)

        s_struct = (
            float(w_recon) * e_recon + float(w_pred) * e_pred + float(w_grad) * e_grad
        ).astype(np.float32)
        s_parts.append(s_struct)
        o_parts.append(order_p.astype(np.float32))
    if not s_parts:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.concatenate(s_parts, axis=0), np.concatenate(o_parts, axis=0)


def quantile_thresholds(scores: np.ndarray) -> dict[str, float]:
    s = np.asarray(scores, dtype=np.float64)
    mean = float(s.mean())
    std = float(s.std())
    return {
        "p95": float(np.quantile(s, 0.95)),
        "p97": float(np.quantile(s, 0.97)),
        "p99": float(np.quantile(s, 0.99)),
        "p99.5": float(np.quantile(s, 0.995)),
        "p99.7": float(np.quantile(s, 0.997)),
        "3sigma": float(mean + 3.0 * std),
        "mean": mean,
        "std": std,
    }
