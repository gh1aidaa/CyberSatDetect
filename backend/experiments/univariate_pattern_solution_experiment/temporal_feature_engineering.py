"""
Temporal feature engineering for univariate windows (experimental only).

Input window x has shape (100, 1). Output shape (100, 8).
No labels, no future leakage beyond explicitly causal rolling features.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6


def _rolling_mean_causal(x1: np.ndarray, win: int) -> np.ndarray:
    """x1: (T,) causal rolling mean with window `win`, pad early steps with expanding mean."""
    T = int(x1.shape[0])
    out = np.zeros_like(x1, dtype=np.float64)
    for t in range(T):
        s = max(0, t - win + 1)
        out[t] = float(np.mean(x1[s : t + 1]))
    return out.astype(np.float32)


def _rolling_std_causal(x1: np.ndarray, win: int) -> np.ndarray:
    T = int(x1.shape[0])
    out = np.zeros_like(x1, dtype=np.float64)
    for t in range(T):
        s = max(0, t - win + 1)
        seg = x1[s : t + 1]
        if seg.size <= 1:
            out[t] = 0.0
        else:
            out[t] = float(np.std(seg, ddof=0))
    return out.astype(np.float32)


def _local_slope_causal_lr(x1: np.ndarray, win: int) -> np.ndarray:
    """
    Causal local linear regression slope at each t using samples x1[s..t] where
    s = max(0, t - win + 1). Regress x on relative time indices 0..L-1.
    """
    T = int(x1.shape[0])
    out = np.zeros(T, dtype=np.float64)
    for t in range(T):
        s = max(0, t - win + 1)
        seg = x1[s : t + 1].astype(np.float64)
        L = int(seg.size)
        if L < 2:
            out[t] = 0.0
            continue
        tt = np.arange(L, dtype=np.float64)
        tm = float(tt.mean())
        xm = float(seg.mean())
        cov = float(((tt - tm) * (seg - xm)).mean()) * L / max(L - 1, 1)  # match np.cov unnormalized
        var_t = float(((tt - tm) ** 2).mean()) * L / max(L - 1, 1)
        out[t] = 0.0 if var_t < EPS else cov / (var_t + EPS)
    return out.astype(np.float32)


def augment_univariate_window(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    Args:
        x: shape (100, 1), float-like.

    Returns:
        feats: shape (100, 8) columns:
            0 original x(t)
            1 first difference dx(t)
            2 second difference ddx(t)
            3 rolling mean (causal, win=5)
            4 rolling std (causal, win=5)
            5 global-window z-score using mean/std over full 100 samples
            6 local slope (causal LR, win=5)
            7 relative position t/99
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or int(x.shape[0]) != 100 or int(x.shape[1]) != 1:
        raise ValueError(f"Expected x shape (100, 1), got {x.shape}")

    sig = np.nan_to_num(x[:, 0].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    T = 100

    f0 = sig.astype(np.float32)

    dx = np.zeros(T, dtype=np.float32)
    dx[1:] = sig[1:] - sig[:-1]

    ddx = np.zeros(T, dtype=np.float32)
    ddx[1:] = dx[1:] - dx[:-1]

    rm = _rolling_mean_causal(sig, 5)
    rs = _rolling_std_causal(sig, 5)

    gmean = float(np.mean(sig))
    gstd = float(np.std(sig))
    z = (sig - gmean) / (gstd + float(eps))
    z = z.astype(np.float32)

    slope = _local_slope_causal_lr(sig, 5)

    rel = (np.arange(T, dtype=np.float32) / 99.0).astype(np.float32)

    feats = np.stack([f0, dx, ddx, rm, rs, z, slope, rel], axis=-1)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return feats


def augment_univariate_batch(X: np.ndarray, eps: float = EPS) -> np.ndarray:
    """X: (B,100,1) -> (B,100,8)"""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"Expected X ndim=3, got {X.ndim}")
    B = int(X.shape[0])
    out = np.zeros((B, 100, 8), dtype=np.float32)
    for i in range(B):
        out[i] = augment_univariate_window(X[i], eps=eps)
    return out
