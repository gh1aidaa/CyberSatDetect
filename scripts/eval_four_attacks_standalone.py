#!/usr/bin/env python3
"""
Four-attack evaluation (noise, spike, drift, freeze only).

Normal loading matches backend/models/evaluate_model_strict_v2.load_windows_npy:
  - Each test .npy is a stack of windows (B, T) or (B, T, C); 2D is promoted to (B, T, 1).
  - No sliding-window re-segmentation on normal files (same as Chapter 7 / strict v2).

Attacked NPZs: same as strict v2 pre-windowed format; optional sliding only for raw 1D/2D series.
Filter: keep files whose attack_type (metadata or filename) is in {noise, spike, drift, freeze}.

Read-only inputs; writes ONLY under --out-dir.

Score: score = e_recon + e_pred + e_grad (unweighted), same branching as evaluate_model_strict_v2.compute_scores_strict
      plus sequence pred head (B, T-1, C) when applicable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
import warnings
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ALLOWED_ATTACKS = frozenset({"noise", "spike", "drift", "freeze"})
SIGNAL_KEYS = ("X", "signal", "telemetry", "x", "data")
LABEL_KEYS = ("y_timestep", "attack_mask", "labels", "label", "mask", "y")


def _safe_out(root: Path, name: str) -> Path:
    out = (root / name).resolve()
    root_r = root.resolve()
    try:
        out.relative_to(root_r)
    except ValueError as e:
        raise ValueError(f"Refusing to write outside --out-dir: {out}") from e
    return out


def load_split_filenames(split_path: Path, split_key: str) -> List[str]:
    with split_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if split_key not in obj:
        raise KeyError(f"Split key '{split_key}' not in {split_path}. Keys: {list(obj.keys())}")
    names = [str(x) for x in obj[split_key]]
    if not names:
        raise ValueError(f"Empty list for split key '{split_key}' in {split_path}")
    return names


def load_windows_npy(path: Path) -> np.ndarray:
    """
    Identical semantics to evaluate_model_strict_v2.load_windows_npy:
    (B, T) -> (B, T, 1); (B, T, C) unchanged.
    """
    x = np.load(path, mmap_mode="r")
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[..., None]
    if x.ndim != 3:
        raise ValueError(f"Unsupported ndarray ndim={x.ndim} for {path} (shape={x.shape})")
    return x


def load_normal_test_file(path: Path, window_t: int) -> Tuple[np.ndarray, np.ndarray]:
    """All windows from one normal chunk; y = 0 (strict v2 convention)."""
    x = load_windows_npy(path)
    if int(x.shape[1]) != int(window_t):
        raise ValueError(f"T mismatch for {path}: expected {window_t}, got {x.shape}")
    b = int(x.shape[0])
    return x.astype(np.float32), np.zeros((b,), dtype=np.uint8)


def infer_attack_type_from_filename(path: Path) -> Optional[str]:
    stem = path.stem.lower()
    for at in sorted(ALLOWED_ATTACKS, key=len, reverse=True):
        if stem == at or stem.startswith(at + "_") or stem.startswith(at + "-"):
            return at
        if f"_{at}_" in stem or stem.endswith("_" + at):
            return at
    return None


def sliding_windows(
    signal: np.ndarray,
    labels: Optional[np.ndarray],
    win: int,
    stride: int,
    min_frac: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if signal.ndim == 1:
        sig = signal.astype(np.float32)[:, None]
    elif signal.ndim == 2:
        sig = signal.astype(np.float32)
    else:
        raise ValueError(f"Signal must be 1D or 2D, got {signal.shape}")

    l = int(sig.shape[0])
    c = int(sig.shape[1])
    if l < win:
        return np.zeros((0, win, c), dtype=np.float32), np.zeros((0,), dtype=np.uint8)

    if labels is None:
        lab = np.zeros((l,), dtype=np.float32)
    else:
        lab = np.asarray(labels)
        if lab.ndim == 2:
            lab = np.max(lab.astype(np.float32), axis=1)
        lab = lab.astype(np.float32).reshape(-1)
        if lab.shape[0] != l:
            raise ValueError(f"Label length {lab.shape[0]} != signal length {l}")

    xs: List[np.ndarray] = []
    ys: List[int] = []
    k = int(np.ceil(float(min_frac) * win))
    k = max(1, min(k, win))

    for s in range(0, l - win + 1, stride):
        w = sig[s : s + win].copy()
        lw = lab[s : s + win]
        attacked = float(np.mean((lw > 0.5).astype(np.float64)))
        yw = 1 if attacked >= min_frac else 0
        xs.append(w)
        ys.append(yw)

    if not xs:
        return np.zeros((0, win, c), dtype=np.float32), np.zeros((0,), dtype=np.uint8)
    return np.stack(xs, axis=0).astype(np.float32), np.asarray(ys, dtype=np.uint8)


def y_window_from_timestep(y_ts: np.ndarray, min_fraction: float) -> np.ndarray:
    if y_ts.ndim != 2:
        raise ValueError(f"y_timestep expected (B,T), got {y_ts.shape}")
    t = int(y_ts.shape[1])
    k = int(np.ceil(float(min_fraction) * t))
    k = max(1, min(k, t))
    s = np.sum(y_ts.astype(np.uint8), axis=1)
    return (s >= k).astype(np.uint8)


def load_attacked_npz(
    path: Path, win: int, stride: int, min_frac: float
) -> Tuple[np.ndarray, np.ndarray, str]:
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)

        if "attack_type" in keys:
            attack_type = str(z["attack_type"].item()).strip().lower()
            if attack_type not in ALLOWED_ATTACKS:
                raise ValueError(f"attack_type '{attack_type}' excluded (not in four-type set)")
        else:
            attack_type = infer_attack_type_from_filename(path)
            if attack_type is None:
                raise ValueError("no attack_type in NPZ and filename did not match allowed types")

        def _labels_from_keys() -> Optional[np.ndarray]:
            for lk in LABEL_KEYS:
                if lk in keys:
                    return np.asarray(z[lk])
            return None

        if "X" in keys:
            x = z["X"].astype(np.float32)
            if x.ndim == 2:
                if int(x.shape[1]) == win:
                    x = x[..., None]
                else:
                    lab = _labels_from_keys()
                    xw, yw = sliding_windows(x, lab, win, stride, min_frac)
                    return xw, yw.astype(np.uint8), attack_type
            if x.ndim == 1:
                lab = _labels_from_keys()
                xw, yw = sliding_windows(x, lab, win, stride, min_frac)
                return xw, yw.astype(np.uint8), attack_type
            if x.ndim == 3:
                if int(x.shape[1]) == win:
                    if "y_window" in keys:
                        yw = z["y_window"].astype(np.uint8)
                    elif "y_timestep" in keys:
                        yw = y_window_from_timestep(z["y_timestep"].astype(np.uint8), min_frac)
                    else:
                        raise KeyError(f"{path}: pre-windowed X requires y_window or y_timestep")
                    if yw.ndim != 1 or int(yw.shape[0]) != int(x.shape[0]):
                        raise ValueError(f"y_window mismatch in {path}")
                    return x.astype(np.float32), yw.astype(np.uint8), attack_type
                raise ValueError(f"{path}: unsupported 3D X shape {x.shape}")

        sig = None
        for sk in SIGNAL_KEYS:
            if sk in keys and sk != "X":
                sig = np.asarray(z[sk], dtype=np.float32)
                break
        if sig is None:
            raise KeyError(f"{path}: no signal array")
        lab = _labels_from_keys()
        xw, yw = sliding_windows(sig, lab, win, stride, min_frac)
        return xw, yw.astype(np.uint8), attack_type


# --- Keras load (match api.py) ---
_KERAS_PATCH_DONE = False


def _ensure_keras_dense_quant_compat() -> None:
    global _KERAS_PATCH_DONE
    if _KERAS_PATCH_DONE:
        return
    try:
        from tensorflow.keras.layers import Dense

        orig = Dense.from_config.__func__

        @classmethod
        def from_config(cls, config):  # noqa: N805
            if isinstance(config, dict) and "quantization_config" in config:
                config = dict(config)
                config.pop("quantization_config", None)
            return orig(cls, config)

        Dense.from_config = from_config
        _KERAS_PATCH_DONE = True
    except Exception:
        pass


def load_keras_model_robust(model_path: Path, t: int, c: int) -> Any:
    import tensorflow as tf
    from tensorflow.keras.models import load_model as tf_load_model

    _ensure_keras_dense_quant_compat()
    keras_mod = tf.keras
    kconf = getattr(keras_mod, "config", None)
    if kconf is not None and hasattr(kconf, "enable_unsafe_deserialization"):
        try:
            kconf.enable_unsafe_deserialization()
        except Exception:
            pass

    try:
        try:
            return tf_load_model(str(model_path), compile=False, safe_mode=False)
        except TypeError:
            return tf_load_model(str(model_path), compile=False)
    except Exception as e:
        warnings.warn(f"Direct Keras load failed ({type(e).__name__}): {e}. Trying weights fallback.")

    models_dir = Path(__file__).resolve().parent.parent / "backend" / "models"
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Expected backend/models at {models_dir}")
    if str(models_dir) not in sys.path:
        sys.path.insert(0, str(models_dir))
    from train_hybrid_model import build_model as build_hybrid_model  # type: ignore

    model = build_hybrid_model(int(t), int(c))
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(model_path, "r") as zf:
            weight_member = None
            for candidate in ("model.weights.h5", "weights.h5"):
                if candidate in zf.namelist():
                    weight_member = candidate
                    break
            if weight_member is None:
                weight_member = next((m for m in zf.namelist() if m.endswith(".h5")), None)
            if weight_member is None:
                raise RuntimeError(f"No .h5 weights in {model_path}")
            extracted = zf.extract(weight_member, tmpdir)
        model.load_weights(extracted)
    return model


def model_predict_outputs(model: Any, x: np.ndarray, batch_size: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    out = model.predict(x, verbose=0, batch_size=int(batch_size))
    if isinstance(out, (list, tuple)):
        if len(out) >= 2:
            return out[0], out[1]
        if len(out) == 1:
            return out[0], None
    return out, None


def compute_scores_unweighted(model: Any, x: np.ndarray, batch_size: int) -> np.ndarray:
    recon, pred = model_predict_outputs(model, x, batch_size=batch_size)
    recon = np.asarray(recon, dtype=np.float32)
    t = int(x.shape[1])

    e_recon = np.mean((x - recon) ** 2, axis=(1, 2))

    dx_true = x[:, 1:, :] - x[:, :-1, :]
    dx_recon = recon[:, 1:, :] - recon[:, :-1, :]
    e_grad = np.mean((dx_true - dx_recon) ** 2, axis=(1, 2))

    if pred is None:
        e_pred = np.zeros_like(e_recon, dtype=np.float32)
    else:
        pred = np.asarray(pred, dtype=np.float32)
        if pred.ndim == 3 and int(pred.shape[1]) == t - 1:
            y_true = x[:, 1:, :]
            e_pred = np.mean((y_true - pred) ** 2, axis=(1, 2))
        elif pred.ndim == 2:
            pred_exp = pred[:, None, :]
            e_pred = np.mean((x[:, -1:, :] - pred_exp) ** 2, axis=(1, 2))
        elif pred.ndim == 3:
            if int(pred.shape[1]) != 1:
                pred = pred[:, :1, :]
            e_pred = np.mean((x[:, -1:, :] - pred) ** 2, axis=(1, 2))
        else:
            e_pred = np.zeros_like(e_recon, dtype=np.float32)

    return (e_recon + e_pred + e_grad).astype(np.float32)


_CHANNEL_ALIGN_WARNED = False


def align_X_to_model_input(x: np.ndarray, model: Any) -> np.ndarray:
    global _CHANNEL_ALIGN_WARNED
    x = np.asarray(x, dtype=np.float32)
    shp = getattr(model, "input_shape", None)
    if not shp or len(shp) < 3 or shp[2] is None:
        return x
    c_need = int(shp[2])
    c_have = int(x.shape[2])
    if c_have == c_need:
        return x
    if c_need == 1 and c_have > 1:
        if not _CHANNEL_ALIGN_WARNED:
            warnings.warn(
                f"Data has C={c_have} but model expects C={c_need}; using first channel only."
            )
            _CHANNEL_ALIGN_WARNED = True
        return x[..., :1]
    raise ValueError(f"Cannot align channels: C={c_have} vs model C={c_need}")


def load_threshold_values(thresholds_path: Path, normal_scores: np.ndarray) -> Dict[str, float]:
    with thresholds_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    inner = cfg.get("thresholds", {})
    if not isinstance(inner, dict):
        inner = {}

    s = np.asarray(normal_scores, dtype=np.float64)

    def q(p: float) -> float:
        return float(np.quantile(s, p))

    out: Dict[str, float] = {}
    out["p95"] = float(inner["p95"]) if "p95" in inner else q(0.95)
    out["p97"] = float(inner["p97"]) if "p97" in inner else q(0.97)
    out["p99"] = float(inner["p99"]) if "p99" in inner else q(0.99)
    if "p99.5" in inner:
        out["p99_5"] = float(inner["p99.5"])
    elif "p995" in inner:
        out["p99_5"] = float(inner["p995"])
    else:
        out["p99_5"] = q(0.995)
    if "p99.7" in inner:
        out["p99_7"] = float(inner["p99.7"])
    elif "p997" in inner:
        out["p99_7"] = float(inner["p997"])
    else:
        out["p99_7"] = q(0.997)
    if "3sigma" in inner:
        out["3sigma"] = float(inner["3sigma"])
    else:
        out["3sigma"] = float(s.mean() + 3.0 * s.std())
    return out


def confusion_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> Tuple[int, int, int, int]:
    y_true = np.asarray(y_true).astype(np.uint8)
    y_pred = (np.asarray(y_score) > float(thr)).astype(np.uint8)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return tp, tn, fp, fn


def metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    tot = tp + tn + fp + fn
    acc = (tp + tn) / tot if tot else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    bal = 0.5 * (recall + tnr)
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "far": float(far),
        "f1": float(f1),
        "balanced_accuracy": float(bal),
    }


def sweep_best_f1(
    y_true: np.ndarray, y_score: np.ndarray, n_points: int
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(np.uint8)
    y_score = np.asarray(y_score, dtype=np.float64)
    lo = float(np.min(y_score))
    hi = float(np.max(y_score))
    if hi <= lo:
        hi = lo + 1e-12
    grid = np.linspace(lo, hi, int(n_points))
    best_f1 = -1.0
    best_thr = float("nan")
    f1_arr = np.zeros_like(grid)
    for i, thr in enumerate(grid):
        tp, tn, fp, fn = confusion_at_threshold(y_true, y_score, float(thr))
        f1v = float(metrics_from_counts(tp, tn, fp, fn)["f1"])
        f1_arr[i] = f1v
        if f1v > best_f1:
            best_f1 = f1v
            best_thr = float(thr)
    recall_arr = np.zeros_like(grid)
    far_arr = np.zeros_like(grid)
    bal_arr = np.zeros_like(grid)
    for i, thr in enumerate(grid):
        tp, tn, fp, fn = confusion_at_threshold(y_true, y_score, float(thr))
        m = metrics_from_counts(tp, tn, fp, fn)
        recall_arr[i] = m["recall"]
        far_arr[i] = m["far"]
        bal_arr[i] = m["balanced_accuracy"]
    return best_f1, best_thr, grid, f1_arr, recall_arr, far_arr, bal_arr


def per_attack_subset(
    scores_normal: np.ndarray,
    y_normal: np.ndarray,
    scores_by_type: Dict[str, np.ndarray],
    y_by_type: Dict[str, np.ndarray],
    attack: str,
) -> Tuple[np.ndarray, np.ndarray]:
    sn = scores_normal.astype(np.float64)
    yn = y_normal.astype(np.uint8)
    if attack not in scores_by_type:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.uint8)
    sa = scores_by_type[attack].astype(np.float64)
    ya = y_by_type[attack].astype(np.uint8)
    return np.concatenate([sn, sa]), np.concatenate([yn, ya])


def _setup_plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_score_distribution(out_dir: Path, scores_n: np.ndarray, scores_a: np.ndarray, plt: Any) -> None:
    """Overlapping density histograms: normal test vs four attack types."""
    scores_n = np.asarray(scores_n, dtype=np.float64)
    scores_a = np.asarray(scores_a, dtype=np.float64)
    combined = np.concatenate([scores_n, scores_a])
    hi = float(np.percentile(combined, 99.95))
    hi = max(hi, 0.05)
    bins = np.linspace(0.0, hi, 120)

    plt.figure(figsize=(10, 6))
    plt.hist(
        scores_n,
        bins=bins,
        alpha=0.55,
        density=True,
        histtype="stepfilled",
        label=f"Normal test n={len(scores_n):,}",
        color="#2ecc71",
        edgecolor="white",
        linewidth=0.3,
    )
    plt.hist(
        scores_a,
        bins=bins,
        alpha=0.55,
        density=True,
        histtype="stepfilled",
        label=f"Attacked (4 types) n={len(scores_a):,}",
        color="#e74c3c",
        edgecolor="white",
        linewidth=0.3,
    )
    plt.xlim(0.0, hi)
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.title("Score distribution (four attack types)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(_safe_out(out_dir, "score_distribution_4attacks.png"), dpi=140)
    plt.close()

    # Alias for validation / thesis folders
    alias = _safe_out(out_dir, "normal_vs_attack_score_distribution.png")
    src = _safe_out(out_dir, "score_distribution_4attacks.png")
    if alias != src and src.is_file():
        alias.write_bytes(src.read_bytes())


def _curves_from_scores(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """ROC/PR curves and AUCs without sklearn (aligned with evaluate_model_strict_v2)."""
    y_true = np.asarray(y_true).astype(np.uint8)
    y_score = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)
    p = float(y_true.sum())
    n = float(len(y_true) - p)
    tpr = tps / max(p, 1.0)
    fpr = fps / max(n, 1.0)
    precision = tps / np.maximum(tps + fps, 1.0)
    recall = tpr
    fpr_full = np.concatenate([[0.0], fpr, [1.0]])
    tpr_full = np.concatenate([[0.0], tpr, [1.0]])
    roc_auc = float(np.trapz(tpr_full, fpr_full))
    rec_full = np.concatenate([[0.0], recall, [1.0]])
    prec_full = np.concatenate([[1.0], precision, [precision[-1] if len(precision) else 0.0]])
    order_pr = np.argsort(rec_full)
    pr_auc = float(np.trapz(prec_full[order_pr], rec_full[order_pr]))
    info = {"roc_auc": roc_auc, "pr_auc": pr_auc}
    curves = {"roc_fpr": fpr, "roc_tpr": tpr, "pr_recall": recall, "pr_precision": precision}
    return info, curves


def plot_roc_pr(out_dir: Path, y_true: np.ndarray, y_score: np.ndarray, plt: Any) -> None:
    info, curves = _curves_from_scores(y_true, y_score)
    fpr, tpr = curves["roc_fpr"], curves["roc_tpr"]
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, lw=2, color="#2980b9", label=f"ROC AUC={info['roc_auc']:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC curve (four attack types)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(_safe_out(out_dir, "roc_curve_4attacks.png"), dpi=140)
    plt.close()

    rec, prec = curves["pr_recall"], curves["pr_precision"]
    plt.figure(figsize=(7, 6))
    plt.plot(rec, prec, lw=2, color="#c0392b", label=f"PR AUC={info['pr_auc']:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall curve (four attack types)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(_safe_out(out_dir, "pr_curve_4attacks.png"), dpi=140)
    plt.close()


def plot_confusion(out_dir: Path, y_true: np.ndarray, y_score: np.ndarray, thr: float, name: str, plt: Any) -> None:
    tp, tn, fp, fn = confusion_at_threshold(y_true, y_score, float(thr))
    cm = np.array([[tn, fp], [fn, tp]], dtype=np.int64)
    plt.figure(figsize=(5.5, 4.8))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion matrix ({name}, thr={thr:.6g})")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(int(v)), ha="center", va="center", color="black")
    plt.colorbar(fraction=0.046)
    plt.tight_layout()
    plt.savefig(_safe_out(out_dir, f"confusion_matrix_{name}_4attacks.png"), dpi=140)
    plt.close()


def plot_per_attack_bars(
    out_dir: Path,
    attacks: Sequence[str],
    metric_name: str,
    values: Dict[str, float],
    suffix: str,
    plt: Any,
) -> None:
    xs = np.arange(len(attacks))
    ys = [float(values.get(a, 0.0)) for a in attacks]
    plt.figure(figsize=(8, 4.5))
    plt.bar(xs, ys, color="#3498db")
    plt.xticks(xs, list(attacks), rotation=0)
    plt.ylabel(metric_name)
    plt.title(f"Per-attack {metric_name} ({suffix})")
    plt.tight_layout()
    plt.savefig(_safe_out(out_dir, f"per_attack_bars_{suffix}_4attacks.png"), dpi=140)
    plt.close()


def plot_threshold_curves(
    out_dir: Path,
    grid: np.ndarray,
    f1_arr: np.ndarray,
    recall_arr: np.ndarray,
    far_arr: np.ndarray,
    bal_arr: np.ndarray,
    plt: Any,
) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(grid, f1_arr, color="#8e44ad", lw=1.6)
    plt.xlabel("Threshold")
    plt.ylabel("F1")
    plt.title("Threshold vs F1 (linear sweep)")
    plt.tight_layout()
    plt.savefig(_safe_out(out_dir, "threshold_vs_f1_4attacks.png"), dpi=140)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(grid, recall_arr, label="Recall", color="#27ae60", lw=1.6)
    plt.plot(grid, far_arr, label="FAR", color="#c0392b", lw=1.6)
    plt.xlabel("Threshold")
    plt.ylabel("Rate")
    plt.title("Threshold vs recall / FAR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(_safe_out(out_dir, "threshold_vs_recall_far_4attacks.png"), dpi=140)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(grid, bal_arr, color="#16a085", lw=1.6)
    plt.xlabel("Threshold")
    plt.ylabel("Balanced accuracy")
    plt.title("Threshold vs balanced accuracy")
    plt.tight_layout()
    plt.savefig(_safe_out(out_dir, "threshold_vs_balanced_accuracy_4attacks.png"), dpi=140)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Four-attack evaluation (strict normal loading + NPZ filter).")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--thresholds", type=str, required=True)
    ap.add_argument("--split", type=str, required=True)
    ap.add_argument("--normal-dir", type=str, required=True)
    ap.add_argument("--attacked-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--window-size", type=int, default=100)
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--anomaly-frac", type=float, default=0.10, help="Fraction of attacked timesteps for y=1 window")
    ap.add_argument("--split-key", type=str, default="test", help="JSON key for normal test file list")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--sweep-points", type=int, default=500)
    args = ap.parse_args()

    win = int(args.window_size)
    stride = int(args.stride)
    min_frac = float(args.anomaly_frac)

    model_path = Path(args.model).resolve()
    thr_path = Path(args.thresholds).resolve()
    split_path = Path(args.split).resolve()
    normal_dir = Path(args.normal_dir).resolve()
    attacked_dir = Path(args.attacked_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    for p, lab in (
        (model_path, "--model"),
        (thr_path, "--thresholds"),
        (split_path, "--split"),
        (normal_dir, "--normal-dir"),
        (attacked_dir, "--attacked-dir"),
    ):
        if not p.exists():
            raise FileNotFoundError(f"{lab} not found: {p}")

    out_dir.mkdir(parents=True, exist_ok=True)

    names = load_split_filenames(split_path, args.split_key)

    t_sample: Optional[int] = None
    c_sample: Optional[int] = None
    for fname in names:
        fp = (normal_dir / fname).resolve()
        if not fp.is_file():
            continue
        try:
            x0 = load_windows_npy(fp)
            if int(x0.shape[1]) != win:
                continue
            t_sample = int(x0.shape[1])
            c_sample = int(x0.shape[2])
            break
        except Exception:
            continue
    if t_sample is None:
        raise RuntimeError(f"No normal test file under {normal_dir} with T={win} (check split + --window-size).")

    import tensorflow as tf  # noqa: F401

    model = load_keras_model_robust(model_path, t_sample, max(1, c_sample))
    inp = getattr(model, "input_shape", None)
    if isinstance(inp, tuple) and len(inp) >= 3 and inp[1] is not None and inp[2] is not None:
        t_model, c_model = int(inp[1]), int(inp[2])
    else:
        t_model, c_model = win, 1

    if t_model != win:
        warnings.warn(f"Model T={t_model} vs --window-size={win}; using model T for checks.")

    bs = int(args.batch_size)
    sweep_n = int(args.sweep_points)

    normal_scores: List[np.ndarray] = []
    normal_y: List[np.ndarray] = []
    n_ok = n_skip = 0
    missing_files = 0

    for fname in names:
        fp = (normal_dir / fname).resolve()
        if not fp.is_file():
            warnings.warn(f"[MISSING] Normal file not on disk: {fname}")
            missing_files += 1
            continue
        try:
            xw, yw = load_normal_test_file(fp, win)
            if int(xw.shape[1]) != t_model:
                n_skip += 1
                continue
            xw = align_X_to_model_input(xw, model)
            if int(xw.shape[2]) != c_model:
                n_skip += 1
                continue
            s = compute_scores_unweighted(model, xw, bs)
            normal_scores.append(s.astype(np.float64))
            normal_y.append(yw.astype(np.uint8))
            n_ok += 1
        except Exception as e:
            warnings.warn(f"[SKIP] Normal {fname}: {type(e).__name__}: {e}")
            n_skip += 1

    if not normal_scores:
        raise RuntimeError("No normal windows scored; check --normal-dir, --split, and --window-size.")

    scores_n = np.concatenate(normal_scores)
    y_n = np.concatenate(normal_y)

    thr_values = load_threshold_values(thr_path, scores_n)

    scores_by_type: Dict[str, List[np.ndarray]] = {k: [] for k in ALLOWED_ATTACKS}
    y_by_type: Dict[str, List[np.ndarray]] = {k: [] for k in ALLOWED_ATTACKS}
    used_files: Dict[str, List[str]] = {k: [] for k in ALLOWED_ATTACKS}
    atk_skip = 0

    for p in sorted(attacked_dir.glob("*.npz")):
        try:
            xw, yw, at = load_attacked_npz(p, win, stride, min_frac)
        except ValueError as e:
            if "excluded" in str(e).lower() or "not in allowed" in str(e).lower():
                continue
            if "did not match" in str(e).lower() or "no attack_type" in str(e).lower():
                warnings.warn(f"[SKIP] {p.name}: {e}")
                atk_skip += 1
                continue
            warnings.warn(f"[SKIP] {p.name}: {e}")
            atk_skip += 1
            continue
        except Exception as e:
            warnings.warn(f"[SKIP] Attacked {p.name}: {type(e).__name__}: {e}")
            atk_skip += 1
            continue

        if xw.shape[0] == 0:
            atk_skip += 1
            continue
        if int(xw.shape[1]) != t_model:
            atk_skip += 1
            continue
        try:
            xw = align_X_to_model_input(xw, model)
            if int(xw.shape[2]) != c_model:
                atk_skip += 1
                continue
            s = compute_scores_unweighted(model, xw, bs)
        except Exception as e:
            warnings.warn(f"[SKIP] Score {p.name}: {type(e).__name__}: {e}")
            atk_skip += 1
            continue
        scores_by_type[at].append(s.astype(np.float64))
        y_by_type[at].append(yw.astype(np.uint8))
        used_files[at].append(p.name)

    for k in sorted(ALLOWED_ATTACKS):
        if not scores_by_type[k]:
            warnings.warn(f"[WARN] No windows for attack type '{k}' after filtering.")

    scores_a_parts: List[np.ndarray] = []
    y_a_parts: List[np.ndarray] = []
    scores_by_type_arr: Dict[str, np.ndarray] = {}
    y_by_type_arr: Dict[str, np.ndarray] = {}

    for at in sorted(ALLOWED_ATTACKS):
        if scores_by_type[at]:
            sa = np.concatenate(scores_by_type[at])
            ya = np.concatenate(y_by_type[at])
            scores_by_type_arr[at] = sa
            y_by_type_arr[at] = ya
            scores_a_parts.append(sa)
            y_a_parts.append(ya)

    if not scores_a_parts:
        raise RuntimeError("No attacked windows after four-type filter.")

    scores_a = np.concatenate(scores_a_parts)
    y_a = np.concatenate(y_a_parts)

    y_all = np.concatenate([y_n, y_a]).astype(np.uint8)
    s_all = np.concatenate([scores_n, scores_a]).astype(np.float64)

    p_pos = float(np.sum(y_all))
    p_neg = float(len(y_all) - p_pos)
    imbalance = (p_neg / p_pos) if p_pos > 0 else float("inf")

    best_f1, best_thr, grid, f1_arr, recall_arr, far_arr, bal_arr = sweep_best_f1(
        y_all, s_all, sweep_n
    )

    thr_row_order = [
        ("p95", thr_values["p95"]),
        ("p97", thr_values["p97"]),
        ("p99", thr_values["p99"]),
        ("p99_5", thr_values["p99_5"]),
        ("p99_7", thr_values["p99_7"]),
        ("3sigma", thr_values["3sigma"]),
        ("best_f1", float(best_thr)),
    ]

    overall_rows: List[Dict[str, Any]] = []
    for tname, tval in thr_row_order:
        tp, tn, fp, fn = confusion_at_threshold(y_all, s_all, float(tval))
        m = metrics_from_counts(tp, tn, fp, fn)
        overall_rows.append(
            {
                "threshold_name": tname,
                "threshold": float(tval),
                **m,
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
            }
        )

    ocsv = _safe_out(out_dir, "overall_threshold_metrics_4attacks.csv")
    fields = [
        "threshold_name",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "far",
        "f1",
        "balanced_accuracy",
        "TP",
        "FP",
        "TN",
        "FN",
    ]
    with ocsv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in overall_rows:
            w.writerow({k: r[k] for k in fields})

    attacks_sorted = sorted(ALLOWED_ATTACKS)
    thr_p99 = float(thr_values["p99"])
    thr_bf1 = float(best_thr)

    per_rows: List[Dict[str, Any]] = []
    for at in attacks_sorted:
        for tlabel, tval in (("p99", thr_p99), ("best_f1", thr_bf1)):
            ss, yy = per_attack_subset(scores_n, y_n, scores_by_type_arr, y_by_type_arr, at)
            if len(ss) == 0:
                per_rows.append(
                    {
                        "attack_type": at,
                        "threshold_name": tlabel,
                        "threshold": tval,
                        "n_windows": 0,
                        "accuracy": float("nan"),
                        "precision": float("nan"),
                        "recall": float("nan"),
                        "far": float("nan"),
                        "f1": float("nan"),
                        "balanced_accuracy": float("nan"),
                        "TP": 0,
                        "TN": 0,
                        "FP": 0,
                        "FN": 0,
                    }
                )
                continue
            tp, tn, fp, fn = confusion_at_threshold(yy, ss, float(tval))
            m = metrics_from_counts(tp, tn, fp, fn)
            per_rows.append(
                {
                    "attack_type": at,
                    "threshold_name": tlabel,
                    "threshold": float(tval),
                    "n_windows": int(len(ss)),
                    **{k: m[k] for k in ("accuracy", "precision", "recall", "far", "f1", "balanced_accuracy")},
                    "TP": tp,
                    "TN": tn,
                    "FP": fp,
                    "FN": fn,
                }
            )

    pcsv = _safe_out(out_dir, "per_attack_full_metrics_4attacks.csv")
    pfields = [
        "attack_type",
        "threshold_name",
        "threshold",
        "n_windows",
        "accuracy",
        "precision",
        "recall",
        "far",
        "f1",
        "balanced_accuracy",
        "TP",
        "TN",
        "FP",
        "FN",
    ]
    with pcsv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pfields)
        w.writeheader()
        for r in per_rows:
            w.writerow({k: r[k] for k in pfields})

    thr_p995 = float(thr_values["p99_5"])
    cm_summary_rows: List[Dict[str, Any]] = []
    for label, tv in (("p99", thr_p99), ("p99_5", thr_p995), ("best_f1", thr_bf1)):
        tp, tn, fp, fn = confusion_at_threshold(y_all, s_all, float(tv))
        cm_summary_rows.append(
            {
                "threshold_name": label,
                "threshold_value": float(tv),
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
            }
        )
    cms_path = _safe_out(out_dir, "confusion_matrix_summary_4attacks.csv")
    with cms_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["threshold_name", "threshold_value", "TP", "TN", "FP", "FN"],
        )
        w.writeheader()
        for r in cm_summary_rows:
            w.writerow(r)

    curve_info, _curve_arrays = _curves_from_scores(y_all, s_all)
    roc_auc = float(curve_info["roc_auc"])
    pr_auc = float(curve_info["pr_auc"])

    n_anom = int((y_a == 1).sum())
    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "window_size": win,
        "stride": stride,
        "anomaly_timestep_fraction_for_positive_window": min_frac,
        "split_key": str(args.split_key),
        "model": str(model_path),
        "thresholds_file": str(thr_path),
        "split_file": str(split_path),
        "normal_dir": str(normal_dir),
        "attacked_dir": str(attacked_dir),
        "normal_test_files_listed": int(len(names)),
        "normal_test_files_scored_ok": int(n_ok),
        "normal_test_files_skipped": int(n_skip),
        "normal_test_files_missing": int(missing_files),
        "windows_normal": int(len(scores_n)),
        "windows_attacked_four_types": int(len(scores_a)),
        "anomalous_windows_in_attacked_subset": int(n_anom),
        "imbalance_ratio_neg_to_pos": float(imbalance),
        "attacked_npz_skipped": int(atk_skip),
        "attack_types": list(attacks_sorted),
        "windows_per_attack": {a: int(len(scores_by_type_arr[a])) for a in scores_by_type_arr},
        "files_per_attack": {k: used_files[k] for k in attacks_sorted if used_files[k]},
        "thresholds_evaluated": {k: float(v) for k, v in thr_values.items()},
        "best_f1_sweep": {
            "best_f1": float(best_f1),
            "best_threshold": float(best_thr),
            "sweep_points": sweep_n,
        },
        "curves": {"roc_auc": roc_auc, "pr_auc": pr_auc},
    }
    with _safe_out(out_dir, "evaluation_summary_4attacks.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with _safe_out(out_dir, "best_f1_from_sweep_4attacks.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_f1": float(best_f1),
                "best_threshold": float(best_thr),
                "sweep_points": sweep_n,
                "grid_min": float(grid[0]),
                "grid_max": float(grid[-1]),
            },
            f,
            indent=2,
        )

    plt = _setup_plt()
    plot_score_distribution(out_dir, scores_n, scores_a, plt)
    plot_roc_pr(out_dir, y_all, s_all, plt)
    plot_confusion(out_dir, y_all, s_all, thr_p99, "p99", plt)
    plot_confusion(out_dir, y_all, s_all, thr_p995, "p99_5", plt)
    plot_confusion(out_dir, y_all, s_all, thr_bf1, "best_f1", plt)

    def collect_metric(atk_list: Sequence[str], tval: float, key: str) -> Dict[str, float]:
        outm: Dict[str, float] = {}
        for at in atk_list:
            ss, yy = per_attack_subset(scores_n, y_n, scores_by_type_arr, y_by_type_arr, at)
            if len(ss) == 0:
                outm[at] = float("nan")
                continue
            tp, tn, fp, fn = confusion_at_threshold(yy, ss, float(tval))
            m = metrics_from_counts(tp, tn, fp, fn)
            outm[at] = float(m[key])
        return outm

    f1_p99 = collect_metric(attacks_sorted, thr_p99, "f1")
    plot_per_attack_bars(out_dir, attacks_sorted, "F1", f1_p99, "p99", plt)
    f1_bf = collect_metric(attacks_sorted, thr_bf1, "f1")
    plot_per_attack_bars(out_dir, attacks_sorted, "F1", f1_bf, "best_f1", plt)

    plot_threshold_curves(out_dir, grid, f1_arr, recall_arr, far_arr, bal_arr, plt)

    print("\n=== Four-attack evaluation (strict normal loader) ===\n")
    print(f"Split key: {args.split_key!r} | Normal files listed: {len(names)} | scored OK: {n_ok} | missing: {missing_files} | skipped: {n_skip}")
    print(f"Normal windows: {len(scores_n):,} | Attacked windows (4 types): {len(scores_a):,} | Anomalous (y=1 in attacked): {n_anom:,}")
    print(f"Imbalance (neg/pos): {imbalance:.4f}")
    print(f"ROC-AUC: {roc_auc:.6f} | PR-AUC: {pr_auc:.6f}")
    print(f"Best F1 (sweep): {best_f1:.4f} @ threshold={best_thr:.6g}\n")
    hdr = f"{'Threshold':<12} {'Thr value':>14} {'Acc':>8} {'P':>8} {'R':>8} {'FAR':>8} {'F1':>8} {'BAcc':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in overall_rows:
        print(
            f"{r['threshold_name']:<12} {r['threshold']:>14.6g} "
            f"{r['accuracy']:>8.4f} {r['precision']:>8.4f} {r['recall']:>8.4f} "
            f"{r['far']:>8.4f} {r['f1']:>8.4f} {r['balanced_accuracy']:>8.4f}"
        )
    print(f"\nOutputs written under: {out_dir}\n")


if __name__ == "__main__":
    main()
