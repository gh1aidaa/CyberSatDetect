"""
Strict unsupervised evaluation v2 for CyberSatDetect.

Key rules (per user request):
- Inference only. Never fit/train. Never save model. Never touch weights.
- Use ONLY normal test split from data/reduced to compute base statistical thresholds.
- Use ONLY attacked_v2 generated NPZs (with y_window ground-truth) for final evaluation.
- Do NOT use old data/attacked in the final evaluation.
- Score must match training formula:
    score = reconstruction_error + prediction_error + gradient_error
  (with the same shapes as in training; if model has no prediction head, prediction_error=0 with warning)

Outputs go to: backend/app/evaluation_strict_v2/
  - evaluation_summary.json
  - threshold_comparison.csv
  - window_scores.csv
  - confusion_matrix.png
  - roc_curve.png
  - precision_recall_curve.png
  - score_distribution.png
  - run_metadata.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import warnings
import zipfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _maybe_tqdm():
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm
    except Exception:
        return None


def _maybe_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception as e:
        warnings.warn(f"(matplotlib not available -> plots will be skipped) {type(e).__name__}: {e}")
        return None


def load_split_filenames(split_json: Path, split_key: str = "test") -> List[str]:
    with split_json.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if split_key not in obj:
        raise KeyError(f"Split key '{split_key}' not found. Available keys: {list(obj.keys())}")
    names = list(obj[split_key])
    if not names:
        raise ValueError(f"No filenames under split '{split_key}' in {split_json}")
    return names


def load_windows_npy(path: Path) -> np.ndarray:
    X = np.load(path).astype(np.float32)
    if X.ndim == 2:
        X = X[..., None]
    if X.ndim != 3:
        raise ValueError(f"Unsupported ndarray ndim={X.ndim} for {path} (shape={getattr(X, 'shape', None)})")
    return X


def infer_T_C_from_sample(normal_dir: Path, pattern: str = "chunk_*.npy") -> Tuple[int, int]:
    sample = next(normal_dir.glob(pattern), None)
    if sample is None:
        raise FileNotFoundError(f"No '{pattern}' files found in {normal_dir}")
    X = load_windows_npy(sample)
    return int(X.shape[1]), int(X.shape[2])


def load_keras_model_robust(model_path: Path, T: int, C: int):
    """
    Robust loader for .keras models that may fail with direct load due to Keras version mismatch.
    Strategy:
      - Try tf.keras.models.load_model(compile=False)
      - If fails, build the architecture from backend/models/train_hybrid_model.py and load weights from zip member.

    NOTE: This does NOT save anything and does NOT train.
    """
    import tensorflow as tf  # local import to keep script importable without TF

    try:
        return tf.keras.models.load_model(model_path, custom_objects={"tf": tf}, compile=False)
    except Exception as e:
        warnings.warn(f"Direct load failed ({type(e).__name__}): {e}. Falling back to weights-only load.")

    # Build architecture exactly as training code defines it.
    # We do NOT modify training code, only import its build_model.
    models_dir = Path(__file__).resolve().parent
    if str(models_dir) not in os.sys.path:
        os.sys.path.insert(0, str(models_dir))
    from train_hybrid_model import build_model as build_hybrid_model  # noqa: E402

    model = build_hybrid_model(T, C)

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(model_path, "r") as zf:
            members = zf.namelist()
            weight_member = None
            for candidate in ("model.weights.h5", "weights.h5"):
                if candidate in members:
                    weight_member = candidate
                    break
            if weight_member is None:
                weight_member = next((m for m in members if m.endswith(".h5")), None)
            if weight_member is None:
                raise RuntimeError(f"Could not find .h5 weights inside {model_path}. Members: {members[:25]} ...")
            extracted = zf.extract(weight_member, tmpdir)

        model.load_weights(extracted)

    return model


def model_predict_outputs(model, X: np.ndarray, batch_size: int = 256):
    """
    Supports:
      - recon only: predict -> ndarray
      - recon + pred: predict -> (recon, pred)
    """
    out = model.predict(X, verbose=0, batch_size=batch_size)
    if isinstance(out, (list, tuple)):
        if len(out) >= 2:
            return out[0], out[1]
        if len(out) == 1:
            return out[0], None
    return out, None


def compute_scores_strict(model, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """
    score = recon_error + pred_error + grad_error
    Using the same shapes as training:
      - recon: (B,T,C)
      - pred: (B,C) or (B,1,C) depending on model; we reshape to (B,1,C)
      - prediction error compares predicted next step against the LAST timestep (as used in train_hybrid_model.compute_scores)
    """
    recon, pred = model_predict_outputs(model, X, batch_size=batch_size)
    recon = np.asarray(recon, dtype=np.float32)

    e_recon = np.mean((X - recon) ** 2, axis=(1, 2))

    # gradient error
    dx_true = X[:, 1:, :] - X[:, :-1, :]
    dx_recon = recon[:, 1:, :] - recon[:, :-1, :]
    e_grad = np.mean((dx_true - dx_recon) ** 2, axis=(1, 2))

    # prediction error (optional)
    if pred is None:
        e_pred = np.zeros_like(e_recon, dtype=np.float32)
    else:
        pred = np.asarray(pred, dtype=np.float32)
        if pred.ndim == 2:  # (B,C)
            pred = pred[:, None, :]
        elif pred.ndim == 3:
            # ok (B,1,C) or (B,T?,C) - we only use first step
            if pred.shape[1] != 1:
                pred = pred[:, :1, :]
        else:
            warnings.warn(f"Unexpected pred ndim={pred.ndim} -> ignoring prediction head in score.")
            pred = None

        if pred is None:
            e_pred = np.zeros_like(e_recon, dtype=np.float32)
        else:
            e_pred = np.mean((X[:, -1:, :] - pred) ** 2, axis=(1, 2))

    return (e_recon + e_pred + e_grad).astype(np.float32)


def thresholds_from_normal_scores(scores: np.ndarray) -> Dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    mean = float(scores.mean())
    std = float(scores.std())
    return {
        "p99": float(np.quantile(scores, 0.99)),
        "p995": float(np.quantile(scores, 0.995)),
        "p997": float(np.quantile(scores, 0.997)),
        "3sigma": float(mean + 3.0 * std),
        "mean": mean,
        "std": std,
    }


@dataclass(frozen=True)
class Confusion:
    TP: int
    TN: int
    FP: int
    FN: int


def confusion_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> Confusion:
    y_true = np.asarray(y_true).astype(np.uint8)
    y_pred = (np.asarray(y_score) > float(thr)).astype(np.uint8)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return Confusion(TP=tp, TN=tn, FP=fp, FN=fn)


def metrics_from_confusion(cm: Confusion) -> Dict[str, float]:
    tp, tn, fp, fn = cm.TP, cm.TN, cm.FP, cm.FN
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0  # TPR
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0  # FPR
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    bal_acc = 0.5 * (recall + tnr)
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "tpr": float(recall),
        "f1": float(f1),
        "tnr": float(tnr),
        "far": float(far),
        "fpr": float(far),
        "fnr": float(fnr),
        "balanced_accuracy": float(bal_acc),
    }


def compute_curves_and_auc(y_true: np.ndarray, y_score: np.ndarray):
    """
    Computes ROC curve and PR curve and AUCs without sklearn.
    Returns:
      - info dict (roc_auc, pr_auc, best_f1_threshold, best_f1, best_youden_threshold, best_youden_j)
      - curves dict with arrays
    """
    y_true = np.asarray(y_true).astype(np.uint8)
    y_score = np.asarray(y_score).astype(np.float64)

    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)
    P = float(y_true.sum())
    N = float(len(y_true) - P)

    tpr = tps / max(P, 1.0)
    fpr = fps / max(N, 1.0)

    precision = tps / np.maximum(tps + fps, 1.0)
    recall = tpr

    # ROC AUC (trapezoid with endpoints)
    fpr_full = np.concatenate([[0.0], fpr, [1.0]])
    tpr_full = np.concatenate([[0.0], tpr, [1.0]])
    roc_auc = float(np.trapz(tpr_full, fpr_full))

    # PR AUC
    rec_full = np.concatenate([[0.0], recall, [1.0]])
    prec_full = np.concatenate([[1.0], precision, [precision[-1] if len(precision) else 0.0]])
    order_pr = np.argsort(rec_full)
    pr_auc = float(np.trapz(prec_full[order_pr], rec_full[order_pr]))

    # Best F1 threshold
    f1_curve = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_f1_idx = int(np.argmax(f1_curve)) if len(f1_curve) else 0
    best_f1 = float(f1_curve[best_f1_idx]) if len(f1_curve) else 0.0
    best_f1_thr = float(y_score_sorted[best_f1_idx]) if len(y_score_sorted) else float("nan")

    # Best Youden J = TPR - FPR
    youden = tpr - fpr
    best_y_idx = int(np.argmax(youden)) if len(youden) else 0
    best_y_j = float(youden[best_y_idx]) if len(youden) else 0.0
    best_y_thr = float(y_score_sorted[best_y_idx]) if len(y_score_sorted) else float("nan")

    info = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "best_f1": best_f1,
        "best_f1_threshold": best_f1_thr,
        "best_youden_j": best_y_j,
        "best_youden_threshold": best_y_thr,
    }
    curves = {
        "roc_fpr": fpr,
        "roc_tpr": tpr,
        "pr_precision": precision,
        "pr_recall": recall,
    }
    return info, curves


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def iter_attacked_npz(attacked_dir: Path) -> Iterable[Path]:
    yield from sorted(attacked_dir.glob("*.npz"))


def load_attacked_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, str | int]]:
    """
    Returns:
      X: (B,T,C)
      y_window: (B,)
      meta: dict with attack_type, attack_start, attack_end, source_file
    """
    with np.load(path, allow_pickle=False) as z:
        X = z["X"].astype(np.float32)
        y_window = z["y_window"].astype(np.uint8)
        attack_type = str(z["attack_type"].item()) if "attack_type" in z else "unknown"
        attack_start = int(z["attack_start"].item()) if "attack_start" in z else -1
        attack_end = int(z["attack_end"].item()) if "attack_end" in z else -1
        source_file = str(z["source_file"].item()) if "source_file" in z else path.stem + ".npy"
    if X.ndim == 2:
        X = X[..., None]
    if X.ndim != 3:
        raise ValueError(f"Bad X shape in {path}: {X.shape}")
    if y_window.ndim != 1 or y_window.shape[0] != X.shape[0]:
        raise ValueError(f"Bad y_window shape in {path}: y={y_window.shape}, X={X.shape}")
    meta = {
        "attack_type": attack_type,
        "attack_start": attack_start,
        "attack_end": attack_end,
        "source_file": source_file,
    }
    return X, y_window, meta


def write_window_scores_csv(
    out_csv: Path,
    normal_rows: Iterable[Tuple[float, int, str]],
    attacked_rows: Iterable[Tuple[float, int, str, str]],
    limit_rows: int = 0,
):
    """
    Writes per-window scores for auditing.
      - normal_rows: (score, y, source_file)
      - attacked_rows: (score, y, source_file, attack_type)
    """
    n_written = 0
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "score", "y_true", "source_file", "attack_type"])

        for score, y, src in normal_rows:
            w.writerow(["normal", f"{float(score):.8g}", int(y), src, "normal"])
            n_written += 1
            if limit_rows and n_written >= limit_rows:
                return n_written

        for score, y, src, atype in attacked_rows:
            w.writerow(["attacked", f"{float(score):.8g}", int(y), src, atype])
            n_written += 1
            if limit_rows and n_written >= limit_rows:
                return n_written

    return n_written


def plot_all(
    out_dir: Path,
    scores_normal: np.ndarray,
    scores_attacked: np.ndarray,
    thresholds: Dict[str, float],
    curves: Dict[str, np.ndarray],
    cm_best: Optional[Confusion],
    best_name: str,
):
    plt = _maybe_matplotlib()
    if plt is None:
        return

    # score distribution
    plt.figure(figsize=(10, 6))
    plt.hist(scores_normal, bins=140, alpha=0.6, density=True, label=f"Normal (test) n={len(scores_normal)}", color="#2ecc71")
    plt.hist(scores_attacked, bins=140, alpha=0.6, density=True, label=f"Attacked_v2 n={len(scores_attacked)}", color="#e74c3c")
    for k in ("p99", "p995", "p997", "3sigma"):
        if k in thresholds:
            plt.axvline(thresholds[k], linestyle="--", linewidth=1.4, label=f"{k}={thresholds[k]:.4g}")
    plt.xlabel("Anomaly score")
    plt.ylabel("Density")
    plt.title("Score distribution (normal test vs attacked_v2)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "score_distribution.png", dpi=140)
    plt.close()

    # ROC
    fpr = curves["roc_fpr"]
    tpr = curves["roc_tpr"]
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, linewidth=2, color="#2980b9")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC curve")
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=140)
    plt.close()

    # PR
    prec = curves["pr_precision"]
    rec = curves["pr_recall"]
    plt.figure(figsize=(7, 6))
    plt.plot(rec, prec, linewidth=2, color="#c0392b")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.tight_layout()
    plt.savefig(out_dir / "precision_recall_curve.png", dpi=140)
    plt.close()

    # Confusion matrix (best)
    if cm_best is not None:
        # simple 2x2 heatmap without seaborn
        mat = np.array([[cm_best.TN, cm_best.FP], [cm_best.FN, cm_best.TP]], dtype=np.int64)
        plt.figure(figsize=(6, 5))
        plt.imshow(mat, cmap="Blues")
        plt.title(f"Confusion matrix ({best_name})")
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        for (i, j), val in np.ndenumerate(mat):
            plt.text(j, i, str(int(val)), ha="center", va="center", color="black")
        plt.tight_layout()
        plt.savefig(out_dir / "confusion_matrix.png", dpi=140)
        plt.close()


def main():
    ap = argparse.ArgumentParser(description="Strict unsupervised evaluation v2 (normal test vs attacked_v2).")
    ap.add_argument("--model", type=str, required=True, help="Path to .keras model (best_model.keras recommended)")
    ap.add_argument("--normal-dir", type=str, required=True, help="Path to data/reduced")
    ap.add_argument("--attacked-dir", type=str, required=True, help="Path to data/attacked_v2 (NPZ files)")
    ap.add_argument("--split-json", type=str, required=True, help="Path to backend/config/data_split.json")
    ap.add_argument("--output-dir", type=str, required=True, help="Output directory backend/app/evaluation_strict_v2")
    ap.add_argument("--window-size", type=int, required=True, help="Window size T (100)")
    ap.add_argument("--stride", type=int, required=True, help="Stride (50) - metadata/consistency only")
    ap.add_argument("--split-key", type=str, default="test", help="Split key inside data_split.json (default: test)")
    ap.add_argument("--batch-size", type=int, default=256, help="Inference batch size")
    ap.add_argument("--max-files", type=int, default=0, help="If >0, limit number of normal test files")
    ap.add_argument("--max-attacked-files", type=int, default=0, help="If >0, limit number of attacked npz files")
    ap.add_argument("--window-scores-csv-limit", type=int, default=500000, help="Max rows to write in window_scores.csv (0 = no limit)")
    args = ap.parse_args()

    model_path = Path(args.model).resolve()
    normal_dir = Path(args.normal_dir).resolve()
    attacked_dir = Path(args.attacked_dir).resolve()
    split_json = Path(args.split_json).resolve()
    out_dir = Path(args.output_dir).resolve()

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not normal_dir.exists():
        raise FileNotFoundError(normal_dir)
    if not attacked_dir.exists():
        raise FileNotFoundError(attacked_dir)
    if not split_json.exists():
        raise FileNotFoundError(split_json)

    safe_mkdir(out_dir)

    run_meta = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(model_path),
        "normal_dir": str(normal_dir),
        "attacked_dir": str(attacked_dir),
        "split_json": str(split_json),
        "split_key": str(args.split_key),
        "window_size": int(args.window_size),
        "stride": int(args.stride),
        "batch_size": int(args.batch_size),
        "limits": {
            "max_files": int(args.max_files),
            "max_attacked_files": int(args.max_attacked_files),
            "window_scores_csv_limit": int(args.window_scores_csv_limit),
        },
        "notes": [
            "Statistical thresholds computed from NORMAL TEST scores only (unsupervised rule).",
            "Attacked_v2 labels are used ONLY for evaluation metrics (supervised-for-measurement).",
        ],
    }

    # Load model
    T_data, C_data = infer_T_C_from_sample(normal_dir)
    if int(args.window_size) != int(T_data):
        warnings.warn(f"window-size arg={args.window_size} but data sample T={T_data}. Proceeding with arg for validation.")

    model = load_keras_model_robust(model_path, T_data, C_data)

    # Normal test scores
    names = load_split_filenames(split_json, args.split_key)
    if args.max_files and int(args.max_files) > 0:
        names = names[: int(args.max_files)]

    tqdm = _maybe_tqdm()
    normal_iter: Iterable[str] = names
    if tqdm is not None:
        normal_iter = tqdm(names, desc="Scoring normal test", unit="file")

    normal_scores_parts: List[np.ndarray] = []
    normal_rows_for_csv: List[Tuple[float, int, str]] = []
    normal_windows = 0
    normal_failed = 0

    for fname in normal_iter:
        fp = (normal_dir / fname).resolve()
        try:
            X = load_windows_npy(fp)
            if X.shape[1] != int(args.window_size):
                raise ValueError(f"T mismatch for {fp}: {X.shape}")
            s = compute_scores_strict(model, X, batch_size=int(args.batch_size))
            normal_scores_parts.append(s)
            normal_windows += int(len(s))
            # y_true=0 for normal
            if len(normal_rows_for_csv) < max(0, int(args.window_scores_csv_limit)):
                take = min(len(s), max(0, int(args.window_scores_csv_limit)) - len(normal_rows_for_csv))
                normal_rows_for_csv.extend((float(sc), 0, fname) for sc in s[:take])
        except Exception as e:
            normal_failed += 1
            warnings.warn(f"[NORMAL FAIL] {fname}: {type(e).__name__}: {e}")
            continue

    if not normal_scores_parts:
        raise RuntimeError("No normal scores produced; cannot evaluate.")
    scores_normal = np.concatenate(normal_scores_parts).astype(np.float64)

    # Compute statistical thresholds from normal scores ONLY
    thr_stats = thresholds_from_normal_scores(scores_normal)
    thresholds = {k: thr_stats[k] for k in ("p99", "p995", "p997", "3sigma")}

    # Attacked_v2 scores + labels
    attacked_files = list(iter_attacked_npz(attacked_dir))
    if args.max_attacked_files and int(args.max_attacked_files) > 0:
        attacked_files = attacked_files[: int(args.max_attacked_files)]

    attack_type_counts: Dict[str, int] = {}
    attacked_iter: Iterable[Path] = attacked_files
    if tqdm is not None:
        attacked_iter = tqdm(attacked_files, desc="Scoring attacked_v2", unit="file")

    attacked_scores_parts: List[np.ndarray] = []
    attacked_labels_parts: List[np.ndarray] = []
    attacked_rows_for_csv: List[Tuple[float, int, str, str]] = []
    attacked_windows = 0
    attacked_failed = 0

    for p in attacked_iter:
        try:
            X_att, y_w, meta = load_attacked_npz(p)
            if X_att.shape[1] != int(args.window_size):
                raise ValueError(f"T mismatch in {p}: X={X_att.shape}")
            s = compute_scores_strict(model, X_att, batch_size=int(args.batch_size))
            attacked_scores_parts.append(s)
            attacked_labels_parts.append(y_w.astype(np.uint8))
            attacked_windows += int(len(s))

            atype = str(meta.get("attack_type", "unknown"))
            attack_type_counts[atype] = attack_type_counts.get(atype, 0) + 1

            if len(attacked_rows_for_csv) < max(0, int(args.window_scores_csv_limit)):
                remaining = max(0, int(args.window_scores_csv_limit)) - len(attacked_rows_for_csv)
                take = min(len(s), remaining)
                src = str(meta.get("source_file", p.stem + ".npy"))
                attacked_rows_for_csv.extend((float(sc), int(y), src, atype) for sc, y in zip(s[:take], y_w[:take]))

        except Exception as e:
            attacked_failed += 1
            warnings.warn(f"[ATTACK FAIL] {p.name}: {type(e).__name__}: {e}")
            continue

    if not attacked_scores_parts:
        raise RuntimeError("No attacked_v2 scores produced; did you generate data/attacked_v2?")

    scores_attacked = np.concatenate(attacked_scores_parts).astype(np.float64)
    y_attacked = np.concatenate(attacked_labels_parts).astype(np.uint8)

    # Build unified arrays for curve/AUC calculations
    y_true = np.concatenate([np.zeros(len(scores_normal), dtype=np.uint8), y_attacked])
    y_score = np.concatenate([scores_normal, scores_attacked])

    curves_info, curves = compute_curves_and_auc(y_true, y_score)

    # Threshold candidates
    # Base statistical thresholds (from normal only)
    candidates: List[Tuple[str, float]] = [(k, float(v)) for k, v in thresholds.items()]

    # Analysis-only thresholds
    candidates.append(("best_f1", float(curves_info["best_f1_threshold"])))
    candidates.append(("best_youden_j", float(curves_info["best_youden_threshold"])))

    # FAR constraints: computed against normal scores only
    def thr_for_far(target_far: float) -> float:
        # FAR = P(score > thr | normal). Need smallest thr such that FAR <= target.
        # Equivalent to quantile at (1 - target_far).
        q = max(0.0, min(1.0, 1.0 - float(target_far)))
        return float(np.quantile(scores_normal, q))

    candidates.append(("far_le_1pct", thr_for_far(0.01)))
    candidates.append(("far_le_0.5pct", thr_for_far(0.005)))

    # Compute metrics for each candidate
    rows = []
    best_by_f1 = None
    for name, thr in candidates:
        cm = confusion_at_threshold(y_true, y_score, thr)
        m = metrics_from_confusion(cm)
        row = {
            "name": name,
            "threshold": float(thr),
            **m,
            "TP": cm.TP,
            "TN": cm.TN,
            "FP": cm.FP,
            "FN": cm.FN,
        }
        rows.append(row)
        if best_by_f1 is None or row["f1"] > best_by_f1["f1"]:
            best_by_f1 = row

    # Choose "best operating point" for report: best F1 among candidates (analysis)
    best_op = best_by_f1 or rows[0]
    best_cm = Confusion(TP=int(best_op["TP"]), TN=int(best_op["TN"]), FP=int(best_op["FP"]), FN=int(best_op["FN"]))

    # Save threshold comparison CSV
    comp_csv = out_dir / "threshold_comparison.csv"
    with comp_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "name",
            "threshold",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "tnr",
            "far",
            "fnr",
            "TP",
            "TN",
            "FP",
            "FN",
        ])
        for r in rows:
            w.writerow([
                r["name"],
                f"{r['threshold']:.10g}",
                f"{r['accuracy']:.10g}",
                f"{r['balanced_accuracy']:.10g}",
                f"{r['precision']:.10g}",
                f"{r['recall']:.10g}",
                f"{r['f1']:.10g}",
                f"{r['tnr']:.10g}",
                f"{r['far']:.10g}",
                f"{r['fnr']:.10g}",
                r["TP"],
                r["TN"],
                r["FP"],
                r["FN"],
            ])

    # Save window_scores.csv (limited for practicality)
    scores_csv = out_dir / "window_scores.csv"
    lim = int(args.window_scores_csv_limit)
    written = write_window_scores_csv(
        scores_csv,
        normal_rows=normal_rows_for_csv,
        attacked_rows=attacked_rows_for_csv,
        limit_rows=lim if lim > 0 else 0,
    )

    # Prepare summary JSON
    summary = {
        "normal": {
            "files_used": len(names),
            "files_failed": int(normal_failed),
            "windows": int(normal_windows),
            "score_stats": {
                "mean": float(scores_normal.mean()),
                "std": float(scores_normal.std()),
                "min": float(scores_normal.min()),
                "max": float(scores_normal.max()),
            },
        },
        "attacked_v2": {
            "files_used": len(attacked_files),
            "files_failed": int(attacked_failed),
            "windows": int(attacked_windows),
            "windows_anomaly": int(y_attacked.sum()),
            "windows_normal_inside_attacked": int((y_attacked == 0).sum()),
            "attack_type_distribution": attack_type_counts,
            "score_stats": {
                "mean": float(scores_attacked.mean()),
                "std": float(scores_attacked.std()),
                "min": float(scores_attacked.min()),
                "max": float(scores_attacked.max()),
            },
        },
        "thresholds": {
            "computed_from": "normal_test_only",
            "stats": {"mean": thr_stats["mean"], "std": thr_stats["std"]},
            "statistical": thresholds,
            "analysis": {
                "best_f1_threshold": float(curves_info["best_f1_threshold"]),
                "best_youden_threshold": float(curves_info["best_youden_threshold"]),
                "far_le_1pct": float(thr_for_far(0.01)),
                "far_le_0.5pct": float(thr_for_far(0.005)),
            },
        },
        "curves": {
            "roc_auc": float(curves_info["roc_auc"]),
            "pr_auc": float(curves_info["pr_auc"]),
        },
        "best_operating_point_by_f1_among_candidates": best_op,
        "artifacts": {
            "threshold_comparison_csv": str(comp_csv),
            "window_scores_csv": str(scores_csv),
            "window_scores_rows_written": int(written),
        },
    }

    (out_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    run_meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out_dir / "run_metadata.json").write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    plot_all(
        out_dir=out_dir,
        scores_normal=scores_normal,
        scores_attacked=scores_attacked,
        thresholds=thresholds,
        curves=curves,
        cm_best=best_cm,
        best_name=str(best_op.get("name", "best")),
    )

    print("Done.")
    print(f"Output: {out_dir}")
    print(f"ROC-AUC: {curves_info['roc_auc']:.6f} | PR-AUC: {curves_info['pr_auc']:.6f}")
    print(f"Best candidate by F1: {best_op.get('name')} thr={best_op.get('threshold'):.6g} f1={best_op.get('f1'):.4f}")


if __name__ == "__main__":
    # Make running from anywhere consistent with repo-relative paths (matches other scripts).
    os.chdir(Path(__file__).resolve().parents[2])
    main()

