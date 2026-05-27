"""
Grid-search anomaly score weights WITHOUT retraining.

We keep the model fixed and only change how we combine the three error terms:
  score = w_recon * e_recon + w_pred * e_pred + w_grad * e_grad

Strict evaluation rules:
- Thresholds for "statistical" operating points (p99/p995/p997/3sigma) are computed from NORMAL TEST ONLY.
- Attacked_v2 labels are used ONLY for measurement.
- No training / no saving model / no weight updates.

Outputs:
  <output-dir>/
    - weight_grid_results.csv
    - best_weights.json
    - run_metadata.json
    - (optional) cached_errors.npz  (to avoid recomputing recon/pred/grad errors)
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import time
import warnings
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def _maybe_tqdm():
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm
    except Exception:
        return None


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


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
    import tensorflow as tf  # local import

    try:
        return tf.keras.models.load_model(model_path, custom_objects={"tf": tf}, compile=False)
    except Exception as e:
        warnings.warn(f"Direct load failed ({type(e).__name__}): {e}. Falling back to weights-only load.")

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
    out = model.predict(X, verbose=0, batch_size=batch_size)
    if isinstance(out, (list, tuple)):
        if len(out) >= 2:
            return out[0], out[1]
        if len(out) == 1:
            return out[0], None
    return out, None


def compute_errors(model, X: np.ndarray, batch_size: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns per-window (B,) arrays:
      e_recon, e_pred, e_grad
    Prediction term follows training code convention: compare pred vs last timestep.
    If model has no prediction head, e_pred is zeros.
    """
    recon, pred = model_predict_outputs(model, X, batch_size=batch_size)
    recon = np.asarray(recon, dtype=np.float32)

    e_recon = np.mean((X - recon) ** 2, axis=(1, 2)).astype(np.float32)

    dx_true = X[:, 1:, :] - X[:, :-1, :]
    dx_recon = recon[:, 1:, :] - recon[:, :-1, :]
    e_grad = np.mean((dx_true - dx_recon) ** 2, axis=(1, 2)).astype(np.float32)

    if pred is None:
        e_pred = np.zeros_like(e_recon, dtype=np.float32)
    else:
        pred = np.asarray(pred, dtype=np.float32)
        if pred.ndim == 2:
            pred = pred[:, None, :]
        elif pred.ndim == 3 and pred.shape[1] != 1:
            pred = pred[:, :1, :]
        if pred.ndim != 3:
            warnings.warn(f"Unexpected pred shape {pred.shape}; using e_pred=0.")
            e_pred = np.zeros_like(e_recon, dtype=np.float32)
        else:
            e_pred = np.mean((X[:, -1:, :] - pred) ** 2, axis=(1, 2)).astype(np.float32)

    return e_recon, e_pred, e_grad


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


def compute_curves_and_auc(y_true: np.ndarray, y_score: np.ndarray):
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

    fpr_full = np.concatenate([[0.0], fpr, [1.0]])
    tpr_full = np.concatenate([[0.0], tpr, [1.0]])
    roc_auc = float(np.trapz(tpr_full, fpr_full))

    rec_full = np.concatenate([[0.0], recall, [1.0]])
    prec_full = np.concatenate([[1.0], precision, [precision[-1] if len(precision) else 0.0]])
    order_pr = np.argsort(rec_full)
    pr_auc = float(np.trapz(prec_full[order_pr], rec_full[order_pr]))

    f1_curve = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_f1_idx = int(np.argmax(f1_curve)) if len(f1_curve) else 0
    best_f1 = float(f1_curve[best_f1_idx]) if len(f1_curve) else 0.0
    best_f1_thr = float(y_score_sorted[best_f1_idx]) if len(y_score_sorted) else float("nan")

    youden = tpr - fpr
    best_y_idx = int(np.argmax(youden)) if len(youden) else 0
    best_y_j = float(youden[best_y_idx]) if len(youden) else 0.0
    best_y_thr = float(y_score_sorted[best_y_idx]) if len(y_score_sorted) else float("nan")

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "best_f1": best_f1,
        "best_f1_threshold": best_f1_thr,
        "best_youden_j": best_y_j,
        "best_youden_threshold": best_y_thr,
    }


def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> Dict[str, float | int]:
    y_true = np.asarray(y_true).astype(np.uint8)
    y_pred = (np.asarray(y_score) > float(thr)).astype(np.uint8)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    bal = 0.5 * (rec + tnr)
    return {
        "threshold": float(thr),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tnr": float(tnr),
        "far": float(far),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def far_threshold_from_normal(scores_normal: np.ndarray, target_far: float) -> float:
    q = max(0.0, min(1.0, 1.0 - float(target_far)))
    return float(np.quantile(scores_normal, q))


def iter_attacked_npz(attacked_dir: Path) -> Iterable[Path]:
    yield from sorted(attacked_dir.glob("*.npz"))


def load_attacked_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        X = z["X"].astype(np.float32)
        y_window = z["y_window"].astype(np.uint8)
    if X.ndim == 2:
        X = X[..., None]
    if X.ndim != 3:
        raise ValueError(f"Bad X shape in {path}: {X.shape}")
    if y_window.ndim != 1 or y_window.shape[0] != X.shape[0]:
        raise ValueError(f"Bad y_window shape in {path}: y={y_window.shape}, X={X.shape}")
    return X, y_window


def main():
    ap = argparse.ArgumentParser(description="Grid-search anomaly score weights (no retraining).")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--normal-dir", type=str, required=True)
    ap.add_argument("--attacked-dir", type=str, required=True)
    ap.add_argument("--split-json", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--window-size", type=int, required=True)
    ap.add_argument("--stride", type=int, required=True)
    ap.add_argument("--split-key", type=str, default="test")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--cache-errors", action="store_true", help="Cache computed errors to NPZ for reuse")
    ap.add_argument("--use-cache", action="store_true", help="Use cached errors if present")
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--max-attacked-files", type=int, default=0)
    ap.add_argument("--w-recon", type=str, default="0.5,1.0,1.5,2.0")
    ap.add_argument("--w-pred", type=str, default="0.5,1.0,2.0,3.0,4.0")
    ap.add_argument("--w-grad", type=str, default="0.5,1.0,2.0,3.0,4.0")
    args = ap.parse_args()

    model_path = Path(args.model).resolve()
    normal_dir = Path(args.normal_dir).resolve()
    attacked_dir = Path(args.attacked_dir).resolve()
    split_json = Path(args.split_json).resolve()
    out_dir = Path(args.output_dir).resolve()
    safe_mkdir(out_dir)

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not normal_dir.exists():
        raise FileNotFoundError(normal_dir)
    if not attacked_dir.exists():
        raise FileNotFoundError(attacked_dir)
    if not split_json.exists():
        raise FileNotFoundError(split_json)

    cache_path = out_dir / "cached_errors.npz"

    # Load or compute errors
    if args.use_cache and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as z:
            eR_n = z["eR_n"].astype(np.float32)
            eP_n = z["eP_n"].astype(np.float32)
            eG_n = z["eG_n"].astype(np.float32)
            eR_a = z["eR_a"].astype(np.float32)
            eP_a = z["eP_a"].astype(np.float32)
            eG_a = z["eG_a"].astype(np.float32)
            y_a = z["y_a"].astype(np.uint8)
        scores_cached = True
    else:
        scores_cached = False
        T, C = infer_T_C_from_sample(normal_dir)
        model = load_keras_model_robust(model_path, T, C)

        names = load_split_filenames(split_json, args.split_key)
        if args.max_files and int(args.max_files) > 0:
            names = names[: int(args.max_files)]

        tqdm = _maybe_tqdm()
        it_names: Iterable[str] = names
        if tqdm is not None:
            it_names = tqdm(names, desc="Computing errors (normal)", unit="file")

        eR_n_parts: List[np.ndarray] = []
        eP_n_parts: List[np.ndarray] = []
        eG_n_parts: List[np.ndarray] = []

        for fname in it_names:
            fp = (normal_dir / fname).resolve()
            try:
                X = load_windows_npy(fp)
                if X.shape[1] != int(args.window_size):
                    raise ValueError(f"T mismatch {X.shape}")
                eR, eP, eG = compute_errors(model, X, batch_size=int(args.batch_size))
                eR_n_parts.append(eR)
                eP_n_parts.append(eP)
                eG_n_parts.append(eG)
            except Exception as e:
                warnings.warn(f"[NORMAL FAIL] {fname}: {type(e).__name__}: {e}")
                continue

        if not eR_n_parts:
            raise RuntimeError("No normal errors computed.")
        eR_n = np.concatenate(eR_n_parts)
        eP_n = np.concatenate(eP_n_parts)
        eG_n = np.concatenate(eG_n_parts)

        attacked_files = list(iter_attacked_npz(attacked_dir))
        if args.max_attacked_files and int(args.max_attacked_files) > 0:
            attacked_files = attacked_files[: int(args.max_attacked_files)]

        it_att: Iterable[Path] = attacked_files
        if tqdm is not None:
            it_att = tqdm(attacked_files, desc="Computing errors (attacked_v2)", unit="file")

        eR_a_parts: List[np.ndarray] = []
        eP_a_parts: List[np.ndarray] = []
        eG_a_parts: List[np.ndarray] = []
        y_parts: List[np.ndarray] = []

        for p in it_att:
            try:
                X, y_w = load_attacked_npz(p)
                if X.shape[1] != int(args.window_size):
                    raise ValueError(f"T mismatch {X.shape}")
                eR, eP, eG = compute_errors(model, X, batch_size=int(args.batch_size))
                eR_a_parts.append(eR)
                eP_a_parts.append(eP)
                eG_a_parts.append(eG)
                y_parts.append(y_w)
            except Exception as e:
                warnings.warn(f"[ATTACK FAIL] {p.name}: {type(e).__name__}: {e}")
                continue

        if not eR_a_parts:
            raise RuntimeError("No attacked errors computed.")
        eR_a = np.concatenate(eR_a_parts)
        eP_a = np.concatenate(eP_a_parts)
        eG_a = np.concatenate(eG_a_parts)
        y_a = np.concatenate(y_parts).astype(np.uint8)

        if args.cache_errors:
            np.savez_compressed(
                cache_path,
                eR_n=eR_n,
                eP_n=eP_n,
                eG_n=eG_n,
                eR_a=eR_a,
                eP_a=eP_a,
                eG_a=eG_a,
                y_a=y_a,
            )

    # Parse weights grid
    def parse_list(s: str) -> List[float]:
        return [float(x.strip()) for x in str(s).split(",") if x.strip()]

    W1 = parse_list(args.w_recon)
    W2 = parse_list(args.w_pred)
    W3 = parse_list(args.w_grad)

    y_true = np.concatenate([np.zeros(len(eR_n), dtype=np.uint8), y_a])

    out_csv = out_dir / "weight_grid_results.csv"
    best = None
    best_key = None

    # Selection policy (honest and explicit):
    # primary: maximize ROC-AUC
    # secondary: maximize F1 at BEST-F1 threshold (analysis-only)
    # tertiary: minimize FAR at that point
    def better(a, b):
        if b is None:
            return True
        if a["roc_auc"] != b["roc_auc"]:
            return a["roc_auc"] > b["roc_auc"]
        if a["best_f1"] != b["best_f1"]:
            return a["best_f1"] > b["best_f1"]
        return a["best_f1_far"] < b["best_f1_far"]

    # Prepare writer
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "w_recon",
            "w_pred",
            "w_grad",
            "roc_auc",
            "pr_auc",
            "best_f1",
            "best_f1_threshold",
            "best_f1_accuracy",
            "best_f1_balanced_accuracy",
            "best_f1_precision",
            "best_f1_recall",
            "best_f1_far",
            "p99_threshold",
            "p99_f1",
            "p99_balanced_accuracy",
            "far_le_1pct_threshold",
            "far_le_1pct_recall",
            "far_le_0.5pct_threshold",
            "far_le_0.5pct_recall",
        ])

        combos = list(itertools.product(W1, W2, W3))
        tqdm = _maybe_tqdm()
        it = combos
        if tqdm is not None:
            it = tqdm(combos, desc="Grid search weights", unit="combo")

        for w1, w2, w3 in it:
            # Compute scores
            scores_n = (w1 * eR_n + w2 * eP_n + w3 * eG_n).astype(np.float64)
            scores_a = (w1 * eR_a + w2 * eP_a + w3 * eG_a).astype(np.float64)
            y_score = np.concatenate([scores_n, scores_a])

            curves = compute_curves_and_auc(y_true, y_score)

            # Statistical thresholds from NORMAL only
            thr_stats = thresholds_from_normal_scores(scores_n)
            thr_p99 = float(thr_stats["p99"])
            thr_far1 = far_threshold_from_normal(scores_n, 0.01)
            thr_far05 = far_threshold_from_normal(scores_n, 0.005)

            m_best = metrics_at_threshold(y_true, y_score, float(curves["best_f1_threshold"]))
            m_p99 = metrics_at_threshold(y_true, y_score, thr_p99)
            m_far1 = metrics_at_threshold(y_true, y_score, thr_far1)
            m_far05 = metrics_at_threshold(y_true, y_score, thr_far05)

            row = {
                "w_recon": float(w1),
                "w_pred": float(w2),
                "w_grad": float(w3),
                "roc_auc": float(curves["roc_auc"]),
                "pr_auc": float(curves["pr_auc"]),
                "best_f1": float(curves["best_f1"]),
                "best_f1_threshold": float(curves["best_f1_threshold"]),
                "best_f1_far": float(m_best["far"]),
            }

            if better(row, best):
                best = row
                best_key = (float(w1), float(w2), float(w3))

            w.writerow([
                f"{w1:.6g}",
                f"{w2:.6g}",
                f"{w3:.6g}",
                f"{curves['roc_auc']:.10g}",
                f"{curves['pr_auc']:.10g}",
                f"{curves['best_f1']:.10g}",
                f"{curves['best_f1_threshold']:.10g}",
                f"{m_best['accuracy']:.10g}",
                f"{m_best['balanced_accuracy']:.10g}",
                f"{m_best['precision']:.10g}",
                f"{m_best['recall']:.10g}",
                f"{m_best['far']:.10g}",
                f"{thr_p99:.10g}",
                f"{m_p99['f1']:.10g}",
                f"{m_p99['balanced_accuracy']:.10g}",
                f"{thr_far1:.10g}",
                f"{m_far1['recall']:.10g}",
                f"{thr_far05:.10g}",
                f"{m_far05['recall']:.10g}",
            ])

    meta = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(model_path),
        "normal_dir": str(normal_dir),
        "attacked_dir": str(attacked_dir),
        "split_json": str(split_json),
        "split_key": str(args.split_key),
        "window_size": int(args.window_size),
        "stride": int(args.stride),
        "batch_size": int(args.batch_size),
        "grid": {"w_recon": W1, "w_pred": W2, "w_grad": W3},
        "used_cache": bool(scores_cached),
        "cache_path": str(cache_path) if cache_path.exists() else None,
        "results_csv": str(out_csv),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "best_by_policy": {"w_recon": best_key[0], "w_pred": best_key[1], "w_grad": best_key[2]} if best_key else None,
        "policy": "maximize ROC-AUC; then maximize best_f1; then minimize FAR at best_f1 threshold",
        "notes": [
            "Statistical thresholds computed from NORMAL TEST scores only (unsupervised rule).",
            "best_f1 threshold is analysis-only (uses labels for measurement).",
        ],
    }

    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    if best_key is not None:
        (out_dir / "best_weights.json").write_text(
            json.dumps(
                {
                    "w_recon": best_key[0],
                    "w_pred": best_key[1],
                    "w_grad": best_key[2],
                    "selection_policy": meta["policy"],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    print("Done.")
    print("Output:", out_dir)
    if best_key is not None:
        print(f"Best weights (by policy): w_recon={best_key[0]} w_pred={best_key[1]} w_grad={best_key[2]}")


if __name__ == "__main__":
    # Keep repo-relative paths consistent.
    os.chdir(Path(__file__).resolve().parents[2])
    main()

