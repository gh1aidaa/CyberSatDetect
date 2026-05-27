"""
Optimize thresholds from saved window scores (no model inference).

This script is intentionally "evaluation-only":
- Reads window-level scores and labels from window_scores.csv
- Sweeps many thresholds efficiently
- Finds thresholds for:
  - best F1
  - best Balanced Accuracy
  - FAR <= 1%
  - FAR <= 0.5%
  - Recall >= 90%
  - best Youden J

IMPORTANT:
- FAR constraints are computed using normal windows only.
- This script does NOT compute thresholds from attacked data except for analysis metrics
  (threshold candidates are derived from score distribution / unique thresholds).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Confusion:
    TP: int
    TN: int
    FP: int
    FN: int


def metrics_from_cm(cm: Confusion) -> Dict[str, float]:
    tp, tn, fp, fn = cm.TP, cm.TN, cm.FP, cm.FN
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0  # TPR
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0  # FPR
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    bal_acc = 0.5 * (rec + tnr)
    youden_j = rec - far
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "precision": float(prec),
        "recall": float(rec),
        "tpr": float(rec),
        "tnr": float(tnr),
        "far": float(far),
        "fpr": float(far),
        "fnr": float(fnr),
        "f1": float(f1),
        "youden_j": float(youden_j),
    }


def read_scores_csv(path: Path, limit_rows: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns: scores, y_true, is_normal_mask
      - scores: float64
      - y_true: uint8
      - is_normal_mask: bool (rows that are normal split)
    """
    scores: List[float] = []
    y: List[int] = []
    is_normal: List[bool] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError("Empty CSV header")
        for i, row in enumerate(r, 1):
            try:
                s = float(row["score"])
                yy = int(row["y_true"])
                split = str(row.get("split", "")).strip().lower()
                scores.append(s)
                y.append(yy)
                is_normal.append(split == "normal")
            except Exception:
                continue
            if limit_rows and i >= limit_rows:
                break

    if not scores:
        raise ValueError("No rows parsed from scores CSV")

    scores_a = np.asarray(scores, dtype=np.float64)
    y_a = np.asarray(y, dtype=np.uint8)
    normal_mask = np.asarray(is_normal, dtype=bool)
    return scores_a, y_a, normal_mask


def sweep_thresholds_fast(scores: np.ndarray, y_true: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Efficient sweep over all unique score thresholds.
    Uses sorted scores descending and cumulative TP/FP.
    """
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.uint8)

    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    y_sorted = y_true[order]

    P = int(y_true.sum())
    N = int(len(y_true) - P)

    tps = np.cumsum(y_sorted).astype(np.int64)
    fps = np.cumsum(1 - y_sorted).astype(np.int64)

    # We need metrics at distinct thresholds (when score changes).
    change = np.r_[True, s_sorted[1:] != s_sorted[:-1]]
    idx = np.where(change)[0]

    thr = s_sorted[idx]
    tp = tps[idx]
    fp = fps[idx]
    fn = P - tp
    tn = N - fp

    # Convert to float metrics
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(P, 1)
    tnr = tn / np.maximum(tn + fp, 1)
    far = fp / np.maximum(fp + tn, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    acc = (tp + tn) / max(P + N, 1)
    bal_acc = 0.5 * (recall + tnr)
    youden = recall - far
    fnr = fn / np.maximum(fn + tp, 1)

    return {
        "threshold": thr,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": precision,
        "recall": recall,
        "tnr": tnr,
        "far": far,
        "fnr": fnr,
        "f1": f1,
        "youden_j": youden,
    }


def pick_first_idx_where(arr: np.ndarray, cond: np.ndarray) -> int:
    idx = np.where(cond)[0]
    return int(idx[0]) if idx.size else -1


def main():
    ap = argparse.ArgumentParser(description="Optimize thresholds from window_scores.csv (no inference).")
    ap.add_argument("--scores", type=str, required=True, help="Path to window_scores.csv")
    ap.add_argument("--output-dir", type=str, required=True, help="Output directory")
    ap.add_argument("--limit-rows", type=int, default=0, help="If >0, read only first N rows")
    ap.add_argument("--max-thresholds", type=int, default=0, help="If >0, downsample thresholds to at most this many points")
    args = ap.parse_args()

    scores_path = Path(args.scores).resolve()
    out_dir = Path(args.output_dir).resolve()
    if not scores_path.exists():
        raise FileNotFoundError(scores_path)
    safe_mkdir(out_dir)

    scores, y_true, normal_mask = read_scores_csv(scores_path, limit_rows=int(args.limit_rows))

    if normal_mask.sum() == 0:
        warnings.warn("No 'normal' rows detected in CSV. FAR constraints will be invalid.")

    # Full sweep on all rows for "best F1/balanced/youdon"
    sweep = sweep_thresholds_fast(scores, y_true)

    # Optional downsample for output size
    n_thr = len(sweep["threshold"])
    take_idx = np.arange(n_thr)
    if args.max_thresholds and int(args.max_thresholds) > 0 and n_thr > int(args.max_thresholds):
        step = int(math.ceil(n_thr / int(args.max_thresholds)))
        take_idx = take_idx[::step]

    # For FAR constraints, compute FAR on normal-only rows at same threshold values:
    # FAR(thr) = P(score_normal > thr)
    scores_norm = scores[normal_mask]
    if scores_norm.size > 0:
        # Efficient: sort normal scores ascending once
        norm_sorted = np.sort(scores_norm)
        # FAR = fraction above thr => 1 - CDF(thr)
        # idx_left = first index where norm_sorted > thr
        # far = (n - idx_left)/n
        nN = float(len(norm_sorted))

        def far_on_normal(thr_arr: np.ndarray) -> np.ndarray:
            thr_arr = np.asarray(thr_arr, dtype=np.float64)
            idx_left = np.searchsorted(norm_sorted, thr_arr, side="right")
            return ((len(norm_sorted) - idx_left) / max(nN, 1.0)).astype(np.float64)

        far_norm = far_on_normal(sweep["threshold"])
    else:
        far_norm = np.full_like(sweep["threshold"], np.nan, dtype=np.float64)

    # Picks
    best_f1_idx = int(np.argmax(sweep["f1"]))
    best_bal_idx = int(np.argmax(sweep["balanced_accuracy"]))
    best_youd_idx = int(np.argmax(sweep["youden_j"]))

    # Constraints: choose the smallest threshold that satisfies constraint (to maximize recall usually)
    far_1_idx = pick_first_idx_where(sweep["threshold"], far_norm <= 0.01)
    far_05_idx = pick_first_idx_where(sweep["threshold"], far_norm <= 0.005)
    rec_90_idx = pick_first_idx_where(sweep["threshold"], sweep["recall"] >= 0.90)

    picks = {
        "best_f1": best_f1_idx,
        "best_balanced_accuracy": best_bal_idx,
        "best_youden_j": best_youd_idx,
        "far_le_1pct": far_1_idx,
        "far_le_0.5pct": far_05_idx,
        "recall_ge_90pct": rec_90_idx,
    }

    def row_at(i: int) -> Dict[str, float | int]:
        if i < 0:
            return {"available": 0}
        cm = Confusion(
            TP=int(sweep["TP"][i]),
            TN=int(sweep["TN"][i]),
            FP=int(sweep["FP"][i]),
            FN=int(sweep["FN"][i]),
        )
        m = metrics_from_cm(cm)
        out = {
            "available": 1,
            "threshold": float(sweep["threshold"][i]),
            "TP": cm.TP,
            "TN": cm.TN,
            "FP": cm.FP,
            "FN": cm.FN,
            **m,
            "far_on_normal": float(far_norm[i]) if np.isfinite(far_norm[i]) else None,
        }
        return out

    summary = {
        "source_scores_csv": str(scores_path),
        "n_rows": int(len(scores)),
        "n_normal_rows": int(normal_mask.sum()),
        "n_attacked_rows": int((~normal_mask).sum()),
        "label_stats": {"positives": int(y_true.sum()), "negatives": int(len(y_true) - int(y_true.sum()))},
        "picks": {name: row_at(idx) for name, idx in picks.items()},
        "notes": [
            "best_* picks are computed using all rows and y_true.",
            "FAR constraints are computed using only rows with split='normal' from the CSV.",
            "If your window_scores.csv was generated with a row limit, results reflect only that subset.",
        ],
    }

    # Write full sweep CSV (downsampled optionally)
    out_csv = out_dir / "threshold_sweep.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "threshold",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "tnr",
            "far",
            "fnr",
            "youden_j",
            "TP",
            "TN",
            "FP",
            "FN",
            "far_on_normal",
        ])
        for i in take_idx:
            w.writerow([
                f"{float(sweep['threshold'][i]):.10g}",
                f"{float(sweep['accuracy'][i]):.10g}",
                f"{float(sweep['balanced_accuracy'][i]):.10g}",
                f"{float(sweep['precision'][i]):.10g}",
                f"{float(sweep['recall'][i]):.10g}",
                f"{float(sweep['f1'][i]):.10g}",
                f"{float(sweep['tnr'][i]):.10g}",
                f"{float(sweep['far'][i]):.10g}",
                f"{float(sweep['fnr'][i]):.10g}",
                f"{float(sweep['youden_j'][i]):.10g}",
                int(sweep["TP"][i]),
                int(sweep["TN"][i]),
                int(sweep["FP"][i]),
                int(sweep["FN"][i]),
                f"{float(far_norm[i]):.10g}" if np.isfinite(far_norm[i]) else "",
            ])

    (out_dir / "threshold_optimization_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Done.")
    print("Output:", out_dir)
    for k, v in summary["picks"].items():
        if v.get("available") != 1:
            print(f"{k}: not found")
        else:
            print(
                f"{k}: thr={v['threshold']:.6g} f1={v['f1']:.4f} "
                f"bal_acc={v['balanced_accuracy']:.4f} recall={v['recall']:.4f} far={v['far']:.4f} "
                f"far_on_normal={v['far_on_normal']}"
            )


if __name__ == "__main__":
    main()

