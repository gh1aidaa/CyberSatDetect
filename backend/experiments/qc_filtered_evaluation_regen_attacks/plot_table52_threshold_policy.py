"""
Table 5.2 threshold policy figure (QC-filtered regen attacks).

Uses calibrated threshold VALUES from thesis Table 5.2 and performance metrics from
the same QC regen evaluation as metrics_qc_filtered.csv (re-scores test split if needed
for best_f1 at thr=0.0509).

Output:
  results/table52_threshold_policy_metrics.csv
  results/table52_threshold_policy_bar_compare.png
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
ATTACKED_OUT = EXP_DIR / "attacked_qc_v2"

MODEL_PATH = BACKEND / "app" / "best_model_qc_filtered.keras"
SPLIT_JSON = BACKEND / "config" / "data_split_qc_filtered.json"
NORMAL_DIR = ROOT / "data" / "reduced"
METRICS_CSV = RESULTS_DIR / "metrics_qc_filtered.csv"

# Table 5.2 (thesis) — threshold values only; metrics filled from evaluation.
TABLE52_THRESHOLDS: List[Tuple[str, float, str]] = [
    ("p99", 0.202873, "p99\n(calibrated)"),
    ("p995", 0.299654, "p99.5\n(calibrated)"),
    ("p997", 0.569635, "p99.7\n(calibrated)"),
    ("3sigma", 0.213348, "3-sigma\n(calibrated)"),
    ("best_f1", 0.0509, "Best-F1\n(analysis)"),
]

# Map thesis / table names -> metrics_qc_filtered.csv row keys
CSV_KEY = {
    "p99": "p99",
    "p995": "p995",
    "p997": "p997",
    "3sigma": "3sigma",
}


def _load_metrics_csv() -> Dict[str, Dict[str, float]]:
    rows: Dict[str, Dict[str, float]] = {}
    with METRICS_CSV.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows[str(r["name"]).strip()] = {k: float(r[k]) for k in r if k != "name"}
    return rows


def _score_evaluation() -> Tuple[np.ndarray, np.ndarray, float, float]:
    models_dir = BACKEND / "models"
    if str(models_dir) not in sys.path:
        sys.path.insert(0, str(models_dir))

    from evaluate_model_strict_v2 import (  # type: ignore
        compute_curves_and_auc,
        compute_scores_strict,
        confusion_at_threshold,
        load_attacked_npz,
        load_keras_model_robust,
        load_split_filenames,
        load_windows_npy,
        metrics_from_confusion,
    )

    X_sample = load_windows_npy(next(NORMAL_DIR.glob("chunk_*.npy")))
    T, C = int(X_sample.shape[1]), int(X_sample.shape[2])
    model = load_keras_model_robust(MODEL_PATH, T, C)

    test_names = load_split_filenames(SPLIT_JSON, "test")
    normal_parts: List[np.ndarray] = []
    for fname in test_names:
        fp = NORMAL_DIR / fname
        X = load_windows_npy(fp)
        normal_parts.append(compute_scores_strict(model, X, batch_size=256).astype(np.float64))

    scores_normal = np.concatenate(normal_parts)
    attacked_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []
    for p in sorted(ATTACKED_OUT.glob("*.npz")):
        X_att, y_w, _ = load_attacked_npz(p)
        s = compute_scores_strict(model, X_att, batch_size=256)
        attacked_parts.append(s.astype(np.float64))
        label_parts.append(y_w.astype(np.uint8))

    scores_attacked = np.concatenate(attacked_parts)
    y_attacked = np.concatenate(label_parts)
    y_true = np.concatenate([np.zeros(len(scores_normal), dtype=np.uint8), y_attacked])
    y_score = np.concatenate([scores_normal, scores_attacked])
    curves_info, _ = compute_curves_and_auc(y_true, y_score)
    return y_true, y_score, float(curves_info["roc_auc"]), float(curves_info["pr_auc"])


def _build_policy_rows(y_true: np.ndarray, y_score: np.ndarray) -> List[Dict[str, Any]]:
    from evaluate_model_strict_v2 import confusion_at_threshold, metrics_from_confusion  # type: ignore

    cached = _load_metrics_csv()
    out: List[Dict[str, Any]] = []
    for name, thr_table, label in TABLE52_THRESHOLDS:
        if name == "best_f1":
            cm = confusion_at_threshold(y_true, y_score, float(thr_table))
            m = metrics_from_confusion(cm)
            row = {
                "name": name,
                "threshold": float(thr_table),
                "recall": m["recall"],
                "far": m["far"],
                "f1": m["f1"],
                "label": label,
            }
        else:
            key = CSV_KEY[name]
            if key not in cached:
                raise KeyError(f"Missing {key} in {METRICS_CSV}")
            c = cached[key]
            row = {
                "name": name,
                "threshold": float(thr_table),
                "recall": float(c["recall"]),
                "far": float(c["far"]),
                "f1": float(c["f1"]),
                "label": label,
            }
        out.append(row)
    return out


def _write_metrics_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "threshold", "recall", "far", "f1", "label"])
        for r in rows:
            w.writerow([r["name"], r["threshold"], r["recall"], r["far"], r["f1"], r["label"]])


def _plot(rows: List[Dict[str, Any]], roc_auc: float, pr_auc: float, out_png: Path) -> None:
    models_dir = BACKEND / "models"
    if str(models_dir) not in sys.path:
        sys.path.insert(0, str(models_dir))
    import generate_ml_engineering_figures as gm  # type: ignore

    plt = gm._ensure_matplotlib()
    gm._style(plt)

    labels = [r["label"] for r in rows]
    thr = [r["threshold"] for r in rows]
    recall = [r["recall"] * 100.0 for r in rows]
    far = [r["far"] * 100.0 for r in rows]
    f1 = [r["f1"] for r in rows]

    x = np.arange(len(rows), dtype=float)
    fig, ax1 = plt.subplots(figsize=(13.2, 4.8))
    ax1.set_title("QC-filtered calibrated thresholds — Table 5.2 policy trade-offs")
    w_bar = 0.34
    ax1.bar(x - w_bar / 2, recall, width=w_bar, color="#1f77b4", alpha=0.85, label="Recall (%)")
    ax1.bar(x + w_bar / 2, far, width=w_bar, color="#d62728", alpha=0.75, label="FAR (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ymax = max(100.0, float(np.max(recall) + 5), float(np.max(far) + 5))
    ax1.set_ylim(0, ymax)
    ax1.set_ylabel("Percent (%)")
    ax1.grid(True, axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, f1, color="#2ca02c", marker="o", linewidth=2.0, label="F1")
    ax2.set_ylabel("F1-score")
    ax2.set_ylim(0.0, 1.05)

    for i, t in enumerate(thr):
        ax1.text(i, max(recall[i], far[i]) + 1.2, f"thr={t:.4g}", ha="center", va="bottom", fontsize=8.5)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")

    fig.text(0.5, 0.02, f"ROC-AUC={roc_auc:.4f} | PR-AUC={pr_auc:.4f}", ha="center", fontsize=9.5, color="#333333")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    os.chdir(ROOT)
    if not METRICS_CSV.exists():
        raise FileNotFoundError(METRICS_CSV)
    if not ATTACKED_OUT.exists():
        raise FileNotFoundError(f"Missing attacked NPZs: {ATTACKED_OUT}")

    print("Scoring QC regen evaluation (for Best-F1 @ 0.0509)...")
    y_true, y_score, roc_auc, pr_auc = _score_evaluation()
    rows = _build_policy_rows(y_true, y_score)

    metrics_out = RESULTS_DIR / "table52_threshold_policy_metrics.csv"
    png_out = RESULTS_DIR / "table52_threshold_policy_bar_compare.png"
    _write_metrics_csv(rows, metrics_out)
    summ = RESULTS_DIR / "table52_evaluation_curves.json"
    summ.write_text(json.dumps({"curves": {"roc_auc": roc_auc, "pr_auc": pr_auc}}, indent=2), encoding="utf-8")
    _plot(rows, roc_auc, pr_auc, png_out)

    print("Wrote:", metrics_out)
    print("Wrote:", png_out)
    for r in rows:
        print(
            f"  {r['name']:8s} thr={r['threshold']:.6g}  "
            f"Recall={r['recall']*100:.2f}%  FAR={r['far']*100:.4f}%  F1={r['f1']:.4f}"
        )


if __name__ == "__main__":
    main()
