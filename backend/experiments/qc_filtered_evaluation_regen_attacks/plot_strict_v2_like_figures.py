"""
Generate strict_v2-like figures for the NEW QC evaluation (with regenerated attacked data).

All threshold-dependent visuals use the Best-F1 operating point only
(threshold from compute_curves_and_auc / strict_v2-style curve analysis).

Creates NEW PNGs under:
  backend/experiments/qc_filtered_evaluation_regen_attacks/results/

Figures:
  - score_distribution_qc_filtered.png  (vertical line = best F1 thr only)
  - roc_curve_qc_filtered.png           (curve + marker at best-F1 operating point)
  - precision_recall_curve_qc_filtered.png (curve + marker at best-F1 operating point)
  - confusion_matrix_best_f1_qc_filtered.png
  - threshold_policy_qc_filtered.png    (Recall%, FAR%, F1 at Best-F1 threshold)

Safety:
  - Read-only from model/split/data.
  - Write-only inside this experiment directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
EXP_DIR = Path(__file__).resolve().parent

MODEL_PATH = BACKEND / "app" / "best_model_qc_filtered.keras"
SPLIT_JSON = BACKEND / "config" / "data_split_qc_filtered.json"
THRESHOLDS_PATH = BACKEND / "app" / "thresholds_qc_filtered.json"

NORMAL_DIR = ROOT / "data" / "reduced"
ATTACKED_DIR = EXP_DIR / "attacked_qc_v2"

OUT_DIR = EXP_DIR / "results"
OUT_SCORE_DIST = OUT_DIR / "score_distribution_qc_filtered.png"
OUT_ROC = OUT_DIR / "roc_curve_qc_filtered.png"
OUT_PR = OUT_DIR / "precision_recall_curve_qc_filtered.png"
OUT_CM = OUT_DIR / "confusion_matrix_best_f1_qc_filtered.png"
OUT_POLICY = OUT_DIR / "threshold_policy_qc_filtered.png"


def _maybe_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    return plt


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

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

    for p in (MODEL_PATH, SPLIT_JSON, THRESHOLDS_PATH, NORMAL_DIR, ATTACKED_DIR):
        if not p.exists():
            raise FileNotFoundError(p)

    thr_doc = __import__("json").loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    thr_map = dict(thr_doc.get("thresholds", {}))
    thresholds = {k: float(thr_map[k]) for k in ("p99", "p995", "p997", "3sigma") if k in thr_map}

    # Load model robustly
    X_sample = load_windows_npy(next(NORMAL_DIR.glob("chunk_*.npy")))
    T, C = int(X_sample.shape[1]), int(X_sample.shape[2])
    model = load_keras_model_robust(MODEL_PATH, T, C)

    # Score normal test
    names = load_split_filenames(SPLIT_JSON, "test")
    normal_scores_parts: List[np.ndarray] = []
    for fname in names:
        X = load_windows_npy((NORMAL_DIR / fname).resolve())
        s = compute_scores_strict(model, X, batch_size=256)
        normal_scores_parts.append(np.asarray(s, dtype=np.float64))
    scores_normal = np.concatenate(normal_scores_parts).astype(np.float64)

    # Score attacked regenerated
    attacked_scores_parts: List[np.ndarray] = []
    attacked_labels_parts: List[np.ndarray] = []
    for npz in sorted(ATTACKED_DIR.glob("*.npz")):
        X_att, y_w, _meta = load_attacked_npz(npz)
        s = compute_scores_strict(model, X_att, batch_size=256)
        attacked_scores_parts.append(np.asarray(s, dtype=np.float64))
        attacked_labels_parts.append(np.asarray(y_w, dtype=np.uint8))
    scores_attacked = np.concatenate(attacked_scores_parts).astype(np.float64)
    y_attacked = np.concatenate(attacked_labels_parts).astype(np.uint8)

    # Curves / AUC
    y_true = np.concatenate([np.zeros(len(scores_normal), dtype=np.uint8), y_attacked])
    y_score = np.concatenate([scores_normal, scores_attacked])
    if len(y_true) != len(y_score):
        raise RuntimeError(
            f"y_true / y_score length mismatch: {len(y_true)} vs {len(y_score)} "
            f"(normal_scores={len(scores_normal)}, attacked_scores={len(scores_attacked)}, "
            f"y_attacked={len(y_attacked)})"
        )
    curves_info, curves = compute_curves_and_auc(y_true, y_score)

    roc_auc = float(curves_info["roc_auc"])
    pr_auc = float(curves_info["pr_auc"])
    best_f1_thr = float(curves_info["best_f1_threshold"])

    plt = _maybe_matplotlib()

    cm_bf = confusion_at_threshold(y_true, y_score, best_f1_thr)
    m_bf = metrics_from_confusion(cm_bf)
    fpr_op = float(m_bf["far"])
    tpr_op = float(m_bf["recall"])
    prec_op = float(m_bf["precision"])
    rec_op = float(m_bf["recall"])

    # Score distribution — show Best-F1 plus reference statistical thresholds
    plt.figure(figsize=(10, 6))
    plt.hist(
        scores_normal,
        bins=140,
        alpha=0.6,
        density=True,
        label=f"Normal (test) n={len(scores_normal)}",
        color="#2ecc71",
    )
    plt.hist(
        scores_attacked,
        bins=140,
        alpha=0.6,
        density=True,
        label=f"Attacked_v2 n={len(scores_attacked)}",
        color="#e74c3c",
    )
    plt.axvline(
        best_f1_thr,
        linestyle="-",
        linewidth=2.6,
        color="#1f4e79",
        label=f"Best-F1 thr={best_f1_thr:.5g}",
    )
    for k, c in (("p99", "#7f8c8d"), ("p995", "#95a5a6"), ("p997", "#bdc3c7"), ("3sigma", "#34495e")):
        if k in thresholds:
            plt.axvline(
                thresholds[k],
                linestyle="--",
                linewidth=1.4,
                color=c,
                alpha=0.95,
                label=f"{k}={thresholds[k]:.5g}",
            )
    plt.xlabel("Anomaly score")
    plt.ylabel("Density")
    plt.title("Score distribution (Best-F1 + reference thresholds)")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_SCORE_DIST, dpi=140)
    plt.close()

    # ROC — curve + Best-F1 operating point
    fpr = curves["roc_fpr"]
    tpr = curves["roc_tpr"]
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, linewidth=2, color="#2980b9", label="ROC")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.scatter(
        [fpr_op],
        [tpr_op],
        s=80,
        zorder=5,
        color="#e67e22",
        edgecolors="black",
        linewidths=0.8,
        label=f"Best-F1 op. (thr={best_f1_thr:.5g})",
    )
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC curve (Best-F1 operating point marked)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_ROC, dpi=140)
    plt.close()

    # PR — curve + Best-F1 operating point
    prec = curves["pr_precision"]
    rec = curves["pr_recall"]
    plt.figure(figsize=(7, 6))
    plt.plot(rec, prec, linewidth=2, color="#c0392b", label="PR curve")
    plt.scatter(
        [rec_op],
        [prec_op],
        s=80,
        zorder=5,
        color="#e67e22",
        edgecolors="black",
        linewidths=0.8,
        label=f"Best-F1 op. (thr={best_f1_thr:.5g})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall curve (Best-F1 operating point marked)")
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_PR, dpi=140)
    plt.close()

    # Confusion matrix at best_f1
    cm = cm_bf
    mat = np.array([[cm.TN, cm.FP], [cm.FN, cm.TP]], dtype=np.int64)
    plt.figure(figsize=(6, 5))
    plt.imshow(mat, cmap="Blues")
    plt.title(f"Confusion matrix (Best-F1, thr={best_f1_thr:.5g})")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    for (i, j), val in np.ndenumerate(mat):
        plt.text(j, i, str(int(val)), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(OUT_CM, dpi=140)
    plt.close()

    # Threshold summary — single operating point (Best-F1)
    recall_pct = 100.0 * float(m_bf["recall"])
    far_pct = 100.0 * float(m_bf["far"])
    f1 = float(m_bf["f1"])
    xb = np.arange(3)
    labels_bar = ["Recall (%)", "FAR (%)", "F1"]
    heights = [recall_pct, far_pct, 100.0 * f1]
    colors_bar = ["#1f77b4", "#d62728", "#2ca02c"]
    plt.figure(figsize=(8, 4.2))
    bars = plt.bar(xb, heights, color=colors_bar, alpha=0.85, edgecolor="black", linewidth=0.6)
    plt.xticks(xb, labels_bar)
    plt.ylabel("Value")
    plt.ylim(0, max(100.0, max(heights) * 1.08))
    plt.title(f"Metrics at Best-F1 threshold (thr = {best_f1_thr:.6g})")
    plt.grid(True, axis="y", alpha=0.25)
    for rect, h in zip(bars, heights):
        plt.text(
            rect.get_x() + rect.get_width() / 2,
            min(rect.get_height() + 1.5, plt.ylim()[1] - 1),
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.figtext(
        0.5,
        0.02,
        f"ROC-AUC={roc_auc:.4f} | PR-AUC={pr_auc:.4f} | Best-F1 thr={best_f1_thr:.6g}",
        ha="center",
        fontsize=9,
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(OUT_POLICY, dpi=140, bbox_inches="tight")
    plt.close()

    # Small printout for convenience
    m_best = metrics_from_confusion(cm)
    print("Wrote:")  # keep simple for logs
    print(str(OUT_SCORE_DIST))
    print(str(OUT_ROC))
    print(str(OUT_PR))
    print(str(OUT_CM))
    print(str(OUT_POLICY))
    print(f"ROC-AUC={roc_auc:.6f} PR-AUC={pr_auc:.6f} best_f1_thr={best_f1_thr:.6g}")
    print(
        f"best_f1 metrics: acc={m_best['accuracy']:.6f} bal_acc={m_best['balanced_accuracy']:.6f} "
        f"prec={m_best['precision']:.6f} rec={m_best['recall']:.6f} f1={m_best['f1']:.6f} far={m_best['far']:.6f}"
    )


if __name__ == "__main__":
    os.chdir(ROOT)
    main()

