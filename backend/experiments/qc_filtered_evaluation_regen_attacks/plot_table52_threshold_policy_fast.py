"""Fast Table 5.2 policy figure (no model scoring).

Metrics: QC regen attacks run (metrics_qc_filtered.csv) for p99 / p99.5 / p99.7 / 3-sigma;
Best-F1 @ 0.0509 on the same QC regen windows (requires table52_threshold_policy_metrics.csv
from plot_table52_threshold_policy.py) OR falls back to eval_four_attacks row if missing.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
METRICS_CSV = RESULTS_DIR / "metrics_qc_filtered.csv"
BEST_F1_METRICS = RESULTS_DIR / "table52_threshold_policy_metrics.csv"
EVAL4_CSV = ROOT / "eval_four_attacks_qc_filtered_output" / "overall_threshold_metrics_4attacks.csv"

TABLE52: List[Dict[str, Any]] = [
    {"name": "p99", "threshold": 0.202873, "label": "p99\n(calibrated)", "csv_key": "p99"},
    {"name": "p995", "threshold": 0.299654, "label": "p99.5\n(calibrated)", "csv_key": "p995"},
    {"name": "p997", "threshold": 0.569635, "label": "p99.7\n(calibrated)", "csv_key": "p997"},
    {"name": "3sigma", "threshold": 0.213348, "label": "3-sigma\n(calibrated)", "csv_key": "3sigma"},
    {"name": "best_f1", "threshold": 0.0509, "label": "Best-F1\n(analysis)", "csv_key": None},
]

METRICS_PRECOMPUTED = RESULTS_DIR / "table52_threshold_policy_metrics.csv"


def _load_csv(path: Path) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out[str(r["name"]).strip()] = {k: float(r[k]) for k in r if k != "name"}
    return out


def _load_precomputed_table52() -> List[Dict[str, Any]] | None:
    if not METRICS_PRECOMPUTED.exists():
        return None
    rows: List[Dict[str, Any]] = []
    with METRICS_PRECOMPUTED.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            name = str(r["name"]).strip()
            spec = next((s for s in TABLE52 if s["name"] == name), None)
            rows.append(
                {
                    "name": name,
                    "threshold": float(r["threshold"]),
                    "label": spec["label"] if spec else str(r.get("label", name)),
                    "recall": float(r["recall"]),
                    "far": float(r["far"]),
                    "f1": float(r["f1"]),
                }
            )
    return rows if len(rows) == len(TABLE52) else None


def _build_rows() -> tuple[List[Dict[str, Any]], float, float]:
    pre = _load_precomputed_table52()
    if pre is not None:
        qc = _load_csv(METRICS_CSV)
        return pre, float(qc["p99"]["roc_auc"]), float(qc["p99"]["pr_auc"])

    qc = _load_csv(METRICS_CSV)
    roc = float(qc["p99"]["roc_auc"])
    pr = float(qc["p99"]["pr_auc"])

    best_f1_row: Dict[str, float] | None = None
    if EVAL4_CSV.exists():
        with EVAL4_CSV.open("r", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if str(r["threshold_name"]).strip() == "best_f1":
                    best_f1_row = {
                        "recall": float(r["recall"]),
                        "far": float(r["far"]),
                        "f1": float(r["f1"]),
                    }
                    break

    rows: List[Dict[str, Any]] = []
    for spec in TABLE52:
        if spec["name"] == "best_f1":
            if best_f1_row is None:
                raise RuntimeError("No best_f1 metrics; run plot_table52_threshold_policy.py first.")
            m = best_f1_row
        else:
            m = qc[str(spec["csv_key"])]
        rows.append(
            {
                "name": spec["name"],
                "threshold": float(spec["threshold"]),
                "label": spec["label"],
                "recall": float(m["recall"]),
                "far": float(m["far"]),
                "f1": float(m["f1"]),
            }
        )
    return rows, roc, pr


def main() -> None:
    models_dir = BACKEND / "models"
    if str(models_dir) not in sys.path:
        sys.path.insert(0, str(models_dir))
    import generate_ml_engineering_figures as gm  # type: ignore

    rows, roc_auc, pr_auc = _build_rows()
    out_png = RESULTS_DIR / "table52_threshold_policy_bar_compare.png"

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
    ax1.set_ylim(0, max(100.0, float(np.max(recall) + 5), float(np.max(far) + 5)))
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

    metrics_out = RESULTS_DIR / "table52_threshold_policy_metrics.csv"
    with metrics_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "threshold", "recall", "far", "f1", "label"])
        for r in rows:
            w.writerow([r["name"], r["threshold"], r["recall"], r["far"], r["f1"], r["label"]])

    print("Wrote:", out_png)
    print("Wrote:", metrics_out)


if __name__ == "__main__":
    main()
