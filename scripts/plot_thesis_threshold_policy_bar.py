"""
Bar+line threshold policy figure for thesis (Table 5.2 policies) from
overall_threshold_metrics_4attacks.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


POLICIES = [
    ("p99", "p99\n(calibrated)"),
    ("p99_5", "p99.5\n(calibrated)"),
    ("p99_7", "p99.7\n(calibrated)"),
    ("3sigma", "3-sigma\n(calibrated)"),
    ("best_f1", "Best-F1\n(analysis)"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("thesis_official_evaluation_figures/overall_threshold_metrics_4attacks.csv"),
    )
    ap.add_argument(
        "--summary-json",
        type=Path,
        default=Path("thesis_official_evaluation_figures/evaluation_summary_4attacks.json"),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("thesis_official_evaluation_figures/threshold_policy_bar_table52.png"),
    )
    args = ap.parse_args()

    rows: dict[str, dict[str, float]] = {}
    with args.metrics_csv.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows[str(r["threshold_name"]).strip()] = {k: float(r[k]) for k in r if k != "threshold_name"}

    roc_auc = pr_auc = float("nan")
    if args.summary_json.exists():
        summ = json.loads(args.summary_json.read_text(encoding="utf-8"))
        curves = summ.get("curves", {})
        roc_auc = float(curves.get("roc_auc", float("nan")))
        pr_auc = float(curves.get("pr_auc", float("nan")))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thr, recall, far, f1, labels = [], [], [], [], []
    for key, lab in POLICIES:
        if key not in rows:
            raise KeyError(f"Missing {key} in {args.metrics_csv}")
        r = rows[key]
        thr.append(float(r["threshold"]))
        recall.append(float(r["recall"]) * 100.0)
        far.append(float(r["far"]) * 100.0)
        f1.append(float(r["f1"]))
        labels.append(lab)

    x = np.arange(len(POLICIES), dtype=float)
    fig, ax1 = plt.subplots(figsize=(13.2, 4.8))
    ax1.set_title("Calibrated threshold policies — four-attack evaluation (Table 5.2)")
    w = 0.34
    ax1.bar(x - w / 2, recall, width=w, color="#1f77b4", alpha=0.85, label="Recall (%)")
    ax1.bar(x + w / 2, far, width=w, color="#d62728", alpha=0.75, label="FAR (%)")
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

    if np.isfinite(roc_auc) and np.isfinite(pr_auc):
        fig.text(0.5, 0.02, f"ROC-AUC={roc_auc:.4f} | PR-AUC={pr_auc:.4f}", ha="center", fontsize=9.5, color="#333333")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(args.output, dpi=220)
    plt.close(fig)
    print("Wrote:", args.output.resolve())


if __name__ == "__main__":
    main()
