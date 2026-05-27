"""
Alternative visualizations for Table 5.2 threshold policies (same data as bar+line chart).

Outputs under thesis_official_evaluation_figures/:
  - threshold_policy_small_multiples_table52.png  (3 panels, no dual axis)
  - threshold_policy_recall_far_scatter_table52.png (trade-off map)
  - threshold_policy_heatmap_table52.png (compact metric grid)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

POLICIES: List[Tuple[str, str]] = [
    ("p99", "p99"),
    ("p99_5", "p99.5"),
    ("p99_7", "p99.7"),
    ("3sigma", "3σ"),
    ("best_f1", "Best-F1"),
]

COLORS = {
    "p99": "#1f77b4",
    "p99_5": "#6baed6",
    "p99_7": "#9ecae1",
    "3sigma": "#2ca02c",
    "best_f1": "#d62728",
}


def _load(metrics_csv: Path, summary_json: Path) -> Tuple[List[dict], float, float]:
    rows_map: Dict[str, Dict[str, float]] = {}
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows_map[str(r["threshold_name"]).strip()] = {k: float(r[k]) for k in r if k != "threshold_name"}

    data = []
    for key, short in POLICIES:
        r = rows_map[key]
        data.append(
            {
                "key": key,
                "label": short,
                "thr": float(r["threshold"]),
                "recall": float(r["recall"]) * 100.0,
                "far": float(r["far"]) * 100.0,
                "f1": float(r["f1"]),
            }
        )

    roc = pr = float("nan")
    if summary_json.exists():
        curves = json.loads(summary_json.read_text(encoding="utf-8")).get("curves", {})
        roc = float(curves.get("roc_auc", float("nan")))
        pr = float(curves.get("pr_auc", float("nan")))
    return data, roc, pr


def plot_small_multiples(data: List[dict], out: Path, roc: float, pr: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [d["label"] for d in data]
    x = np.arange(len(data))
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharex=True)
    fig.suptitle("Threshold policy comparison (Table 5.2)", fontsize=12, y=1.02)

    specs = [
        ("recall", "Recall (%)", "#1f77b4", (0, 105)),
        ("far", "FAR (%)", "#d62728", (0, max(2.5, max(d["far"] for d in data) * 1.35))),
        ("f1", "F1-score", "#2ca02c", (0, 1.05)),
    ]
    for ax, (field, ylab, color, ylim) in zip(axes, specs):
        vals = [d[field] for d in data]
        bars = ax.bar(x, vals, color=color, alpha=0.88, edgecolor="white", linewidth=0.6)
        ax.set_ylabel(ylab)
        ax.set_ylim(ylim)
        ax.grid(True, axis="y", alpha=0.25)
        for i, (b, d) in enumerate(zip(bars, data)):
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                h + (ylim[1] - ylim[0]) * 0.02,
                f"{h:.2f}" if field == "f1" else f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            if field == "recall":
                ax.text(b.get_x() + b.get_width() / 2, -ylim[1] * 0.12, f"τ={d['thr']:.3g}", ha="center", fontsize=7, color="#555")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[0].set_xticks(x)
    axes[2].set_xticks(x)

    if np.isfinite(roc):
        fig.text(0.5, -0.02, f"ROC-AUC={roc:.4f} | PR-AUC={pr:.4f}", ha="center", fontsize=9, color="#444")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_recall_far_scatter(data: List[dict], out: Path, roc: float, pr: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.set_title("Recall–FAR trade-off by threshold policy")
    ax.set_xlabel("False alarm rate FAR (%)")
    ax.set_ylabel("Recall (%)")

    for d in data:
        c = COLORS.get(d["key"], "#333")
        ax.scatter(d["far"], d["recall"], s=120 + 280 * d["f1"], c=c, alpha=0.85, edgecolors="#333", linewidths=0.6, zorder=3)
        ax.annotate(
            f"{d['label']}\nτ={d['thr']:.3g}, F1={d['f1']:.3f}",
            (d["far"], d["recall"]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=8.5,
            color=c,
            arrowprops=dict(arrowstyle="-", color="#999", lw=0.8),
        )

    ax.set_xlim(-0.2, max(d["far"] for d in data) * 1.45 + 0.3)
    ax.set_ylim(30, 102)
    ax.grid(True, alpha=0.3)
    ax.axhline(75.4, color="#1f77b4", ls=":", lw=0.9, alpha=0.5)
    ax.text(0.98, 0.02, "Marker size ∝ F1", transform=ax.transAxes, ha="right", fontsize=8, color="#666")

    if np.isfinite(roc):
        fig.text(0.5, 0.01, f"ROC-AUC={roc:.4f} | PR-AUC={pr:.4f}", ha="center", fontsize=9, color="#444")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(data: List[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["Recall (%)", "FAR (%)", "F1"]
    mat = np.array([[d["recall"], d["far"], d["f1"] * 100.0] for d in data], dtype=float)
    row_labels = [f"{d['label']}\n(τ={d['thr']:.3g})" for d in data]

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(metrics)
    ax.set_yticks(np.arange(len(data)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title("Threshold policies — metric heatmap")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            txt = f"{val:.1f}" if j < 2 else f"{data[i]['f1']:.3f}"
            ax.text(j, i, txt, ha="center", va="center", color="black" if val > 50 else "#111", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, label="Scale (F1 shown as %×100 in cell text)")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


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
        "--out-dir",
        type=Path,
        default=Path("thesis_official_evaluation_figures"),
    )
    args = ap.parse_args()

    data, roc, pr = _load(args.metrics_csv, args.summary_json)
    out_dir = args.out_dir

    p1 = out_dir / "threshold_policy_small_multiples_table52.png"
    p2 = out_dir / "threshold_policy_recall_far_scatter_table52.png"
    p3 = out_dir / "threshold_policy_heatmap_table52.png"

    plot_small_multiples(data, p1, roc, pr)
    plot_recall_far_scatter(data, p2, roc, pr)
    plot_heatmap(data, p3)

    for p in (p1, p2, p3):
        print("Wrote:", p.resolve())


if __name__ == "__main__":
    main()
