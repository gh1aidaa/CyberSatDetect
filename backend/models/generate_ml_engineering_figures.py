"""
Generate ML-engineering style figures for reports:
1) Model architecture schematic (baseline hybrid AE as implemented in `train_hybrid_model.py`)
2) Threshold policy figure from strict evaluation outputs (uses YOUR CSV/JSON artifacts)
3) Training dynamics curves IF a JSON history file is provided (optional)

Outputs default to: backend/app/ml_engineering_figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _ensure_matplotlib():
    import importlib.util

    if importlib.util.find_spec("matplotlib") is None:
        raise RuntimeError("matplotlib is not installed. Run: python -m pip install matplotlib")
    import matplotlib.pyplot as plt  # noqa: WPS433

    return plt


def _style(plt):
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )


def plot_architecture_schematic(plt, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    ax.axis("off")
    ax.set_title(title, pad=14)

    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    boxes = [
        ("Input\n(B,T,C)", (0.02, 0.55, 0.10, 0.30)),
        ("Conv1D\nfeature extract", (0.15, 0.55, 0.12, 0.30)),
        ("Temporal\nencoder\n(LSTM/GRU)", (0.30, 0.55, 0.14, 0.30)),
        ("Latent\nrepresentation", (0.47, 0.55, 0.12, 0.30)),
        ("Decoder\nreconstruction", (0.62, 0.55, 0.14, 0.30)),
        ("Reconstruction\nhead\nX̂", (0.79, 0.72, 0.18, 0.22)),
        ("Prediction\nhead\np̂", (0.79, 0.40, 0.18, 0.22)),
    ]

    drawn = []
    for text, (x, y, w, h) in boxes:
        bb = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            linewidth=1.1,
            edgecolor="#333333",
            facecolor="#f2f2f2",
        )
        ax.add_patch(bb)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)
        drawn.append((x, y, w, h))

    # arrows along main trunk (first 5 boxes)
    for i in range(5 - 1):
        x1, y1, w1, h1 = drawn[i]
        x2, y2, w2, h2 = drawn[i + 1]
        arr = FancyArrowPatch(
            (x1 + w1, y1 + h1 / 2),
            (x2, y2 + h2 / 2),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.1,
            color="#444444",
        )
        ax.add_patch(arr)

    # split from decoder to heads (last trunk box index=4)
    x4, y4, w4, h4 = drawn[4]
    xr1, yr1, wr1, hr1 = drawn[5]
    xr2, yr2, wr2, hr2 = drawn[6]

    arr_top = FancyArrowPatch(
        (x4 + w4, y4 + h4 * 0.72),
        (xr1, yr1 + hr1 / 2),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.05,
        color="#444444",
        connectionstyle="arc3,rad=-0.12",
    )
    arr_bot = FancyArrowPatch(
        (x4 + w4, y4 + h4 * 0.28),
        (xr2, yr2 + hr2 / 2),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.05,
        color="#444444",
        connectionstyle="arc3,rad=0.12",
    )
    ax.add_patch(arr_top)
    ax.add_patch(arr_bot)

    ax.text(
        0.02,
        0.12,
        "Inference score (evaluation/inference):\n"
        "score = e_recon + e_pred + e_grad\n"
        "(Separation loss is training-only regularization.)",
        fontsize=9.5,
        color="#222222",
        va="top",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _read_threshold_rows(csv_path: Path) -> Dict[str, Dict[str, float]]:
    import csv

    rows: Dict[str, Dict[str, float]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            name = str(r["name"]).strip()
            rows[name] = {k: float(r[k]) for k in r.keys() if k != "name"}
    return rows


def plot_threshold_policy(plt, out_path: Path, csv_path: Path, summary_json: Optional[Path], title: str):
    rows = _read_threshold_rows(csv_path)

    wanted = ["p99", "best_f1", "far_le_1pct", "far_le_0.5pct"]
    labels = ["p99\n(normal-derived)", "best F1\n(analysis)", "FAR ≤ 1%\n(constraint)", "FAR ≤ 0.5%\n(constraint)"]
    keys = wanted

    thr = [rows[k]["threshold"] for k in keys]
    far = [rows[k]["far"] * 100.0 for k in keys]
    recall = [rows[k]["recall"] * 100.0 for k in keys]
    f1 = [rows[k]["f1"] for k in keys]

    x = np.arange(len(keys), dtype=float)

    fig, ax1 = plt.subplots(figsize=(11.8, 4.8))
    ax1.set_title(title)
    w = 0.34
    b1 = ax1.bar(x - w / 2, recall, width=w, color="#1f77b4", alpha=0.85, label="Recall (%)")
    b2 = ax1.bar(x + w / 2, far, width=w, color="#d62728", alpha=0.75, label="FAR (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, max(100.0, float(np.max(recall) + 5), float(np.max(far) + 5)))
    ax1.set_ylabel("Percent (%)")
    ax1.grid(True, axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, f1, color="#2ca02c", marker="o", linewidth=2.0, label="F1")
    ax2.set_ylabel("F1-score")
    ax2.set_ylim(0.0, 1.05)

    # annotate thresholds above bars
    for i, t in enumerate(thr):
        ax1.text(i, max(recall[i], far[i]) + 1.2, f"thr={t:.4g}", ha="center", va="bottom", fontsize=8.5, rotation=0)

    # legends merged
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")

    # Optional subtitle from summary json (ROC-AUC etc.)
    if summary_json is not None and summary_json.exists():
        summ = json.loads(summary_json.read_text(encoding="utf-8"))
        curves = summ.get("curves", {}) or {}
        subt = f"ROC-AUC={curves.get('roc_auc'):.4f} | PR-AUC={curves.get('pr_auc'):.4f}"
        fig.text(0.5, 0.02, subt, ha="center", fontsize=9.5, color="#333333")

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path)
    plt.close(fig)


def plot_threshold_axis(plt, out_path: Path, csv_path: Path, title: str):
    """
    Secondary view: show thresholds on x-axis vs FAR/Recall/F1.
    """
    rows = _read_threshold_rows(csv_path)
    keys = ["best_f1", "p99", "far_le_1pct", "far_le_0.5pct"]
    labels = ["best_f1", "p99", "far≤1%", "far≤0.5%"]

    thr = np.array([rows[k]["threshold"] for k in keys], dtype=float)
    order = np.argsort(thr)
    thr = thr[order]
    labels = [labels[i] for i in order]

    far = np.array([rows[keys[i]]["far"] * 100.0 for i in order], dtype=float)
    recall = np.array([rows[keys[i]]["recall"] * 100.0 for i in order], dtype=float)
    f1 = np.array([rows[keys[i]]["f1"] for i in order], dtype=float)

    fig, ax1 = plt.subplots(figsize=(11.8, 4.6))
    ax1.set_title(title)
    ax1.plot(thr, recall, marker="o", color="#1f77b4", linewidth=2.0, label="Recall (%)")
    ax1.plot(thr, far, marker="o", color="#d62728", linewidth=2.0, label="FAR (%)")
    ax1.set_xlabel("Threshold value")
    ax1.set_ylabel("Percent (%)")
    ax1.grid(True, alpha=0.25)

    for t, lab in zip(thr, labels):
        ax1.axvline(t, color="#bbbbbb", linestyle="--", linewidth=1.0, alpha=0.8)
        ax1.text(t, 2.0, lab, rotation=90, va="bottom", ha="right", fontsize=8.5, color="#555555")

    ax2 = ax1.twinx()
    ax2.plot(thr, f1, marker="s", color="#2ca02c", linewidth=2.0, label="F1")
    ax2.set_ylabel("F1-score")
    ax2.set_ylim(0.0, 1.05)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center left", bbox_to_anchor=(1.12, 0.55))

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _load_training_history(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_training_history(plt, out_path: Path, hist: Dict[str, Any], title: str):
    """
    Accepts common Keras history dict: {"loss":[...], "val_loss":[...], ...}
    """
    fig, ax = plt.subplots(figsize=(11.2, 4.6))
    ax.set_title(title)

    def pick_series(*names: str) -> Optional[List[float]]:
        for n in names:
            v = hist.get(n)
            if isinstance(v, list) and len(v) > 0:
                return [float(x) for x in v]
        return None

    loss = pick_series("loss")
    val_loss = pick_series("val_loss")
    if loss:
        ax.plot(np.arange(1, len(loss) + 1), loss, label="train loss", color="#1f77b4", linewidth=2.0)
    if val_loss:
        ax.plot(np.arange(1, len(val_loss) + 1), val_loss, label="val loss", color="#ff7f0e", linewidth=2.0)

    ax.set_xlabel("Epoch (index in history)")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_readme(out_dir: Path, notes: str):
    (out_dir / "README.txt").write_text(notes, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--threshold-csv",
        type=Path,
        default=Path("backend/app/evaluation_strict_v2/threshold_comparison.csv"),
    )
    ap.add_argument(
        "--summary-json",
        type=Path,
        default=Path("backend/app/evaluation_strict_v2/evaluation_summary.json"),
    )
    ap.add_argument("--training-history-json", type=Path, default=None, help="Optional Keras-like history JSON.")
    ap.add_argument("--output-dir", type=Path, default=Path("backend/app/ml_engineering_figures"))
    ap.add_argument(
        "--policy-title",
        type=str,
        default=None,
        help="Title for threshold policy figures (defaults to strict_v2 wording).",
    )
    ap.add_argument(
        "--skip-architecture",
        action="store_true",
        help="Do not emit 00_model_architecture_schematic.png (faster report-only runs).",
    )
    args = ap.parse_args()

    plt = _ensure_matplotlib()
    _style(plt)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_architecture:
        plot_architecture_schematic(
            plt,
            args.output_dir / "00_model_architecture_schematic.png",
            title="CyberSatDetect hybrid model (schematic; baseline training architecture)",
        )

    if not args.threshold_csv.exists():
        raise FileNotFoundError(f"Missing threshold CSV: {args.threshold_csv}")

    bar_title = args.policy_title or (
        "Threshold policy on YOUR strict_v2 results (p99 vs best-F1 vs FAR constraints)"
    )
    lines_title = (
        (args.policy_title + " — threshold sweep (sorted by threshold)")
        if args.policy_title
        else "Threshold sweep view (sorted by threshold) — strict_v2"
    )

    plot_threshold_policy(
        plt,
        args.output_dir / "01_threshold_policy_bar_compare.png",
        csv_path=args.threshold_csv,
        summary_json=args.summary_json,
        title=bar_title,
    )
    plot_threshold_axis(
        plt,
        args.output_dir / "02_threshold_policy_lines.png",
        csv_path=args.threshold_csv,
        title=lines_title,
    )

    notes_lines = [
        "Generated by: backend/models/generate_ml_engineering_figures.py",
        "",
        "Figures:",
    ]
    if not args.skip_architecture:
        notes_lines.append(
            "- 00_model_architecture_schematic.png : schematic aligned with train_hybrid_model.py (not auto-parsed)."
        )
    else:
        notes_lines.append("- 00_model_architecture_schematic.png : skipped (--skip-architecture).")
    notes_lines.extend(
        [
            "- 01_threshold_policy_bar_compare.png : uses the provided threshold_comparison.csv.",
            "- 02_threshold_policy_lines.png : alternate visualization from the same CSV.",
            "",
            "Training dynamics:",
        ]
    )

    if args.training_history_json and args.training_history_json.exists():
        hist = _load_training_history(args.training_history_json)
        plot_training_history(
            plt,
            args.output_dir / "03_training_loss_curves.png",
            hist=hist,
            title="Training dynamics (from provided history JSON)",
        )
        notes_lines.append(f"- 03_training_loss_curves.png : from {args.training_history_json.as_posix()}")
    else:
        notes_lines.append(
            "- No training history JSON was provided/found in-repo, so training curves were NOT fabricated."
        )
        notes_lines.append(
            "  If you export Keras history to JSON, rerun with: --training-history-json <path>"
        )

    write_readme(args.output_dir, "\n".join(notes_lines) + "\n")

    meta = {
        "threshold_csv": str(args.threshold_csv.as_posix()),
        "summary_json": str(args.summary_json.as_posix()) if args.summary_json.exists() else None,
        "training_history_json": str(args.training_history_json.as_posix()) if args.training_history_json else None,
    }
    (args.output_dir / "figure_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Wrote figures to:", args.output_dir.resolve())
    for p in sorted(args.output_dir.glob("*.png")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
