"""
Matplotlib visualizations for the univariate pattern experiment (best-effort).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _read_baseline(baseline_csv: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not baseline_csv.is_file():
        return out
    with baseline_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            k = f"{row.get('section','')}|{row.get('metric','')}"
            try:
                out[k] = float(row.get("value", "nan"))
            except Exception:
                continue
    return out


def _pick_best_candidate(sel_rows: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    best = None
    best_v = -1.0
    for r in sel_rows:
        try:
            v = float(r.get("pattern_shift_recall", -1.0))
        except Exception:
            continue
        if v > best_v:
            best_v = v
            best = r
    return best


def make_all_plots(*, out_dir: Path, sel_rows: List[Dict[str, Any]], baseline_csv: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    bl = _read_baseline(baseline_csv)
    base_ps = float(bl.get("per_attack_p99|pattern_shift_recall", 0.1141194898187514))
    base_far = float(bl.get("overall_p99|far", 0.034051387741803496))
    base_f1 = float(bl.get("overall_p99|f1", 0.7943689592379886))
    base_bal = float(bl.get("overall_p99|balanced_accuracy", 0.8736036112484018))

    labels = [f"{r.get('model_name','')}\nw={r.get('W_order_score')}" for r in sel_rows]
    ps = [float(r.get("pattern_shift_recall", 0.0)) for r in sel_rows]
    far = [float(r.get("FAR", 0.0)) for r in sel_rows]
    f1 = [float(r.get("F1", 0.0)) for r in sel_rows]
    bal = [float(r.get("balanced_accuracy", 0.0)) for r in sel_rows]

    def _bar_compare(path: str, vals: List[float], title: str, ylab: str, base: float) -> None:
        if not sel_rows:
            return
        x = np.arange(len(vals))
        fig, ax = plt.subplots(figsize=(max(10, len(vals) * 1.2), 5))
        ax.bar(x, vals, color="#3498db", alpha=0.85, label="candidate")
        ax.axhline(base, color="#e74c3c", lw=2, ls="--", label="baseline (p99)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / path, dpi=140)
        plt.close(fig)

    _bar_compare("pattern_shift_recall_comparison.png", ps, "pattern_shift recall @p99", "recall", base_ps)
    _bar_compare("far_comparison.png", far, "FAR @p99", "FAR", base_far)
    _bar_compare("f1_comparison.png", f1, "Overall F1 @p99", "F1", base_f1)
    _bar_compare("balanced_accuracy_comparison.png", bal, "Balanced accuracy @p99", "balanced acc", base_bal)

    cand = _pick_best_candidate(sel_rows)
    if cand is not None:
        fig, ax = plt.subplots(figsize=(9, 5))
        attacks = ["noise", "spike", "drift", "freeze", "pattern_shift"]
        base_map = {
            "noise": float(bl.get("per_attack_p99|noise_recall", 1.0)),
            "spike": float(bl.get("per_attack_p99|spike_recall", 0.9998952550539436)),
            "drift": float(bl.get("per_attack_p99|drift_recall", 1.0)),
            "freeze": float(bl.get("per_attack_p99|freeze_recall", 0.9969097651421508)),
            "pattern_shift": float(bl.get("per_attack_p99|pattern_shift_recall", 0.1141194898187514)),
        }
        cand_map = {
            "noise": float(cand.get("noise_recall_delta", 0.0)) + base_map["noise"],
            "spike": float(cand.get("spike_recall_delta", 0.0)) + base_map["spike"],
            "drift": float(cand.get("drift_recall_delta", 0.0)) + base_map["drift"],
            "freeze": float(cand.get("freeze_recall_delta", 0.0)) + base_map["freeze"],
            "pattern_shift": float(cand.get("pattern_shift_recall", 0.0)),
        }
        x = np.arange(len(attacks))
        w = 0.38
        ax.bar(x - w / 2, [base_map[a] for a in attacks], width=w, label="baseline", color="#95a5a6")
        ax.bar(x + w / 2, [cand_map[a] for a in attacks], width=w, label="best candidate", color="#2980b9")
        ax.set_xticks(x)
        ax.set_xticklabels(attacks, rotation=20, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.set_title("Per-attack recall: baseline vs best candidate (p99)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "per_attack_recall_baseline_vs_candidate.png", dpi=140)
        plt.close(fig)

    # Score distribution placeholders (requires optional score dump file)
    dump = out_dir / "score_dump_example.npz"
    if dump.is_file():
        z = np.load(dump)
        sn = z["s_normal"].astype(np.float64)
        sps = z["s_ps"].astype(np.float64)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(sn, bins=80, alpha=0.55, density=True, label="normal (candidate scores)")
        ax.hist(sps, bins=80, alpha=0.55, density=True, label="pattern_shift (candidate scores)")
        ax.set_title("Score distribution (candidate model)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "score_distribution_pattern_shift_baseline_vs_candidate.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(sn, bins=80, alpha=0.55, density=True, label="normal")
        s_att = z["s_attacked"].astype(np.float64) if "s_attacked" in z.files else sps
        ax.hist(s_att, bins=80, alpha=0.55, density=True, label="attacked (all types)")
        ax.set_title("Score distribution: normal vs attacked (candidate)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "score_distribution_normal_vs_attack_candidate.png", dpi=140)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Score histograms require score_dump_example.npz\n(re-run evaluate with --score-dump)",
            ha="center",
            va="center",
        )
        fig.savefig(out_dir / "score_distribution_pattern_shift_baseline_vs_candidate.png", dpi=140)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Score histograms require score_dump_example.npz\n(re-run evaluate with --score-dump)",
            ha="center",
            va="center",
        )
        fig.savefig(out_dir / "score_distribution_normal_vs_attack_candidate.png", dpi=140)
        plt.close(fig)
