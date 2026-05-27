"""
Markdown report generator for the univariate pattern experiment.
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


def write_report(out_md: Path, *, sel_rows: List[Dict[str, Any]], baseline_csv: Path) -> None:
    bl = _read_baseline(baseline_csv)
    best = None
    best_ps = -1.0
    for r in sel_rows:
        try:
            v = float(r.get("pattern_shift_recall", -1.0))
        except Exception:
            continue
        if v > best_ps:
            best_ps = v
            best = r

    lines: List[str] = []
    lines.append("# Univariate Pattern Solution — Experimental Report\n")
    lines.append("## 1. Problem Statement\n")
    lines.append(
        "The production detector operates on univariate windows `(100, 1)`, which weakens sensitivity to "
        "`pattern_shift` attacks because temporal order can change while marginal statistics remain near-normal.\n"
    )
    lines.append("## 2. Why Univariate `(100,1)` Makes Pattern-Shift Hard\n")
    lines.append(
        "With a single feature per timestep, reconstruction, one-step prediction, and gradient consistency can "
        "remain small when values are permuted but still plausible; temporal ordering violations are not explicitly modeled.\n"
    )
    lines.append("## 3. Temporal Feature Engineering\n")
    lines.append(
        "We augment each window to `(100, 8)` using causal rolling statistics, finite differences, global z-score, "
        "causal local slope, and a normalized time index `t/99` (see `temporal_feature_engineering.py`).\n"
    )
    lines.append("## 4. Multi-Step Prediction\n")
    lines.append(
        "The model predicts the **last 10** samples of the original channel, encouraging it to represent short-horizon "
        "temporal evolution rather than only the final value.\n"
    )
    lines.append("## 5. Pattern-Order Self-Supervised Head\n")
    lines.append(
        "A small MLP head outputs a pseudo-probability that a window resembles synthetically reordered windows generated "
        "from **normal training data only** (no `attacked_v2` in training).\n"
    )
    lines.append("## 6. Why The System Remains Unsupervised / Self-Supervised\n")
    lines.append(
        "Attack labels are not used for training. Pseudo-shifts are synthetic transformations of normal windows used only "
        "as auxiliary objectives (separation + order).\n"
    )
    lines.append("## 7. Training Setup\n")
    lines.append(
        "- Data: `data/reduced` chunks listed under `train` in `data_split_qc_filtered.json`.\n"
        "- Validation: normal-only validation split for threshold calibration.\n"
        "- Models: six candidates from `{original_only, all_features}` × `{W_order_train ∈ {0.1,0.25,0.5}}`.\n"
    )
    lines.append("## 8. Threshold Calibration\n")
    lines.append(
        "Thresholds are quantiles of anomaly scores on **validation-normal only**, saved to "
        "`results/univariate_pattern_thresholds.json` (does not modify `backend/app/thresholds.json`).\n"
    )
    lines.append("## 9. Evaluation Protocol\n")
    lines.append(
        "Strict window-level evaluation on `test-normal` + `attacked_v2` with `y_window` labels (or derived `>=10%` rule). "
        "This mirrors the Chapter 7 style evaluation used for reporting.\n"
    )
    lines.append("## 10. Results\n")
    if not sel_rows:
        lines.append("_No candidate rows were produced (missing models, thresholds, or data)._ \n")
    else:
        lines.append("| model | W_order_train | W_order_score | PS recall | FAR | F1 | BalAcc | recommendation |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|\n")
        for r in sel_rows:
            lines.append(
                f"| {r.get('model_name','')} | {r.get('W_order_train','')} | {r.get('W_order_score','')} | "
                f"{r.get('pattern_shift_recall','')} | {r.get('FAR','')} | {r.get('F1','')} | {r.get('balanced_accuracy','')} | "
                f"{r.get('recommendation','')} |\n"
            )
    lines.append("## 11. Per-Attack Analysis\n")
    lines.append("See `results/per_attack_univariate_pattern_metrics.csv` and the per-attack bar chart.\n")
    lines.append("## 12. Pattern-Shift Analysis\n")
    bps = bl.get("per_attack_p99|pattern_shift_recall", float("nan"))
    if isinstance(bps, float) and np.isfinite(bps):
        lines.append(f"Baseline `pattern_shift` recall @p99 (Chapter 7 extract): {bps:.6f}\n")
    else:
        lines.append("Baseline `pattern_shift` recall @p99: (missing baseline CSV row)\n")
    if best is not None:
        lines.append(
            f"Best candidate in this run (by `pattern_shift` recall): `{best.get('model_name')}` "
            f"with `pattern_shift_recall={best.get('pattern_shift_recall')}`.\n"
        )
    lines.append("## 13. Candidate Selection Decision\n")
    lines.append(
        "Candidates are evaluated against explicit acceptance rules in `model_selection_comparison.csv`. "
        "Even `ACCEPT_CANDIDATE` remains **research-only** and must not be wired into production automatically.\n"
    )
    lines.append("## 14. Deployment Safety Note\n")
    lines.append(
        "Do not replace `backend/app/best_model_qc_filtered.keras`, do not edit `thresholds.json`, and do not change API "
        "routes or official pipelines based on this experiment without a separate production review.\n"
    )
    lines.append("## 15. Future Work\n")
    lines.append(
        "- Contrastive temporal learning on window pairs.\n"
        "- Transformer / attention encoders for long-range interactions.\n"
        "- Multivariate telemetry fusion when additional sensors exist.\n"
    )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("".join(lines), encoding="utf-8")
