"""
Regenerate attacked_v2 for QC-filtered test split (NEW output dir) and re-run evaluation.

Strict safety:
- Never modify/delete/move any existing dataset under data/.
- Writes ONLY under this experiment directory.

Inputs:
- backend/app/best_model_qc_filtered.keras
- backend/app/thresholds_qc_filtered.json
- backend/config/data_split_qc_filtered.json
- data/reduced/

Outputs (all NEW):
- attacked_qc_v2/  (new NPZs + generation_metadata.json)
- results/evaluation_qc_filtered.json
- results/metrics_qc_filtered.csv
- results/threshold_comparison.csv (strict_v2-style four policy points)
- results/evaluation_summary_strict_style.json (ROC/PR for figure subtitle)
- results/01_threshold_policy_bar_compare_qc_regen.png (Recall/FAR bars + F1 line)
- results/confusion_matrix_qc_filtered.json
- logs/evaluation.log

Use `--evaluate-only` to skip regeneration when attacked_qc_v2/ already exists.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
EXP_DIR = Path(__file__).resolve().parent
ATTACKED_OUT = EXP_DIR / "attacked_qc_v2"
RESULTS_DIR = EXP_DIR / "results"
LOGS_DIR = EXP_DIR / "logs"

MODEL_PATH = BACKEND / "app" / "best_model_qc_filtered.keras"
THRESHOLDS_PATH = BACKEND / "app" / "thresholds_qc_filtered.json"
SPLIT_JSON = BACKEND / "config" / "data_split_qc_filtered.json"
NORMAL_DIR = ROOT / "data" / "reduced"


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _ensure_within_experiments(p: Path) -> None:
    base = (BACKEND / "experiments").resolve()
    rp = p.resolve()
    if base not in rp.parents and rp != base:
        raise ValueError(f"Refusing to write outside backend/experiments: {rp}")


def _tee_stdout(log_file: Path):
    class Tee:
        def __init__(self, *files):
            self.files = files

        def write(self, s):
            for f in self.files:
                f.write(s)
                f.flush()

        def flush(self):
            for f in self.files:
                f.flush()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_within_experiments(LOGS_DIR)
    log_fh = log_file.open("w", encoding="utf-8")
    tee = Tee(sys.stdout, log_fh)
    return tee, log_fh


def regenerate_attacked() -> int:
    """
    Uses the project's original generator script with explicit paths,
    but writes to ATTACKED_OUT (new directory) only.
    """
    ATTACKED_OUT.mkdir(parents=True, exist_ok=True)
    _ensure_within_experiments(ATTACKED_OUT)

    models_dir = BACKEND / "models"
    script = models_dir / "regenerate_attacked_dataset.py"
    if not script.exists():
        raise FileNotFoundError(script)

    import subprocess

    cmd = [
        sys.executable,
        str(script),
        "--normal-dir",
        str(NORMAL_DIR),
        "--split-json",
        str(SPLIT_JSON),
        "--output-dir",
        str(ATTACKED_OUT),
        "--window-size",
        "100",
        "--stride",
        "50",
        "--seed",
        "42",
        "--split-key",
        "test",
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    return len(list(ATTACKED_OUT.glob("*.npz")))


def evaluate(attacked_dir: Path) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_within_experiments(RESULTS_DIR)
    _ensure_within_experiments(LOGS_DIR)

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
        thresholds_from_normal_scores,
    )

    for p in (MODEL_PATH, THRESHOLDS_PATH, SPLIT_JSON, NORMAL_DIR, attacked_dir):
        if not p.exists():
            raise FileNotFoundError(p)

    log_path = LOGS_DIR / "evaluation.log"
    _ensure_within_experiments(log_path)
    tee, log_fh = _tee_stdout(log_path)
    old_stdout = sys.stdout
    sys.stdout = tee
    try:
        print("=== QC Filtered Evaluation (regen attacked_v2) ===")
        print("started_at:", _now_utc())
        print("model:", str(MODEL_PATH))
        print("thresholds:", str(THRESHOLDS_PATH))
        print("split_json:", str(SPLIT_JSON))
        print("normal_dir:", str(NORMAL_DIR))
        print("attacked_dir:", str(attacked_dir))

        thr_doc = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
        thr_map = dict(thr_doc.get("thresholds", {}))
        thr_keys = ["3sigma", "p99", "p995", "p997"]
        for k in thr_keys:
            if k not in thr_map:
                raise KeyError(f"Missing threshold '{k}' in {THRESHOLDS_PATH}")

        X_sample = load_windows_npy(next(NORMAL_DIR.glob("chunk_*.npy")))
        T, C = int(X_sample.shape[1]), int(X_sample.shape[2])
        model = load_keras_model_robust(MODEL_PATH, T, C)

        test_names = load_split_filenames(SPLIT_JSON, "test")
        print("normal test files:", len(test_names))

        normal_scores_parts: List[np.ndarray] = []
        normal_failed = 0
        normal_windows = 0
        for fname in test_names:
            fp = (NORMAL_DIR / fname).resolve()
            try:
                X = load_windows_npy(fp)
                s = compute_scores_strict(model, X, batch_size=256)
                normal_scores_parts.append(np.asarray(s, dtype=np.float64))
                normal_windows += int(len(s))
            except Exception as e:
                normal_failed += 1
                print("[NORMAL FAIL]", fname, type(e).__name__, str(e))

        scores_normal = np.concatenate(normal_scores_parts).astype(np.float64)

        attacked_npz = sorted(attacked_dir.glob("*.npz"))
        print("attacked npz files:", len(attacked_npz))
        attacked_scores_parts: List[np.ndarray] = []
        attacked_labels_parts: List[np.ndarray] = []
        attacked_failed = 0
        attacked_windows = 0
        for p in attacked_npz:
            try:
                X_att, y_w, _meta = load_attacked_npz(p)
                s = compute_scores_strict(model, X_att, batch_size=256)
                attacked_scores_parts.append(np.asarray(s, dtype=np.float64))
                attacked_labels_parts.append(np.asarray(y_w, dtype=np.uint8))
                attacked_windows += int(len(y_w))
            except Exception as e:
                attacked_failed += 1
                print("[ATTACK FAIL]", p.name, type(e).__name__, str(e))

        scores_attacked = np.concatenate(attacked_scores_parts).astype(np.float64)
        y_attacked = np.concatenate(attacked_labels_parts).astype(np.uint8)

        y_true = np.concatenate([np.zeros(len(scores_normal), dtype=np.uint8), y_attacked])
        y_score = np.concatenate([scores_normal, scores_attacked])

        curves_info, _curves = compute_curves_and_auc(y_true, y_score)
        roc_auc = float(curves_info["roc_auc"])
        pr_auc = float(curves_info["pr_auc"])

        rows = []
        best_by_acc = None
        for name in thr_keys:
            thr = float(thr_map[name])
            cm = confusion_at_threshold(y_true, y_score, thr)
            m = metrics_from_confusion(cm)
            row = {
                "name": name,
                "threshold": thr,
                **m,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "TP": cm.TP,
                "TN": cm.TN,
                "FP": cm.FP,
                "FN": cm.FN,
            }
            rows.append(row)
            if best_by_acc is None or row["accuracy"] > best_by_acc["accuracy"]:
                best_by_acc = row

        metrics_csv = RESULTS_DIR / "metrics_qc_filtered.csv"
        _ensure_within_experiments(metrics_csv)
        with metrics_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "name",
                    "threshold",
                    "accuracy",
                    "balanced_accuracy",
                    "precision",
                    "recall",
                    "tpr",
                    "f1",
                    "far",
                    "roc_auc",
                    "pr_auc",
                    "TP",
                    "TN",
                    "FP",
                    "FN",
                ]
            )
            for r in rows:
                w.writerow(
                    [
                        r["name"],
                        f"{r['threshold']:.12g}",
                        f"{r['accuracy']:.10g}",
                        f"{r['balanced_accuracy']:.10g}",
                        f"{r['precision']:.10g}",
                        f"{r['recall']:.10g}",
                        f"{r['tpr']:.10g}",
                        f"{r['f1']:.10g}",
                        f"{r['far']:.10g}",
                        f"{r['roc_auc']:.10g}",
                        f"{r['pr_auc']:.10g}",
                        r["TP"],
                        r["TN"],
                        r["FP"],
                        r["FN"],
                    ]
                )

        # Strict_v2-style policy comparison (same four points as ML engineering bar chart).
        thr_stats = thresholds_from_normal_scores(scores_normal)
        p99_derived = float(thr_stats["p99"])

        def thr_for_far(target_far: float) -> float:
            q = max(0.0, min(1.0, 1.0 - float(target_far)))
            return float(np.quantile(scores_normal, q))

        policy_specs = [
            ("p99", p99_derived),
            ("best_f1", float(curves_info["best_f1_threshold"])),
            ("far_le_1pct", thr_for_far(0.01)),
            ("far_le_0.5pct", thr_for_far(0.005)),
        ]
        policy_rows = []
        for pname, thr in policy_specs:
            cm_p = confusion_at_threshold(y_true, y_score, thr)
            m_p = metrics_from_confusion(cm_p)
            policy_rows.append(
                {
                    "name": pname,
                    "threshold": thr,
                    **m_p,
                    "TP": cm_p.TP,
                    "TN": cm_p.TN,
                    "FP": cm_p.FP,
                    "FN": cm_p.FN,
                }
            )

        tc_csv = RESULTS_DIR / "threshold_comparison.csv"
        _ensure_within_experiments(tc_csv)
        with tc_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "name",
                    "threshold",
                    "accuracy",
                    "balanced_accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "tnr",
                    "far",
                    "fnr",
                    "TP",
                    "TN",
                    "FP",
                    "FN",
                ]
            )
            for r in policy_rows:
                w.writerow(
                    [
                        r["name"],
                        f"{r['threshold']:.10g}",
                        f"{r['accuracy']:.10g}",
                        f"{r['balanced_accuracy']:.10g}",
                        f"{r['precision']:.10g}",
                        f"{r['recall']:.10g}",
                        f"{r['f1']:.10g}",
                        f"{r['tnr']:.10g}",
                        f"{r['far']:.10g}",
                        f"{r['fnr']:.10g}",
                        r["TP"],
                        r["TN"],
                        r["FP"],
                        r["FN"],
                    ]
                )

        summ_json = RESULTS_DIR / "evaluation_summary_strict_style.json"
        _ensure_within_experiments(summ_json)
        summ_json.write_text(
            json.dumps({"curves": {"roc_auc": roc_auc, "pr_auc": pr_auc}}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        try:
            import generate_ml_engineering_figures as gm  # type: ignore

            plt = gm._ensure_matplotlib()
            gm._style(plt)
            fig_path = RESULTS_DIR / "01_threshold_policy_bar_compare_qc_regen.png"
            _ensure_within_experiments(fig_path)
            gm.plot_threshold_policy(
                plt,
                fig_path,
                csv_path=tc_csv,
                summary_json=summ_json,
                title="QC-filtered regen attacks — threshold policy (p99 vs best-F1 vs FAR constraints)",
            )
            print("Wrote threshold policy figure:", str(fig_path))
        except Exception as e:
            print("[WARN] Could not render threshold policy figure:", type(e).__name__, str(e))

        best = best_by_acc or rows[0]
        cm_json = RESULTS_DIR / "confusion_matrix_qc_filtered.json"
        _ensure_within_experiments(cm_json)
        cm_json.write_text(
            json.dumps(
                {
                    "best_threshold_name": best["name"],
                    "threshold": best["threshold"],
                    "TP": best["TP"],
                    "TN": best["TN"],
                    "FP": best["FP"],
                    "FN": best["FN"],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        eval_json = RESULTS_DIR / "evaluation_qc_filtered.json"
        _ensure_within_experiments(eval_json)
        eval_json.write_text(
            json.dumps(
                {
                    "started_at": _now_utc(),
                    "model": str(MODEL_PATH),
                    "thresholds_file": str(THRESHOLDS_PATH),
                    "split_json": str(SPLIT_JSON),
                    "attacked_dir": str(attacked_dir),
                    "counts": {
                        "normal_test_files": len(test_names),
                        "normal_test_files_failed": int(normal_failed),
                        "attacked_npz_files": len(attacked_npz),
                        "attacked_npz_files_failed": int(attacked_failed),
                        "normal_windows": int(normal_windows),
                        "attacked_windows": int(attacked_windows),
                        "attacked_anomaly_windows": int(int(y_attacked.sum())),
                    },
                    "thresholds_used": {k: float(thr_map[k]) for k in thr_keys},
                    "metrics_by_threshold": rows,
                    "best_threshold_by_accuracy": best_by_acc,
                    "curves": {"roc_auc": roc_auc, "pr_auc": pr_auc},
                    "artifacts": {
                        "evaluation_qc_filtered_json": str(eval_json),
                        "metrics_qc_filtered_csv": str(metrics_csv),
                        "confusion_matrix_qc_filtered_json": str(cm_json),
                        "evaluation_log": str(log_path),
                        "threshold_comparison_csv": str(tc_csv),
                        "evaluation_summary_strict_style_json": str(summ_json),
                        "threshold_policy_bar_png": str(
                            RESULTS_DIR / "01_threshold_policy_bar_compare_qc_regen.png"
                        ),
                    },
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print("finished_at:", _now_utc())
        print("ROC-AUC:", roc_auc, "PR-AUC:", pr_auc)
        print("Best threshold by accuracy:", best["name"], "thr=", best["threshold"])
        print("Output dir:", str(EXP_DIR))

    finally:
        sys.stdout = old_stdout
        try:
            log_fh.close()
        except Exception:
            pass


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip attacked NPZ regeneration; only score model and write results/ artifacts.",
    )
    args = ap.parse_args()

    # Make relative imports consistent.
    os.chdir(ROOT)
    if not args.evaluate_only:
        n_npz = regenerate_attacked()
        print("Regenerated attacked NPZ files:", n_npz)
    else:
        if not ATTACKED_OUT.exists() or not any(ATTACKED_OUT.glob("*.npz")):
            raise FileNotFoundError(f"--evaluate-only requires existing NPZs under {ATTACKED_OUT}")
        print("Skipping regenerate_attacked (--evaluate-only). Using:", str(ATTACKED_OUT))
    evaluate(ATTACKED_OUT)


if __name__ == "__main__":
    main()

