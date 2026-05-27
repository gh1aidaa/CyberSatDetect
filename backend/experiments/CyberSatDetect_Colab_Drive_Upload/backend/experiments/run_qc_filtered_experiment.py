"""
QC-filtered training + evaluation experiment.

Writes ONLY under backend/experiments/qc_filtered_run[_NNN]/ — never modifies app/, thresholds.json, or existing models.

Imports training logic from backend.models.train_hybrid_model without modifying that file (runtime path monkeypatch).
"""

from __future__ import annotations

import csv
import json
import os
import sys
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Repo root (parents[2] from backend/experiments/)
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
MODELS_DIR = BACKEND / "models"
EXP_ROOT_BASE = BACKEND / "experiments"

if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))


def _allocate_run_dir() -> Path:
    """Pick qc_filtered_run or qc_filtered_run_001 ... if directory exists."""
    stem = "qc_filtered_run"
    candidate = EXP_ROOT_BASE / stem
    if not candidate.exists():
        return candidate
    i = 1
    while True:
        cand = EXP_ROOT_BASE / f"{stem}_{i:03d}"
        if not cand.exists():
            return cand
        i += 1
        if i > 999:
            raise RuntimeError("Could not allocate experiment directory")


def _validate_write_path(p: Path) -> None:
    ep = p.resolve()
    base = EXP_ROOT_BASE.resolve()
    if base not in ep.parents and ep != base:
        raise ValueError(f"Refusing to write outside experiments: {ep}")


def run_training_monkeypatch(exp_dir: Path) -> None:
    import train_hybrid_model as thm

    model_d = exp_dir / "model"
    res_d = exp_dir / "results"
    logs_d = exp_dir / "logs"
    for d in (model_d, res_d, logs_d):
        d.mkdir(parents=True, exist_ok=True)

    split_path = BACKEND / "config" / "data_split_qc_filtered.json"
    if not split_path.exists():
        raise FileNotFoundError(split_path)

    thm.ROOT = ROOT
    thm.BACKEND = BACKEND
    thm.APP_DIR = model_d
    thm.DATA_DIR = ROOT / "data" / "reduced"
    thm.SPLIT_FILE = split_path
    thm.MODEL_OUT = model_d / "final_model_qc_filtered.keras"
    thm.THRESH_OUT = res_d / "thresholds_qc_filtered.json"
    thm.CHECKPOINT_MODEL = model_d / "checkpoint_model.keras"
    thm.CHECKPOINT_INFO = model_d / "checkpoint_info.json"

    for p in (thm.MODEL_OUT, thm.THRESH_OUT, thm.CHECKPOINT_MODEL, thm.CHECKPOINT_INFO):
        _validate_write_path(p)

    log_path = logs_d / "training.log"
    _validate_write_path(log_path)

    class Tee:
        def __init__(self, *files):
            self.files = files

        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()

        def flush(self):
            for f in self.files:
                f.flush()

    tee = Tee(sys.stdout, open(log_path, "w", encoding="utf-8"))
    old_stdout = sys.stdout
    sys.stdout = tee
    try:
        thm.main()
    finally:
        sys.stdout = old_stdout
        tee.files[1].close()

    best_src = model_d / "best_model.keras"
    best_dst = model_d / "best_model_qc_filtered.keras"
    if best_src.exists():
        shutil.copy2(best_src, best_dst)
        _validate_write_path(best_dst)


def generate_attacked_for_qc_test(exp_dir: Path) -> Path:
    """Same attacked_v2 protocol; output only under experiment dir (read-only on data/reduced)."""
    out_dir = exp_dir / "attacked_qc_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    _validate_write_path(out_dir)
    script = MODELS_DIR / "regenerate_attacked_dataset.py"
    cmd = [
        sys.executable,
        str(script),
        "--normal-dir",
        str(ROOT / "data" / "reduced"),
        "--split-json",
        str(BACKEND / "config" / "data_split_qc_filtered.json"),
        "--output-dir",
        str(out_dir),
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
    return out_dir


def load_split(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_npy(path: Path) -> np.ndarray:
    x = np.load(path).astype(np.float32)
    if x.ndim == 2:
        x = x[..., None]
    return x


def run_evaluation(exp_dir: Path) -> None:
    import tensorflow as tf

    from train_hybrid_model import compute_scores
    from evaluate_model_strict_v2 import (
        load_attacked_npz,
        iter_attacked_npz,
        confusion_at_threshold,
        metrics_from_confusion,
        compute_curves_and_auc,
    )

    model_dir = exp_dir / "model"
    res_dir = exp_dir / "results"
    logs_dir = exp_dir / "logs"
    model_path = model_dir / "best_model_qc_filtered.keras"
    if not model_path.exists():
        model_path = model_dir / "best_model.keras"
    thr_path = res_dir / "thresholds_qc_filtered.json"

    split_json = BACKEND / "config" / "data_split_qc_filtered.json"
    normal_dir = ROOT / "data" / "reduced"
    attacked_dir = exp_dir / "attacked_qc_test"

    for p in (model_path, thr_path, split_json, normal_dir, attacked_dir):
        if not Path(p).exists():
            raise FileNotFoundError(p)

    split = load_split(split_json)
    test_names = list(split["test"])

    with thr_path.open("r", encoding="utf-8") as f:
        thr_doc = json.load(f)
    stat_thr = thr_doc["thresholds"]

    # Infer T, C from first test file
    fp0 = normal_dir / test_names[0]
    X0 = load_npy(fp0)
    T, C = int(X0.shape[1]), int(X0.shape[2])

    model = tf.keras.models.load_model(model_path, custom_objects={"tf": tf}, compile=False)

    # ----- Normal TEST scores (weighted, same as training compute_scores) -----
    normal_scores_parts: List[np.ndarray] = []
    for fname in test_names:
        fp = normal_dir / fname
        X = load_npy(fp)
        if X.shape[1] != T:
            raise ValueError(f"T mismatch {fp}: {X.shape}")
        s = compute_scores(model, X)
        normal_scores_parts.append(s)
    scores_normal = np.concatenate(normal_scores_parts).astype(np.float64)

    # ----- Attacked: all NPZ under experiment attacked_qc_test (built from QC test split) -----
    attacked_scores_parts: List[np.ndarray] = []
    attacked_labels_parts: List[np.ndarray] = []
    for npz_path in iter_attacked_npz(attacked_dir):
        X_att, y_w, meta = load_attacked_npz(npz_path)
        if X_att.shape[1] != T:
            raise ValueError(f"T mismatch {npz_path}: {X_att.shape}")
        s = compute_scores(model, X_att)
        attacked_scores_parts.append(s)
        attacked_labels_parts.append(y_w.astype(np.uint8))

    if not attacked_scores_parts:
        raise RuntimeError("No attacked NPZ found under experiment attacked_qc_test/.")

    scores_attacked = np.concatenate(attacked_scores_parts).astype(np.float64)
    y_attacked = np.concatenate(attacked_labels_parts).astype(np.uint8)

    y_true = np.concatenate(
        [np.zeros(len(scores_normal), dtype=np.uint8), y_attacked]
    )
    y_score = np.concatenate([scores_normal, scores_attacked])

    curves_info, curves = compute_curves_and_auc(y_true, y_score)

    # Statistical thresholds from VALIDATION (calibrated during training — already in JSON)
    candidates: List[Tuple[str, float]] = [
        ("3sigma", float(stat_thr["3sigma"])),
        ("p99", float(stat_thr["p99"])),
        ("p995", float(stat_thr["p995"])),
        ("p997", float(stat_thr["p997"])),
    ]

    candidates.append(("best_f1", float(curves_info["best_f1_threshold"])))
    candidates.append(("best_youden_j", float(curves_info["best_youden_threshold"])))

    def thr_for_far(target_far: float) -> float:
        q = max(0.0, min(1.0, 1.0 - float(target_far)))
        return float(np.quantile(scores_normal, q))

    candidates.append(("far_le_1pct", thr_for_far(0.01)))
    candidates.append(("far_le_0.5pct", thr_for_far(0.005)))

    rows = []
    best_by_f1 = None
    best_by_acc = None
    for name, thr in candidates:
        cm = confusion_at_threshold(y_true, y_score, thr)
        m = metrics_from_confusion(cm)
        row = {
            "name": name,
            "threshold": float(thr),
            **m,
            "roc_auc": float(curves_info["roc_auc"]),
            "pr_auc": float(curves_info["pr_auc"]),
            "TP": cm.TP,
            "TN": cm.TN,
            "FP": cm.FP,
            "FN": cm.FN,
        }
        rows.append(row)
        if best_by_f1 is None or row["f1"] > best_by_f1["f1"]:
            best_by_f1 = row
        if best_by_acc is None or row["accuracy"] > best_by_acc["accuracy"]:
            best_by_acc = row

    metrics_csv = res_dir / "metrics_qc_filtered.csv"
    _validate_write_path(metrics_csv)
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
                "fpr",
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
                    f"{r['fpr']:.10g}",
                    f"{r['roc_auc']:.10g}",
                    f"{r['pr_auc']:.10g}",
                    r["TP"],
                    r["TN"],
                    r["FP"],
                    r["FN"],
                ]
            )

    ck_info_path = model_dir / "checkpoint_info.json"
    best_val_loss = None
    if ck_info_path.exists():
        try:
            best_val_loss = float(json.loads(ck_info_path.read_text(encoding="utf-8")).get("best_val_loss"))
        except Exception:
            pass

    eval_summary = {
        "experiment_dir": str(exp_dir.resolve()),
        "split_json": str(split_json.resolve()),
        "model": str(model_path.resolve()),
        "thresholds_source": str(thr_path.resolve()),
        "attacked_dir": str(attacked_dir.resolve()),
        "best_val_loss_from_checkpoint": best_val_loss,
        "counts": {
            "train_files": len(split["train"]),
            "validation_files": len(split["validation"]),
            "test_files_used_normal": len(test_names),
            "normal_windows": int(len(scores_normal)),
            "attacked_windows": int(len(scores_attacked)),
            "attacked_anomaly_windows": int(y_attacked.sum()),
        },
        "validation_calibration_stats": {
            "mean": thr_doc.get("mean"),
            "std": thr_doc.get("std"),
            "thresholds": stat_thr,
        },
        "curves": {
            "roc_auc": float(curves_info["roc_auc"]),
            "pr_auc": float(curves_info["pr_auc"]),
        },
        "best_operating_point_by_f1_among_candidates": best_by_f1,
        "best_operating_point_by_accuracy_among_candidates": best_by_acc,
        "all_candidates": rows,
    }

    eval_json = res_dir / "evaluation_qc_filtered.json"
    _validate_write_path(eval_json)
    eval_json.write_text(json.dumps(eval_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_txt = logs_dir / "summary_qc_filtered.txt"
    _validate_write_path(summary_txt)
    bf = best_by_f1 or rows[0]
    lines = [
        "=== QC-filtered experiment summary ===",
        f"Experiment dir: {exp_dir}",
        f"Train / Val / Test file counts: {len(split['train'])} / {len(split['validation'])} / {len(split['test'])}",
        f"Best validation loss (checkpoint): {best_val_loss}",
        f"Thresholds (validation-normal): {json.dumps(stat_thr, indent=2)}",
        f"Best threshold (by F1 among candidates, same protocol as strict v2): {bf.get('name')} = {bf.get('threshold')}",
        f"Accuracy: {bf.get('accuracy')} | Balanced acc: {bf.get('balanced_accuracy')}",
        f"Precision: {bf.get('precision')} | Recall: {bf.get('recall')} | F1: {bf.get('f1')} | FAR: {bf.get('far')}",
        f"ROC-AUC: {curves_info['roc_auc']} | PR-AUC: {curves_info['pr_auc']}",
        f"Model: {model_dir / 'best_model_qc_filtered.keras'}",
        f"Results: {eval_json}",
        "",
    ]
    summary_txt.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))


def main():
    exp_dir = _allocate_run_dir()
    exp_dir.mkdir(parents=True, exist_ok=True)
    _validate_write_path(exp_dir / "results" / "evaluation_qc_filtered.json")

    marker = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "path": str(exp_dir)}
    (exp_dir / "run_manifest.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")

    os.chdir(ROOT)
    run_training_monkeypatch(exp_dir)
    generate_attacked_for_qc_test(exp_dir)
    run_evaluation(exp_dir)


if __name__ == "__main__":
    main()
