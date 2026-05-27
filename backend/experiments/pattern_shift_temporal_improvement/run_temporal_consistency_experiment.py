"""
Research-only: temporal consistency scoring on top of hybrid recon/pred/grad.
Writes ONLY under --output-dir. Does not modify production, api, thresholds, or the .keras file.
"""

from __future__ import annotations

import argparse
import csv
import os
import importlib.util
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

W_TEMP_GRID = (0.1, 0.25, 0.5, 1.0, 2.0)
MAX_ACF_LAG = 20


def _resolve(repo_root: Path, p: str | Path) -> Path:
    path = Path(p)
    return (repo_root / path).resolve() if not path.is_absolute() else path.resolve()


def _safe_out(output_dir: Path, *parts: str) -> Path:
    out = output_dir.joinpath(*parts).resolve()
    root = output_dir.resolve()
    try:
        out.relative_to(root)
    except ValueError as e:
        raise ValueError(f"Refusing to write outside output_dir: {out}") from e
    return out


def _load_eval(repo_root: Path):
    mod_path = repo_root / "backend" / "models" / "evaluate_model_strict_v2.py"
    spec = importlib.util.spec_from_file_location("_pst_eval", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def y_window_from_timestep(y_ts: np.ndarray, min_fraction: float = 0.10) -> np.ndarray:
    if y_ts.ndim != 2:
        raise ValueError(y_ts.shape)
    t = y_ts.shape[1]
    k = int(np.ceil(float(min_fraction) * t))
    k = max(1, min(k, t))
    s = np.sum(y_ts.astype(np.uint8), axis=1)
    return (s >= k).astype(np.uint8)


def load_attacked_npz_fallback(path: Path, ev: Any) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        X = z["X"].astype(np.float32)
        if "y_window" in keys:
            y_w = z["y_window"].astype(np.uint8)
        elif "y_timestep" in keys:
            y_w = y_window_from_timestep(z["y_timestep"].astype(np.uint8))
        else:
            raise KeyError(path)
        attack_type = str(z["attack_type"].item()) if "attack_type" in z else "unknown"
    if X.ndim == 2:
        X = X[..., None]
    meta = {"attack_type": attack_type}
    return X, y_w, meta


def load_attacked_dir(attacked_dir: Path, window_t: int, ev: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    sp, yp, tparts, rej = [], [], [], 0
    for p in sorted(attacked_dir.glob("*.npz")):
        try:
            with np.load(p, allow_pickle=False) as z:
                keys = set(z.files)
            if "y_window" in keys:
                X, y_w, meta = ev.load_attacked_npz(p)
            else:
                X, y_w, meta = load_attacked_npz_fallback(p, ev)
            if X.shape[1] != window_t:
                rej += 1
                continue
            sp.append(X.astype(np.float32))
            yp.append(np.asarray(y_w).astype(np.uint8))
            at = str(meta.get("attack_type", "unknown"))
            tparts.append(np.array([at] * len(y_w), dtype=object))
        except Exception:
            rej += 1
            continue
    if not sp:
        raise RuntimeError("No valid attacked NPZ.")
    return np.concatenate(sp), np.concatenate(yp), np.concatenate(tparts), rej


def hybrid_base_score(
    model: Any, X: np.ndarray, weights: Dict[str, Any], batch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    recon, pred = model.predict(X, verbose=0, batch_size=int(batch_size))
    recon = np.asarray(recon, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    w = weights if isinstance(weights, dict) else {}
    w_recon = float(w.get("W_RECON", 1.0))
    w_pred = float(w.get("W_PRED", 2.0))
    w_grad = float(w.get("W_GRAD", 2.0))
    e_recon = np.mean((X - recon) ** 2, axis=(1, 2))
    dx_true = X[:, 1:, :] - X[:, :-1, :]
    dx_recon = recon[:, 1:, :] - recon[:, :-1, :]
    e_grad = np.mean((dx_true - dx_recon) ** 2, axis=(1, 2))
    t = X.shape[1]
    if pred.ndim == 3 and pred.shape[1] == t - 1:
        y_true = X[:, 1:, :]
        e_pred = np.mean((y_true - pred) ** 2, axis=(1, 2))
    elif pred.ndim == 2:
        pred_exp = pred[:, None, :]
        e_pred = np.mean((X[:, -1:, :] - pred_exp) ** 2, axis=(1, 2))
    elif pred.ndim == 3:
        if pred.shape[1] != 1:
            pred = pred[:, :1, :]
        e_pred = np.mean((X[:, -1:, :] - pred) ** 2, axis=(1, 2))
    else:
        e_pred = np.zeros(X.shape[0], dtype=np.float32)
    base = (w_recon * e_recon + w_pred * e_pred + w_grad * e_grad).astype(np.float32)
    return base, recon


def acf_corrs(x_bt: np.ndarray, max_lag: int) -> np.ndarray:
    """x_bt (B,T) centered internally per row; returns (B, L) Pearson at lags 1..L."""
    B, T = x_bt.shape
    L = min(max_lag, max(0, T - 2))
    if L < 1:
        return np.zeros((B, 1), dtype=np.float64)
    xm = x_bt.astype(np.float64) - x_bt.mean(axis=1, keepdims=True).astype(np.float64)
    out = np.zeros((B, L), dtype=np.float64)
    for k in range(1, L + 1):
        a = xm[:, :-k]
        b = xm[:, k:]
        num = np.sum(a * b, axis=1)
        den = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1) + 1e-12)
        out[:, k - 1] = num / den
    return out


def temporal_consistency_score(X: np.ndarray, recon: np.ndarray, max_lag: int = MAX_ACF_LAG) -> np.ndarray:
    """
    Combined temporal score: mean of autocorrelation MSE and FFT magnitude MSE (per window).
    Uses mean over channels for univariate/multivariate windows.
    """
    x = np.mean(X.astype(np.float64), axis=2)
    r = np.mean(recon.astype(np.float64), axis=2)
    acx = acf_corrs(x, max_lag)
    acr = acf_corrs(r, max_lag)
    mse_ac = np.mean((acx - acr) ** 2, axis=1)
    fx = np.abs(np.fft.rfft(x, axis=1))
    fr = np.abs(np.fft.rfft(r, axis=1))
    mse_sp = np.mean((fx - fr) ** 2, axis=1)
    return (0.5 * mse_ac + 0.5 * mse_sp).astype(np.float64)


def accumulate_scores_for_files(
    model: Any,
    weights: Dict[str, Any],
    normal_dir: Path,
    names: List[str],
    window_t: int,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    bases, temps = [], []
    ok = rej = 0
    for fname in names:
        fp = (normal_dir / fname).resolve()
        if not fp.is_file():
            rej += 1
            continue
        try:
            X = np.load(fp).astype(np.float32)
            if X.ndim == 2:
                X = X[..., None]
            if X.ndim != 3 or X.shape[1] != window_t:
                rej += 1
                continue
            b, r = hybrid_base_score(model, X, weights, batch_size)
            t = temporal_consistency_score(X, r)
            bases.append(b.astype(np.float64))
            temps.append(t)
            ok += 1
        except Exception:
            rej += 1
            continue
    if not bases:
        raise RuntimeError("No windows scored.")
    return np.concatenate(bases), np.concatenate(temps), ok, rej


@dataclass
class CM:
    TP: int
    TN: int
    FP: int
    FN: int


def confusion(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> CM:
    y_true = np.asarray(y_true).astype(np.uint8)
    y_pred = (np.asarray(y_score) > float(thr)).astype(np.uint8)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return CM(tp, tn, fp, fn)


def metrics(cm: CM) -> Dict[str, float]:
    tp, tn, fp, fn = cm.TP, cm.TN, cm.FP, cm.FN
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    bal = 0.5 * (rec + tnr)
    return {"precision": prec, "recall": rec, "far": far, "f1": f1, "balanced_accuracy": bal}


def per_attack_metrics(
    atk_scores: np.ndarray,
    atk_y: np.ndarray,
    atk_type: np.ndarray,
    thr: float,
) -> Dict[str, Dict[str, float]]:
    pred = atk_scores > float(thr)
    out: Dict[str, Dict[str, float]] = {}
    for at in sorted(set(str(x) for x in atk_type.tolist())):
        msk = atk_type == at
        y_sub = atk_y[msk].astype(np.uint8)
        p_sub = pred[msk]
        tp = int(np.sum(p_sub & (y_sub == 1)))
        fn = int(np.sum((~p_sub) & (y_sub == 1)))
        fp = int(np.sum(p_sub & (y_sub == 0)))
        tw = int(np.sum(y_sub == 1))
        rec = float(tp / (tp + fn)) if (tp + fn) else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        out[at] = {"recall": rec, "precision": prec, "f1": f1, "total_positive_windows": float(tw)}
    return out


def write_baseline_csv(ch7: Path, out_csv: Path) -> Dict[str, float]:
    overall = ch7 / "overall_threshold_metrics.csv"
    per_atk = ch7 / "per_attack_threshold_metrics.csv"
    rows_out: List[Dict[str, str]] = []
    extras: Dict[str, float] = {}

    with overall.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["threshold_name"] in ("p99", "best_f1"):
                name = row["threshold_name"]
                rows_out.append(
                    {
                        "metric_group": name,
                        "precision": row["precision"],
                        "recall": row["recall"],
                        "far": row["far"],
                        "f1": row["f1"],
                        "balanced_accuracy": row["balanced_accuracy"],
                        "threshold_value": row["threshold_value"],
                        "source": str(overall),
                    }
                )
                if name == "p99":
                    extras["baseline_p99_far"] = float(row["far"])
                    extras["baseline_p99_f1"] = float(row["f1"])
                    extras["baseline_p99_recall"] = float(row["recall"])
                if name == "best_f1":
                    extras["baseline_best_f1_far"] = float(row["far"])
                    extras["baseline_best_f1_f1"] = float(row["f1"])
                    extras["baseline_best_f1_recall"] = float(row["recall"])

    with per_atk.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["attack_type"] == "pattern_shift" and row["threshold_name"] == "p99":
                rows_out.append(
                    {
                        "metric_group": "pattern_shift_at_p99",
                        "precision": row["precision"],
                        "recall": row["recall"],
                        "far": "",
                        "f1": row["f1"],
                        "balanced_accuracy": "",
                        "threshold_value": row["threshold_value"],
                        "source": str(per_atk),
                    }
                )
                extras["baseline_pattern_shift_recall_p99"] = float(row["recall"])
                extras["baseline_pattern_shift_f1_p99"] = float(row["f1"])
            if row["attack_type"] == "pattern_shift" and row["threshold_name"] == "best_f1":
                extras["baseline_pattern_shift_recall_best_f1_row"] = float(row["recall"])
                extras["baseline_pattern_shift_f1_best_f1_row"] = float(row["f1"])
                rows_out.append(
                    {
                        "metric_group": "pattern_shift_at_best_f1_row",
                        "precision": row["precision"],
                        "recall": row["recall"],
                        "far": "",
                        "f1": row["f1"],
                        "balanced_accuracy": "",
                        "threshold_value": row["threshold_value"],
                        "source": str(per_atk),
                    }
                )

    fieldnames = [
        "metric_group",
        "precision",
        "recall",
        "far",
        "f1",
        "balanced_accuracy",
        "threshold_value",
        "source",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    return extras


def pick_best_w_temp(rows: List[Dict[str, Any]], baseline_far: float) -> Tuple[float, str]:
    """Prefer higher pattern_shift_recall with overall FAR <= baseline_far + 0.01."""
    cap = baseline_far + 0.01
    feasible = [r for r in rows if float(r["overall_far"]) <= cap]
    if feasible:
        best = max(
            feasible,
            key=lambda r: (float(r["pattern_shift_recall"]), -float(r["overall_far"])),
        )
        return float(best["W_TEMP"]), (
            f"Best W_TEMP under FAR cap (baseline p99 FAR + 0.01 = {cap:.6f}): "
            f"W_TEMP={best['W_TEMP']}, pattern_shift_recall={best['pattern_shift_recall']:.6f}, "
            f"overall_FAR={best['overall_far']:.6f}"
        )
    best = max(rows, key=lambda r: float(r["pattern_shift_recall"]))
    return float(best["W_TEMP"]), (
        "No W_TEMP met FAR cap; chose max pattern_shift_recall (FAR may exceed cap). "
        f"W_TEMP={best['W_TEMP']}, pattern_shift_recall={best['pattern_shift_recall']:.6f}, "
        f"overall_FAR={best['overall_far']:.6f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, required=True)
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--thresholds-path", type=str, required=True)
    ap.add_argument("--split-file", type=str, required=True)
    ap.add_argument("--normal-dir", type=str, required=True)
    ap.add_argument("--attacked-dir", type=str, required=True)
    ap.add_argument("--chapter7-results", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    repo_root = Path(args.repo_root).resolve()
    out_dir = _resolve(repo_root, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ch7 = _resolve(repo_root, args.chapter7_results)
    baseline_path = _safe_out(out_dir, "baseline_summary.csv")
    extras = write_baseline_csv(ch7, baseline_path)
    baseline_far = extras["baseline_p99_far"]

    ev = _load_eval(repo_root)
    model_path = _resolve(repo_root, args.model_path)
    thresholds_path = _resolve(repo_root, args.thresholds_path)
    split_file = _resolve(repo_root, args.split_file)
    normal_dir = _resolve(repo_root, args.normal_dir)
    attacked_dir = _resolve(repo_root, args.attacked_dir)

    with thresholds_path.open(encoding="utf-8") as f:
        thr_cfg = json.load(f)
    weights = thr_cfg.get("weights", {})
    if not isinstance(weights, dict):
        weights = {}

    split_obj = json.loads(split_file.read_text(encoding="utf-8"))
    val_names = list(split_obj.get("validation", []))
    test_names = list(split_obj.get("test", []))

    T, C = ev.infer_T_C_from_sample(normal_dir)
    model = ev.load_keras_model_robust(model_path, T, C)

    t0 = time.time()
    warnings.filterwarnings("ignore", category=UserWarning)
    val_b, val_t, _, _ = accumulate_scores_for_files(model, weights, normal_dir, val_names, T, int(args.batch_size))
    test_b, test_t, _, _ = accumulate_scores_for_files(model, weights, normal_dir, test_names, T, int(args.batch_size))
    atk_X, atk_y, atk_type, _ = load_attacked_dir(attacked_dir, T, ev)
    atk_parts_b, atk_parts_t = [], []
    bs = int(args.batch_size)
    for i in range(0, len(atk_X), bs):
        chunk = atk_X[i : i + bs]
        b, r = hybrid_base_score(model, chunk, weights, bs)
        atk_parts_b.append(b.astype(np.float64))
        atk_parts_t.append(temporal_consistency_score(chunk, r))
    atk_b = np.concatenate(atk_parts_b)
    atk_t = np.concatenate(atk_parts_t)

    y_true = np.concatenate([np.zeros(len(test_b), dtype=np.uint8), atk_y.astype(np.uint8)])

    results_rows: List[Dict[str, Any]] = []
    float_rows: List[Dict[str, float]] = []
    per_rows: List[Dict[str, Any]] = []

    for w_temp in W_TEMP_GRID:
        w_temp = float(w_temp)
        val_e = val_b + w_temp * val_t
        thr = float(np.quantile(val_e, 0.99))
        test_e = test_b + w_temp * test_t
        atk_e = atk_b + w_temp * atk_t
        y_score = np.concatenate([test_e, atk_e])
        cm = confusion(y_true, y_score, thr)
        m = metrics(cm)
        pam = per_attack_metrics(atk_e, atk_y, atk_type, thr)
        row_float: Dict[str, Any] = {
            "W_TEMP": w_temp,
            "threshold_used": thr,
            "overall_precision": m["precision"],
            "overall_recall": m["recall"],
            "overall_far": m["far"],
            "overall_f1": m["f1"],
            "overall_balanced_accuracy": m["balanced_accuracy"],
            "pattern_shift_recall": pam.get("pattern_shift", {}).get("recall", 0.0),
            "pattern_shift_f1": pam.get("pattern_shift", {}).get("f1", 0.0),
            "pattern_shift_precision": pam.get("pattern_shift", {}).get("precision", 0.0),
            "noise_recall": pam.get("noise", {}).get("recall", float("nan")),
            "spike_recall": pam.get("spike", {}).get("recall", float("nan")),
            "drift_recall": pam.get("drift", {}).get("recall", float("nan")),
            "freeze_recall": pam.get("freeze", {}).get("recall", float("nan")),
        }
        float_rows.append(dict(row_float))
        csv_row: Dict[str, Any] = {"calibration": "p99_on_validation_enhanced_score"}
        for k, v in row_float.items():
            csv_row[k] = f"{float(v):.12g}" if isinstance(v, (float, np.floating)) else v
        results_rows.append(csv_row)

        for at, pm in pam.items():
            per_rows.append(
                {
                    "W_TEMP": str(w_temp),
                    "attack_type": at,
                    "recall": f"{pm['recall']:.12g}",
                    "precision": f"{pm['precision']:.12g}",
                    "f1": f"{pm['f1']:.12g}",
                    "threshold_used": f"{thr:.12g}",
                }
            )

    best_w, best_note = pick_best_w_temp(float_rows, baseline_far)
    best_ps = max(float_rows, key=lambda r: r["pattern_shift_recall"])["pattern_shift_recall"]
    best_ps_row = max(float_rows, key=lambda r: r["pattern_shift_recall"])
    worth = (
        best_ps > extras["baseline_pattern_shift_recall_p99"] + 0.02
        and float(best_ps_row["overall_far"]) <= baseline_far + 0.015
    )
    needs_retrain = not worth or best_ps < extras["baseline_pattern_shift_recall_p99"] + 0.05

    p_res = _safe_out(out_dir, "temporal_consistency_results.csv")
    keys = list(results_rows[0].keys())
    with p_res.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results_rows)

    p_per = _safe_out(out_dir, "temporal_consistency_per_attack.csv")
    with p_per.open("w", newline="", encoding="utf-8") as f:
        ww = csv.DictWriter(f, fieldnames=["W_TEMP", "attack_type", "recall", "precision", "f1", "threshold_used"])
        ww.writeheader()
        ww.writerows(per_rows)

    elapsed = time.time() - t0
    rep = _safe_out(out_dir, "pattern_shift_temporal_improvement_report.md")
    rep.write_text(
        "\n".join(
            [
                "# Pattern shift — temporal consistency experiment (research)",
                "",
                f"Elapsed scoring time (approx): **{elapsed / 60:.1f} min**",
                "",
                "## 1. Problem",
                "",
                "- **pattern_shift** exhibits low recall under the original hybrid score because the attack often preserves marginal value statistics while altering **temporal ordering**.",
                "- Reconstruction-focused objectives can still fit such windows, so the base anomaly score stays relatively small.",
                "",
                "## 2. Baseline (Chapter 7, read-only)",
                "",
                "Values copied into `baseline_summary.csv` from:",
                f"- `{ch7 / 'overall_threshold_metrics.csv'}`",
                f"- `{ch7 / 'per_attack_threshold_metrics.csv'}`",
                "",
                f"- **p99 (overall):** recall={extras['baseline_p99_recall']:.6f}, FAR={extras['baseline_p99_far']:.6f}, F1={extras['baseline_p99_f1']:.6f}",
                f"- **best_f1 row (overall):** recall={extras['baseline_best_f1_recall']:.6f}, FAR={extras['baseline_best_f1_far']:.6f}, F1={extras['baseline_best_f1_f1']:.6f}",
                f"- **pattern_shift @ p99 calibration:** recall={extras['baseline_pattern_shift_recall_p99']:.6f}, F1={extras['baseline_pattern_shift_f1_p99']:.6f}",
                f"- **pattern_shift @ best_f1 row (Chapter 7 table):** recall={extras.get('baseline_pattern_shift_recall_best_f1_row', float('nan')):.6f}, F1={extras.get('baseline_pattern_shift_f1_best_f1_row', float('nan')):.6f}",
                "",
                "## 3. Temporal consistency experiment",
                "",
                "- **temporal_consistency_score** = 0.5 × MSE(autocorr(X), autocorr(X_hat)) + 0.5 × MSE(|FFT(X)|, |FFT(X_hat)|), averaged over channels.",
                "- **enhanced_score** = base_hybrid + **W_TEMP** × temporal_consistency_score (no separation loss; no retraining).",
                "- For each **W_TEMP**, threshold = **p99** of enhanced scores on **validation-normal** only (same governance idea as Chapter 7).",
                "",
                "See **`temporal_consistency_results.csv`** and **`temporal_consistency_per_attack.csv`**.",
                "",
                "### W_TEMP sweep (this run)",
                "",
                "| W_TEMP | overall_FAR | overall_F1 | overall_recall | bal_acc | pattern_shift_recall | pattern_shift_F1 |",
                "|---:|---:|---:|---:|---:|---:|---:|",
                *[
                    f"| {r['W_TEMP']:.4g} | {r['overall_far']:.6f} | {r['overall_f1']:.6f} | {r['overall_recall']:.6f} | "
                    f"{r['overall_balanced_accuracy']:.6f} | {r['pattern_shift_recall']:.6f} | {r['pattern_shift_f1']:.6f} |"
                    for r in float_rows
                ],
                "",
                "### Trade-off summary",
                "",
                f"- **Best pattern_shift recall in grid:** {best_ps:.6f} (W_TEMP={best_ps_row['W_TEMP']}), FAR={best_ps_row['overall_far']:.6f}, F1={best_ps_row['overall_f1']:.6f}",
                f"- **Heuristic preferred W_TEMP (FAR cap vs baseline p99 FAR):** {best_w} — {best_note}",
                f"- **Worth deploying as experimental overlay?** `{'yes (marginal, monitor FAR)' if worth else 'limited — see recommendation'}`",
                "",
                "## 4. Ablation study discussion",
                "",
                "Ablation study was intentionally not recomputed in this run to avoid retraining or modifying the production model. "
                "The current evaluation therefore focuses on operational robustness, threshold governance, strict window-level evaluation, and per-attack analysis.",
                "",
                "**Future work (training-time, not executed here):** removing or shrinking the prediction head; ablating the gradient term; "
                "training without separation loss; comparing LSTM-only vs GRU-only vs the hybrid stack under identical splits.",
                "",
                "## 5. Future model improvements",
                "",
                "- **Multi-step prediction head** — see `../multi_step_prediction_design.md`",
                "- **Order-prediction auxiliary task** — see `../order_prediction_auxiliary_task_design.md`",
                "- **Contrastive temporal learning** (future): learn embeddings invariant to benign noise but sensitive to permutations / phase breaks.",
                "",
                "## 6. Recommendation",
                "",
                "- If **`temporal_consistency_results.csv`** shows materially higher **pattern_shift_recall** at acceptable **FAR** vs baseline p99 FAR, a **non-production scoring branch** could trial this term before any API merge.",
                f"- If gains are small or FAR rises sharply, **retraining objectives** (multi-step / order / contrastive) are likely required — post-hoc scoring alone may not suffice.",
                "",
                f"_Auto flags: needs_architecture_change_likely={str(needs_retrain).lower()}_",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("Done.", baseline_path, p_res, p_per, rep, flush=True)


if __name__ == "__main__":
    main()
