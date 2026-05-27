"""
CyberSatDetect / NebulaSec — isolated Validation & Evaluation pipeline.

Reads only configured inputs; writes ONLY under --output-dir.
Does not train models, does not modify thresholds.json, api, or .env.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Output safety
# ---------------------------------------------------------------------------
def _resolve_under(repo_root: Path, p: str | Path) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    else:
        path = path.resolve()
    return path


def _safe_out(output_dir: Path, *parts: str) -> Path:
    out = output_dir.joinpath(*parts).resolve()
    root = output_dir.resolve()
    try:
        out.relative_to(root)
    except ValueError as e:
        raise ValueError(f"Refusing to write outside output_dir: {out}") from e
    return out


def _load_eval_strict(repo_root: Path):
    mod_path = repo_root / "backend" / "models" / "evaluate_model_strict_v2.py"
    if not mod_path.is_file():
        raise FileNotFoundError(mod_path)
    spec = importlib.util.spec_from_file_location("_val_eval_strict", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _maybe_plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception as e:
        warnings.warn(f"matplotlib unavailable: {e}")
        return None


def compute_scores_hybrid_weighted(
    model: Any,
    X: np.ndarray,
    weights: Dict[str, Any],
    batch_size: int = 256,
) -> np.ndarray:
    """Same hybrid score as production inference (no separation term)."""
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

    return (w_recon * e_recon + w_pred * e_pred + w_grad * e_grad).astype(np.float32)


@dataclass(frozen=True)
class Confusion:
    TP: int
    TN: int
    FP: int
    FN: int


def confusion_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> Confusion:
    y_true = np.asarray(y_true).astype(np.uint8)
    y_pred = (np.asarray(y_score) > float(thr)).astype(np.uint8)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return Confusion(TP=tp, TN=tn, FP=fp, FN=fn)


def metrics_from_confusion(cm: Confusion) -> Dict[str, float]:
    tp, tn, fp, fn = cm.TP, cm.TN, cm.FP, cm.FN
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    tpr = recall
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    bal_acc = 0.5 * (recall + tnr)
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "tpr": float(tpr),
        "far": float(far),
        "fpr": float(far),
        "tnr": float(tnr),
        "specificity": float(tnr),
        "f1": float(f1),
        "balanced_accuracy": float(bal_acc),
    }


def quantiles_from_scores(scores: np.ndarray) -> Dict[str, float]:
    s = np.asarray(scores, dtype=np.float64)
    mean = float(s.mean())
    std = float(s.std())
    return {
        "p95": float(np.quantile(s, 0.95)),
        "p97": float(np.quantile(s, 0.97)),
        "p99": float(np.quantile(s, 0.99)),
        "p99.5": float(np.quantile(s, 0.995)),
        "p99.7": float(np.quantile(s, 0.997)),
        "3sigma": float(mean + 3.0 * std),
        "mean": mean,
        "std": std,
    }


def official_threshold_map(cfg: Dict[str, Any]) -> Dict[str, float]:
    raw = cfg.get("thresholds", {})
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in raw.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    canon = {}
    if "p95" in out:
        canon["p95"] = out["p95"]
    if "p97" in out:
        canon["p97"] = out["p97"]
    if "p99" in out:
        canon["p99"] = out["p99"]
    if "p99.5" in out:
        canon["p99.5"] = out["p99.5"]
    elif "p995" in out:
        canon["p99.5"] = out["p995"]
    if "p99.7" in out:
        canon["p99.7"] = out["p99.7"]
    elif "p997" in out:
        canon["p99.7"] = out["p997"]
    if "3sigma" in out:
        canon["3sigma"] = out["3sigma"]
    if "best_f1" in out:
        canon["best_f1"] = out["best_f1"]
    return canon


def score_normal_split(
    ev: Any,
    model: Any,
    weights: Dict[str, Any],
    normal_dir: Path,
    names: List[str],
    window_t: int,
    batch_size: int,
    max_files: int = 0,
) -> Tuple[np.ndarray, int, int]:
    """Returns (scores, n_ok_files, n_rejected). If max_files>0, only first N listed files are tried."""
    if max_files and max_files > 0:
        names = names[: int(max_files)]
    parts: List[np.ndarray] = []
    ok = 0
    rej = 0
    for fname in names:
        fp = (normal_dir / fname).resolve()
        if not fp.is_file():
            rej += 1
            continue
        try:
            x = ev.load_windows_npy(fp)
            if x.shape[1] != window_t:
                rej += 1
                continue
            s = compute_scores_hybrid_weighted(model, x, weights, batch_size=batch_size)
            parts.append(s.astype(np.float64))
            ok += 1
        except Exception:
            rej += 1
            continue
    if not parts:
        raise RuntimeError("No valid normal windows scored.")
    return np.concatenate(parts), ok, rej


def load_attacked_all(
    ev: Any,
    model: Any,
    weights: Dict[str, Any],
    attacked_dir: Path,
    window_t: int,
    batch_size: int,
    max_files: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """scores, y_window, attack_type per window, n_rejected files."""
    files = sorted(attacked_dir.glob("*.npz"))
    if max_files and max_files > 0:
        files = files[: int(max_files)]
    sparts: List[np.ndarray] = []
    yparts: List[np.ndarray] = []
    tparts: List[np.ndarray] = []
    rej = 0
    for p in files:
        try:
            x_att, y_w, meta = ev.load_attacked_npz(p)
            if x_att.shape[1] != window_t:
                rej += 1
                continue
            s = compute_scores_hybrid_weighted(model, x_att, weights, batch_size=batch_size)
            sparts.append(s.astype(np.float64))
            yparts.append(np.asarray(y_w).astype(np.uint8))
            at = str(meta.get("attack_type", "unknown"))
            tparts.append(np.array([at] * int(len(s)), dtype=object))
        except Exception:
            rej += 1
            continue
    if not sparts:
        raise RuntimeError("No attacked_v2 windows scored.")
    return (
        np.concatenate(sparts),
        np.concatenate(yparts),
        np.concatenate(tparts),
        rej,
    )


def sweep_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    grid_n: int = 500,
) -> Tuple[np.ndarray, List[Dict[str, Any]], float, float]:
    """Linear sweep between min and max score."""
    y_true = np.asarray(y_true).astype(np.uint8)
    y_score = np.asarray(y_score).astype(np.float64)
    lo = float(np.min(y_score))
    hi = float(np.max(y_score))
    if hi <= lo:
        hi = lo + 1e-12
    thrs = np.linspace(lo, hi, int(grid_n))
    rows: List[Dict[str, Any]] = []
    best_f1 = -1.0
    best_thr = float("nan")
    for thr in thrs:
        cm = confusion_at_threshold(y_true, y_score, float(thr))
        m = metrics_from_confusion(cm)
        row = {"threshold": float(thr), **m, "tp": cm.TP, "tn": cm.TN, "fp": cm.FP, "fn": cm.FN}
        rows.append(row)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thr = float(thr)
    return thrs, rows, best_f1, best_thr


def plot_confusion(cm: Confusion, title: str, out_path: Path, plt: Any) -> None:
    mat = np.array([[cm.TN, cm.FP], [cm.FN, cm.TP]], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(title)
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, str(int(v)), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def find_ablation_tables(repo_root: Path) -> List[Path]:
    hits: List[Path] = []
    root = repo_root / "backend" / "experiments"
    if root.is_dir():
        for p in root.rglob("*.csv"):
            try:
                with p.open("r", encoding="utf-8", errors="ignore") as f:
                    head = f.readline().lower()
                if "wrecon" in head or "ablation" in head or "without" in head:
                    hits.append(p)
            except OSError:
                continue
    ab = repo_root / "ablation_study" / "results.csv"
    if ab.is_file() and ab not in hits:
        hits.append(ab)
    return hits


def ablation_has_data(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            rows = list(r)
        return len(rows) > 0 and any(any((v or "").strip() for v in row.values()) for row in rows)
    except OSError:
        return False


def parse_sections_arg(raw: str) -> List[int]:
    """
    Contiguous pipeline sections 1..k only (k<=7).
    - 'all' -> [1,2,3,4,5,6,7]
    - '1' -> [1]
    - '1,2,3' -> [1,2,3]
    """
    t = raw.strip().lower()
    if t == "all":
        return [1, 2, 3, 4, 5, 6, 7]
    parts = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    if not parts or min(parts) < 1 or max(parts) > 7:
        raise ValueError("--sections must list integers in 1..7, or 'all'")
    expected = list(range(1, max(parts) + 1))
    if parts != expected:
        raise ValueError("--sections must be contiguous from 1 (e.g. 1,2,3 not 2,3 alone)")
    return parts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, required=True)
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--thresholds-path", type=str, required=True)
    ap.add_argument("--split-file", type=str, required=True)
    ap.add_argument("--normal-dir", type=str, required=True)
    ap.add_argument("--attacked-dir", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--sweep-points", type=int, default=500)
    ap.add_argument("--stride", type=int, default=0, help="Stride used in windowing (0=auto from metadata or 50)")
    ap.add_argument(
        "--max-validation-files",
        type=int,
        default=0,
        help="If >0, cap validation files (lighter run). 0 = all.",
    )
    ap.add_argument("--max-test-files", type=int, default=0, help="If >0, cap normal test files. 0 = all.")
    ap.add_argument("--max-attacked-files", type=int, default=0, help="If >0, cap attacked_v2 npz files. 0 = all.")
    ap.add_argument("--skip-plots", action="store_true", help="Skip all PNG generation (faster, less RAM).")
    ap.add_argument(
        "--sections",
        type=str,
        default="all",
        help="Comma-separated contiguous sections 1..k (k<=7), or 'all'. "
        "1=Validation, 2=Evaluation+overall metrics, 3=Per-attack, 4=Sweep CSV+plots, "
        "5=ROC/PR/dist/confusion/bars, 6=Ablation+evaluation_summary.json, 7=Markdown report.",
    )
    args = ap.parse_args()
    sections_list = parse_sections_arg(args.sections)
    sec_set = set(sections_list)
    max_sec = max(sections_list)

    repo_root = Path(args.repo_root).resolve()
    model_path = _resolve_under(repo_root, args.model_path)
    thresholds_path = _resolve_under(repo_root, args.thresholds_path)
    split_file = _resolve_under(repo_root, args.split_file)
    normal_dir = _resolve_under(repo_root, args.normal_dir)
    attacked_dir = _resolve_under(repo_root, args.attacked_dir)
    output_dir = _resolve_under(repo_root, args.output_dir)

    for p, lab in (
        (model_path, "model"),
        (thresholds_path, "thresholds"),
        (split_file, "split"),
        (normal_dir, "normal-dir"),
        (attacked_dir, "attacked-dir"),
    ):
        if not p.exists():
            raise FileNotFoundError(f"{lab}: {p}")

    output_dir.mkdir(parents=True, exist_ok=True)
    plt = None if args.skip_plots else _maybe_plt()

    manifest: List[Dict[str, Any]] = []

    def _manifest(sec: int, title: str, outputs: List[str]) -> None:
        manifest.append({"section": sec, "title": title, "outputs": outputs})

    ev = _load_eval_strict(repo_root)
    with thresholds_path.open("r", encoding="utf-8") as f:
        thr_cfg = json.load(f)
    weights = thr_cfg.get("weights", {})
    if not isinstance(weights, dict):
        weights = {}

    official = official_threshold_map(thr_cfg)

    t_sample, c_sample = ev.infer_T_C_from_sample(normal_dir)
    model = ev.load_keras_model_robust(model_path, t_sample, c_sample)

    stride = int(args.stride) if int(args.stride) > 0 else 50
    meta_path = attacked_dir / "generation_metadata.json"
    if meta_path.is_file():
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                gm = json.load(f)
            if isinstance(gm, dict) and "stride" in gm:
                stride = int(gm["stride"])
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    with split_file.open("r", encoding="utf-8") as f:
        split_obj = json.load(f)
    val_names = list(split_obj.get("validation", []))
    test_names = list(split_obj.get("test", []))
    if not val_names:
        raise ValueError("split JSON missing non-empty 'validation' list")
    if not test_names:
        raise ValueError("split JSON missing non-empty 'test' list")

    print(f"\n=== Sections to run: {sections_list} (max={max_sec}) ===\n", flush=True)

    # ----- SECTION 1 — Validation (normal-only) -----
    print("=== Section 1/7: Validation (normal-only) ===", flush=True)
    val_scores, val_files_ok, val_rej = score_normal_split(
        ev,
        model,
        weights,
        normal_dir,
        val_names,
        t_sample,
        int(args.batch_size),
        max_files=int(args.max_validation_files),
    )
    val_q = quantiles_from_scores(val_scores)

    rows_vcomp: List[Dict[str, Any]] = []
    for key in ("p95", "p97", "p99", "p99.5", "p99.7", "3sigma"):
        comp = val_q[key]
        off = official.get(key)
        rows_vcomp.append(
            {
                "threshold_name": key,
                "validation_computed": comp,
                "official_json_value": "" if off is None else off,
                "absolute_delta": "" if off is None else float(comp) - float(off),
            }
        )

    p_vcomp = _safe_out(output_dir, "validation_thresholds_computed.csv")
    with p_vcomp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["threshold_name", "validation_computed", "official_json_value", "absolute_delta"],
        )
        w.writeheader()
        for row in rows_vcomp:
            w.writerow(row)

    val_summary = {
        "num_validation_files_listed": len(val_names),
        "num_validation_files_scored_ok": val_files_ok,
        "num_rejected_or_invalid": val_rej,
        "num_validation_windows": int(len(val_scores)),
        "window_size": int(t_sample),
        "stride": int(stride),
        "input_shape_timesteps": int(t_sample),
        "input_shape_channels": int(c_sample),
        "score_mean": float(val_scores.mean()),
        "score_std": float(val_scores.std()),
        "score_min": float(val_scores.min()),
        "score_max": float(val_scores.max()),
        "score_median": float(np.median(val_scores)),
    }
    if int(args.max_validation_files) > 0:
        val_summary["max_validation_files_cap"] = int(args.max_validation_files)
    p_vsum = _safe_out(output_dir, "validation_score_summary.csv")
    with p_vsum.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(val_summary.keys()))
        w.writeheader()
        w.writerow(val_summary)

    if plt is not None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(val_scores, bins=80, color="#3498db", alpha=0.75, edgecolor="white")
        ax.set_title("Validation-normal anomaly score distribution")
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(_safe_out(output_dir, "validation_score_distribution.png"), dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(val_scores, bins=80, color="#ecf0f1", edgecolor="white", label="scores")
        for k, c in [
            ("p95", "#95a5a6"),
            ("p97", "#7f8c8d"),
            ("p99", "#e74c3c"),
            ("p99.5", "#c0392b"),
            ("p99.7", "#8e44ad"),
            ("3sigma", "#27ae60"),
        ]:
            ax.axvline(val_q[k], color=c, linestyle="--", linewidth=1.5, label=f"{k}={val_q[k]:.4g}")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title("Validation thresholds (computed on validation-normal only)")
        ax.set_xlabel("Score")
        fig.tight_layout()
        fig.savefig(_safe_out(output_dir, "validation_threshold_lines.png"), dpi=140)
        plt.close(fig)

    _manifest(
        1,
        "Validation",
        ["validation_thresholds_computed.csv", "validation_score_summary.csv"]
        + ([] if plt is None else ["validation_score_distribution.png", "validation_threshold_lines.png"]),
    )

    if max_sec == 1:
        note = _safe_out(output_dir, "validation_only_run.txt")
        note.write_text(
            "Stopped after Section 1 (--sections 1).\n"
            "Run with --sections 1,2 or --sections all to continue.\n",
            encoding="utf-8",
        )
        print("\n=== Stopped after Section 1 ===", flush=True)
        for p in (p_vcomp, p_vsum, note):
            print(f"  {p}")
        if plt is not None:
            for name in ("validation_score_distribution.png", "validation_threshold_lines.png"):
                print(f"  {_safe_out(output_dir, name)}")
        man_path = _safe_out(output_dir, "pipeline_sections_manifest.json")
        man_path.write_text(json.dumps({"sections_run": [1], "manifest": manifest}, indent=2), encoding="utf-8")
        print(f"  {man_path}")
        return

    # ----- SECTION 2 — Evaluation scores + ROC/PR + sweep (in memory) + overall metrics -----
    print("=== Section 2/7: Evaluation (test-normal + attacked_v2) + overall threshold metrics ===", flush=True)
    test_scores, test_ok, test_rej = score_normal_split(
        ev,
        model,
        weights,
        normal_dir,
        test_names,
        t_sample,
        int(args.batch_size),
        max_files=int(args.max_test_files),
    )
    atk_scores, atk_y, atk_type, atk_rej = load_attacked_all(
        ev,
        model,
        weights,
        attacked_dir,
        t_sample,
        int(args.batch_size),
        max_files=int(args.max_attacked_files),
    )

    y_true = np.concatenate([np.zeros(len(test_scores), dtype=np.uint8), atk_y.astype(np.uint8)])
    y_score = np.concatenate([test_scores, atk_scores])

    n_attack_windows = int(len(atk_scores))
    n_anomalous = int(np.sum(atk_y == 1))
    n_normal_inside_attacked = int(np.sum(atk_y == 0))
    imbalance = float((len(test_scores) + n_normal_inside_attacked) / max(n_anomalous, 1))

    curves_info, curves = ev.compute_curves_and_auc(y_true, y_score)
    roc_auc = float(curves_info["roc_auc"])
    pr_auc = float(curves_info["pr_auc"])
    _, sweep_rows, sweep_best_f1, sweep_best_thr = sweep_thresholds(
        y_true, y_score, grid_n=int(args.sweep_points)
    )

    best_f1_payload = {
        "best_f1": float(sweep_best_f1),
        "best_f1_threshold": float(sweep_best_thr),
        "note": "Analysis only; not written to thresholds.json.",
        "sweep_range": [float(np.min(y_score)), float(np.max(y_score))],
        "num_sweep_points": int(args.sweep_points),
    }

    # Evaluation thresholds: validation-calibrated names + best_f1 from sweep
    eval_thresholds: List[Tuple[str, float]] = [
        ("p95", val_q["p95"]),
        ("p97", val_q["p97"]),
        ("p99", val_q["p99"]),
        ("p99.5", val_q["p99.5"]),
        ("p99.7", val_q["p99.7"]),
        ("3sigma", val_q["3sigma"]),
        ("best_f1", float(sweep_best_thr)),
    ]

    overall_rows: List[Dict[str, Any]] = []
    for name, thr in eval_thresholds:
        cm = confusion_at_threshold(y_true, y_score, thr)
        m = metrics_from_confusion(cm)
        overall_rows.append(
            {
                "threshold_name": name,
                "threshold_value": float(thr),
                "TP": cm.TP,
                "TN": cm.TN,
                "FP": cm.FP,
                "FN": cm.FN,
                **m,
            }
        )

    p_over = _safe_out(output_dir, "overall_threshold_metrics.csv")
    ocols = [
        "threshold_name",
        "threshold_value",
        "TP",
        "TN",
        "FP",
        "FN",
        "accuracy",
        "precision",
        "recall",
        "tpr",
        "far",
        "fpr",
        "tnr",
        "specificity",
        "f1",
        "balanced_accuracy",
    ]
    with p_over.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ocols)
        w.writeheader()
        for row in overall_rows:
            w.writerow({k: row.get(k) for k in ocols})

    _manifest(2, "Evaluation + overall threshold metrics", ["overall_threshold_metrics.csv"])
    if max_sec == 2:
        man_path = _safe_out(output_dir, "pipeline_sections_manifest.json")
        man_path.write_text(
            json.dumps({"sections_run": sections_list, "manifest": manifest}, indent=2),
            encoding="utf-8",
        )
        print(f"\n=== Stopped after Section 2 ===\n  {man_path}\n", flush=True)
        return

    # ----- SECTION 3 — Per-attack -----
    print("=== Section 3/7: Per-attack threshold metrics ===", flush=True)
    atk_arr = np.asarray(atk_type, dtype=object)
    per_rows: List[Dict[str, Any]] = []

    for thr_name, thr_val in eval_thresholds:
        pred = atk_scores > float(thr_val)
        types_seen = sorted(set(str(x) for x in atk_arr.tolist()))
        for at in types_seen:
            msk = atk_arr == at
            if not np.any(msk):
                continue
            s_sub = atk_scores[msk]
            y_sub = atk_y[msk].astype(np.uint8)
            p_sub = pred[msk]
            tp = int(np.sum(p_sub & (y_sub == 1)))
            fn = int(np.sum((~p_sub) & (y_sub == 1)))
            fp = int(np.sum(p_sub & (y_sub == 0)))
            tw = int(np.sum(y_sub == 1))
            det = int(tp)
            missed = int(fn)
            rec = float(tp / (tp + fn)) if (tp + fn) else 0.0
            miss_r = float(fn / tw) if tw else 0.0
            prec = float(tp / (tp + fp)) if (tp + fp) else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
            per_rows.append(
                {
                    "attack_type": at,
                    "threshold_name": thr_name,
                    "threshold_value": float(thr_val),
                    "total_attack_windows": tw,
                    "detected_windows": det,
                    "missed_windows": missed,
                    "detection_rate": rec,
                    "recall": rec,
                    "missed_rate": miss_r,
                    "precision": prec,
                    "f1": f1,
                    "average_score": float(np.mean(s_sub)) if len(s_sub) else 0.0,
                    "max_score": float(np.max(s_sub)) if len(s_sub) else 0.0,
                    "min_score": float(np.min(s_sub)) if len(s_sub) else 0.0,
                }
            )

    p_per = _safe_out(output_dir, "per_attack_threshold_metrics.csv")
    pcols = [
        "attack_type",
        "threshold_name",
        "threshold_value",
        "total_attack_windows",
        "detected_windows",
        "missed_windows",
        "detection_rate",
        "recall",
        "missed_rate",
        "precision",
        "f1",
        "average_score",
        "max_score",
        "min_score",
    ]
    with p_per.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pcols)
        w.writeheader()
        for row in per_rows:
            w.writerow({k: row.get(k) for k in pcols})

    # best_threshold_per_attack
    by_at: Dict[str, List[Dict[str, Any]]] = {}
    for r in per_rows:
        by_at.setdefault(str(r["attack_type"]), []).append(r)

    thr_by_name = {str(r["threshold_name"]): float(r["threshold_value"]) for r in overall_rows}

    def operational_reason(at: str) -> str:
        return (
            f"العتبة التشغيلية المقترحة p99 (معايرة validation) توازن FAR/Recall؛ "
            f"لهجوم {at}: Freeze أسهل عند قلة التغير الزمني؛ Spike عند magnitude عالٍ؛ "
            f"Drift أصعب لكونه تدريجياً؛ pattern_shift يغيّر العلاقة الزمنية؛ "
            f"noise يعتمد الشدة؛ scale يعتمد beta؛ drop/masking يعتمد أسلوب الإخفاء."
        )

    best_attack_rows: List[Dict[str, str]] = []
    for at, lst in sorted(by_at.items()):
        if not lst:
            continue
        br = max(lst, key=lambda x: float(x["recall"]))
        bf = max(lst, key=lambda x: float(x["f1"]))
        p99_row = next((x for x in lst if x["threshold_name"] == "p99"), None)
        op_thr = "p99"
        if p99_row is None:
            op_thr = str(max(lst, key=lambda x: float(x["recall"]))["threshold_name"])
        best_attack_rows.append(
            {
                "attack_type": at,
                "best_threshold_by_recall": str(br["threshold_name"]),
                "best_threshold_by_f1": str(bf["threshold_name"]),
                "best_operational_threshold": op_thr,
                "reason": operational_reason(at),
            }
        )

    p_batk = _safe_out(output_dir, "best_threshold_per_attack.csv")
    with p_batk.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "attack_type",
                "best_threshold_by_recall",
                "best_threshold_by_f1",
                "best_operational_threshold",
                "reason",
            ],
        )
        w.writeheader()
        for row in best_attack_rows:
            w.writerow(row)

    _manifest(
        3,
        "Per-attack",
        ["per_attack_threshold_metrics.csv", "best_threshold_per_attack.csv"],
    )
    if max_sec == 3:
        man_path = _safe_out(output_dir, "pipeline_sections_manifest.json")
        man_path.write_text(
            json.dumps({"sections_run": sections_list, "manifest": manifest}, indent=2),
            encoding="utf-8",
        )
        print(f"\n=== Stopped after Section 3 ===\n  {man_path}\n", flush=True)
        return

    # ----- SECTION 4 — Threshold sweep (CSV + JSON + sweep line plots) -----
    print("=== Section 4/7: Threshold sweep (analysis only) ===", flush=True)
    p_sweep = _safe_out(output_dir, "threshold_sweep_results.csv")
    with p_sweep.open("w", newline="", encoding="utf-8") as f:
        cols = [
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "tpr",
            "far",
            "fpr",
            "tnr",
            "specificity",
            "f1",
            "balanced_accuracy",
            "tp",
            "tn",
            "fp",
            "fn",
        ]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in sweep_rows:
            w.writerow({k: row.get(k) for k in cols})

    p_bf1 = _safe_out(output_dir, "best_f1_from_sweep.json")
    with p_bf1.open("w", encoding="utf-8") as f:
        json.dump(best_f1_payload, f, indent=2)

    thv = [r["threshold"] for r in sweep_rows]
    if plt is not None:
        f1v = [r["f1"] for r in sweep_rows]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(thv, f1v, color="#8e44ad")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("F1")
        ax.set_title("Threshold sweep: F1")
        fig.tight_layout()
        fig.savefig(_safe_out(output_dir, "threshold_vs_f1.png"), dpi=140)
        plt.close(fig)

        recv = [r["recall"] for r in sweep_rows]
        farv = [r["far"] for r in sweep_rows]
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax1.plot(thv, recv, color="#2980b9", label="Recall")
        ax1.set_xlabel("Threshold")
        ax1.set_ylabel("Recall", color="#2980b9")
        ax2 = ax1.twinx()
        ax2.plot(thv, farv, color="#e67e22", label="FAR")
        ax2.set_ylabel("FAR", color="#e67e22")
        ax1.set_title("Threshold sweep: Recall vs FAR")
        fig.tight_layout()
        fig.savefig(_safe_out(output_dir, "threshold_vs_recall_far.png"), dpi=140)
        plt.close(fig)

        bal = [r["balanced_accuracy"] for r in sweep_rows]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(thv, bal, color="#16a085")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Balanced accuracy")
        ax.set_title("Threshold sweep: balanced accuracy")
        fig.tight_layout()
        fig.savefig(_safe_out(output_dir, "threshold_vs_balanced_accuracy.png"), dpi=140)
        plt.close(fig)

    _manifest(
        4,
        "Threshold sweep",
        ["threshold_sweep_results.csv", "best_f1_from_sweep.json"]
        + (
            []
            if plt is None
            else [
                "threshold_vs_f1.png",
                "threshold_vs_recall_far.png",
                "threshold_vs_balanced_accuracy.png",
            ]
        ),
    )
    if max_sec == 4:
        man_path = _safe_out(output_dir, "pipeline_sections_manifest.json")
        man_path.write_text(
            json.dumps({"sections_run": sections_list, "manifest": manifest}, indent=2),
            encoding="utf-8",
        )
        print(f"\n=== Stopped after Section 4 ===\n  {man_path}\n", flush=True)
        return

    # ----- SECTION 5 — ROC / PR / distributions / confusion / per-attack bars -----
    print("=== Section 5/7: Curves and visuals ===", flush=True)
    if plt is not None:
        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.plot(curves["roc_fpr"], curves["roc_tpr"], color="#2980b9", lw=2)
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"ROC (AUC={roc_auc:.4f})")
        fig.tight_layout()
        fig.savefig(_safe_out(output_dir, "roc_curve.png"), dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.plot(curves["pr_recall"], curves["pr_precision"], color="#c0392b", lw=2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision–Recall (AUC={pr_auc:.4f})")
        fig.tight_layout()
        fig.savefig(_safe_out(output_dir, "precision_recall_curve.png"), dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        combined = np.concatenate([test_scores, atk_scores]).astype(np.float64)
        hi = float(np.percentile(combined, 99.95))
        hi = max(hi, 0.05)
        bins = np.linspace(0.0, hi, 120)
        ax.hist(
            test_scores,
            bins=bins,
            alpha=0.55,
            density=True,
            histtype="stepfilled",
            label=f"Normal test n={len(test_scores):,}",
            color="#2ecc71",
            edgecolor="white",
            linewidth=0.3,
        )
        ax.hist(
            atk_scores,
            bins=bins,
            alpha=0.55,
            density=True,
            histtype="stepfilled",
            label=f"Attacked_v2 n={len(atk_scores):,}",
            color="#e74c3c",
            edgecolor="white",
            linewidth=0.3,
        )
        ax.set_xlim(0.0, hi)
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.legend(loc="upper right")
        ax.set_title("Normal test vs attacked_v2 score distribution (overlap)")
        fig.tight_layout()
        fig.savefig(_safe_out(output_dir, "normal_vs_attack_score_distribution.png"), dpi=140)
        plt.close(fig)

        thr_p99 = thr_by_name["p99"]
        thr_p995 = thr_by_name["p99.5"]
        thr_bf1 = thr_by_name["best_f1"]
        plot_confusion(
            confusion_at_threshold(y_true, y_score, thr_p99),
            f"Confusion @ p99 (thr={thr_p99:.4g})",
            _safe_out(output_dir, "p99_confusion_matrix.png"),
            plt,
        )
        plot_confusion(
            confusion_at_threshold(y_true, y_score, thr_p995),
            f"Confusion @ p99.5 (thr={thr_p995:.4g})",
            _safe_out(output_dir, "p99_5_confusion_matrix.png"),
            plt,
        )
        plot_confusion(
            confusion_at_threshold(y_true, y_score, thr_bf1),
            f"Confusion @ best_f1 sweep (thr={thr_bf1:.4g})",
            _safe_out(output_dir, "best_f1_confusion_matrix.png"),
            plt,
        )

        # Per-attack detection @ p99 / best_f1
        def detection_bar(thr_key: str, fname: str) -> None:
            thr_v = thr_by_name[thr_key]
            pred = atk_scores > float(thr_v)
            types = sorted(set(str(x) for x in atk_arr.tolist()))
            rates: List[float] = []
            for at in types:
                msk = atk_arr == at
                y_sub = atk_y[msk].astype(np.uint8)
                p_sub = pred[msk]
                tw = int(np.sum(y_sub == 1))
                if tw == 0:
                    rates.append(0.0)
                else:
                    tp = int(np.sum(p_sub & (y_sub == 1)))
                    rates.append(tp / tw)
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.bar(types, rates, color="#3498db")
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Detection rate (recall on y=1)")
            ax.set_title(f"Per-attack detection rate @ {thr_key}")
            ax.tick_params(axis="x", rotation=35)
            fig.tight_layout()
            fig.savefig(_safe_out(output_dir, fname), dpi=140)
            plt.close(fig)

        detection_bar("p99", "per_attack_detection_rate_p99.png")
        detection_bar("best_f1", "per_attack_detection_rate_best_f1.png")

    sec5_files = (
        []
        if plt is None
        else [
            "roc_curve.png",
            "precision_recall_curve.png",
            "normal_vs_attack_score_distribution.png",
            "p99_confusion_matrix.png",
            "p99_5_confusion_matrix.png",
            "best_f1_confusion_matrix.png",
            "per_attack_detection_rate_p99.png",
            "per_attack_detection_rate_best_f1.png",
        ]
    )
    _manifest(5, "ROC/PR/distributions/confusion/per-attack bars", sec5_files)
    if max_sec == 5:
        man_path = _safe_out(output_dir, "pipeline_sections_manifest.json")
        man_path.write_text(
            json.dumps({"sections_run": sections_list, "manifest": manifest}, indent=2),
            encoding="utf-8",
        )
        print(f"\n=== Stopped after Section 5 ===\n  {man_path}\n", flush=True)
        return

    # ----- SECTION 6 — Ablation scan + evaluation_summary.json -----
    print("=== Section 6/7: Ablation discovery + evaluation_summary.json ===", flush=True)
    ablation_paths = find_ablation_tables(repo_root)
    ablation_usable = [p for p in ablation_paths if ablation_has_data(p)]

    # ----- evaluation_summary.json -----
    summary = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_path": str(model_path),
        "thresholds_path": str(thresholds_path),
        "split_file": str(split_file),
        "validation": val_summary,
        "evaluation": {
            "test_normal_windows": int(len(test_scores)),
            "test_files_ok": test_ok,
            "test_files_rejected": test_rej,
            "attacked_windows": n_attack_windows,
            "attacked_files_rejected": atk_rej,
            "anomalous_windows_attacked": n_anomalous,
            "normal_windows_inside_attacked_files": n_normal_inside_attacked,
            "imbalance_ratio_neg_to_pos": imbalance,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "best_f1_sweep": best_f1_payload,
            "official_p99_in_json": official.get("p99"),
            "validation_computed_p99": val_q["p99"],
        },
        "ablation_csv_candidates": [str(p) for p in ablation_paths],
        "ablation_included_in_report": [str(p) for p in ablation_usable],
        "light_run_options": {
            "sections_requested": sections_list,
            "max_section": max_sec,
            "max_validation_files": int(args.max_validation_files),
            "max_test_files": int(args.max_test_files),
            "max_attacked_files": int(args.max_attacked_files),
            "batch_size": int(args.batch_size),
            "sweep_points": int(args.sweep_points),
            "skip_plots": bool(args.skip_plots),
        },
    }
    p_sumj = _safe_out(output_dir, "evaluation_summary.json")
    with p_sumj.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _manifest(6, "Ablation + evaluation_summary", ["evaluation_summary.json"])
    if max_sec == 6:
        man_path = _safe_out(output_dir, "pipeline_sections_manifest.json")
        man_path.write_text(
            json.dumps({"sections_run": sections_list, "manifest": manifest}, indent=2),
            encoding="utf-8",
        )
        print(f"\n=== Stopped after Section 6 ===\n  {man_path}\n", flush=True)
        return

    # ----- SECTION 7 — Final Markdown report -----
    print("=== Section 7/7: Markdown report ===", flush=True)
    report_path = _safe_out(output_dir, "validation_evaluation_report.md")
    lines: List[str] = []
    lines.append("# CyberSatDetect / NebulaSec — Validation & Evaluation Report\n\n")
    lines.append(f"> Pipeline sections run: **{sections_list}** (see `pipeline_sections_manifest.json`).\n\n")
    lines.append("> Generated offline; production thresholds and API were not modified.\n\n")

    lines.append("## 1. Validation Phase (normal-only)\n\n")
    lines.append("### 1.1 Validation set description\n\n")
    lines.append(f"- Files listed: {len(val_names)}, scored OK: {val_files_ok}, rejected: {val_rej}\n")
    lines.append(f"- Windows: {len(val_scores)}, window size T={t_sample}, channels={c_sample}, stride={stride}\n\n")

    lines.append("### 1.2 Why validation must be normal-only\n\n")
    lines.append(
        "| السبب | الشرح |\n"
        "|---|---|\n"
        "| خلو التلوث | أي نافذة هجومية في validation ترفع المئينيات وتُضعف تمييز السلوك الطبيعي. |\n"
        "| معايرة FAR | العتبات تُفسر كـ quantiles على السلوك الطبيعي فقط. |\n"
        "| فصل الأدوار | validation للمعايرة؛ test+attacked للقياس بدون تسريب معايرة إلى نفس شريحة القياس عند اتباع بروتوكول صارم. |\n\n"
    )

    lines.append("### 1.3 Why calibrate thresholds on validation — not test\n\n")
    lines.append(
        "- استخدام **test** لاختيار عتبة يُكيف النموذج/العتبة على مجموعة القياس ويُضخّم التفاؤل.\n"
        "- **Validation** يوفّر تقديراً خارج عيّنة التدريب دون لمس test حتى تقييم الإغلاق.\n\n"
    )

    lines.append("### 1.4 Validation score distribution & calibration\n\n")
    lines.append(
        f"- Mean={val_summary['score_mean']:.6g}, std={val_summary['score_std']:.6g}, "
        f"min={val_summary['score_min']:.6g}, max={val_summary['score_max']:.6g}\n\n"
    )
    lines.append("| Threshold | Validation computed | Official JSON | Delta |\n")
    lines.append("|---:|---:|---:|---:|\n")
    for row in rows_vcomp:
        off = row["official_json_value"]
        dlt = row["absolute_delta"]
        off_s = f"{float(off):.6g}" if off != "" else "—"
        dlt_s = f"{float(dlt):.6g}" if dlt != "" else "—"
        lines.append(
            f"| {row['threshold_name']} | {float(row['validation_computed']):.6g} | {off_s} | {dlt_s} |\n"
        )
    lines.append("\n### 1.5 Why p99 remains a sound production threshold\n\n")
    lines.append(
        "- p99 على بيانات **طبيعية** يحدد معدل إنذار كاذب تقريبي قابل للتفسير دون تحسين مباشر على هجمات الاختبار.\n"
        "- أعلى من p99 (p99.5/p99.7) يقلل FAR لكن قد يخفض Recall على هجمات خفيفة.\n"
        "- `best_f1` من sweep هو تحسين أكاديمي على خليط test+attack ولا يُوصى به كعتبة إنتاج ثابتة دون مراجعة FAR.\n\n"
    )

    lines.append("\n### 1.6 Methodology (summary table)\n\n")
    lines.append(
        "| الموضوع | التوصية |\n"
        "|---|---|\n"
        "| Validation normal-only | يمنع رفع المئينيات بسبب هجمات ويحافظ على تفسير FAR. |\n"
        "| معايرة على validation وليس test | يقلل تسريب اختيار العتبة إلى مجموعة القياس النهائية. |\n"
        "| p99 كعتبة إنتاج | مئينيات على الطبيعي = حوكمة؛ أقل حساسية لـ overfitting على هجمات الاختبار من `best_f1`. |\n\n"
    )

    lines.append("## 2. Evaluation protocol\n\n")
    lines.append(
        "- **Strict window-level** labels; **no** point adjustment; **no** sequence-only shortcut.\n"
        "- Normal test windows: label 0. Attacked windows: label 1 iff ≥10% timesteps attacked (`y_window` in attacked_v2).\n"
        "- Scores: weighted recon + pred + grad (no separation at inference).\n\n"
    )

    lines.append("## 3. Dataset statistics (evaluation)\n\n")
    lines.append(f"- Normal test windows: {len(test_scores)} (files ok {test_ok}, rejected {test_rej})\n")
    lines.append(f"- Attacked windows: {n_attack_windows} (files rejected {atk_rej})\n")
    lines.append(f"- Anomalous windows (y=1) in attacked: {n_anomalous}\n")
    lines.append(f"- Normal windows inside attacked files (y=0): {n_normal_inside_attacked}\n")
    lines.append(f"- Imbalance ratio (neg/pos): {imbalance:.4f}\n\n")

    lines.append("## 4. Overall threshold results (validation-calibrated + best_f1 sweep)\n\n")
    lines.append(
        "| name | thr | Acc | Prec | Rec | FAR | F1 | BalAcc |\n|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    for r in overall_rows:
        lines.append(
            f"| {r['threshold_name']} | {r['threshold_value']:.6g} | {r['accuracy']:.4f} | "
            f"{r['precision']:.4f} | {r['recall']:.4f} | {r['far']:.4f} | {r['f1']:.4f} | {r['balanced_accuracy']:.4f} |\n"
        )
    lines.append(f"\n- ROC AUC = **{roc_auc:.4f}**, PR AUC = **{pr_auc:.4f}**\n\n")

    lines.append("## 5. Per-attack results\n\n")
    lines.append("See `per_attack_threshold_metrics.csv` and `best_threshold_per_attack.csv`.\n\n")
    lines.append("### 5.1 Attack difficulty (qualitative)\n\n")
    lines.append(
        "- **Freeze:** غالباً أسهل عند قلة التغير الزمني.\n"
        "- **Spike:** أسهل إذا كان magnitude عالياً.\n"
        "- **Drift:** أصعب لأنه تدريجي.\n"
        "- **pattern_shift:** أصعب لأنه يغيّر العلاقة الزمنية.\n"
        "- **noise:** يعتمد على شدة التشويش.\n"
        "- **scale:** يعتمد على beta.\n"
        "- **drop/masking:** يعتمد على طريقة الإخفاء.\n\n"
    )

    lines.append("## 6. Threshold selection analysis\n\n")
    lines.append(
        "| | Academic (`best_f1` sweep) | Operational (`p99` validation-calibrated) |\n"
        "|---|---|---|\n"
        "| الهدف | تعظيم F1 على خليط القياس | ضبط FAR مع تفسير إحصائي على الطبيعي |\n"
        "| المخاطر | FAR أعلى غالباً | Recall أقل على هجمات خفيفة |\n\n"
    )
    lines.append(
        "- **p99.5 / p99.7:** FAR أقل من p99 لكن Recall يميل للانخفاض (تجارة واضحة).\n\n"
    )

    lines.append("## 7. Recommended thresholds\n\n")
    lines.append(
        f"- **أفضل عتبة أكاديمياً (F1 على هذا الجمع):** `best_f1` ≈ {sweep_best_thr:.6g} (F1={sweep_best_f1:.4f}).\n"
        f"- **أفضل عتبة تشغيلية مقترحة:** `p99` validation-calibrated = **{thr_by_name['p99']:.6g}** "
        "(العتبة الرسمية في `thresholds.json` لم تُعدل؛ p99 الإنتاجي الملفّي للمقارنة فقط في القسم 1).\n\n"
    )

    lines.append("## 8. Error analysis\n\n")
    lines.append(
        "- هجمات خفيفة أو drift قد تبقى تحت العتبة بسبب تداخل التوزيعات بين normal و attack.\n"
        "- تداخل الذيلين (overlap) يفرض trade-off بين Recall و FAR.\n\n"
    )

    lines.append("## 9. Limitations\n\n")
    lines.append(
        "- الهجمات الاصطناعية لا تغطي كل سيناريوهات الواقع.\n"
        "- عتبة ثابتة وليست تكيفية.\n"
        "- النموذج يفترض أن الطبيعي قابل للاستنساخ؛ انحراف القناة الطبيعي قد يسبب FP.\n"
        "- قنوات متعددة/علاقات متعددة المتغيرات: هذا المسار أحادي القناة داخل النافذة.\n\n"
    )

    lines.append("## 10. Ablation study\n\n")
    if ablation_usable:
        lines.append("تم العثور على ملفات CSV تحتوي بيانات ablation وتمت الإشارة إليها في `evaluation_summary.json`.\n\n")
        for p in ablation_usable:
            lines.append(f"- `{p}`\n")
        lines.append("\nراجع هذه الملفات يدوياً لدمج الجداول (Full / without Lpred / …).\n\n")
    else:
        lines.append(
            "**Ablation study requires separately trained variants and was not recomputed in this run "
            "to avoid modifying the production system.**\n\n"
            "Expected variants when available: Full model, without Lpred, without Lgrad, without Lsep, "
            "reconstruction only, LSTM only, GRU only — metrics: F1, Recall, FAR, Balanced Accuracy.\n\n"
        )

    lines.append("## 11. Conclusion\n\n")
    lines.append(
        "أُجريت معايرة على **validation-normal** وتقييم صارم على **test-normal + attacked_v2** "
        "بدون point adjustment. تُظهر الجداول والمنحنيات trade-off بين FAR وRecall؛ "
        "يُنصح بالإبقاء على **p99** كمرجع إنتاجي مع مراقبة FAR، بينما يُستخدم **best_f1** من الـ sweep "
        "للتحليل الأكاديمي فقط.\n"
    )

    report_path.write_text("".join(lines), encoding="utf-8")

    _manifest(7, "Final report", ["validation_evaluation_report.md"])

    man_path = _safe_out(output_dir, "pipeline_sections_manifest.json")
    man_path.write_text(
        json.dumps({"sections_run": sections_list, "manifest": manifest}, indent=2),
        encoding="utf-8",
    )

    written: List[Path] = [p_vcomp, p_vsum, p_over, p_per, p_batk, p_sweep, p_bf1, p_sumj, report_path, man_path]

    if plt is not None:
        written.extend(
            [
                _safe_out(output_dir, "validation_score_distribution.png"),
                _safe_out(output_dir, "validation_threshold_lines.png"),
                _safe_out(output_dir, "normal_vs_attack_score_distribution.png"),
                _safe_out(output_dir, "roc_curve.png"),
                _safe_out(output_dir, "precision_recall_curve.png"),
                _safe_out(output_dir, "threshold_vs_f1.png"),
                _safe_out(output_dir, "threshold_vs_recall_far.png"),
                _safe_out(output_dir, "threshold_vs_balanced_accuracy.png"),
                _safe_out(output_dir, "p99_confusion_matrix.png"),
                _safe_out(output_dir, "p99_5_confusion_matrix.png"),
                _safe_out(output_dir, "best_f1_confusion_matrix.png"),
                _safe_out(output_dir, "per_attack_detection_rate_p99.png"),
                _safe_out(output_dir, "per_attack_detection_rate_best_f1.png"),
            ]
        )

    # ----- Print outputs -----
    print("\n=== Validation & Evaluation complete ===\n")
    for p in written:
        ok = p.is_file()
        print(f"  [{'ok' if ok else 'MISSING'}] {p}")


if __name__ == "__main__":
    main()
