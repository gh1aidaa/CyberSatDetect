"""
Evaluate univariate pattern experiment models on test-normal + attacked_v2 (evaluation only).

Writes under --output-dir; if --attack-types-only is set, metrics/plots/report go to
<output-dir>/eval_four_attacks_only/ so a full multi-attack run in the parent folder stays intact.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import numpy as np

_EXP_DIR = Path(__file__).resolve().parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from temporal_feature_engineering import augment_univariate_batch  # noqa: E402
from score_univariate_pattern import compute_window_scores, compute_window_scores_decomposed  # noqa: E402
from conservative_pattern_fusion import (  # noqa: E402
    apply_tail_order_boost,
    calibrate_tail_boost,
    save_fusion_config,
)


def _resolve(repo: Path, p: str | Path) -> Path:
    path = Path(p)
    return (repo / path).resolve() if not path.is_absolute() else path.resolve()


def _load_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_split(repo: Path, split_file: str) -> Dict[str, List[str]]:
    return _load_json(_resolve(repo, split_file))


@dataclass
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
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    bal_acc = 0.5 * (recall + tnr)
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "far": float(far),
        "tnr": float(tnr),
        "f1": float(f1),
        "balanced_accuracy": float(bal_acc),
    }


def compute_curves_and_auc(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(np.uint8)
    y_score = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)
    P = float(y_true.sum())
    N = float(len(y_true) - P)
    tpr = tps / max(P, 1.0)
    fpr = fps / max(N, 1.0)
    precision = tps / np.maximum(tps + fps, 1.0)
    recall = tpr
    fpr_full = np.concatenate([[0.0], fpr, [1.0]])
    tpr_full = np.concatenate([[0.0], tpr, [1.0]])
    roc_auc = float(np.trapz(tpr_full, fpr_full))
    rec_full = np.concatenate([[0.0], recall, [1.0]])
    prec_full = np.concatenate([[1.0], precision, [precision[-1] if len(precision) else 0.0]])
    order_pr = np.argsort(rec_full)
    pr_auc = float(np.trapz(prec_full[order_pr], rec_full[order_pr]))
    return {"roc_auc": roc_auc, "pr_auc": pr_auc}


def _npz_attack_type_only(path: Path) -> str:
    """Read attack_type without loading large arrays (npz is a zip archive)."""
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            if "attack_type.npy" not in names:
                with np.load(path, allow_pickle=False) as z:
                    return str(z["attack_type"].item()) if "attack_type" in z.files else "unknown"
            raw = zf.read("attack_type.npy")
        v = np.load(io.BytesIO(raw), allow_pickle=False)
        return str(v.item()) if v.ndim == 0 else str(v.flat[0])
    except Exception:
        with np.load(path, allow_pickle=False) as z:
            return str(z["attack_type"].item()) if "attack_type" in z.files else "unknown"


def _filtered_npz_paths(
    attacked_dir: Path,
    max_files: int,
    allowed: Optional[FrozenSet[str]],
) -> List[Path]:
    files = sorted(attacked_dir.glob("*.npz"))
    if allowed is not None:
        filt: List[Path] = []
        for p in files:
            try:
                at = _npz_attack_type_only(p)
            except Exception:
                continue
            if at in allowed:
                filt.append(p)
        files = filt
    if max_files and max_files > 0:
        files = files[: int(max_files)]
    return list(files)


def _load_npz_safe(p: Path) -> Tuple[np.ndarray, np.ndarray, str]:
    with np.load(p, allow_pickle=False) as z:
        X = z["X"].astype(np.float32)
        if X.ndim == 2:
            X = X[..., None]
        if "y_window" in z.files:
            yw = z["y_window"].astype(np.uint8)
        elif "y" in z.files:
            y = z["y"].astype(np.float32)
            if y.ndim == 2 and int(y.shape[0]) == int(X.shape[0]):
                frac = (np.mean(np.abs(y) > 1e-6, axis=1) >= 0.10).astype(np.uint8)
            else:
                frac = np.zeros((X.shape[0],), dtype=np.uint8)
            yw = frac
        else:
            yw = np.zeros((X.shape[0],), dtype=np.uint8)
        at = str(z["attack_type"].item()) if "attack_type" in z.files else "unknown"
    return X, yw, at


def _collect_scores_normal(
    model: Any,
    *,
    recon_target: str,
    w_order_score: float,
    normal_dir: Path,
    files: List[str],
    max_files: int,
    batch_size: int,
) -> np.ndarray:
    parts: List[np.ndarray] = []
    use = files[: int(max_files)] if max_files and max_files > 0 else files
    for name in use:
        fp = (normal_dir / name).resolve()
        if not fp.is_file():
            continue
        x = np.load(fp).astype(np.float32)
        if x.ndim == 2:
            x = x[..., None]
        if x.ndim != 3 or int(x.shape[1]) != 100:
            continue
        xa = augment_univariate_batch(x)
        s = compute_window_scores(
            model,
            x,
            xa,
            recon_target=recon_target,
            w_order=float(w_order_score),
            batch_size=int(batch_size),
        )
        parts.append(s.astype(np.float64))
    return np.concatenate(parts, axis=0) if parts else np.zeros((0,), dtype=np.float64)


def _collect_scores_attacked(
    model: Any,
    *,
    recon_target: str,
    w_order_score: float,
    attacked_dir: Path,
    max_files: int,
    batch_size: int,
    allowed_attack_types: Optional[FrozenSet[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    atypes: List[np.ndarray] = []
    files = _filtered_npz_paths(attacked_dir, max_files, allowed_attack_types)
    for p in files:
        try:
            X, yw, at = _load_npz_safe(p)
        except Exception:
            continue
        xa = augment_univariate_batch(X)
        s = compute_window_scores(
            model,
            X,
            xa,
            recon_target=recon_target,
            w_order=float(w_order_score),
            batch_size=int(batch_size),
        )
        scores.append(s.astype(np.float64))
        labels.append(yw.astype(np.uint8))
        atypes.append(np.array([at] * int(len(s)), dtype=object))
    if not scores:
        return (
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.uint8),
            np.zeros((0,), dtype=object),
        )
    return (
        np.concatenate(scores, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(atypes, axis=0),
    )


def _collect_decomposed_normal(
    model: Any,
    *,
    recon_target: str,
    normal_dir: Path,
    files: List[str],
    max_files: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    s_list: List[np.ndarray] = []
    o_list: List[np.ndarray] = []
    use = files[: int(max_files)] if max_files and max_files > 0 else files
    for name in use:
        fp = (normal_dir / name).resolve()
        if not fp.is_file():
            continue
        x = np.load(fp).astype(np.float32)
        if x.ndim == 2:
            x = x[..., None]
        if x.ndim != 3 or int(x.shape[1]) != 100:
            continue
        xa = augment_univariate_batch(x)
        s, o = compute_window_scores_decomposed(
            model,
            x,
            xa,
            recon_target=recon_target,
            batch_size=int(batch_size),
        )
        s_list.append(np.asarray(s, dtype=np.float64))
        o_list.append(np.asarray(o, dtype=np.float64))
    if not s_list:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    return np.concatenate(s_list, axis=0), np.concatenate(o_list, axis=0)


def _collect_decomposed_attacked(
    model: Any,
    *,
    recon_target: str,
    attacked_dir: Path,
    max_files: int,
    batch_size: int,
    allowed_attack_types: Optional[FrozenSet[str]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s_list: List[np.ndarray] = []
    o_list: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    atypes: List[np.ndarray] = []
    files = _filtered_npz_paths(attacked_dir, max_files, allowed_attack_types)
    for p in files:
        try:
            X, yw, at = _load_npz_safe(p)
        except Exception:
            continue
        xa = augment_univariate_batch(X)
        s, o = compute_window_scores_decomposed(
            model,
            X,
            xa,
            recon_target=recon_target,
            batch_size=int(batch_size),
        )
        s_list.append(np.asarray(s, dtype=np.float64))
        o_list.append(np.asarray(o, dtype=np.float64))
        labels.append(yw.astype(np.uint8))
        atypes.append(np.array([at] * int(len(s)), dtype=object))
    if not s_list:
        z = np.zeros((0,), dtype=np.float64)
        return z, z, np.zeros((0,), dtype=np.uint8), np.zeros((0,), dtype=object)
    return (
        np.concatenate(s_list, axis=0),
        np.concatenate(o_list, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(atypes, axis=0),
    )


def _baseline_map(baseline_csv: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not baseline_csv.is_file():
        return out
    with baseline_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            key = f"{row.get('section','')}|{row.get('metric','')}"
            try:
                out[key] = float(row.get("value", "nan"))
            except Exception:
                continue
    return out


def _recommendation(
    *,
    ps_rec: float,
    ps_gain: float,
    far: float,
    far_delta: float,
    f1: float,
    f1_delta: float,
    bal: float,
    bal_delta: float,
    noise_d: float,
    spike_d: float,
    drift_d: float,
    freeze_d: float,
) -> Tuple[str, str]:
    _ = ps_gain, far, f1, bal
    if far_delta > 0.01:
        return "REJECT_HIGH_FAR", "FAR increased more than +1% vs baseline p99"
    if f1_delta < -0.02:
        return "REJECT_LOW_F1", "Overall F1 dropped more than 2% vs baseline p99"
    if bal_delta < -0.02:
        return "REJECT_LOW_F1", "Balanced accuracy dropped more than 2% vs baseline p99"
    if min(noise_d, spike_d, drift_d, freeze_d) < -0.02:
        return "REJECT_ATTACK_REGRESSION", "At least one baseline attack recall dropped more than 2%"
    if ps_rec < 0.30:
        return "REJECT_WEAK_PATTERN_GAIN", "pattern_shift recall did not reach 30% target"
    return "ACCEPT_CANDIDATE", "Meets acceptance rules (research candidate only; not production)"


def _recommendation_attack_subset(
    *,
    far_delta: float,
    f1_delta: float,
    bal_delta: float,
    noise_d: float,
    spike_d: float,
    drift_d: float,
    freeze_d: float,
) -> Tuple[str, str]:
    if far_delta > 0.01:
        return "REJECT_HIGH_FAR", "FAR increased more than +1% vs baseline p99"
    if f1_delta < -0.02:
        return "REJECT_LOW_F1", "Overall F1 dropped more than 2% vs baseline p99"
    if bal_delta < -0.02:
        return "REJECT_LOW_F1", "Balanced accuracy dropped more than 2% vs baseline p99"
    if min(noise_d, spike_d, drift_d, freeze_d) < -0.02:
        return "REJECT_ATTACK_REGRESSION", "At least one baseline attack recall dropped more than 2%"
    return (
        "SUBSET_EVAL_ONLY",
        "Attack-type subset (pattern_shift excluded); aggregate F1/FAR vs full-threat baseline is not directly comparable",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, required=True)
    ap.add_argument("--baseline-results", type=str, required=True)
    ap.add_argument("--split-file", type=str, required=True)
    ap.add_argument("--normal-dir", type=str, required=True)
    ap.add_argument("--attacked-dir", type=str, required=True)
    ap.add_argument("--models-dir", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--max-attacked-files", type=int, default=0)
    ap.add_argument(
        "--score-dump",
        type=int,
        default=0,
        help="If >0, writes score_dump_example.npz with up to N scores for the first (model,w_order_score) combo",
    )
    ap.add_argument(
        "--conservative-fusion",
        type=str,
        default="off",
        choices=("off", "on"),
        help="If on: structural score only + tail-calibrated order boost (validation-normal); "
        "writes fusion CSVs; does not replace standard w_order sweep unless fusion-only path runs.",
    )
    ap.add_argument(
        "--attack-types-only",
        type=str,
        default="",
        help="Comma-separated attack_type values to include from attacked_v2 (e.g. noise,freeze,spike,drift). "
        "When set, all outputs go under <output-dir>/eval_four_attacks_only/ so full-run CSVs stay untouched; "
        "thresholds are still read from <output-dir>/univariate_pattern_thresholds.json.",
    )
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    split = _load_split(repo, args.split_file)
    test_files = list(split.get("test", []))
    val_files = list(split.get("validation", []))
    normal_dir = _resolve(repo, args.normal_dir)
    attacked_dir = _resolve(repo, args.attacked_dir)
    models_dir = _resolve(repo, args.models_dir)
    out_dir = _resolve(repo, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    at_arg = (getattr(args, "attack_types_only", None) or "").strip()
    allowed_attack_types: Optional[FrozenSet[str]] = None
    if at_arg:
        allowed_attack_types = frozenset(x.strip() for x in at_arg.split(",") if x.strip())
    write_dir = out_dir / "eval_four_attacks_only" if allowed_attack_types else out_dir
    write_dir.mkdir(parents=True, exist_ok=True)
    if allowed_attack_types:
        meta_subset = {
            "attack_types_only": sorted(allowed_attack_types),
            "thresholds_json": str(out_dir / "univariate_pattern_thresholds.json"),
            "writes_to": str(write_dir),
        }
        (write_dir / "eval_subset_metadata.json").write_text(
            json.dumps(meta_subset, indent=2),
            encoding="utf-8",
        )

    baseline_csv = _resolve(repo, Path(args.baseline_results) / "baseline_comparison_summary.csv")
    if not baseline_csv.is_file():
        baseline_csv = _resolve(repo, "backend/experiments/univariate_pattern_solution_experiment/results/baseline_comparison_summary.csv")
    bl = _baseline_map(baseline_csv)
    def bget(section: str, metric: str, default: float = float("nan")) -> float:
        return float(bl.get(f"{section}|{metric}", default))

    thr_path = out_dir / "univariate_pattern_thresholds.json"
    thresholds = _load_json(thr_path) if thr_path.is_file() else {}

    import tensorflow as tf

    model_paths = sorted([p for p in models_dir.glob("univariate_pattern_*.keras") if p.is_file()])
    if not model_paths:
        print("[WARN] No trained models found; writing placeholder metrics files.")
        (write_dir / "overall_univariate_pattern_metrics.csv").write_text(
            "status,no_models_found\n", encoding="utf-8"
        )
        sel_rows: List[Dict[str, Any]] = []
        try:
            from univariate_pattern_plots import make_all_plots  # type: ignore

            make_all_plots(out_dir=write_dir, sel_rows=sel_rows, baseline_csv=baseline_csv)
        except Exception as e:
            print(f"[WARN] plots skipped: {e}")
        try:
            from univariate_pattern_report import write_report  # type: ignore

            write_report(
                write_dir / "univariate_pattern_solution_report.md",
                sel_rows=sel_rows,
                baseline_csv=baseline_csv,
            )
        except Exception as e:
            print(f"[WARN] report skipped: {e}")
        return 0

    w_order_scores = [0.1, 0.25, 0.5, 1.0]
    thr_names = ["p95", "p97", "p99", "p99.5", "p99.7", "3sigma"]

    overall_rows: List[Dict[str, Any]] = []
    per_attack_rows: List[Dict[str, Any]] = []
    ps_rows: List[Dict[str, Any]] = []
    fusion_overall: List[Dict[str, Any]] = []
    fusion_per_attack: List[Dict[str, Any]] = []

    max_files = int(args.max_files or 0)
    max_att = int(args.max_attacked_files or 0)
    score_dump = int(args.score_dump or 0)
    dump_written = False

    for mp in model_paths:
        stem = mp.stem
        meta_path = models_dir / f"{stem}_meta.json"
        recon_target = "original_only"
        if meta_path.is_file():
            meta = _load_json(meta_path)
            recon_target = str(meta.get("recon_target", recon_target))

        try:
            model = tf.keras.models.load_model(mp, compile=False)
        except Exception as e:
            print(f"[WARN] skip model {mp.name}: {e}")
            continue

        model_thr = thresholds.get(stem, {}) if isinstance(thresholds, dict) else {}

        if str(args.conservative_fusion).lower() == "on":
            vs, vo = _collect_decomposed_normal(
                model,
                recon_target=recon_target,
                normal_dir=normal_dir,
                files=val_files,
                max_files=max_files,
                batch_size=int(args.batch_size),
            )
            if len(vs) == 0:
                print(f"[WARN] fusion: no validation scores for {stem}")
            else:
                thr_ref = float(np.quantile(vs, 0.99))
                gate, boost, meta = calibrate_tail_boost(vs, vo)
                val_fused = apply_tail_order_boost(vs, vo, thr_ref, gate_ratio=gate, boost_max=boost)
                thr_op = float(np.quantile(val_fused, 0.99))
                cfg = {
                    "model_stem": stem,
                    "model_file": mp.name,
                    "reconstruction_target": recon_target,
                    "thr_structural_p99_ref": thr_ref,
                    "thr_fused_p99_operational": thr_op,
                    "gate_ratio": gate,
                    "boost_max": boost,
                    "calibration_meta": meta,
                }
                save_fusion_config(write_dir / f"conservative_fusion_{stem}.json", cfg)

                ts, to = _collect_decomposed_normal(
                    model,
                    recon_target=recon_target,
                    normal_dir=normal_dir,
                    files=test_files,
                    max_files=max_files,
                    batch_size=int(args.batch_size),
                )
                as_, ao, y_a, at_a = _collect_decomposed_attacked(
                    model,
                    recon_target=recon_target,
                    attacked_dir=attacked_dir,
                    max_files=max_att,
                    batch_size=int(args.batch_size),
                    allowed_attack_types=allowed_attack_types,
                )
                if len(ts) == 0 or len(as_) == 0:
                    print(f"[WARN] fusion: insufficient test/attack data for {stem}")
                else:
                    s_n_f = apply_tail_order_boost(ts, to, thr_ref, gate_ratio=gate, boost_max=boost)
                    s_a_f = apply_tail_order_boost(as_, ao, thr_ref, gate_ratio=gate, boost_max=boost)
                    y_all = np.concatenate([np.zeros(len(s_n_f), dtype=np.uint8), y_a.astype(np.uint8)], axis=0)
                    s_all = np.concatenate([s_n_f, s_a_f], axis=0)
                    aucs = compute_curves_and_auc(y_all, s_all)
                    cm = confusion_at_threshold(y_all, s_all, thr_op)
                    m = metrics_from_confusion(cm)
                    fusion_overall.append(
                        {
                            "model": mp.name,
                            "model_stem": stem,
                            "reconstruction_target": recon_target,
                            "score_mode": "conservative_tail_order_fusion",
                            "threshold_name": "p99_fused_val",
                            "threshold_value": thr_op,
                            "gate_ratio": gate,
                            "boost_max": boost,
                            "TP": cm.TP,
                            "TN": cm.TN,
                            "FP": cm.FP,
                            "FN": cm.FN,
                            **m,
                            "roc_auc": aucs["roc_auc"],
                            "pr_auc": aucs["pr_auc"],
                        }
                    )
                    for atype in sorted(set(at_a.tolist())):
                        mask = at_a == atype
                        y_sub = y_a[mask]
                        s_sub = s_a_f[mask]
                        if y_sub.size == 0:
                            continue
                        tp = int(np.sum((s_sub > thr_op) & (y_sub == 1)))
                        fn = int(np.sum((s_sub <= thr_op) & (y_sub == 1)))
                        fp = int(np.sum((s_sub > thr_op) & (y_sub == 0)))
                        rec = tp / (tp + fn) if (tp + fn) else 0.0
                        prec = tp / (tp + fp) if (tp + fp) else 0.0
                        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
                        missed = fn / (tp + fn) if (tp + fn) else 0.0
                        fusion_per_attack.append(
                            {
                                "model": mp.name,
                                "attack_type": str(atype),
                                "recall": rec,
                                "precision": prec,
                                "f1": f1,
                                "missed_rate": missed,
                                "avg_score": float(np.mean(s_sub)) if len(s_sub) else 0.0,
                                "min_score": float(np.min(s_sub)) if len(s_sub) else 0.0,
                                "max_score": float(np.max(s_sub)) if len(s_sub) else 0.0,
                            }
                        )

        for wos in w_order_scores:
            wkey = f"w_order_score_{wos}"
            thr_block = model_thr.get(wkey, {}) if isinstance(model_thr, dict) else {}

            s_n = _collect_scores_normal(
                model,
                recon_target=recon_target,
                w_order_score=float(wos),
                normal_dir=normal_dir,
                files=test_files,
                max_files=max_files,
                batch_size=int(args.batch_size),
            )
            s_a, y_a, at_a = _collect_scores_attacked(
                model,
                recon_target=recon_target,
                w_order_score=float(wos),
                attacked_dir=attacked_dir,
                max_files=max_att,
                batch_size=int(args.batch_size),
                allowed_attack_types=allowed_attack_types,
            )
            if len(s_n) == 0 or len(s_a) == 0:
                print(f"[WARN] insufficient scoring data for {stem} wos={wos}")
                continue

            if score_dump > 0 and (not dump_written) and mp == model_paths[0] and abs(float(wos) - float(w_order_scores[0])) < 1e-9:
                k = int(score_dump)
                dump_at = "pattern_shift"
                if allowed_attack_types is not None and "pattern_shift" not in allowed_attack_types:
                    dump_at = sorted(allowed_attack_types)[0] if allowed_attack_types else "noise"
                mask_dump = at_a == dump_at
                s_ps = s_a[mask_dump][:k] if int(np.sum(mask_dump)) else np.zeros((0,), dtype=np.float64)
                np.savez_compressed(
                    write_dir / "score_dump_example.npz",
                    s_normal=np.asarray(s_n[:k], dtype=np.float64),
                    s_ps=np.asarray(s_ps, dtype=np.float64),
                    s_attacked=np.asarray(s_a[:k], dtype=np.float64),
                    dump_attack_type=np.array([dump_at], dtype=object),
                )
                dump_written = True

            y_all = np.concatenate([np.zeros(len(s_n), dtype=np.uint8), y_a.astype(np.uint8)], axis=0)
            s_all = np.concatenate([s_n, s_a], axis=0)
            aucs = compute_curves_and_auc(y_all, s_all)

            for tn in thr_names:
                thr = float(thr_block.get(tn, float("nan")))
                if not np.isfinite(thr):
                    continue
                cm = confusion_at_threshold(y_all, s_all, thr)
                m = metrics_from_confusion(cm)
                overall_rows.append(
                    {
                        "model": mp.name,
                        "model_stem": stem,
                        "reconstruction_target": recon_target,
                        "w_order_score": float(wos),
                        "threshold_name": tn,
                        "threshold_value": thr,
                        "TP": cm.TP,
                        "TN": cm.TN,
                        "FP": cm.FP,
                        "FN": cm.FN,
                        **m,
                        "roc_auc": aucs["roc_auc"],
                        "pr_auc": aucs["pr_auc"],
                    }
                )

                for atype in sorted(set(at_a.tolist())):
                    mask = at_a == atype
                    y_sub = y_a[mask]
                    s_sub = s_a[mask]
                    if y_sub.size == 0:
                        continue
                    tp = int(np.sum((s_sub > thr) & (y_sub == 1)))
                    fn = int(np.sum((s_sub <= thr) & (y_sub == 1)))
                    fp = int(np.sum((s_sub > thr) & (y_sub == 0)))
                    tn_cm = int(np.sum((s_sub <= thr) & (y_sub == 0)))
                    rec = tp / (tp + fn) if (tp + fn) else 0.0
                    prec = tp / (tp + fp) if (tp + fp) else 0.0
                    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
                    missed = fn / (tp + fn) if (tp + fn) else 0.0
                    per_attack_rows.append(
                        {
                            "model": mp.name,
                            "w_order_score": float(wos),
                            "threshold_name": tn,
                            "attack_type": str(atype),
                            "recall": rec,
                            "precision": prec,
                            "f1": f1,
                            "missed_rate": missed,
                            "avg_score": float(np.mean(s_sub)) if len(s_sub) else 0.0,
                            "min_score": float(np.min(s_sub)) if len(s_sub) else 0.0,
                            "max_score": float(np.max(s_sub)) if len(s_sub) else 0.0,
                        }
                    )
                    if str(atype) == "pattern_shift":
                        ps_rows.append(
                            {
                                "model": mp.name,
                                "w_order_score": float(wos),
                                "threshold_name": tn,
                                "pattern_shift_recall": rec,
                                "pattern_shift_precision": prec,
                                "pattern_shift_f1": f1,
                                "pattern_shift_missed_rate": missed,
                            }
                        )

    def _wcsv(path: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)

    _wcsv(write_dir / "overall_univariate_pattern_metrics.csv", overall_rows)
    _wcsv(write_dir / "per_attack_univariate_pattern_metrics.csv", per_attack_rows)
    _wcsv(write_dir / "pattern_shift_focused_metrics.csv", ps_rows)
    _wcsv(write_dir / "overall_conservative_fusion_metrics.csv", fusion_overall)
    _wcsv(write_dir / "per_attack_conservative_fusion_metrics.csv", fusion_per_attack)

    # Model selection vs baseline @ p99
    sel_rows: List[Dict[str, Any]] = []
    base_ps = bget("per_attack_p99", "pattern_shift_recall", 0.1141194898187514)
    base_far = bget("overall_p99", "far", 0.034051387741803496)
    base_f1 = bget("overall_p99", "f1", 0.7943689592379886)
    base_bal = bget("overall_p99", "balanced_accuracy", 0.8736036112484018)
    base_noise = bget("per_attack_p99", "noise_recall", 1.0)
    base_spike = bget("per_attack_p99", "spike_recall", 0.9998952550539436)
    base_drift = bget("per_attack_p99", "drift_recall", 1.0)
    base_freeze = bget("per_attack_p99", "freeze_recall", 0.9969097651421508)

    def _recall_pa(model_name: str, wos: float, atype: str) -> float:
        row = next(
            (
                x
                for x in per_attack_rows
                if x.get("model") == model_name
                and float(x.get("w_order_score", -1.0)) == float(wos)
                and x.get("attack_type") == atype
                and x.get("threshold_name") == "p99"
            ),
            None,
        )
        return float(row["recall"]) if row else float("nan")

    for r in overall_rows:
        if r.get("threshold_name") != "p99":
            continue
        stem = str(r.get("model_stem", ""))
        wtrain = float("nan")
        meta_p = models_dir / f"{stem}_meta.json"
        if meta_p.is_file():
            wtrain = float(_load_json(meta_p).get("w_order_train", float("nan")))

        wos = float(r.get("w_order_score", 0.0))
        model_name = str(r.get("model", ""))

        ps_rec = _recall_pa(model_name, wos, "pattern_shift")
        noise_r = _recall_pa(model_name, wos, "noise")
        spike_r = _recall_pa(model_name, wos, "spike")
        drift_r = _recall_pa(model_name, wos, "drift")
        freeze_r = _recall_pa(model_name, wos, "freeze")
        def _finite_delta(v: float, base: float) -> float:
            d = float(v - base)
            return d if np.isfinite(d) else 0.0

        far = float(r.get("far", 0.0))
        f1 = float(r.get("f1", 0.0))
        bal = float(r.get("balanced_accuracy", 0.0))
        subset_no_ps = (
            allowed_attack_types is not None and "pattern_shift" not in allowed_attack_types
        )
        if subset_no_ps:
            rec, reason = _recommendation_attack_subset(
                far_delta=float(far - base_far),
                f1_delta=float(f1 - base_f1),
                bal_delta=float(bal - base_bal),
                noise_d=_finite_delta(noise_r, base_noise),
                spike_d=_finite_delta(spike_r, base_spike),
                drift_d=_finite_delta(drift_r, base_drift),
                freeze_d=_finite_delta(freeze_r, base_freeze),
            )
        else:
            rec, reason = _recommendation(
                ps_rec=ps_rec,
                ps_gain=float(ps_rec - base_ps),
                far=far,
                far_delta=float(far - base_far),
                f1=f1,
                f1_delta=float(f1 - base_f1),
                bal=bal,
                bal_delta=float(bal - base_bal),
                noise_d=_finite_delta(noise_r, base_noise),
                spike_d=_finite_delta(spike_r, base_spike),
                drift_d=_finite_delta(drift_r, base_drift),
                freeze_d=_finite_delta(freeze_r, base_freeze),
            )
        ps_gain_v = float(ps_rec - base_ps) if np.isfinite(ps_rec) else float("nan")
        sel_rows.append(
            {
                "model_name": r.get("model"),
                "reconstruction_target": r.get("reconstruction_target"),
                "W_order_train": wtrain,
                "W_order_score": wos,
                "threshold_name": "p99",
                "input_shape": "(100,8)_aug",
                "evaluation_scope": "attack_subset" if subset_no_ps else "full",
                "pattern_shift_recall": ps_rec,
                "pattern_shift_gain": ps_gain_v,
                "FAR": far,
                "FAR_delta": float(far - base_far),
                "F1": f1,
                "F1_delta": float(f1 - base_f1),
                "balanced_accuracy": bal,
                "balanced_accuracy_delta": float(bal - base_bal),
                "noise_recall_delta": float(noise_r - base_noise) if np.isfinite(noise_r) else float("nan"),
                "spike_recall_delta": float(spike_r - base_spike) if np.isfinite(spike_r) else float("nan"),
                "drift_recall_delta": float(drift_r - base_drift) if np.isfinite(drift_r) else float("nan"),
                "freeze_recall_delta": float(freeze_r - base_freeze) if np.isfinite(freeze_r) else float("nan"),
                "recommendation": rec,
                "reason": reason,
            }
        )

    _wcsv(write_dir / "model_selection_comparison.csv", sel_rows)

    # Visualizations (best-effort)
    try:
        from univariate_pattern_plots import make_all_plots  # type: ignore

        make_all_plots(out_dir=write_dir, sel_rows=sel_rows, baseline_csv=baseline_csv)
    except Exception as e:
        print(f"[WARN] plots skipped: {e}")

    # Report
    try:
        from univariate_pattern_report import write_report  # type: ignore

        write_report(
            write_dir / "univariate_pattern_solution_report.md",
            sel_rows=sel_rows,
            baseline_csv=baseline_csv,
        )
    except Exception as e:
        print(f"[WARN] report skipped: {e}")

    print(f"Wrote evaluation outputs under: {write_dir}")
    if write_dir.resolve() != out_dir.resolve():
        print(f"(Full-run parent dir unchanged: {out_dir}; thresholds read from: {thr_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
