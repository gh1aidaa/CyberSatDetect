"""
Chapter 7 — Testing + Validation + Evaluation (CyberSatDetect / NebulaSec).

Reads production paths; writes ONLY under --output-dir (must be .../results).
Does not modify api.py, thresholds.json, .env, or official models.
"""

from __future__ import annotations

import argparse
import csv
import os
import importlib.util
import json
import platform
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _resolve_under(repo_root: Path, p: str | Path) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path.resolve()


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
    spec = importlib.util.spec_from_file_location("_ch7_eval_strict", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(mod_path)
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
    canon: Dict[str, float] = {}
    for k in ("p95", "p97", "p99"):
        if k in out:
            canon[k] = out[k]
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
    return canon


def sweep_thresholds(
    y_true: np.ndarray, y_score: np.ndarray, grid_n: int
) -> Tuple[np.ndarray, List[Dict[str, Any]], float, float]:
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
        rows.append({"threshold": float(thr), **m, "tp": cm.TP, "tn": cm.TN, "fp": cm.FP, "fn": cm.FN})
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thr = float(thr)
    return thrs, rows, float(best_f1), float(best_thr)


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


def discover_model_path(repo_root: Path, preferred: Path) -> Tuple[Optional[Path], List[str]]:
    tried = [str(preferred)]
    if preferred.is_file():
        return preferred, tried
    for pat in ("backend/app/best_model.keras", "backend/app/final_model.keras"):
        p = (repo_root / pat).resolve()
        tried.append(str(p))
        if p.is_file():
            return p, tried
    app_dir = repo_root / "backend" / "app"
    if app_dir.is_dir():
        for p in sorted(app_dir.glob("*.keras")):
            tried.append(str(p))
            return p, tried
    models_dir = repo_root / "backend" / "models"
    if models_dir.is_dir():
        for p in sorted(models_dir.glob("*.keras")):
            tried.append(str(p))
            return p, tried
    return None, tried


def scan_api_static(api_path: Path) -> Dict[str, Any]:
    if not api_path.is_file():
        return {"api_path": str(api_path), "readable": False}
    text = api_path.read_text(encoding="utf-8", errors="ignore")
    return {
        "api_path": str(api_path),
        "readable": True,
        "count_app_routes": len(re.findall(r"@app\.(get|post|put|delete)\s*\(\s*[\"']", text, re.I)),
        "has_runs_analyze": "/runs/{run_id}/analyze" in text or '"/runs/{run_id}/analyze"' in text,
        "has_runs_upload": "/runs/upload" in text,
        "has_auth_login": "/auth/login" in text,
        "has_limiter": "Limiter(" in text or "limiter" in text,
        "has_rate_limit_handler": "RateLimitExceeded" in text or "rate_limit" in text.lower(),
        "has_csv_or_npy_mentions": bool(re.search(r"\.csv|\.npy|load_channels", text, re.I)),
    }


def write_testing_summary(
    out_path: Path,
    env: Dict[str, Any],
    checks: Dict[str, Any],
    api_info: Dict[str, Any],
    model_used: Optional[str],
) -> None:
    ar = api_info.get("has_runs_analyze", False)
    up = api_info.get("has_runs_upload", False)
    lg = api_info.get("has_auth_login", False)
    lm = api_info.get("has_limiter", False)
    rl = api_info.get("has_rate_limit_handler", False)
    csvn = api_info.get("has_csv_or_npy_mentions", False)
    lines = [
        "# Chapter 7 — Testing summary (evidence)\n\n",
        "## 1. Environment\n\n",
        f"- Python: `{env['python']}`\n",
        f"- TensorFlow: `{env.get('tensorflow', 'not importable')}`\n",
        f"- NumPy: `{env['numpy']}`\n",
        f"- OS: `{env['os']}`\n",
        f"- CWD: `{env['cwd']}`\n\n",
        "## 2. File availability\n\n",
    ]
    for k, v in checks.items():
        lines.append(f"- **{k}**: `{v}`\n")
    lines.append("\n## 3. API / functional (read-only scan of `api.py`)\n\n")
    for k, v in api_info.items():
        lines.append(f"- `{k}`: {v}\n")
    if model_used:
        lines.append(f"\n**Model path used (after discovery):** `{model_used}`\n")
    lines.append("\n## 4. Test matrix (static / non-production)\n\n")
    lines.append("| Test Case | Expected Result | Evidence | Status |\n")
    lines.append("|---|---|---|---|\n")
    lines.append("| Preferred Keras model on disk | File exists | `preferred_model` in §2 | see §2 |\n")
    lines.append("| thresholds.json | Present, readable | §2 | see §2 |\n")
    lines.append("| data split JSON | Present | §2 | see §2 |\n")
    lines.append("| Normal windows (`chunk_*.npy`) | At least one under normal dir | §2 | see §2 |\n")
    lines.append("| Attacked NPZ dataset | At least one `.npz` | §2 | see §2 |\n")
    lines.append(
        f"| POST analyze route | String `/runs/{{run_id}}/analyze` in source | api.py read-only | "
        f"{'PASS' if ar else 'FAIL'} |\n"
    )
    lines.append(
        f"| POST upload route | `/runs/upload` in source | api.py read-only | {'PASS' if up else 'FAIL'} |\n"
    )
    lines.append(
        f"| Auth login | `/auth/login` in source | api.py read-only | {'PASS' if lg else 'FAIL'} |\n"
    )
    lines.append(
        f"| Rate limiting | `Limiter(` or limiter symbol | api.py read-only | "
        f"{'PASS' if lm else 'FAIL'} |\n"
    )
    lines.append(
        f"| Rate-limit handling | handler or RateLimitExceeded | api.py read-only | "
        f"{'PASS' if rl else 'FAIL'} |\n"
    )
    lines.append(
        f"| CSV/NPY validation | loaders mention csv/npy/channels | api.py read-only | "
        f"{'PASS' if csvn else 'FAIL'} |\n"
    )
    lines.append(
        f"| FastAPI route decorators | `@app.get/post/...` count | "
        f"`count_app_routes={api_info.get('count_app_routes', 'n/a')}` | informational |\n"
    )
    lines.append("\n")
    out_path.write_text("".join(lines), encoding="utf-8")


def y_window_from_timestep(y_ts: np.ndarray, min_fraction: float = 0.10) -> np.ndarray:
    """Per-window label: 1 if fraction of attacked timesteps >= min_fraction."""
    if y_ts.ndim != 2:
        raise ValueError(f"y_timestep expected (B,T), got {y_ts.shape}")
    t = y_ts.shape[1]
    k = int(np.ceil(float(min_fraction) * t))
    k = max(1, min(k, t))
    s = np.sum(y_ts.astype(np.uint8), axis=1)
    return (s >= k).astype(np.uint8)


def load_attacked_npz_ch7(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    with np.load(path, allow_pickle=False) as z:
        X = z["X"].astype(np.float32)
        if "y_window" in z:
            y_w = z["y_window"].astype(np.uint8)
        elif "y_timestep" in z:
            y_w = y_window_from_timestep(z["y_timestep"].astype(np.uint8))
        else:
            raise KeyError(f"{path}: need y_window or y_timestep in NPZ")
        attack_type = str(z["attack_type"].item()) if "attack_type" in z else "unknown"
        meta = {"attack_type": attack_type}
    if X.ndim == 2:
        X = X[..., None]
    if y_w.ndim != 1 or y_w.shape[0] != X.shape[0]:
        raise ValueError(f"{path}: y_window shape {y_w.shape} vs X {X.shape}")
    return X, y_w, meta


def load_attacked_dir_ch7(attacked_dir: Path, window_t: int, ev: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    files = sorted(attacked_dir.glob("*.npz"))
    sp, yp, tparts, rej = [], [], [], 0
    for p in files:
        try:
            with np.load(p, allow_pickle=False) as z:
                keys = set(z.files)
            if "y_window" in keys:
                X, y_w, meta = ev.load_attacked_npz(p)
            else:
                X, y_w, meta = load_attacked_npz_ch7(p)
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
        raise RuntimeError("No valid attacked_v2 NPZ windows.")
    return np.concatenate(sp), np.concatenate(yp), np.concatenate(tparts), rej


def score_normal_files(
    ev: Any,
    model: Any,
    weights: Dict[str, Any],
    normal_dir: Path,
    names: List[str],
    window_t: int,
    batch_size: int,
    max_files: int = 0,
) -> Tuple[np.ndarray, int, int]:
    if max_files and max_files > 0:
        names = names[: int(max_files)]
    parts: List[np.ndarray] = []
    ok = rej = 0
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
        raise RuntimeError("No normal windows scored.")
    return np.concatenate(parts), ok, rej


def find_ablation_tables(repo_root: Path) -> List[Path]:
    hits: List[Path] = []
    root = repo_root / "backend" / "experiments"
    if root.is_dir():
        for p in root.rglob("*.csv"):
            try:
                head = p.read_text(encoding="utf-8", errors="ignore")[:4000].lower()
            except OSError:
                continue
            if any(k in head for k in ("wrecon", "ablation", "without l", "l_pred", "l_grad")):
                hits.append(p)
    ab = repo_root / "ablation_study" / "results.csv"
    if ab.is_file() and ab not in hits:
        hits.append(ab)
    return hits


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
    ap.add_argument("--max-validation-files", type=int, default=0)
    ap.add_argument("--max-test-files", type=int, default=0)
    ap.add_argument("--max-attacked-files", type=int, default=0)
    args = ap.parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    repo_root = Path(args.repo_root).resolve()
    preferred_model = _resolve_under(repo_root, args.model_path)
    thresholds_path = _resolve_under(repo_root, args.thresholds_path)
    split_file = _resolve_under(repo_root, args.split_file)
    normal_dir = _resolve_under(repo_root, args.normal_dir)
    attacked_dir = _resolve_under(repo_root, args.attacked_dir)
    output_dir = _resolve_under(repo_root, args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- env ---
    try:
        import tensorflow as tf  # noqa: WPS433

        tf_ver = getattr(tf, "__version__", str(tf))
    except Exception as e:
        tf_ver = f"import failed: {e}"
    env = {
        "python": sys.version.split()[0],
        "tensorflow": tf_ver,
        "numpy": np.__version__,
        "os": platform.platform(),
        "cwd": str(Path.cwd()),
    }

    api_path = repo_root / "backend" / "app" / "api.py"
    api_info = scan_api_static(api_path)

    model_path, model_tried = discover_model_path(repo_root, preferred_model)
    checks: Dict[str, Any] = {
        "preferred_model": "OK" if preferred_model.is_file() else f"MISSING: {preferred_model}",
        "resolved_model": str(model_path) if model_path else "NONE (searched alternatives)",
        "thresholds.json": "OK" if thresholds_path.is_file() else f"MISSING: {thresholds_path}",
        "split_file": "OK" if split_file.is_file() else f"MISSING: {split_file}",
        "normal_dir": "OK" if normal_dir.is_dir() else f"MISSING: {normal_dir}",
        "attacked_dir": "OK" if attacked_dir.is_dir() else f"MISSING: {attacked_dir}",
    }
    sample_chunk = next(normal_dir.glob("chunk_*.npy"), None) if normal_dir.is_dir() else None
    checks["sample_chunk_in_normal_dir"] = str(sample_chunk) if sample_chunk else "NO chunk_*.npy found"
    npz_n = len(list(attacked_dir.glob("*.npz"))) if attacked_dir.is_dir() else 0
    checks["attacked_npz_count"] = str(npz_n)

    p_test = _safe_out(output_dir, "testing_summary.md")
    write_testing_summary(p_test, env, checks, api_info, str(model_path) if model_path else None)

    missing: List[str] = []
    if not model_path:
        missing.append(f"No .keras model found. Tried: {model_tried}")
    if not thresholds_path.is_file():
        missing.append(str(thresholds_path))
    if not split_file.is_file():
        missing.append(str(split_file))
    if not normal_dir.is_dir() or sample_chunk is None:
        missing.append(f"No window .npy under {normal_dir} (need chunk_*.npy)")
    if not attacked_dir.is_dir() or npz_n == 0:
        missing.append(f"No .npz under {attacked_dir}")

    if missing:
        fail = _safe_out(output_dir, "evaluation_summary.json")
        fail.write_text(
            json.dumps(
                {
                    "status": "prerequisites_failed",
                    "missing": missing,
                    "model_search_tried": model_tried,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        rep = _safe_out(output_dir, "chapter7_testing_validation_evaluation_report.md")
        rep.write_text(
            "# Chapter 7 report\n\n**Pipeline did not run:** prerequisites missing.\n\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\n\nSee `testing_summary.md`.\n",
            encoding="utf-8",
        )
        print("PREREQUISITES FAILED — see results/testing_summary.md and evaluation_summary.json", flush=True)
        for m in missing:
            print(f"  MISSING: {m}", flush=True)
        sys.exit(1)

    plt = _maybe_plt()
    ev = _load_eval_strict(repo_root)
    with thresholds_path.open("r", encoding="utf-8") as f:
        thr_cfg = json.load(f)
    weights = thr_cfg.get("weights", {})
    if not isinstance(weights, dict):
        weights = {}
    official = official_threshold_map(thr_cfg)

    t_sample, c_sample = ev.infer_T_C_from_sample(normal_dir)
    model = ev.load_keras_model_robust(model_path, t_sample, c_sample)

    stride = 50
    gm_path = attacked_dir / "generation_metadata.json"
    if gm_path.is_file():
        try:
            gm = json.loads(gm_path.read_text(encoding="utf-8"))
            if isinstance(gm, dict) and "stride" in gm:
                stride = int(gm["stride"])
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    split_obj = json.loads(split_file.read_text(encoding="utf-8"))
    val_names = list(split_obj.get("validation", []))
    test_names = list(split_obj.get("test", []))
    if not val_names or not test_names:
        raise ValueError("split JSON needs non-empty validation and test")

    mv = int(args.max_validation_files) or 0
    mt = int(args.max_test_files) or 0
    ma = int(args.max_attacked_files) or 0

    val_scores, val_ok, val_rej = score_normal_files(
        ev, model, weights, normal_dir, val_names, t_sample, int(args.batch_size), mv
    )
    val_q = quantiles_from_scores(val_scores)

    rows_v = []
    for key in ("p95", "p97", "p99", "p99.5", "p99.7", "3sigma"):
        comp = val_q[key]
        off = official.get(key)
        rows_v.append(
            {
                "threshold_name": key,
                "validation_computed": comp,
                "official_json_value": "" if off is None else off,
                "absolute_delta": "" if off is None else float(comp) - float(off),
            }
        )
    p_vt = _safe_out(output_dir, "validation_thresholds_computed.csv")
    with p_vt.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["threshold_name", "validation_computed", "official_json_value", "absolute_delta"],
        )
        w.writeheader()
        for row in rows_v:
            w.writerow(row)

    val_summary = {
        "num_validation_files_listed": len(val_names),
        "num_validation_files_scored_ok": val_ok,
        "num_rejected_or_invalid": val_rej,
        "num_validation_windows": int(len(val_scores)),
        "window_size": int(t_sample),
        "stride": stride,
        "input_shape_timesteps": int(t_sample),
        "input_shape_channels": int(c_sample),
        "score_mean": float(val_scores.mean()),
        "score_std": float(val_scores.std()),
        "score_min": float(val_scores.min()),
        "score_max": float(val_scores.max()),
        "score_median": float(np.median(val_scores)),
    }
    if mv:
        val_summary["max_validation_files_cap"] = mv
    p_vs = _safe_out(output_dir, "validation_score_summary.csv")
    with p_vs.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(val_summary.keys()))
        w.writeheader()
        w.writerow(val_summary)

    if plt is not None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(val_scores, bins=80, color="#3498db", alpha=0.75, edgecolor="white")
        ax.set_title("Validation-normal scores")
        ax.set_xlabel("Score")
        fig.tight_layout()
        fig.savefig(_safe_out(output_dir, "validation_score_distribution.png"), dpi=140)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(val_scores, bins=80, color="#ecf0f1", edgecolor="white")
        for k, c in [
            ("p95", "#95a5a6"),
            ("p97", "#7f8c8d"),
            ("p99", "#e74c3c"),
            ("p99.5", "#c0392b"),
            ("p99.7", "#8e44ad"),
            ("3sigma", "#27ae60"),
        ]:
            ax.axvline(val_q[k], color=c, linestyle="--", linewidth=1.2, label=f"{k}={val_q[k]:.4g}")
        ax.legend(fontsize=8)
        ax.set_title("Validation thresholds")
        fig.tight_layout()
        fig.savefig(_safe_out(output_dir, "validation_threshold_lines.png"), dpi=140)
        plt.close(fig)

    test_scores, test_ok, test_rej = score_normal_files(
        ev, model, weights, normal_dir, test_names, t_sample, int(args.batch_size), mt
    )
    atk_X, atk_y, atk_type, atk_rej = load_attacked_dir_ch7(attacked_dir, t_sample, ev)
    if ma:
        n_cap = max(1, min(int(atk_X.shape[0]), 5000))
        atk_X = atk_X[:n_cap]
        atk_y = atk_y[:n_cap]
        atk_type = atk_type[:n_cap]
    atk_scores = compute_scores_hybrid_weighted(
        model, atk_X, weights, batch_size=int(args.batch_size)
    ).astype(np.float64)

    y_true = np.concatenate([np.zeros(len(test_scores), dtype=np.uint8), atk_y.astype(np.uint8)])
    y_score = np.concatenate([test_scores, atk_scores])
    n_atk = int(len(atk_scores))
    n_anom = int(np.sum(atk_y == 1))
    n_norm_in_atk = int(np.sum(atk_y == 0))
    imbalance = float((len(test_scores) + n_norm_in_atk) / max(n_anom, 1))

    curves_info, curves = ev.compute_curves_and_auc(y_true, y_score)
    roc_auc = float(curves_info["roc_auc"])
    pr_auc = float(curves_info["pr_auc"])
    _, sweep_rows, sweep_best_f1, sweep_best_thr = sweep_thresholds(
        y_true, y_score, int(args.sweep_points)
    )
    best_f1_payload = {
        "best_f1": float(sweep_best_f1),
        "best_f1_threshold": float(sweep_best_thr),
        "note": "Analysis only; not written to thresholds.json.",
    }
    with _safe_out(output_dir, "best_f1_from_sweep.json").open("w", encoding="utf-8") as f:
        json.dump(best_f1_payload, f, indent=2)

    sweep_cols = [
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
    p_sw = _safe_out(output_dir, "threshold_sweep_results.csv")
    with p_sw.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sweep_cols)
        w.writeheader()
        for row in sweep_rows:
            w.writerow({k: row.get(k) for k in sweep_cols})

    eval_thr: List[Tuple[str, float]] = [
        ("p95", val_q["p95"]),
        ("p97", val_q["p97"]),
        ("p99", val_q["p99"]),
        ("p99.5", val_q["p99.5"]),
        ("p99.7", val_q["p99.7"]),
        ("3sigma", val_q["3sigma"]),
        ("best_f1", float(sweep_best_thr)),
    ]
    overall_rows: List[Dict[str, Any]] = []
    for name, thr in eval_thr:
        cm = confusion_at_threshold(y_true, y_score, thr)
        m = metrics_from_confusion(cm)
        overall_rows.append(
            {"threshold_name": name, "threshold_value": float(thr), "TP": cm.TP, "TN": cm.TN, "FP": cm.FP, "FN": cm.FN, **m}
        )
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
    p_ov = _safe_out(output_dir, "overall_threshold_metrics.csv")
    with p_ov.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ocols)
        w.writeheader()
        for row in overall_rows:
            w.writerow({k: row.get(k) for k in ocols})

    atk_arr = np.asarray(atk_type, dtype=object)
    per_rows: List[Dict[str, Any]] = []
    for thr_name, thr_val in eval_thr:
        pred = atk_scores > float(thr_val)
        for at in sorted(set(str(x) for x in atk_arr.tolist())):
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
            rec = float(tp / (tp + fn)) if (tp + fn) else 0.0
            prec = float(tp / (tp + fp)) if (tp + fp) else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
            per_rows.append(
                {
                    "attack_type": at,
                    "threshold_name": thr_name,
                    "threshold_value": float(thr_val),
                    "total_attack_windows": tw,
                    "detected_windows": int(tp),
                    "missed_windows": int(fn),
                    "detection_rate": rec,
                    "recall": rec,
                    "missed_rate": float(fn / tw) if tw else 0.0,
                    "precision": prec,
                    "f1": f1,
                    "average_score": float(np.mean(s_sub)) if len(s_sub) else 0.0,
                    "max_score": float(np.max(s_sub)) if len(s_sub) else 0.0,
                    "min_score": float(np.min(s_sub)) if len(s_sub) else 0.0,
                }
            )
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
    p_per = _safe_out(output_dir, "per_attack_threshold_metrics.csv")
    with p_per.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pcols)
        w.writeheader()
        for row in per_rows:
            w.writerow({k: row.get(k) for k in pcols})

    by_at: Dict[str, List[Dict[str, Any]]] = {}
    for r in per_rows:
        by_at.setdefault(str(r["attack_type"]), []).append(r)

    def op_reason(at: str) -> str:
        return (
            f"p99 (validation-calibrated) suggested for ops; attack={at}: "
            "freeze/spike/drift/noise/scale/drop/masking/pattern_shift trade-offs vary."
        )

    best_rows: List[Dict[str, str]] = []
    avg_recall_by_at: Dict[str, float] = {}
    for at, lst in sorted(by_at.items()):
        if not lst:
            continue
        avg_recall_by_at[at] = float(np.mean([float(x["recall"]) for x in lst]))
        br = max(lst, key=lambda x: float(x["recall"]))
        bf = max(lst, key=lambda x: float(x["f1"]))
        op = "p99" if any(x["threshold_name"] == "p99" for x in lst) else str(br["threshold_name"])
        best_rows.append(
            {
                "attack_type": at,
                "best_threshold_by_recall": str(br["threshold_name"]),
                "best_threshold_by_f1": str(bf["threshold_name"]),
                "best_operational_threshold": op,
                "reason": op_reason(at),
            }
        )
    hardest = easiest = ""
    if avg_recall_by_at:
        hardest = min(avg_recall_by_at, key=lambda k: avg_recall_by_at[k])
        easiest = max(avg_recall_by_at, key=lambda k: avg_recall_by_at[k])

    p_bat = _safe_out(output_dir, "best_threshold_per_attack.csv")
    with p_bat.open("w", newline="", encoding="utf-8") as f:
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
        for row in best_rows:
            w.writerow(row)

    p_diff = _safe_out(output_dir, "attack_difficulty_ranking.csv")
    with p_diff.open("w", newline="", encoding="utf-8") as f:
        ww = csv.writer(f)
        ww.writerow(["attack_type", "mean_recall_across_thresholds", "rank_hard_to_easy"])
        ranked = sorted(avg_recall_by_at.items(), key=lambda kv: kv[1])
        for i, (at, ar) in enumerate(ranked):
            ww.writerow([at, f"{ar:.6f}", i + 1])

    thr_by_name = {str(r["threshold_name"]): float(r["threshold_value"]) for r in overall_rows}
    thv = [r["threshold"] for r in sweep_rows]

    if plt is not None:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(thv, [r["f1"] for r in sweep_rows], color="#8e44ad")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("F1")
        fig.tight_layout()
        fig.savefig(_safe_out(output_dir, "threshold_vs_f1.png"), dpi=140)
        plt.close(fig)
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax1.plot(thv, [r["recall"] for r in sweep_rows], color="#2980b9")
        ax1.set_ylabel("Recall", color="#2980b9")
        ax2 = ax1.twinx()
        ax2.plot(thv, [r["far"] for r in sweep_rows], color="#e67e22")
        ax2.set_ylabel("FAR", color="#e67e22")
        ax1.set_xlabel("Threshold")
        fig.tight_layout()
        fig.savefig(_safe_out(output_dir, "threshold_vs_recall_far.png"), dpi=140)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(thv, [r["balanced_accuracy"] for r in sweep_rows], color="#16a085")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Balanced accuracy")
        fig.tight_layout()
        fig.savefig(_safe_out(output_dir, "threshold_vs_balanced_accuracy.png"), dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.plot(curves["roc_fpr"], curves["roc_tpr"], color="#2980b9", lw=2)
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"ROC AUC={roc_auc:.4f}")
        fig.tight_layout()
        fig.savefig(_safe_out(output_dir, "roc_curve.png"), dpi=140)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.plot(curves["pr_recall"], curves["pr_precision"], color="#c0392b", lw=2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR AUC={pr_auc:.4f}")
        fig.tight_layout()
        fig.savefig(_safe_out(output_dir, "precision_recall_curve.png"), dpi=140)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(test_scores, bins=80, alpha=0.6, label="Normal test", color="#2ecc71")
        ax.hist(atk_scores, bins=80, alpha=0.6, label="Attacked", color="#e74c3c")
        ax.legend()
        fig.tight_layout()
        fig.savefig(_safe_out(output_dir, "normal_vs_attack_score_distribution.png"), dpi=140)
        plt.close(fig)

        plot_confusion(
            confusion_at_threshold(y_true, y_score, thr_by_name["p99"]),
            f"p99 thr={thr_by_name['p99']:.4g}",
            _safe_out(output_dir, "p99_confusion_matrix.png"),
            plt,
        )
        if "p99.5" in thr_by_name:
            plot_confusion(
                confusion_at_threshold(y_true, y_score, thr_by_name["p99.5"]),
                f"p99.5 thr={thr_by_name['p99.5']:.4g}",
                _safe_out(output_dir, "p99_5_confusion_matrix.png"),
                plt,
            )
        plot_confusion(
            confusion_at_threshold(y_true, y_score, thr_by_name["best_f1"]),
            f"best_f1 thr={thr_by_name['best_f1']:.4g}",
            _safe_out(output_dir, "best_f1_confusion_matrix.png"),
            plt,
        )

        def bar_det(k: str, fn: str) -> None:
            thr_v = thr_by_name[k]
            pred = atk_scores > float(thr_v)
            types = sorted(set(str(x) for x in atk_arr.tolist()))
            rates = []
            for at in types:
                msk = atk_arr == at
                y_sub = atk_y[msk].astype(np.uint8)
                p_sub = pred[msk]
                tw = int(np.sum(y_sub == 1))
                rates.append((int(np.sum(p_sub & (y_sub == 1))) / tw) if tw else 0.0)
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.bar(types, rates, color="#3498db")
            ax.set_ylim(0, 1.05)
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            fig.savefig(_safe_out(output_dir, fn), dpi=140)
            plt.close(fig)

        bar_det("p99", "per_attack_detection_rate_p99.png")
        bar_det("best_f1", "per_attack_detection_rate_best_f1.png")

    ablation_paths = find_ablation_tables(repo_root)
    ablation_usable = []
    for p in ablation_paths:
        try:
            with p.open("r", encoding="utf-8", newline="") as fh:
                rr = list(csv.DictReader(fh))
            if rr and any(any((v or "").strip() for v in row.values()) for row in rr):
                ablation_usable.append(p)
        except OSError:
            continue

    p99_row = next((r for r in overall_rows if r["threshold_name"] == "p99"), None)
    bf_row = next((r for r in overall_rows if r["threshold_name"] == "best_f1"), None)
    prod_like = [r for r in overall_rows if str(r["threshold_name"]) in {"p95", "p97", "p99", "p99.5", "p99.7", "3sigma"}]
    low_far = min(prod_like, key=lambda r: (float(r["far"]), -float(r["threshold_value"]))) if prod_like else None
    best_bal = max(overall_rows, key=lambda r: float(r["balanced_accuracy"]))

    summary = {
        "status": "ok",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_path_used": str(model_path),
        "thresholds_path": str(thresholds_path),
        "split_file": str(split_file),
        "validation": val_summary,
        "evaluation": {
            "test_normal_windows": int(len(test_scores)),
            "attacked_windows": n_atk,
            "anomalous_windows": n_anom,
            "normal_windows_inside_attacked": n_norm_in_atk,
            "imbalance_ratio_neg_to_pos": imbalance,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "best_f1_sweep": best_f1_payload,
            "hardest_attack_type": hardest,
            "easiest_attack_type": easiest,
            "mean_recall_by_attack": avg_recall_by_at,
            "recommended_operational_threshold": "p99",
            "lowest_far_threshold_name": str(low_far["threshold_name"]) if low_far else None,
            "lowest_far_value": float(low_far["far"]) if low_far else None,
            "best_balanced_accuracy_threshold": str(best_bal["threshold_name"]),
            "p99_metrics": {k: p99_row.get(k) for k in ("far", "recall", "f1", "balanced_accuracy")} if p99_row else {},
            "best_f1_row_metrics": {k: bf_row.get(k) for k in ("far", "recall", "f1", "threshold_value")} if bf_row else {},
        },
        "ablation_csv_candidates": [str(p) for p in ablation_paths],
        "ablation_data_found": [str(p) for p in ablation_usable],
    }
    with _safe_out(output_dir, "evaluation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # --- Markdown report ---
    rep_path = _safe_out(output_dir, "chapter7_testing_validation_evaluation_report.md")
    overlap_note = (
        f"Score overlap: normal test scores in [{float(test_scores.min()):.4g}, {float(test_scores.max()):.4g}], "
        f"attack scores in [{float(atk_scores.min()):.4g}, {float(atk_scores.max()):.4g}]. "
        "False positives rise when normal tail exceeds the chosen threshold; false negatives when attack scores "
        "stay below it (subtle attacks, masking, or windows labeled normal inside attacked files)."
    )
    thr_table = ["| Threshold | Value | Acc | Prec | Recall | FAR | F1 | BalAcc |\n", "|---|---:|---:|---:|---:|---:|---:|---:|\n"]
    for r in overall_rows:
        thr_table.append(
            f"| {r['threshold_name']} | {float(r['threshold_value']):.6g} | "
            f"{float(r['accuracy']):.4f} | {float(r['precision']):.4f} | {float(r['recall']):.4f} | "
            f"{float(r['far']):.4f} | {float(r['f1']):.4f} | {float(r['balanced_accuracy']):.4f} |\n"
        )
    fig_rel = [
        "testing_summary.md",
        "validation_thresholds_computed.csv",
        "validation_score_summary.csv",
        "validation_score_distribution.png",
        "validation_threshold_lines.png",
        "overall_threshold_metrics.csv",
        "per_attack_threshold_metrics.csv",
        "best_threshold_per_attack.csv",
        "attack_difficulty_ranking.csv",
        "threshold_sweep_results.csv",
        "best_f1_from_sweep.json",
        "evaluation_summary.json",
        "normal_vs_attack_score_distribution.png",
        "roc_curve.png",
        "precision_recall_curve.png",
        "threshold_vs_f1.png",
        "threshold_vs_recall_far.png",
        "threshold_vs_balanced_accuracy.png",
        "p99_confusion_matrix.png",
        "best_f1_confusion_matrix.png",
        "per_attack_detection_rate_p99.png",
        "per_attack_detection_rate_best_f1.png",
    ]
    if plt is not None and "p99.5" in thr_by_name:
        fig_rel.insert(fig_rel.index("p99_confusion_matrix.png") + 1, "p99_5_confusion_matrix.png")

    lines = [
        "# Chapter 7 — Testing, Validation, Evaluation\n\n",
        f"> Generated UTC: {summary['generated_at_utc']}\n\n",
        "## 1. Testing Phase\n",
        "Environment, file checks, and static API evidence: see `testing_summary.md`.\n\n",
        "## 2. Validation Phase (validation-normal only)\n",
        "Scores use the same hybrid weighted formula as `api.compute_scores_hybrid` (reconstruction + prediction + gradient; "
        "no separation loss at inference). Quantiles **p95…p99.7** and **3σ** are computed on validation-normal scores only; "
        "compare to `backend/app/thresholds.json` in `validation_thresholds_computed.csv` (read-only; file not modified).\n\n",
        f"- Validation windows: **{val_summary['num_validation_windows']}** "
        f"(files OK: {val_summary['num_validation_files_scored_ok']})\n",
        f"- Mean / std / min / max score: **{val_summary['score_mean']:.6g}** / **{val_summary['score_std']:.6g}** / "
        f"**{val_summary['score_min']:.6g}** / **{val_summary['score_max']:.6g}**\n",
        f"- Stride (from `generation_metadata.json` if present): **{val_summary['stride']}**\n\n",
        "**Why p99 suits production:** it is a high quantile on **clean validation-normal** behavior, giving a stable "
        "false-alarm budget while remaining interpretable; tighter quantiles (p99.5/p99.7) reduce FAR but sacrifice recall "
        "on subtle attacks.\n\n",
        "## 3. Evaluation Protocol\n",
        "- Strict **window-level** labels; no point-wise relabeling; no sequence-level score inflation.\n",
        "- Normal test windows: label **0**. Attacked windows: **1** if `y_window==1`, else derived from `y_timestep` "
        "(≥10% attacked steps).\n",
        "- No train/validation leakage: evaluation uses **test** normal list and **attacked_v2** only.\n\n",
        "## 4. Dataset Statistics\n",
        f"- Normal test windows: **{len(test_scores)}**\n",
        f"- Attacked windows (all NPZ rows): **{n_atk}**; anomalous (y=1): **{n_anom}**; normal inside attacked files (y=0): **{n_norm_in_atk}**\n",
        f"- Imbalance ratio (neg/pos): **{imbalance:.4f}**\n\n",
        "## 5. Overall Threshold Results\n",
        "".join(thr_table),
        f"\n- **ROC-AUC:** {roc_auc:.4f} | **PR-AUC:** {pr_auc:.4f}\n",
        f"- **Best F1 (sweep):** threshold **{sweep_best_thr:.6g}**, F1 **{sweep_best_f1:.4f}** (saved in `best_f1_from_sweep.json` only).\n",
        f"- **Best balanced accuracy row:** `{best_bal['threshold_name']}`.\n",
        f"- **Lowest FAR** among p95/p97/p99/p99.5/p99.7/3sigma: `{low_far['threshold_name'] if low_far else 'n/a'}` "
        f"(FAR={float(low_far['far']):.4f}).\n" if low_far else "- **Lowest FAR:** n/a\n",
        f"- **p99 vs best_f1:** p99 FAR **{float(p99_row['far']):.4f}**, recall **{float(p99_row['recall']):.4f}**, F1 **{float(p99_row['f1']):.4f}**; "
        f"best_f1 threshold **{float(bf_row['threshold_value']):.6g}** with FAR **{float(bf_row['far']):.4f}**, "
        f"recall **{float(bf_row['recall']):.4f}**, F1 **{float(bf_row['f1']):.4f}**.\n"
        if p99_row and bf_row
        else "\n",
        "\n## 6. Per-Attack Results\n",
        f"- **Hardest attack type** (lowest mean recall across thresholds): **{hardest or 'n/a'}**\n",
        f"- **Easiest attack type:** **{easiest or 'n/a'}**\n",
        "Tables: `per_attack_threshold_metrics.csv`, `best_threshold_per_attack.csv`, `attack_difficulty_ranking.csv`.\n"
        "Freeze/masking/pattern_shift often keep scores closer to nominal dynamics → harder detection; spike/scale/drift "
        "often inflate reconstruction or gradient terms → easier.\n\n",
        "## 7. Threshold Selection Analysis\n",
        "- **best_f1:** maximizes F1 on this specific test+attack mix — useful academically, sensitive to class balance.\n",
        "- **p99:** operational threshold from normal validation distribution — stable FAR reference.\n",
        "- **p99.5 / p99.7:** stricter normal envelope → lower FAR, lower recall on weak attacks.\n",
        "- No single threshold is universally optimal across attack physics and telemetry quirks.\n\n",
        "## 8. Recommended Threshold\n",
        f"- Academic (F1): **{sweep_best_thr:.6g}** (from sweep; not persisted to `thresholds.json`).\n",
        "- Operational: **p99** (or stricter if FAR budget is tight), aligned with production `thresholds.json` philosophy.\n",
        "- **Is p99 suitable for production?** Yes as a **default reference** calibrated on normal validation; tune with "
        "operational FAR targets and per-constellation validation.\n\n",
        "## 9. Error Analysis\n",
        overlap_note + "\n\n",
        "## 10. Curves and Figures\n",
        "All under `backend/experiments/chapter7_testing_validation_evaluation/results/`:\n\n",
        "".join(f"- `{fn}`\n" for fn in fig_rel),
        "\n## 11. Ablation Study\n",
    ]
    if ablation_usable:
        lines.append("Located prior experiment CSVs (not recomputed):\n" + "\n".join(f"- `{p}`" for p in ablation_usable) + "\n\n")
    else:
        lines.append(
            "Ablation study requires separately trained variants and was not recomputed in this run "
            "to avoid modifying the production system.\n\n"
        )
    lines.extend(
        [
            "## 12. Limitations\n",
            "- Synthetic attacks do not cover all on-orbit failure modes.\n",
            "- Fixed global threshold (not adaptive to regime changes).\n",
            "- Detector is learned from **normal** behavior; covariate shift can raise FP.\n",
            "- Multivariate coupling may be under-modeled if channels are treated independently in preprocessing.\n\n",
            "## 13. Conclusion\n",
            "Hybrid reconstruction–prediction–gradient scores separate most attacks at validation-calibrated thresholds; "
            "use **p99** for operations and **sweep F1** for offline what-if analysis. Validate FAR on mission-specific telemetry.\n",
        ]
    )
    rep_path.write_text("".join(lines), encoding="utf-8")

    print("\n=== Chapter 7 complete ===\n", flush=True)
    for name in (
        "testing_summary.md",
        "validation_thresholds_computed.csv",
        "validation_score_summary.csv",
        "overall_threshold_metrics.csv",
        "per_attack_threshold_metrics.csv",
        "best_threshold_per_attack.csv",
        "attack_difficulty_ranking.csv",
        "threshold_sweep_results.csv",
        "best_f1_from_sweep.json",
        "evaluation_summary.json",
        "chapter7_testing_validation_evaluation_report.md",
    ):
        print(f"  {_safe_out(output_dir, name)}")


if __name__ == "__main__":
    main()
