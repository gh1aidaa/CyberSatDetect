"""
تقييم نموذج كشف الشذوذ Hybrid LSTM-GRU AE
============================================
Normal  = data/reduced   (Label 0)
Anomaly = data/attacked  (Label 1)

يحسب:
- Accuracy, Precision, Recall, F1, Confusion Matrix لكل threshold
- ROC-AUC, PR-AUC
- العتبة المثلى (Best F1 threshold)
- توزيع الدرجات (histogram + ROC + PR curves)
"""

import os
import sys
import json
import time
import zipfile
import tempfile
from pathlib import Path
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_hybrid_model import build_model as build_hybrid_model

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
APP_DIR = BACKEND / "app"

NORMAL_DIR = ROOT / "data" / "reduced"
ANOMALY_DIR = ROOT / "data" / "attacked"

SPLIT_FILE = BACKEND / "config" / "data_split.json"
SPLIT_KEY = "test"

MODEL_PATH = APP_DIR / "best_model.keras"
THRESH_PATH = APP_DIR / "thresholds.json"

RESULTS_PATH = APP_DIR / "evaluation_results.json"
PLOTS_DIR = APP_DIR / "eval_plots"

BATCH_SIZE = 256

W_RECON = 1.0
W_PRED = 2.0
W_GRAD = 2.0


def load_npy(path: Path) -> np.ndarray:
    x = np.load(path).astype(np.float32)
    if x.ndim == 2:
        x = x[..., None]
    return x


def infer_T_C_from_sample(sample_dir: Path) -> tuple:
    """يستنتج (T, C) من أوّل ملف npy في المجلد."""
    sample_file = next(sample_dir.glob("chunk_*.npy"), None)
    if sample_file is None:
        raise FileNotFoundError(f"No chunk_*.npy in {sample_dir}")
    X = load_npy(sample_file)
    return int(X.shape[1]), int(X.shape[2])


def load_keras_model_robust(model_path: Path, T: int, C: int):
    """يحمّل النموذج بطريقة مقاومة لاختلاف إصدارات Keras.

    يحاول أولاً tf.keras.models.load_model. إذا فشل (مثلاً بسبب
    quantization_config أو أي kwarg جديد من Keras 3)، يبني المعمارية
    من train_hybrid_model.build_model ويحمّل ملف الأوزان فقط من
    .keras (وهو zip يحتوي model.weights.h5).
    """
    try:
        return tf.keras.models.load_model(
            model_path, custom_objects={"tf": tf}, compile=False
        )
    except Exception as e:
        print(f"\n  [WARN] Direct load failed ({type(e).__name__}). "
              f"Falling back to weights-only load.")

    print(f"  Building fresh architecture (T={T}, C={C}) and loading weights...")
    model = build_hybrid_model(T, C)

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(model_path, "r") as zf:
            members = zf.namelist()
            weight_member = None
            for candidate in ("model.weights.h5", "weights.h5"):
                if candidate in members:
                    weight_member = candidate
                    break
            if weight_member is None:
                weight_member = next(
                    (m for m in members if m.endswith(".h5")), None
                )
            if weight_member is None:
                raise RuntimeError(
                    f"Could not find .h5 weights inside {model_path}. "
                    f"Members: {members}"
                )
            extracted = zf.extract(weight_member, tmpdir)

        model.load_weights(extracted)

    print("  Weights loaded successfully into freshly-built architecture.")
    return model


def compute_scores(model, X: np.ndarray) -> np.ndarray:
    """Anomaly score = W_RECON*Recon + W_PRED*Pred + W_GRAD*Grad (نفس صيغة التدريب)"""
    recon, pred = model.predict(X, verbose=0, batch_size=BATCH_SIZE)

    e_recon = np.mean((X - recon) ** 2, axis=(1, 2))

    pred_reshaped = np.reshape(pred, (-1, 1, 1))
    e_pred = np.mean((X[:, -1:, :] - pred_reshaped) ** 2, axis=(1, 2))

    dx_true = X[:, 1:, :] - X[:, :-1, :]
    dx_recon = recon[:, 1:, :] - recon[:, :-1, :]
    e_grad = np.mean((dx_true - dx_recon) ** 2, axis=(1, 2))

    return (W_RECON * e_recon) + (W_PRED * e_pred) + (W_GRAD * e_grad)


def score_files(model, files, label_name: str, source_desc: str) -> np.ndarray:
    if not files:
        raise FileNotFoundError(f"No files to score for {label_name} ({source_desc})")

    print(f"\n[{label_name}] Scoring {len(files)} chunks from {source_desc} ...")
    all_scores = []
    t0 = time.time()
    missing = 0
    for i, fp in enumerate(files, 1):
        fp = Path(fp)
        if not fp.exists():
            missing += 1
            continue
        X = load_npy(fp)
        scores = compute_scores(model, X)
        all_scores.append(scores)
        if i % 50 == 0 or i == len(files):
            elapsed = time.time() - t0
            print(f"  [{label_name}] {i}/{len(files)} | elapsed={elapsed:.1f}s")
    if missing:
        print(f"  [{label_name}] WARNING: {missing} file(s) listed but not found on disk (skipped)")
    if not all_scores:
        raise RuntimeError(f"No scores produced for {label_name}")
    return np.concatenate(all_scores)


def load_test_filenames(split_file: Path, split_key: str = "test") -> list:
    """يقرأ data_split.json ويرجع قائمة بأسماء ملفات الـ test."""
    with open(split_file, "r", encoding="utf-8") as f:
        split = json.load(f)
    if split_key not in split:
        raise KeyError(
            f"Split key '{split_key}' not found in {split_file}. "
            f"Available keys: {list(split.keys())}"
        )
    return list(split[split_key])


def metrics_at_threshold(scores_norm: np.ndarray, scores_anom: np.ndarray, thr: float) -> dict:
    y_true = np.concatenate([np.zeros(len(scores_norm)), np.ones(len(scores_anom))])
    y_pred = np.concatenate([
        (scores_norm > thr).astype(int),
        (scores_anom > thr).astype(int),
    ])

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "threshold": float(thr),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tnr": float(tnr),
        "far": float(far),
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
    }


def compute_auc_curves(scores_norm: np.ndarray, scores_anom: np.ndarray):
    """يحسب ROC-AUC و PR-AUC والعتبة المثلى (أعلى F1) بدون sklearn."""
    y_true = np.concatenate([np.zeros(len(scores_norm)), np.ones(len(scores_anom))])
    y_score = np.concatenate([scores_norm, scores_anom])

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

    f1_curve = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_idx = int(np.argmax(f1_curve))
    best_thr = float(y_score_sorted[best_idx])
    best_f1 = float(f1_curve[best_idx])

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "best_f1": best_f1,
        "best_threshold": best_thr,
    }, (fpr, tpr, precision, recall)


def maybe_plot(scores_norm, scores_anom, thresholds_dict, curves):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"\n(matplotlib not available, skipping plots: {e})")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fpr, tpr, precision, recall = curves

    plt.figure(figsize=(10, 6))
    plt.hist(scores_norm, bins=120, alpha=0.6, label=f"Normal (reduced) n={len(scores_norm)}",
             color="#2ecc71", density=True)
    plt.hist(scores_anom, bins=120, alpha=0.6, label=f"Anomaly (attacked) n={len(scores_anom)}",
             color="#e74c3c", density=True)
    colors = {"3sigma": "#34495e", "p99": "#3498db", "p995": "#9b59b6", "p997": "#e67e22"}
    for name, val in thresholds_dict.items():
        plt.axvline(val, linestyle="--", linewidth=1.5, color=colors.get(name, "k"),
                    label=f"{name} = {val:.4f}")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Density")
    plt.title("Anomaly Score Distribution: Normal vs Attacked")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "score_distribution.png", dpi=130)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label="ROC", color="#2980b9", linewidth=2)
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "roc_curve.png", dpi=130)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color="#c0392b", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pr_curve.png", dpi=130)
    plt.close()

    print(f"\nPlots saved to: {PLOTS_DIR}")


def fmt_pct(x: float) -> str:
    return f"{x*100:6.2f}%"


def print_table(rows):
    header = ("Threshold", "Value", "Accuracy", "Precision", "Recall", "F1", "FAR")
    widths = (12, 12, 11, 11, 11, 11, 11)
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    print(line)
    print("|" + "|".join(f" {h:^{w}} " for h, w in zip(header, widths)) + "|")
    print(line)
    for r in rows:
        cells = (
            r["name"],
            f"{r['threshold']:.5f}",
            fmt_pct(r["accuracy"]),
            fmt_pct(r["precision"]),
            fmt_pct(r["recall"]),
            f"{r['f1']:.4f}",
            fmt_pct(r["far"]),
        )
        print("|" + "|".join(f" {c:^{w}} " for c, w in zip(cells, widths)) + "|")
    print(line)


def main():
    print("=" * 78)
    print("EVALUATION: Hybrid LSTM-GRU Anomaly Detector")
    print(f"  Normal  source : {NORMAL_DIR}  (split='{SPLIT_KEY}' from {SPLIT_FILE.name})")
    print(f"  Anomaly source : {ANOMALY_DIR}  (all chunks)")
    print(f"  Model          : {MODEL_PATH}")
    print("=" * 78)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    if not THRESH_PATH.exists():
        raise FileNotFoundError(THRESH_PATH)
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(SPLIT_FILE)

    print("\nInferring input shape (T, C) from data sample ...")
    T, C = infer_T_C_from_sample(NORMAL_DIR)
    print(f"  T (timesteps) = {T}, C (channels) = {C}")

    print("\nLoading model ...")
    model = load_keras_model_robust(MODEL_PATH, T, C)
    print("Model loaded. Input shape:", model.input_shape)

    with open(THRESH_PATH, "r", encoding="utf-8") as f:
        thresh_data = json.load(f)
    thresholds = thresh_data["thresholds"]
    print("\nLoaded thresholds:")
    for k, v in thresholds.items():
        print(f"  {k:>8s} = {v:.6f}")

    test_names = load_test_filenames(SPLIT_FILE, SPLIT_KEY)
    print(f"\nLoaded {len(test_names)} '{SPLIT_KEY}' filenames from {SPLIT_FILE}")
    print("(train + validation are intentionally EXCLUDED from evaluation)")

    normal_files = [NORMAL_DIR / name for name in test_names]
    anomaly_files = sorted(ANOMALY_DIR.glob("chunk_*.npy"))

    scores_norm = score_files(
        model, normal_files, "NORMAL",
        source_desc=f"data/reduced (test split, {len(normal_files)} files)",
    )
    scores_anom = score_files(
        model, anomaly_files, "ANOMALY",
        source_desc=f"data/attacked (all, {len(anomaly_files)} files)",
    )

    print("\n" + "=" * 78)
    print("SCORE STATISTICS")
    print("=" * 78)
    print(f"  Normal   : n={len(scores_norm):>7d}  "
          f"mean={scores_norm.mean():.6f}  std={scores_norm.std():.6f}  "
          f"min={scores_norm.min():.6f}  max={scores_norm.max():.6f}")
    print(f"  Anomaly  : n={len(scores_anom):>7d}  "
          f"mean={scores_anom.mean():.6f}  std={scores_anom.std():.6f}  "
          f"min={scores_anom.min():.6f}  max={scores_anom.max():.6f}")
    print(f"  Mean separation: {scores_anom.mean() - scores_norm.mean():.6f}")

    print("\n" + "=" * 78)
    print("METRICS PER THRESHOLD")
    print("=" * 78)
    rows = []
    per_thr_results = {}
    for name, thr in thresholds.items():
        m = metrics_at_threshold(scores_norm, scores_anom, thr)
        m["name"] = name
        rows.append(m)
        per_thr_results[name] = m
    print_table(rows)

    auc_info, curves = compute_auc_curves(scores_norm, scores_anom)
    best = metrics_at_threshold(scores_norm, scores_anom, auc_info["best_threshold"])
    best["name"] = "BEST F1*"

    print("\n" + "=" * 78)
    print("BEST OPERATING POINT (max F1 over all thresholds)")
    print("=" * 78)
    print_table([best])
    print(f"  ROC-AUC = {auc_info['roc_auc']:.6f}")
    print(f"  PR-AUC  = {auc_info['pr_auc']:.6f}")

    print("\n" + "=" * 78)
    print("VERDICT (target = 98% accuracy)")
    print("=" * 78)
    best_named = max(rows + [best], key=lambda r: r["accuracy"])
    if best_named["accuracy"] >= 0.98:
        print(f"  PASSED  Best accuracy = {best_named['accuracy']*100:.2f}% "
              f"at threshold '{best_named['name']}' ({best_named['threshold']:.6f})")
    else:
        print(f"  BELOW TARGET  Best accuracy = {best_named['accuracy']*100:.2f}% "
              f"at threshold '{best_named['name']}' ({best_named['threshold']:.6f})")
        print("  Suggestions: try BEST F1 threshold above, or retrain a few more chunks.")

    out = {
        "model_path": str(MODEL_PATH),
        "normal_source": {
            "dir": str(NORMAL_DIR),
            "split_file": str(SPLIT_FILE),
            "split_key": SPLIT_KEY,
            "n_files": len(normal_files),
        },
        "anomaly_source": {
            "dir": str(ANOMALY_DIR),
            "n_files": len(anomaly_files),
        },
        "n_normal": int(len(scores_norm)),
        "n_anomaly": int(len(scores_anom)),
        "score_stats": {
            "normal":  {"mean": float(scores_norm.mean()), "std": float(scores_norm.std()),
                        "min": float(scores_norm.min()),  "max": float(scores_norm.max())},
            "anomaly": {"mean": float(scores_anom.mean()), "std": float(scores_anom.std()),
                        "min": float(scores_anom.min()),  "max": float(scores_anom.max())},
            "mean_separation": float(scores_anom.mean() - scores_norm.mean()),
        },
        "thresholds_used": thresholds,
        "per_threshold": per_thr_results,
        "best_f1_operating_point": {
            "threshold": auc_info["best_threshold"],
            "f1": auc_info["best_f1"],
            "metrics": best,
        },
        "roc_auc": auc_info["roc_auc"],
        "pr_auc": auc_info["pr_auc"],
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4, ensure_ascii=False)
    print(f"\nResults JSON saved to: {RESULTS_PATH}")

    maybe_plot(scores_norm, scores_anom, thresholds, curves)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
