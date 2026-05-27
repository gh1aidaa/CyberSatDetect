from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
ABLATION_DIR = ROOT / "ablation_study"
RESULTS_CSV = ABLATION_DIR / "results.csv"
BACKEND = ROOT / "backend"
MODELS_DIR = BACKEND / "models"
DATA_DIR = ROOT / "data" / "reduced"
ATTACKED_DIR = ROOT / "data" / "attacked_v2"
SPLIT_FILE = BACKEND / "config" / "data_split_qc_filtered.json"


def _configure_paths(repo_root: Path, results_csv: Path | None = None) -> None:
    """Re-point all module paths to a different repo root (Colab/Drive friendly)."""
    global ROOT, ABLATION_DIR, RESULTS_CSV, BACKEND, MODELS_DIR, DATA_DIR, ATTACKED_DIR, SPLIT_FILE
    ROOT = repo_root.resolve()
    ABLATION_DIR = ROOT / "ablation_study"
    RESULTS_CSV = Path(results_csv).resolve() if results_csv else (ABLATION_DIR / "results.csv")
    BACKEND = ROOT / "backend"
    MODELS_DIR = BACKEND / "models"
    DATA_DIR = ROOT / "data" / "reduced"
    ATTACKED_DIR = ROOT / "data" / "attacked_v2"
    SPLIT_FILE = BACKEND / "config" / "data_split_qc_filtered.json"
    if str(MODELS_DIR) not in sys.path:
        sys.path.insert(0, str(MODELS_DIR))


if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

import train_hybrid_model as thm  # noqa: E402
from evaluate_model_strict_v2 import (  # noqa: E402
    compute_curves_and_auc,
    confusion_at_threshold,
    iter_attacked_npz,
    load_attacked_npz,
    metrics_from_confusion,
)


EXPERIMENTS: List[Tuple[str, float, float, float, float]] = [
    ("Baseline", 1.0, 2.0, 2.0, 0.5),
    ("Exp-1", 1.0, 1.0, 1.0, 0.5),
    ("Exp-2", 1.0, 2.0, 1.0, 0.5),
    ("Exp-3", 1.0, 1.0, 2.0, 0.5),
    ("Exp-4", 2.0, 2.0, 2.0, 0.5),
    ("Exp-5", 1.0, 3.0, 3.0, 0.5),
]

RESULT_FIELDS = [
    "experiment",
    "Wrecon",
    "Wpred",
    "Wgrad",
    "Wsep",
    "F1",
    "Precision",
    "Recall",
    "FAR",
    "notes",
]


def reset_seeds() -> None:
    random.seed(thm.SEED)
    np.random.seed(thm.SEED)
    tf.random.set_seed(thm.SEED)


def set_loss_weights(w_recon: float, w_pred: float, w_grad: float, w_sep: float) -> None:
    thm.W_RECON = float(w_recon)
    thm.W_PRED = float(w_pred)
    thm.W_GRAD = float(w_grad)
    thm.W_SEP = float(w_sep)


def load_split() -> Dict[str, List[str]]:
    with SPLIT_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_npy(path: Path) -> np.ndarray:
    x = np.load(path).astype(np.float32)
    if x.ndim == 2:
        x = x[..., None]
    return x


def weighted_scores_np(
    model: tf.keras.Model,
    x: np.ndarray,
    w_recon: float,
    w_pred: float,
    w_grad: float,
) -> np.ndarray:
    recon, pred = model.predict(x, verbose=0, batch_size=thm.BATCH_SIZE)
    recon = np.asarray(recon, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    if pred.ndim == 2:
        pred = pred[:, None, :]
    elif pred.ndim == 3 and pred.shape[1] != 1:
        pred = pred[:, :1, :]

    e_recon = np.mean((x - recon) ** 2, axis=(1, 2))
    e_pred = np.mean((x[:, -1:, :] - pred) ** 2, axis=(1, 2))
    dx_true = x[:, 1:, :] - x[:, :-1, :]
    dx_recon = recon[:, 1:, :] - recon[:, :-1, :]
    e_grad = np.mean((dx_true - dx_recon) ** 2, axis=(1, 2))
    return ((w_recon * e_recon) + (w_pred * e_pred) + (w_grad * e_grad)).astype(np.float32)


def train_from_scratch(
    train_files: List[Path],
    val_files: List[Path],
    w_recon: float,
    w_pred: float,
    w_grad: float,
    w_sep: float,
) -> Tuple[tf.keras.Model, int, float]:
    reset_seeds()
    set_loss_weights(w_recon, w_pred, w_grad, w_sep)

    x0 = load_npy(train_files[0])
    _, t_steps, channels = x0.shape
    model = thm.build_model(t_steps, channels)
    optimizer = tf.keras.optimizers.Adam(thm.LEARNING_RATE)

    best_weights = None
    best_val_loss = float("inf")
    best_chunk = 0
    no_improve_count = 0
    current_lr = thm.LEARNING_RATE

    for chunk_idx, fp in enumerate(train_files[: min(thm.MAX_CHUNKS, len(train_files))]):
        x_train = load_npy(fp)
        ds = (
            tf.data.Dataset.from_tensor_slices(x_train)
            .shuffle(20000, seed=thm.SEED + chunk_idx)
            .batch(thm.BATCH_SIZE)
        )
        chunk_losses: List[float] = []

        for _epoch in range(thm.EPOCHS_PER_CHUNK):
            for batch in ds:
                with tf.GradientTape() as tape:
                    recon, pred = model(batch, training=True)
                    pseudo_batch = thm.build_pseudo_anomalies(batch)
                    pseudo_recon, pseudo_pred = model(pseudo_batch, training=True)

                    loss_recon = thm.mse(batch, recon)
                    pred_reshaped = tf.reshape(pred, [-1, 1, 1])
                    loss_pred = thm.mse(batch[:, -1:, :], pred_reshaped)
                    loss_grad = thm.grad_mse(batch, recon)
                    normal_scores = thm.batch_scores_tf(batch, recon, pred)
                    pseudo_scores = thm.batch_scores_tf(pseudo_batch, pseudo_recon, pseudo_pred)
                    loss_sep = tf.reduce_mean(
                        tf.nn.relu(thm.ANOMALY_MARGIN - (pseudo_scores - normal_scores))
                    )
                    total_loss = (
                        (w_recon * loss_recon)
                        + (w_pred * loss_pred)
                        + (w_grad * loss_grad)
                        + (w_sep * loss_sep)
                    )

                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                chunk_losses.append(float(total_loss))

        avg_train_loss = float(np.mean(chunk_losses)) if chunk_losses else float("nan")
        if (chunk_idx + 1) % thm.VAL_CHECK_INTERVAL == 0 or chunk_idx == 0:
            val_files_sample = random.sample(val_files, min(30, len(val_files)))
            val_scores: List[float] = []
            for val_fp in val_files_sample:
                x_val = load_npy(val_fp)
                scores = weighted_scores_np(model, x_val, w_recon, w_pred, w_grad)
                val_scores.extend(float(s) for s in scores)
            val_loss = float(np.mean(val_scores))
            print(
                f"[{time.strftime('%H:%M:%S')}] chunk {chunk_idx + 1:3d}/{min(thm.MAX_CHUNKS, len(train_files))} "
                f"train_loss={avg_train_loss:.6f} val_loss={val_loss:.6f} lr={current_lr:.6f}",
                flush=True,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_chunk = chunk_idx + 1
                no_improve_count = 0
                best_weights = model.get_weights()
            else:
                no_improve_count += 1

            if no_improve_count >= 5 and no_improve_count % 5 == 0:
                current_lr = max(current_lr * 0.5, 1e-5)
                optimizer = tf.keras.optimizers.Adam(current_lr)

            if no_improve_count >= thm.EARLY_STOP_PATIENCE:
                print(f"Early stopping at chunk {chunk_idx + 1}; best chunk {best_chunk}.", flush=True)
                break

    if best_weights is not None:
        model.set_weights(best_weights)
    return model, best_chunk, best_val_loss


def collect_normal_scores(
    model: tf.keras.Model,
    files: Iterable[Path],
    w_recon: float,
    w_pred: float,
    w_grad: float,
) -> np.ndarray:
    parts = []
    for fp in files:
        x = load_npy(fp)
        parts.append(weighted_scores_np(model, x, w_recon, w_pred, w_grad).astype(np.float64))
    if not parts:
        raise RuntimeError("No normal validation scores produced.")
    return np.concatenate(parts)


def collect_attacked_scores(
    model: tf.keras.Model,
    w_recon: float,
    w_pred: float,
    w_grad: float,
) -> Tuple[np.ndarray, np.ndarray]:
    score_parts = []
    label_parts = []
    for npz_path in iter_attacked_npz(ATTACKED_DIR):
        x_att, y_window, _meta = load_attacked_npz(npz_path)
        score_parts.append(weighted_scores_np(model, x_att, w_recon, w_pred, w_grad).astype(np.float64))
        label_parts.append(y_window.astype(np.uint8))
    if not score_parts:
        raise RuntimeError("No attacked_v2 scores produced.")
    return np.concatenate(score_parts), np.concatenate(label_parts)


def evaluate_model(
    model: tf.keras.Model,
    val_files: List[Path],
    w_recon: float,
    w_pred: float,
    w_grad: float,
) -> Dict[str, float | str]:
    scores_normal = collect_normal_scores(model, val_files, w_recon, w_pred, w_grad)
    scores_attacked, y_attacked = collect_attacked_scores(model, w_recon, w_pred, w_grad)

    y_true = np.concatenate([np.zeros(len(scores_normal), dtype=np.uint8), y_attacked])
    y_score = np.concatenate([scores_normal, scores_attacked])
    curves_info, _curves = compute_curves_and_auc(y_true, y_score)

    thresholds = {
        "3sigma": float(scores_normal.mean() + (3.0 * scores_normal.std())),
        "p99": float(np.quantile(scores_normal, 0.99)),
        "p995": float(np.quantile(scores_normal, 0.995)),
        "p997": float(np.quantile(scores_normal, 0.997)),
        "best_f1": float(curves_info["best_f1_threshold"]),
    }

    best_row = None
    best_name = None
    for name, threshold in thresholds.items():
        cm = confusion_at_threshold(y_true, y_score, threshold)
        metrics = metrics_from_confusion(cm)
        if best_row is None or metrics["f1"] > best_row["f1"]:
            best_row = metrics
            best_name = name

    if best_row is None or best_name is None:
        raise RuntimeError("No evaluation metrics produced.")

    return {
        "F1": float(best_row["f1"]),
        "Precision": float(best_row["precision"]),
        "Recall": float(best_row["recall"]),
        "FAR": float(best_row["far"]),
        "notes": f"threshold={best_name}; validation-normal files={len(val_files)}; attacked_v2 files={len(list(iter_attacked_npz(ATTACKED_DIR)))}",
    }


def init_results_csv_if_missing() -> None:
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    if not RESULTS_CSV.exists():
        with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=RESULT_FIELDS).writeheader()


def already_done_experiments() -> set[str]:
    if not RESULTS_CSV.exists():
        return set()
    with RESULTS_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["experiment"] for row in reader if row.get("experiment")}


def append_result(row: Dict[str, float | str]) -> None:
    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writerow(row)


def run_single_experiment(
    name: str,
    w_recon: float,
    w_pred: float,
    w_grad: float,
    w_sep: float,
    train_files: List[Path],
    val_files: List[Path],
) -> Dict[str, float | str]:
    print(
        f"\n=== Starting {name}: Wrecon={w_recon}, Wpred={w_pred}, Wgrad={w_grad}, Wsep={w_sep} ===",
        flush=True,
    )
    model, best_chunk, best_val_loss = train_from_scratch(
        train_files, val_files, w_recon, w_pred, w_grad, w_sep
    )
    metrics = evaluate_model(model, val_files, w_recon, w_pred, w_grad)
    row: Dict[str, float | str] = {
        "experiment": name,
        "Wrecon": w_recon,
        "Wpred": w_pred,
        "Wgrad": w_grad,
        "Wsep": w_sep,
        "F1": f"{metrics['F1']:.10g}",
        "Precision": f"{metrics['Precision']:.10g}",
        "Recall": f"{metrics['Recall']:.10g}",
        "FAR": f"{metrics['FAR']:.10g}",
        "notes": f"{metrics['notes']}; best_chunk={best_chunk}; best_val_loss={best_val_loss:.10g}",
    }
    append_result(row)
    print(f"=== Completed {name}: F1={row['F1']} ===", flush=True)
    tf.keras.backend.clear_session()
    del model
    return row


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Loss-weight ablation runner (Colab/local).")
    ap.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="Project root containing backend/, data/reduced/, data/attacked_v2/. Defaults to script's parent.",
    )
    ap.add_argument(
        "--split-file",
        type=str,
        default=None,
        help="Split JSON to use. Default: backend/config/data_split_qc_filtered.json.",
    )
    ap.add_argument(
        "--results-csv",
        type=str,
        default=None,
        help="Override path for results.csv (default: ablation_study/results.csv under repo root).",
    )
    ap.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated list of experiment names to run (e.g. 'Baseline,Exp-1'). Default: all not yet done.",
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing results.csv and start over (will overwrite).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.repo_root:
        _configure_paths(Path(args.repo_root), Path(args.results_csv) if args.results_csv else None)
    elif args.results_csv:
        _configure_paths(ROOT, Path(args.results_csv))
    if args.split_file:
        global SPLIT_FILE
        SPLIT_FILE = Path(args.split_file).resolve()

    os.chdir(ROOT)
    for required in (DATA_DIR, ATTACKED_DIR, SPLIT_FILE):
        if not required.exists():
            raise FileNotFoundError(required)

    split = load_split()
    train_files = [DATA_DIR / name for name in split["train"]]
    val_files = [DATA_DIR / name for name in split["validation"]]

    if args.fresh and RESULTS_CSV.exists():
        RESULTS_CSV.unlink()
    init_results_csv_if_missing()

    requested = None
    if args.only:
        requested = {name.strip() for name in args.only.split(",") if name.strip()}

    done = already_done_experiments()
    to_run: List[Tuple[str, float, float, float, float]] = []
    for spec in EXPERIMENTS:
        name = spec[0]
        if requested is not None and name not in requested:
            continue
        if name in done:
            print(f"[skip] {name} already in {RESULTS_CSV.name}", flush=True)
            continue
        to_run.append(spec)

    print(f"Repo root: {ROOT}", flush=True)
    print(f"Results CSV: {RESULTS_CSV}", flush=True)
    print(f"Experiments to run ({len(to_run)}): {[s[0] for s in to_run]}", flush=True)

    for name, w_recon, w_pred, w_grad, w_sep in to_run:
        run_single_experiment(name, w_recon, w_pred, w_grad, w_sep, train_files, val_files)

    # Summary over all rows currently in results.csv
    completed_rows: List[Dict[str, str]] = []
    with RESULTS_CSV.open("r", newline="", encoding="utf-8") as f:
        completed_rows = list(csv.DictReader(f))

    if not completed_rows:
        print("\nNo rows in results.csv yet.", flush=True)
        return

    print("\nSummary")
    print("experiment,Wrecon,Wpred,Wgrad,Wsep,F1,Precision,Recall,FAR")
    for row in completed_rows:
        print(
            f"{row['experiment']},{row['Wrecon']},{row['Wpred']},{row['Wgrad']},{row['Wsep']},"
            f"{row['F1']},{row['Precision']},{row['Recall']},{row['FAR']}"
        )
    best = max(completed_rows, key=lambda r: float(r["F1"]))
    print(
        f"\nBest F1: {best['experiment']} "
        f"(Wrecon={best['Wrecon']}, Wpred={best['Wpred']}, Wgrad={best['Wgrad']}, Wsep={best['Wsep']}) "
        f"with F1={best['F1']}"
    )


if __name__ == "__main__":
    main()
