"""
Train experimental univariate pattern models (writes ONLY under experiment dirs).

Usage:
  python backend/experiments/univariate_pattern_solution_experiment/train_univariate_pattern_model.py ^
    --repo-root . ^
    --split-file backend/config/data_split_qc_filtered.json ^
    --normal-dir data/reduced ^
    --output-dir backend/experiments/univariate_pattern_solution_experiment/results ^
    --models-dir backend/experiments/univariate_pattern_solution_experiment/models ^
    --epochs 3 ^
    --batch-size 256
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np

_EXP_DIR = Path(__file__).resolve().parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from temporal_feature_engineering import augment_univariate_batch  # noqa: E402
from pseudo_pattern_generator import apply_random_pseudo, counts_from_names  # noqa: E402
from model_univariate_pattern import build_univariate_pattern_model  # noqa: E402
from score_univariate_pattern import compute_window_scores, quantile_thresholds  # noqa: E402


def _resolve(repo: Path, p: str | Path) -> Path:
    path = Path(p)
    return (repo / path).resolve() if not path.is_absolute() else path.resolve()


def _load_split(repo: Path, split_file: str) -> Dict[str, List[str]]:
    fp = _resolve(repo, split_file)
    with fp.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_files_chunks(normal_dir: Path, filenames: List[str], *, max_files: int) -> Iterator[np.ndarray]:
    use = filenames[: int(max_files)] if max_files and max_files > 0 else filenames
    for name in use:
        p = (normal_dir / name).resolve()
        if not p.is_file():
            continue
        x = np.load(p).astype(np.float32)
        if x.ndim == 2:
            x = x[..., None]
        if x.ndim != 3 or int(x.shape[1]) != 100 or int(x.shape[2]) != 1:
            continue
        yield x


def _write_sample_visualization(out_png: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    t = np.linspace(0, 1, 100, dtype=np.float32)
    x = (0.4 * np.sin(2 * np.pi * 3 * t) + 0.05 * np.random.randn(100).astype(np.float32))[..., None]
    from temporal_feature_engineering import augment_univariate_window  # noqa: WPS433

    aug = augment_univariate_window(x)
    names = ["x", "dx", "ddx", "roll_mean5", "roll_std5", "z_win", "slope5", "t/99"]
    fig, axs = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
    for k in range(8):
        r, c = divmod(k, 2)
        axs[r][c].plot(aug[:, k], lw=1.2)
        axs[r][c].set_title(names[k])
    fig.suptitle("Augmented univariate window (100,8) — synthetic demo")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def _append_pseudo_log(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.is_file()
    fieldnames = list(row.keys())
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        w.writerow(row)


def _model_filename(recon_key: str, w_order_train: float) -> str:
    wtag = str(w_order_train).replace(".", "_")
    return f"univariate_pattern_{recon_key}_worder_{wtag}.keras"


def _train_step_tf(
    model: Any,
    opt: Any,
    xb: np.ndarray,
    *,
    recon_target: str,
    w_order_train: float,
    rng_np: np.random.Generator,
    margin: float,
    w_recon: float,
    w_pred: float,
    w_grad: float,
    w_sep: float,
) -> Tuple[float, Dict[str, int]]:
    import tensorflow as tf

    bce = tf.keras.losses.BinaryCrossentropy()
    xb_t = tf.constant(xb, dtype=tf.float32)

    x_aug = tf.constant(augment_univariate_batch(xb), dtype=tf.float32)
    x_pseudo_np, names = apply_random_pseudo(xb, rng=rng_np)
    x_pseudo_aug = tf.constant(augment_univariate_batch(x_pseudo_np), dtype=tf.float32)

    y_sig = xb_t
    y_last10 = y_sig[:, -10:, :]

    if recon_target == "original_only":
        y_recon_n = y_sig
        y_recon_p = tf.constant(x_pseudo_np, dtype=tf.float32)
    else:
        y_recon_n = x_aug
        y_recon_p = x_pseudo_aug

    with tf.GradientTape() as tape:
        recon_n, pred_n, order_n = model(x_aug, training=True)
        recon_p, pred_p, order_p = model(x_pseudo_aug, training=True)

        l_recon_n = tf.reduce_mean(tf.square(y_recon_n - recon_n))
        l_recon_p = tf.reduce_mean(tf.square(y_recon_p - recon_p))
        l_recon = 0.5 * (l_recon_n + l_recon_p)

        l_pred = tf.reduce_mean(tf.square(y_last10 - pred_n))

        sig_n = y_sig[:, :, 0]
        r0_n = recon_n[:, :, 0]
        dx_t_n = sig_n[:, 1:] - sig_n[:, :-1]
        dx_r_n = r0_n[:, 1:] - r0_n[:, :-1]
        l_grad_n = tf.reduce_mean(tf.square(dx_t_n - dx_r_n))

        sig_p = tf.constant(x_pseudo_np, dtype=tf.float32)[:, :, 0]
        r0_p = recon_p[:, :, 0]
        dx_t_p = sig_p[:, 1:] - sig_p[:, :-1]
        dx_r_p = r0_p[:, 1:] - r0_p[:, :-1]
        l_grad_p = tf.reduce_mean(tf.square(dx_t_p - dx_r_p))
        l_grad = 0.5 * (l_grad_n + l_grad_p)

        e_recon_n = tf.reduce_mean(tf.square(y_recon_n - recon_n), axis=[1, 2])
        e_pred_n = tf.reduce_mean(tf.square(y_last10 - pred_n), axis=[1, 2])
        e_grad_n = tf.reduce_mean(tf.square(dx_t_n - dx_r_n), axis=1)

        e_recon_p = tf.reduce_mean(tf.square(y_recon_p - recon_p), axis=[1, 2])
        y_last10_p = tf.constant(x_pseudo_np, dtype=tf.float32)[:, -10:, :]
        e_pred_p = tf.reduce_mean(tf.square(y_last10_p - pred_p), axis=[1, 2])
        e_grad_p = tf.reduce_mean(tf.square(dx_t_p - dx_r_p), axis=1)

        s_n = w_recon * e_recon_n + w_pred * e_pred_n + w_grad * e_grad_n
        s_p = w_recon * e_recon_p + w_pred * e_pred_p + w_grad * e_grad_p
        l_sep = tf.reduce_mean(tf.nn.relu(margin - (s_p - s_n)))

        zeros = tf.zeros_like(order_n)
        ones = tf.ones_like(order_p)
        l_order = 0.5 * (tf.reduce_mean(bce(zeros, order_n)) + tf.reduce_mean(bce(ones, order_p)))

        total = (
            w_recon * l_recon
            + w_pred * l_pred
            + w_grad * l_grad
            + w_sep * l_sep
            + float(w_order_train) * l_order
        )

    grads = tape.gradient(total, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return float(total.numpy()), counts_from_names(names)


def _train_model_epochs(
    *,
    model: Any,
    opt: Any,
    recon_target: str,
    w_order_train: float,
    train_files: List[str],
    normal_dir: Path,
    epochs: int,
    batch_size: int,
    max_files: int,
    rng_np: np.random.Generator,
    log_csv: Path,
    model_name: str,
) -> None:
    for ep in range(int(epochs)):
        pseudo_accum: Dict[str, int] = {}
        windows_used = 0
        for chunk in _iter_files_chunks(normal_dir, train_files, max_files=max_files):
            n = int(chunk.shape[0])
            idx = np.arange(n, dtype=np.int64)
            rng_np.shuffle(idx)
            for s in range(0, n, int(batch_size)):
                sel = idx[s : s + int(batch_size)]
                if sel.size == 0:
                    continue
                xb = chunk[sel]
                loss, pc = _train_step_tf(
                    model,
                    opt,
                    xb,
                    recon_target=recon_target,
                    w_order_train=float(w_order_train),
                    rng_np=rng_np,
                    margin=0.15,
                    w_recon=1.0,
                    w_pred=2.0,
                    w_grad=2.0,
                    w_sep=0.5,
                )
                _ = loss
                for k, v in pc.items():
                    pseudo_accum[k] = pseudo_accum.get(k, 0) + int(v)
                windows_used += int(xb.shape[0])

        row = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": model_name,
            "epoch": int(ep + 1),
            "windows_used": int(windows_used),
            **{f"pseudo_count_{k}": int(v) for k, v in sorted(pseudo_accum.items())},
        }
        _append_pseudo_log(log_csv, row)


def _calibrate_model(
    *,
    model: Any,
    recon_target: str,
    val_files: List[str],
    normal_dir: Path,
    max_files: int,
    batch_size: int,
    w_order_scores: List[float],
    thresholds_out: Dict[str, Any],
    val_summary_rows: List[Dict[str, Any]],
    model_key: str,
) -> None:
    parts_x: List[np.ndarray] = []
    for chunk in _iter_files_chunks(normal_dir, val_files, max_files=max_files):
        parts_x.append(chunk)
    if not parts_x:
        return
    Xv = np.concatenate(parts_x, axis=0).astype(np.float32)
    x_aug_v = augment_univariate_batch(Xv)

    thresholds_out[model_key] = {}
    for wos in w_order_scores:
        scores = compute_window_scores(
            model,
            Xv,
            x_aug_v,
            recon_target=recon_target,
            w_recon=1.0,
            w_pred=2.0,
            w_grad=2.0,
            w_order=float(wos),
            batch_size=int(batch_size),
        )
        thr = quantile_thresholds(scores)
        thresholds_out[model_key][f"w_order_score_{wos}"] = thr
        val_summary_rows.append(
            {
                "model_key": model_key,
                "w_order_score": float(wos),
                "num_val_windows": int(len(scores)),
                "score_mean": float(np.mean(scores)),
                "score_std": float(np.std(scores)),
                "thr_p99": float(thr["p99"]),
            }
        )


def _zip_colab_bundle(repo_root: Path, out_zip: Path) -> None:
    paths = [
        Path("backend/experiments/univariate_pattern_solution_experiment"),
        Path("backend/config/data_split_qc_filtered.json"),
    ]
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in paths:
            p = (repo_root / rel).resolve()
            if not p.exists():
                continue
            if p.is_dir():
                for f in p.rglob("*"):
                    if f.is_file() and f.suffix in {".py", ".md", ".csv", ".json"}:
                        arc = f.relative_to(repo_root)
                        zf.write(f, arcname=str(arc).replace("\\", "/"))
            else:
                zf.write(p, arcname=str(rel).replace("\\", "/"))
        readme = (
            "Colab training bundle for univariate_pattern_solution_experiment.\n"
            "1) Unzip alongside your repo root (or adjust PYTHONPATH).\n"
            "2) Upload data/reduced/*.npy matching split JSON.\n"
            "3) Run: python backend/experiments/univariate_pattern_solution_experiment/train_univariate_pattern_model.py "
            "--repo-root . --split-file backend/config/data_split_qc_filtered.json --normal-dir data/reduced "
            "--output-dir backend/experiments/univariate_pattern_solution_experiment/results "
            "--models-dir backend/experiments/univariate_pattern_solution_experiment/models --epochs 3 --batch-size 256\n"
        )
        zf.writestr("COLAB_README.txt", readme)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, required=True)
    ap.add_argument("--split-file", type=str, required=True)
    ap.add_argument("--normal-dir", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--models-dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-files", type=int, default=0, help="If >0, limit train/val files for smoke tests")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    split = _load_split(repo, args.split_file)
    train_files = list(split.get("train", []))
    val_files = list(split.get("validation", []))
    normal_dir = _resolve(repo, args.normal_dir)
    out_dir = _resolve(repo, args.output_dir)
    models_dir = _resolve(repo, args.models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    _write_sample_visualization(out_dir / "sample_augmented_window_visualization.png")

    if not normal_dir.is_dir():
        zip_path = out_dir / "univariate_pattern_colab_training_bundle.zip"
        _zip_colab_bundle(repo, zip_path)
        print(f"[ERROR] normal-dir not found: {normal_dir}")
        print(f"Wrote Colab bundle: {zip_path}")
        return 2

    if next(normal_dir.glob("chunk_*.npy"), None) is None:
        zip_path = out_dir / "univariate_pattern_colab_training_bundle.zip"
        _zip_colab_bundle(repo, zip_path)
        print(f"[ERROR] No chunk_*.npy under {normal_dir}")
        print(f"Wrote Colab bundle: {zip_path}")
        return 2

    import tensorflow as tf

    rng_np = np.random.default_rng(int(args.seed))
    tf.keras.utils.set_random_seed(int(args.seed))

    recon_targets = [("original", "original_only"), ("allfeatures", "all_features")]
    w_orders = [0.1, 0.25, 0.5]
    w_order_scores = [0.1, 0.25, 0.5, 1.0]

    thresholds_all: Dict[str, Any] = {"generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    val_rows: List[Dict[str, Any]] = []

    max_files = int(args.max_files or 0)

    for recon_key, recon_target in recon_targets:
        for w_order_train in w_orders:
            name = _model_filename(recon_key, float(w_order_train))
            model_path = models_dir / name
            weights_path = models_dir / (Path(name).stem + ".weights.h5")
            meta_path = models_dir / (Path(name).stem + "_meta.json")

            if model_path.is_file():
                model = tf.keras.models.load_model(model_path, compile=False)
            else:
                model = build_univariate_pattern_model(T=100, C_in=8, recon_target=recon_target)  # type: ignore[arg-type]
                if weights_path.is_file():
                    model.load_weights(str(weights_path))

            opt = tf.keras.optimizers.Adam(1e-3)

            _train_model_epochs(
                model=model,
                opt=opt,
                recon_target=recon_target,
                w_order_train=float(w_order_train),
                train_files=train_files,
                normal_dir=normal_dir,
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                max_files=max_files,
                rng_np=rng_np,
                log_csv=out_dir / "pseudo_pattern_generation_log.csv",
                model_name=name,
            )

            model.save(model_path)
            try:
                model.save_weights(str(weights_path))
            except Exception:
                pass

            meta = {
                "model_file": str(name),
                "recon_target": recon_target,
                "w_order_train": float(w_order_train),
                "input_shape_aug": [100, 8],
                "trained_epochs": int(args.epochs),
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

            mk = Path(name).stem
            _calibrate_model(
                model=model,
                recon_target=recon_target,
                val_files=val_files,
                normal_dir=normal_dir,
                max_files=max_files,
                batch_size=int(args.batch_size),
                w_order_scores=w_order_scores,
                thresholds_out=thresholds_all,
                val_summary_rows=val_rows,
                model_key=mk,
            )

    (out_dir / "univariate_pattern_thresholds.json").write_text(
        json.dumps(thresholds_all, indent=2),
        encoding="utf-8",
    )
    vcsv = out_dir / "validation_score_summary.csv"
    if val_rows:
        with vcsv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(val_rows[0].keys()))
            w.writeheader()
            for r in val_rows:
                w.writerow(r)

    print(f"Done. Models in: {models_dir}")
    print(f"Thresholds: {out_dir / 'univariate_pattern_thresholds.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
