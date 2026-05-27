"""
Generate publication-style figures for:
- data cleaning (before/after + invalid mask)
- data preparation (histograms)
- structuring (sliding windows + split hierarchy diagram)

This script intentionally duplicates the inference cleaning logic used in `backend/app/api.py::clean_series`
to avoid importing the full FastAPI app as a module.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def clean_series(x: np.ndarray) -> np.ndarray:
    """
    Mirrors `backend/app/api.py::clean_series`:
    - float32
    - replace inf -> nan
    - linear interpolation across finite points, then fill remaining with 0
    """
    # IMPORTANT: input may be memory-mapped read-only; always copy before mutating.
    x = np.asarray(x, dtype=np.float32).reshape(-1).copy()
    x[~np.isfinite(x)] = np.nan

    n = x.size
    idx = np.arange(n, dtype=np.float32)
    mask = np.isfinite(x)

    if mask.sum() == 0:
        return np.zeros_like(x, dtype=np.float32)

    x_interp = x.copy()
    x_interp[~mask] = np.interp(idx[~mask], idx[mask], x[mask]).astype(np.float32)
    x_interp[~np.isfinite(x_interp)] = 0.0
    return x_interp.astype(np.float32)


def make_windows_1ch(x: np.ndarray, win_len: int, stride: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Mirrors `backend/app/api.py::make_windows_1ch`:
    x: (N,) -> Xw: (W, T, 1), spans: [(start,end), ...]
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = x.size
    if n < win_len:
        return np.zeros((0, win_len, 1), dtype=np.float32), []

    spans: List[Tuple[int, int]] = []
    windows: List[np.ndarray] = []
    for s in range(0, n - win_len + 1, stride):
        e = s + win_len
        windows.append(x[s:e])
        spans.append((s, e))

    Xw = np.asarray(windows, dtype=np.float32)[..., None]
    return Xw, spans


def _ensure_matplotlib():
    import importlib.util

    if importlib.util.find_spec("matplotlib") is None:
        raise RuntimeError(
            "matplotlib is not installed. Install it with:\n"
            "  python -m pip install matplotlib\n"
        )
    import matplotlib.pyplot as plt  # noqa: WPS433

    return plt


def _style(plt):
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )


def plot_cleaning_before_after(
    plt,
    out_path: Path,
    raw: np.ndarray,
    cleaned: np.ndarray,
    start: int,
    length: int,
    title: str,
):
    s = int(start)
    e = min(int(s + length), int(raw.size))
    t = np.arange(s, e, dtype=np.int32)

    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(t, raw[s:e], color="#d62728", linewidth=1.0, alpha=0.85, label="Raw (float32; NaN/Inf allowed)")
    ax.plot(t, cleaned[s:e], color="#1f77b4", linewidth=1.2, alpha=0.9, label="Cleaned (interpolated + filled)")
    ax.set_title(title)
    ax.set_xlabel("Timestep index")
    ax.set_ylabel("Telemetry value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_invalid_mask(plt, out_path: Path, raw: np.ndarray, start: int, length: int, title: str):
    s = int(start)
    e = min(int(s + length), int(raw.size))
    t = np.arange(s, e, dtype=np.int32)
    finite = np.isfinite(raw[s:e])
    invalid = (~finite).astype(np.float32)

    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.fill_between(t, 0.0, invalid, color="#ff9896", alpha=0.85, step="mid", label="Invalid timestep (NaN/Inf)")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["valid", "invalid"])
    ax.set_title(title)
    ax.set_xlabel("Timestep index")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_histograms(plt, out_path: Path, raw_finite: np.ndarray, cleaned: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    bins = 80
    ax.hist(raw_finite, bins=bins, density=True, alpha=0.55, color="#d62728", label="Raw (finite values only)")
    ax.hist(cleaned, bins=bins, density=True, alpha=0.45, color="#1f77b4", label="Cleaned (all timesteps)")
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_window_segmentation(
    plt,
    out_path: Path,
    cleaned: np.ndarray,
    win_len: int,
    stride: int,
    start: int,
    max_windows_draw: int,
    title: str,
):
    s = int(start)
    # show enough context for a few windows
    span = win_len + stride * (max_windows_draw - 1) + int(0.15 * win_len)
    e = min(int(s + span), int(cleaned.size))
    t = np.arange(s, e, dtype=np.int32)

    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(t, cleaned[s:e], color="#1f77b4", linewidth=1.1, label="Cleaned series (context)")

    spans = []
    pos = s
    while pos + win_len <= e and len(spans) < max_windows_draw:
        spans.append((pos, pos + win_len))
        pos += stride

    colors = ["#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf"]
    for i, (a, b) in enumerate(spans):
        c = colors[i % len(colors)]
        ax.axvspan(a, b - 1, color=c, alpha=0.18, label=f"Window {i+1}: [{a}, {b})")

    ax.set_title(title + f" (window={win_len}, stride={stride})")
    ax.set_xlabel("Timestep index")
    ax.set_ylabel("Telemetry value")
    ax.grid(True, alpha=0.25)

    # de-dup legend labels
    handles, labels = ax.get_legend_handles_labels()
    uniq = {}
    for h, lab in zip(handles, labels):
        uniq[lab] = h
    ax.legend(list(uniq.values()), list(uniq.keys()), loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_pipeline_diagram(plt, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    ax.axis("off")
    ax.set_title(title, pad=14)

    boxes = [
        ("Raw telemetry\n(CSV/NPY)", (0.05, 0.55, 0.16, 0.30)),
        ("Numeric coercion\n(float32)", (0.24, 0.55, 0.16, 0.30)),
        ("Stabilization\n(Inf→NaN)", (0.43, 0.55, 0.16, 0.30)),
        ("Interpolation +\nfill residual", (0.62, 0.55, 0.16, 0.30)),
        ("Structuring\nwindows (T,C)", (0.81, 0.55, 0.16, 0.30)),
        ("Train / Val / Test\n(data_split.json)", (0.33, 0.12, 0.34, 0.28)),
    ]

    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    drawn = []
    for text, (x, y, w, h) in boxes:
        bb = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            linewidth=1.1,
            edgecolor="#333333",
            facecolor="#f2f2f2",
        )
        ax.add_patch(bb)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)
        drawn.append((x, y, w, h))

    def center(box):
        x, y, w, h = box
        return (x + w / 2, y + h / 2)

    # arrows across top pipeline
    for i in range(len(drawn) - 2):  # stop before last box on top row? actually last is windows; we have 5 in row1 + split below
        pass

    # We built boxes list manually; connect first five in order
    top_boxes = [drawn[i] for i in range(5)]
    for i in range(len(top_boxes) - 1):
        x1, y1, w1, h1 = top_boxes[i]
        x2, y2, w2, h2 = top_boxes[i + 1]
        arr = FancyArrowPatch(
            (x1 + w1, y1 + h1 / 2),
            (x2, y2 + h2 / 2),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.1,
            color="#444444",
        )
        ax.add_patch(arr)

    # arrow from windows to split
    xw, yw, ww, hw = drawn[4]
    xs, ys, ws, hs = drawn[5]
    arr2 = FancyArrowPatch(
        (xw + ww / 2, yw),
        (xs + ws / 2, ys + hs),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.1,
        color="#444444",
        connectionstyle="arc3,rad=0.05",
    )
    ax.add_patch(arr2)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def load_split(split_json: Path) -> Dict[str, List[str]]:
    return json.loads(split_json.read_text(encoding="utf-8"))


def pick_sample_file(data_dir: Path, split_json: Path | None, prefer: str) -> Path:
    if split_json is not None and split_json.exists():
        sp = load_split(split_json)
        files = sp.get(prefer) or []
        for name in files:
            p = data_dir / name
            if p.exists():
                return p

    # fallback: first npy lexicographically
    files = sorted(data_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found under: {data_dir}")
    return files[0]


def _candidate_files_from_split(data_dir: Path, split_json: Path, prefer: str) -> List[Path]:
    sp = load_split(split_json)
    names: List[str] = []
    if prefer in sp and isinstance(sp[prefer], list):
        names.extend([str(x) for x in sp[prefer]])
    # also include other splits as fallback pool (still "real data")
    for k in ("train", "validation", "test"):
        if k == prefer:
            continue
        if k in sp and isinstance(sp[k], list):
            names.extend([str(x) for x in sp[k]])

    out: List[Path] = []
    seen = set()
    for n in names:
        p = (data_dir / n).resolve()
        if p.suffix.lower() != ".npy":
            continue
        if not p.exists():
            continue
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def count_invalid_in_npy(path: Path) -> Tuple[int, int]:
    """
    Returns (n_timesteps, invalid_count) for 1D series or first channel of (N,C).
    Uses mmap when possible to reduce RAM.
    """
    arr = np.load(path, mmap_mode="r")
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim == 2 and x.shape[1] >= 1:
        x = x[:, 0]
    elif x.ndim == 1:
        pass
    else:
        # unexpected shapes: flatten as last resort
        x = x.reshape(-1)

    invalid = int(np.sum(~np.isfinite(x)))
    return int(x.size), invalid


def find_real_file_with_invalid(
    data_dir: Path,
    split_json: Path,
    prefer: str,
    scan_max_files: int,
    seed: int,
    scan_scope: str,
) -> Tuple[Path | None, int, Dict[str, object]]:
    """
    Scan a bounded random subset of npy files to find any file containing NaN/Inf.

    scan_scope:
      - "split": only files listed in data_split.json (train/val/test union, prefer first)
      - "all": any *.npy under data_dir (still real on-disk data)
    """
    if scan_scope == "all":
        candidates = sorted(data_dir.glob("*.npy"))
        if not candidates:
            return None, 0, {"reason": "no_npy_files", "candidates": 0, "scan_scope": scan_scope}
    else:
        candidates = _candidate_files_from_split(data_dir, split_json, prefer)
        if not candidates:
            return None, 0, {"reason": "no_candidates", "candidates": 0, "scan_scope": scan_scope}

    rng = random.Random(int(seed))
    rng.shuffle(candidates)
    scanned = 0
    for p in candidates[: max(1, int(scan_max_files))]:
        scanned += 1
        try:
            n, inv = count_invalid_in_npy(p)
        except Exception as e:
            return None, scanned, {"reason": "read_error", "error": str(e), "file": str(p)}

        if inv > 0:
            return p, scanned, {
                "reason": "found",
                "invalid_timesteps": inv,
                "n_timesteps": n,
                "scan_scope": scan_scope,
            }

    return None, scanned, {"reason": "not_found_in_scan", "candidates_total": len(candidates), "scan_scope": scan_scope}


def pick_segment_showing_invalid(raw: np.ndarray, seg_len: int) -> int:
    """
    Choose a segment start that contains at least one invalid timestep, if possible.
    """
    x = np.asarray(raw, dtype=np.float32).reshape(-1)
    inv = ~np.isfinite(x)
    if not inv.any():
        return 0

    first = int(np.argmax(inv))
    s = max(0, first - int(0.25 * seg_len))
    s = min(s, max(0, int(x.size) - int(seg_len)))
    return int(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/reduced"))
    ap.add_argument("--split-json", type=Path, default=Path("backend/config/data_split.json"))
    ap.add_argument("--output-dir", type=Path, default=Path("backend/app/data_pipeline_figures"))
    ap.add_argument("--window-size", type=int, default=100)
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--prefer-split", type=str, default="test", choices=["train", "validation", "test"])
    ap.add_argument("--segment-start", type=int, default=0)
    ap.add_argument("--segment-len", type=int, default=800)
    ap.add_argument("--max-windows-draw", type=int, default=4)
    ap.add_argument(
        "--scan-max-files",
        type=int,
        default=600,
        help="When not using --demo-artifacts, scan up to N candidate .npy files to find real NaN/Inf.",
    )
    ap.add_argument("--scan-seed", type=int, default=42, help="RNG seed for shuffling candidate files during scan.")
    ap.add_argument(
        "--scan-scope",
        type=str,
        default="split",
        choices=["split", "all"],
        help='Where to search for invalid values: split-listed files ("split") or any npy under data-dir ("all").',
    )
    ap.add_argument(
        "--demo-artifacts",
        action="store_true",
        help=(
            "If set, inject a small controlled amount of NaN/Inf into a copied series for visualization. "
            "This does NOT modify any dataset files on disk."
        ),
    )
    ap.add_argument("--demo-nan", type=int, default=25, help="Number of NaNs to inject (spread across segment).")
    ap.add_argument("--demo-inf", type=int, default=8, help="Number of +/-Infs to inject (spread across segment).")
    args = ap.parse_args()

    plt = _ensure_matplotlib()
    _style(plt)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    artifact_info: Dict[str, object] = {}
    sample_path: Path

    if args.demo_artifacts:
        sample_path = pick_sample_file(args.data_dir, args.split_json, args.prefer_split)
        artifact_info = {"mode": "demo_injection"}
    else:
        found, scanned, info = find_real_file_with_invalid(
            args.data_dir,
            args.split_json,
            args.prefer_split,
            scan_max_files=args.scan_max_files,
            seed=args.scan_seed,
            scan_scope=args.scan_scope,
        )
        artifact_info = {
            "mode": "real_scan",
            "scan_seed": int(args.scan_seed),
            "scan_max_files": int(args.scan_max_files),
            "files_scanned": int(scanned),
            **info,
        }
        if found is not None:
            sample_path = found
        else:
            # No invalid values found within scan budget: still generate real plots from a real file,
            # but invalid mask may be empty.
            sample_path = pick_sample_file(args.data_dir, args.split_json, args.prefer_split)

    raw = np.load(sample_path, mmap_mode="r")
    raw = np.asarray(raw, dtype=np.float32).reshape(-1).copy()

    if args.demo_artifacts:
        # Controlled visualization-only corruption (never written back to disk).
        s0 = int(args.segment_start)
        L = int(args.segment_len)
        e0 = min(s0 + L, int(raw.size))
        if e0 > s0:
            idx = np.linspace(s0, e0 - 1, num=max(2, args.demo_nan), dtype=np.int64)
            raw[idx[: args.demo_nan]] = np.nan

            idx2 = np.linspace(s0, e0 - 1, num=max(2, args.demo_inf), dtype=np.int64)
            signs = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(args.demo_inf)], dtype=np.float32)
            raw[idx2[: args.demo_inf]] = signs * np.float32(np.inf)

    cleaned = clean_series(raw)

    # histogram uses finite subset of raw for readability
    raw_finite = raw[np.isfinite(raw)]

    # If user left default segment but we're in real mode, try to center segment on real invalids.
    seg_start = int(args.segment_start)
    if (not args.demo_artifacts) and seg_start == 0:
        seg_start = pick_segment_showing_invalid(raw, int(args.segment_len))

    meta = {
        "sample_file": str(sample_path.as_posix()),
        "n_timesteps": int(raw.size),
        "invalid_timesteps": int(np.sum(~np.isfinite(raw))),
        "demo_artifacts": bool(args.demo_artifacts),
        "segment_start_used": int(seg_start),
        "segment_len": int(args.segment_len),
        "artifact_search": artifact_info,
        "window_size": int(args.window_size),
        "stride": int(args.stride),
        "num_windows": int(make_windows_1ch(cleaned, args.window_size, args.stride)[0].shape[0]),
        "prefer_split": args.prefer_split,
    }
    (args.output_dir / "figure_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    demo_note = " [demo artifacts]" if args.demo_artifacts else ""

    plot_cleaning_before_after(
        plt,
        args.output_dir / "01_cleaning_before_after.png",
        raw,
        cleaned,
        start=seg_start,
        length=args.segment_len,
        title=f"Cleaning: raw vs cleaned (sample: {sample_path.name}){demo_note}",
    )

    plot_invalid_mask(
        plt,
        args.output_dir / "02_invalid_timesteps_mask.png",
        raw,
        start=seg_start,
        length=args.segment_len,
        title=f"Invalid timesteps mask (NaN/Inf) on raw series{demo_note}",
    )

    plot_histograms(
        plt,
        args.output_dir / "03_value_histogram_raw_vs_cleaned.png",
        raw_finite,
        cleaned,
        title=f"Value distribution: raw (finite) vs cleaned{demo_note}",
    )

    plot_window_segmentation(
        plt,
        args.output_dir / "04_sliding_windows_overlay.png",
        cleaned,
        win_len=args.window_size,
        stride=args.stride,
        start=seg_start,
        max_windows_draw=args.max_windows_draw,
        title=f"Structuring: sliding-window segmentation (overlay){demo_note}",
    )

    plot_pipeline_diagram(
        plt,
        args.output_dir / "00_pipeline_overview.png",
        title=f"Data pipeline overview (cleaning → structuring → splits){demo_note}",
    )

    print("Wrote figures to:", args.output_dir.resolve())
    for p in sorted(args.output_dir.glob("*.png")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
