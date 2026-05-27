"""
Generate small CSV + NPY samples for end-to-end web testing.
Run: python scripts/generate_web_smoke_samples.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "samples" / "web_smoke"
N = 8000  # enough windows with WINDOW_LEN=100, STRIDE=50


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    t = np.linspace(0, 80, N, dtype=np.float32)
    # Normal: smooth signal + small noise
    x = 0.4 * np.sin(t) + 0.05 * rng.standard_normal(N).astype(np.float32)
    # Injected anomaly region: large deviation (should increase scores)
    x[3200:4200] += 2.5

    csv_path = OUT_DIR / "web_smoke_test.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["telemetry_ch1"])
        for v in x:
            w.writerow([float(v)])

    npy_path = OUT_DIR / "web_smoke_test.npy"
    # (N, 1) raw multichannel form used by server for .npy
    np.save(npy_path, x.reshape(-1, 1).astype(np.float32))

    print("Wrote:")
    print(" ", csv_path)
    print(" ", npy_path)
    print("Shape (npy):", np.load(npy_path).shape)


if __name__ == "__main__":
    main()
