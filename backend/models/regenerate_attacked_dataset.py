"""
Regenerate attacked dataset (v2) with strict ground-truth labels.

Constraints (per user request):
- Do NOT touch training code or model files.
- Do NOT modify/delete data/reduced.
- Do NOT use old data/attacked for evaluation (only delete its contents if requested separately).

This script:
- Reads ONLY normal test files from data/reduced using backend/config/data_split.json
- Creates attacked versions under data/attacked_v2
- Saves one NPZ per source file including:
  - X: attacked windows (B, T, C)
  - y_timestep: attacked mask per timestep (B, T) [0/1]
  - y_window: window label per window (B,) [0/1], 1 if attacked fraction >= 10%
  - attack_type: str
  - attack_start, attack_end: int (inclusive start, exclusive end) over timesteps
  - source_file: original filename
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


ATTACK_TYPES = ("freeze", "spike", "drift", "pattern_shift", "noise")


def _maybe_tqdm():
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm
    except Exception:
        return None


def load_split_filenames(split_json: Path, split_key: str = "test") -> List[str]:
    with split_json.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if split_key not in obj:
        raise KeyError(f"Split key '{split_key}' not found. Available keys: {list(obj.keys())}")
    names = list(obj[split_key])
    if not names:
        raise ValueError(f"No filenames under split '{split_key}' in {split_json}")
    return names


def load_windows_npy(path: Path) -> np.ndarray:
    """
    Expected training/eval windows format in this repo:
      - (B, T) or (B, T, C)
    We convert to float32 and ensure (B, T, C).
    """
    X = np.load(path).astype(np.float32)
    if X.ndim == 2:
        X = X[..., None]
    if X.ndim != 3:
        raise ValueError(f"Unsupported array ndim={X.ndim} for {path} (shape={getattr(X, 'shape', None)})")
    return X


def _validate_window_shape(X: np.ndarray, window_size: int, channels: Optional[int] = None) -> Tuple[int, int, int]:
    if X.ndim != 3:
        raise ValueError(f"Expected 3D windows array, got shape {X.shape}")
    B, T, C = int(X.shape[0]), int(X.shape[1]), int(X.shape[2])
    if T != int(window_size):
        raise ValueError(f"Window size mismatch: expected T={window_size}, got {T} (shape={X.shape})")
    if channels is not None and C != int(channels):
        raise ValueError(f"Channel mismatch: expected C={channels}, got {C} (shape={X.shape})")
    return B, T, C


@dataclass(frozen=True)
class AttackSpec:
    attack_type: str
    start: int
    end: int  # exclusive
    params: Dict[str, float]


def choose_attack_spec(
    rng: np.random.Generator,
    window_size: int,
    attack_type: Optional[str] = None,
    min_len: int = 10,
    max_len: Optional[int] = None,
) -> AttackSpec:
    if attack_type is None:
        attack_type = str(rng.choice(ATTACK_TYPES))
    if attack_type not in ATTACK_TYPES:
        raise ValueError(f"Unsupported attack_type: {attack_type} (supported: {ATTACK_TYPES})")

    T = int(window_size)
    max_len = int(max_len or max(min_len, int(0.4 * T)))
    max_len = min(max_len, T)
    L = int(rng.integers(min_len, max_len + 1))
    start = int(rng.integers(0, T - L + 1))
    end = start + L

    # Default params tuned to create detectable but not absurd changes.
    params: Dict[str, float] = {}
    if attack_type == "freeze":
        params = {"offset": float(rng.uniform(0.3, 1.2))}
    elif attack_type == "spike":
        params = {"amplitude": float(rng.uniform(3.0, 8.0))}
    elif attack_type == "drift":
        params = {"strength": float(rng.uniform(0.8, 3.0))}
    elif attack_type == "pattern_shift":
        params = {"shift": float(int(rng.integers(3, max(4, T // 4))))}
    elif attack_type == "noise":
        params = {"sigma": float(rng.uniform(0.3, 1.2))}

    return AttackSpec(attack_type=attack_type, start=start, end=end, params=params)


def apply_attack(
    X: np.ndarray,
    spec: AttackSpec,
    rng: np.random.Generator,
    window_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply attack inside [start:end) timesteps for a SUBSET of windows in a file.
    If window_mask is None -> attack all windows.
    window_mask: boolean array shape (B,) where True means "attack this window".
    Returns:
      X_attacked: (B,T,C)
      y_timestep: (B,T) 0/1
    """
    X = np.asarray(X, dtype=np.float32)
    X_att = X.copy()
    B, T, C = X_att.shape

    if window_mask is None:
        window_mask = np.ones((B,), dtype=bool)
    window_mask = np.asarray(window_mask, dtype=bool).reshape(-1)
    if window_mask.shape[0] != B:
        raise ValueError(f"window_mask length mismatch: expected {B}, got {window_mask.shape[0]}")

    y_ts = np.zeros((B, T), dtype=np.uint8)
    y_ts[window_mask, spec.start:spec.end] = 1

    s, e = spec.start, spec.end
    if spec.attack_type == "freeze":
        # freeze: set segment to a constant baseline + offset (per window, per channel)
        baseline = X_att[window_mask, s : s + 1, :]  # (Bw,1,C)
        offset = float(spec.params.get("offset", 0.8))
        X_att[window_mask, s:e, :] = baseline + offset

    elif spec.attack_type == "spike":
        # spike: inject a single timestep spike inside the segment for each window
        amp = float(spec.params.get("amplitude", 5.0))
        spike_t = int(rng.integers(s, e))
        Bw = int(window_mask.sum())
        if Bw > 0:
            spike = rng.normal(loc=amp, scale=0.2 * amp, size=(Bw, 1, C)).astype(np.float32)
            X_att[window_mask, spike_t : spike_t + 1, :] = X_att[window_mask, spike_t : spike_t + 1, :] + spike

    elif spec.attack_type == "drift":
        # drift: add linear ramp increasing over the segment
        strength = float(spec.params.get("strength", 2.5))
        ramp = np.linspace(0.0, strength, e - s, dtype=np.float32).reshape(1, -1, 1)
        X_att[window_mask, s:e, :] = X_att[window_mask, s:e, :] + ramp

    elif spec.attack_type == "pattern_shift":
        # pattern shift over time dimension: roll the segment by k (circular)
        k = int(spec.params.get("shift", 15))
        seg = X_att[window_mask, s:e, :]
        X_att[window_mask, s:e, :] = np.roll(seg, shift=k, axis=1)

    elif spec.attack_type == "noise":
        sigma = float(spec.params.get("sigma", 0.8))
        Bw = int(window_mask.sum())
        if Bw > 0:
            noise = rng.normal(loc=0.0, scale=sigma, size=(Bw, e - s, C)).astype(np.float32)
            X_att[window_mask, s:e, :] = X_att[window_mask, s:e, :] + noise

    else:
        raise ValueError(f"Unknown attack_type: {spec.attack_type}")

    return X_att.astype(np.float32), y_ts


def y_window_from_timestep(y_timestep: np.ndarray, min_fraction: float = 0.10) -> np.ndarray:
    y_timestep = np.asarray(y_timestep)
    if y_timestep.ndim != 2:
        raise ValueError(f"y_timestep must be 2D (B,T), got shape {y_timestep.shape}")
    frac = y_timestep.mean(axis=1)
    return (frac >= float(min_fraction)).astype(np.uint8)


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Regenerate attacked dataset v2 with strict labels.")
    ap.add_argument("--normal-dir", type=str, required=True, help="Path to data/reduced")
    ap.add_argument("--split-json", type=str, required=True, help="Path to backend/config/data_split.json")
    ap.add_argument("--output-dir", type=str, required=True, help="Output attacked_v2 directory")
    ap.add_argument("--window-size", type=int, required=True, help="Window size T (expected 100)")
    ap.add_argument("--stride", type=int, required=True, help="Stride (kept for metadata/compatibility)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--split-key", type=str, default="test", help="Split key inside data_split.json (default: test)")
    ap.add_argument("--attack-type", type=str, default="mixed", help="One of attack types or 'mixed'")
    ap.add_argument("--min-attack-len", type=int, default=10, help="Minimum attack length in timesteps")
    ap.add_argument("--max-attack-len", type=int, default=40, help="Maximum attack length in timesteps")
    ap.add_argument("--window-anom-frac", type=float, default=0.10, help="Anomaly fraction threshold for y_window")
    ap.add_argument("--max-files", type=int, default=0, help="If >0, limit number of processed files")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing NPZ files if present")
    ap.add_argument(
        "--attack-window-frac",
        type=float,
        default=0.30,
        help="Fraction of windows inside each file to attack (0..1). Keeps attacked files partially normal at window-level.",
    )
    args = ap.parse_args()

    normal_dir = Path(args.normal_dir).resolve()
    split_json = Path(args.split_json).resolve()
    out_dir = Path(args.output_dir).resolve()

    if not normal_dir.exists():
        raise FileNotFoundError(normal_dir)
    if not split_json.exists():
        raise FileNotFoundError(split_json)

    safe_mkdir(out_dir)

    names = load_split_filenames(split_json, args.split_key)
    if args.max_files and int(args.max_files) > 0:
        names = names[: int(args.max_files)]

    rng = np.random.default_rng(int(args.seed))

    attack_mode = str(args.attack_type).strip().lower()
    if attack_mode != "mixed" and attack_mode not in ATTACK_TYPES:
        raise ValueError(f"--attack-type must be one of {ATTACK_TYPES} or 'mixed'")

    tqdm = _maybe_tqdm()
    it = names
    if tqdm is not None:
        it = tqdm(names, desc="Generating attacked_v2", unit="file")

    n_ok = 0
    n_failed = 0
    attack_counts: Dict[str, int] = {k: 0 for k in ATTACK_TYPES}

    for fname in it:
        src = (normal_dir / fname).resolve()
        if not src.exists():
            warnings.warn(f"[SKIP] Source file not found: {src}")
            n_failed += 1
            continue

        out_npz = out_dir / (Path(fname).stem + ".npz")
        if out_npz.exists() and not args.overwrite:
            continue

        try:
            X = load_windows_npy(src)
            B, _, _ = _validate_window_shape(X, window_size=args.window_size)

            spec = choose_attack_spec(
                rng=rng,
                window_size=args.window_size,
                attack_type=None if attack_mode == "mixed" else attack_mode,
                min_len=int(args.min_attack_len),
                max_len=int(args.max_attack_len),
            )

            frac = float(args.attack_window_frac)
            if not (0.0 <= frac <= 1.0):
                raise ValueError("--attack-window-frac must be within [0,1]")
            k = int(round(frac * B))
            k = max(1, k) if frac > 0 else 0
            mask = np.zeros((B,), dtype=bool)
            if k > 0:
                idx = rng.choice(B, size=k, replace=False)
                mask[idx] = True

            X_att, y_ts = apply_attack(X, spec, rng=rng, window_mask=mask)
            y_w = y_window_from_timestep(y_ts, min_fraction=float(args.window_anom_frac))

            # Store metadata as arrays to keep everything inside one .npz
            np.savez_compressed(
                out_npz,
                X=X_att,
                y_timestep=y_ts,
                y_window=y_w,
                attack_type=np.array(spec.attack_type),
                attack_start=np.array(int(spec.start), dtype=np.int32),
                attack_end=np.array(int(spec.end), dtype=np.int32),
                source_file=np.array(str(fname)),
                window_size=np.array(int(args.window_size), dtype=np.int32),
                stride=np.array(int(args.stride), dtype=np.int32),
                seed=np.array(int(args.seed), dtype=np.int32),
                attack_window_frac=np.array(float(args.attack_window_frac), dtype=np.float32),
            )

            attack_counts[spec.attack_type] += 1
            n_ok += 1
        except Exception as e:
            warnings.warn(f"[FAIL] {fname}: {type(e).__name__}: {e}")
            n_failed += 1
            continue

    meta = {
        "normal_dir": str(normal_dir),
        "split_json": str(split_json),
        "split_key": str(args.split_key),
        "output_dir": str(out_dir),
        "window_size": int(args.window_size),
        "stride": int(args.stride),
        "seed": int(args.seed),
        "attack_type_mode": attack_mode,
        "window_anom_frac": float(args.window_anom_frac),
        "min_attack_len": int(args.min_attack_len),
        "max_attack_len": int(args.max_attack_len),
        "files_total": len(names),
        "files_ok": int(n_ok),
        "files_failed": int(n_failed),
        "attack_type_counts": attack_counts,
    }

    (out_dir / "generation_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nDone.")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # Ensure relative imports won't break if launched from repo root.
    os.chdir(Path(__file__).resolve().parents[2])
    main()

