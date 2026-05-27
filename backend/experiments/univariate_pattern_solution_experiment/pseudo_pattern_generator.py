"""
Pseudo pattern-shift generators for self-supervised training only.

Operates on normal windows with shape (100, 1). No attacked_v2 usage.
"""

from __future__ import annotations

import numpy as np

RNG = np.random.default_rng()


def _ensure_tc(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[..., None]
    if x.ndim != 3 or int(x.shape[1]) != 100 or int(x.shape[2]) != 1:
        raise ValueError(f"Expected (B,100,1), got {x.shape}")
    return x


def transform_circular_shift(x: np.ndarray, shift: int | None = None) -> np.ndarray:
    x = _ensure_tc(x)
    B, T, C = x.shape
    if shift is None:
        shift = int(RNG.integers(5, 45))
    y = np.roll(x, shift=int(shift), axis=1)
    return y.astype(np.float32)


def transform_block_shuffle(x: np.ndarray, n_blocks: int = 5) -> np.ndarray:
    x = _ensure_tc(x)
    B, T, C = x.shape
    out = np.empty_like(x)
    for b in range(B):
        k = int(max(2, min(int(n_blocks), T // 5)))
        cuts = sorted(RNG.choice(np.arange(1, T), size=k - 1, replace=False).tolist())
        idxs = [0] + cuts + [T]
        blocks = [x[b, idxs[i] : idxs[i + 1], :].copy() for i in range(len(idxs) - 1)]
        order = RNG.permutation(len(blocks))
        out[b] = np.concatenate([blocks[int(j)] for j in order], axis=0)
    return out


def transform_segment_swap(x: np.ndarray) -> np.ndarray:
    x = _ensure_tc(x)
    B, T, C = x.shape
    out = x.copy()
    for b in range(B):
        len_a = int(RNG.integers(10, 20))
        len_b = int(len_a)
        a0 = int(RNG.integers(5, 40))
        b0 = int(RNG.integers(45, max(46, T - len_b - 1)))
        a1 = a0 + len_a
        b1 = b0 + len_b
        if a1 > T or b1 > T:
            continue
        seg_a = out[b, a0:a1, :].copy()
        seg_b = out[b, b0:b1, :].copy()
        out[b, a0:a1, :] = seg_b
        out[b, b0:b1, :] = seg_a
    return out.astype(np.float32)


def transform_partial_time_reversal(x: np.ndarray) -> np.ndarray:
    x = _ensure_tc(x)
    B, T, C = x.shape
    out = x.copy()
    for b in range(B):
        s = int(RNG.integers(10, 40))
        e = int(RNG.integers(s + 15, min(s + 60, T)))
        out[b, s:e, :] = out[b, s:e, :][::-1, :]
    return out.astype(np.float32)


def transform_local_segment_permutation(x: np.ndarray) -> np.ndarray:
    x = _ensure_tc(x)
    B, T, C = x.shape
    out = x.copy()
    for b in range(B):
        w = int(RNG.integers(8, 20))
        s = int(RNG.integers(0, max(1, T - w)))
        perm = RNG.permutation(w)
        out[b, s : s + w, :] = out[b, s : s + w, :][perm, :]
    return out.astype(np.float32)


_TRANSFORMS = {
    "circular_shift": transform_circular_shift,
    "block_shuffle": transform_block_shuffle,
    "segment_swap": transform_segment_swap,
    "partial_time_reversal": transform_partial_time_reversal,
    "local_segment_permutation": transform_local_segment_permutation,
}


def apply_random_pseudo(x: np.ndarray, rng: np.random.Generator | None = None) -> tuple[np.ndarray, list[str]]:
    """
    Returns pseudo windows and list of transform names per batch element (same length as B).
    """
    x = _ensure_tc(x)
    B = int(x.shape[0])
    g = rng if rng is not None else RNG
    names: list[str] = []
    keys = list(_TRANSFORMS.keys())
    for _ in range(B):
        k = str(keys[int(g.integers(0, len(keys)))])
        names.append(k)
    out = x.copy()
    for i in range(B):
        xi = x[i : i + 1]
        fn = _TRANSFORMS[names[i]]
        yi = fn(xi)
        out[i : i + 1] = yi
    return out.astype(np.float32), names


def counts_from_names(names: list[str]) -> dict[str, int]:
    c: dict[str, int] = {}
    for n in names:
        c[n] = c.get(n, 0) + 1
    return c
