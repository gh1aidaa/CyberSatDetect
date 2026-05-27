"""
Conservative pattern-shift assist (experimental only).

Idea: keep the main anomaly signal as structural (recon + pred + grad) with NO
global inflation from the order head. Apply a small order-based boost only in
the upper tail near the operational threshold, calibrated on validation-normal.

This targets pattern_shift-like temporal disorder while limiting FP inflation
that would hurt FAR and other attack recalls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def apply_tail_order_boost(
    s_struct: np.ndarray,
    order_p: np.ndarray,
    thr_ref: float,
    *,
    gate_ratio: float,
    boost_max: float,
) -> np.ndarray:
    """
    s_fused = s_struct + boost_max * order_p * tail
    tail in [0,1] ramps from 0 at gate_ratio*thr_ref to 1 at thr_ref (clipped above).
    """
    s = np.asarray(s_struct, dtype=np.float64)
    o = np.asarray(order_p, dtype=np.float64).reshape(-1)
    thr = float(thr_ref)
    g = float(gate_ratio)
    bm = float(boost_max)
    span = max(thr * (1.0 - g), 1e-12)
    tail = np.clip((s - g * thr) / span, 0.0, 1.0)
    return (s + bm * o * tail).astype(np.float64)


def calibrate_tail_boost(
    val_s_struct: np.ndarray,
    val_order: np.ndarray,
    *,
    gate_candidates: List[float] | None = None,
    boost_candidates: List[float] | None = None,
    max_p99_inflation_ratio: float = 1.002,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Pick (gate_ratio, boost_max) maximizing boost_max subject to:
      quantile(fused, 0.99) <= quantile(val_s_struct, 0.99) * max_p99_inflation_ratio

    Falls back to (gate, 0.0) if no candidate satisfies constraint.
    """
    vs = np.asarray(val_s_struct, dtype=np.float64)
    vo = np.asarray(val_order, dtype=np.float64).reshape(-1)
    if vs.size == 0:
        return 0.85, 0.0, {"base_p99": float("nan"), "fused_p99": float("nan")}

    base_p99 = float(np.quantile(vs, 0.99))
    cap = base_p99 * float(max_p99_inflation_ratio)

    gates = gate_candidates if gate_candidates is not None else [0.88, 0.82, 0.76, 0.70, 0.64, 0.58]
    boosts = boost_candidates if boost_candidates is not None else [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08]

    best_gate = float(gates[0])
    best_boost = 0.0
    best_fused_p99 = base_p99

    for bm in sorted(boosts, reverse=True):
        feasible_gates: List[float] = []
        for g in gates:
            fused = apply_tail_order_boost(vs, vo, base_p99, gate_ratio=float(g), boost_max=float(bm))
            fp99 = float(np.quantile(fused, 0.99))
            if fp99 <= cap:
                feasible_gates.append(float(g))
        if feasible_gates:
            best_boost = float(bm)
            best_gate = float(min(feasible_gates))
            fused = apply_tail_order_boost(vs, vo, base_p99, gate_ratio=best_gate, boost_max=best_boost)
            best_fused_p99 = float(np.quantile(fused, 0.99))
            break

    meta = {
        "base_p99": base_p99,
        "fused_p99": best_fused_p99,
        "cap_p99": cap,
        "chosen_gate_ratio": best_gate,
        "chosen_boost_max": best_boost,
    }
    return best_gate, best_boost, meta


def save_fusion_config(path: Path, cfg: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
