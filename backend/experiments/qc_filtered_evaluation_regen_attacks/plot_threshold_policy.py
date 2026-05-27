"""
Compatibility wrapper: all QC-filtered figures (including threshold_policy_qc_filtered.png)
are produced by plot_strict_v2_like_figures.main() in one inference pass.

Run either this script or plot_strict_v2_like_figures.py — same result.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    exp_dir = Path(__file__).resolve().parent
    if str(exp_dir) not in sys.path:
        sys.path.insert(0, str(exp_dir))
    import plot_strict_v2_like_figures as figmod  # noqa: PLC0415 — same directory

    os.chdir(ROOT)
    figmod.main()


if __name__ == "__main__":
    main()
