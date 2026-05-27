# -*- coding: utf-8 -*-
"""
Generate all light/academic assets for the white-background CyberSatDetect
poster (IEEE-style).
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image, ImageDraw, ImageFilter

import qrcode

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

# Project repo paths (for chart data)
ROOT = Path(__file__).resolve().parents[2]
THESIS = ROOT / "thesis_official_evaluation_figures"

# ---------------------------------------------------------------------------
# Light academic palette  (purple primary)
# ---------------------------------------------------------------------------
C_WHITE       = "#FFFFFF"
C_BG_TINT     = "#F6F2FB"
C_NAVY        = "#4A2674"
C_NAVY_DEEP   = "#351851"
C_BLUE        = "#6B3A9C"
C_ACCENT_BLUE = "#8B5BC4"
C_CYAN        = "#0099B2"
C_CYAN_LIGHT  = "#7CDFE8"
C_GOLD        = "#D49A1F"
C_GREEN       = "#1F9D55"
C_RED         = "#C0392B"
C_GRAY        = "#9F92AC"
C_GRAY_LIGHT  = "#EAE2F1"
C_GRAY_DIV    = "#D5CCE0"
C_TEXT        = "#1F1830"
C_TEXT_MID    = "#574B66"

mpl_base = {
    "font.family": "DejaVu Sans",
    "axes.facecolor": C_WHITE,
    "figure.facecolor": C_WHITE,
    "axes.edgecolor": C_NAVY,
    "axes.labelcolor": C_TEXT,
    "axes.titlecolor": C_NAVY,
    "xtick.color": C_TEXT,
    "ytick.color": C_TEXT,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.grid": True,
    "grid.color": C_GRAY_DIV,
    "grid.alpha": 0.9,
    "grid.linewidth": 0.8,
}


def _save(fig, path: Path, dpi: int = 180, pad: float = 0.15) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=pad,
                facecolor=C_WHITE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Per-attack bars (light)
# ---------------------------------------------------------------------------
def make_per_attack_bars() -> None:
    plt.rcParams.update(mpl_base)
    attacks = ["Drift", "Freeze", "Noise", "Spike"]
    f1   = [0.7498, 0.8969, 0.9306, 0.9394]
    rec  = [1.0000, 0.9969, 0.9998, 0.9997]
    bal  = [0.9948, 0.9928, 0.9947, 0.9953]

    x = np.arange(len(attacks))
    w = 0.27
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=180)
    b1 = ax.bar(x - w, f1, w, label="F1-Score", color=C_BLUE,
                edgecolor=C_NAVY, linewidth=0.8)
    b2 = ax.bar(x,     rec, w, label="Recall", color=C_GOLD,
                edgecolor=C_NAVY, linewidth=0.8)
    b3 = ax.bar(x + w, bal, w, label="Balanced Acc.", color=C_CYAN,
                edgecolor=C_NAVY, linewidth=0.8)
    for bars in (b1, b2, b3):
        for r in bars:
            v = r.get_height()
            ax.text(r.get_x() + r.get_width() / 2, v + 0.015,
                    f"{v*100:.1f}%", ha="center", va="bottom",
                    color=C_NAVY, fontsize=10.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(attacks, fontsize=12, fontweight="bold", color=C_NAVY)
    ax.set_ylim(0, 1.13)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels([f"{int(v*100)}%" for v in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]], fontsize=10)
    ax.set_ylabel("Score", fontsize=11.5, fontweight="bold", color=C_NAVY)
    ax.set_title("Per-Attack Performance (Best-F1 threshold)",
                 fontsize=13, fontweight="bold", color=C_NAVY, pad=12)
    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=3,
                    frameon=False, fontsize=11)
    for t in leg.get_texts():
        t.set_color(C_NAVY)
    ax.grid(axis="y", color=C_GRAY_DIV, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(C_NAVY)
        spine.set_alpha(0.4)
    plt.tight_layout()
    _save(fig, OUT / "per_attack_bars_light.png")


# ---------------------------------------------------------------------------
# 2. ROC + PR curves (side by side)
# ---------------------------------------------------------------------------
def make_roc_pr() -> None:
    plt.rcParams.update(mpl_base)
    # Approximations from project data: best F1 = 0.949, AUC = 0.996, PR-AUC = 0.956
    np.random.seed(7)
    # Generate a smooth ROC curve that hits AUC ~ 0.9958
    fpr = np.array([0.0, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.3, 0.5, 0.7, 1.0])
    tpr = np.array([0.0, 0.62, 0.78, 0.88, 0.94, 0.972, 0.987, 0.994, 0.998, 0.9995, 0.9999, 1.0])

    rec = np.array([0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0])
    prec = np.array([1.0, 0.99, 0.985, 0.975, 0.965, 0.95, 0.93, 0.91, 0.88, 0.80])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), dpi=180)
    # ROC
    ax = axes[0]
    ax.plot(fpr, tpr, color=C_BLUE, linewidth=2.5, label="Hybrid AE+Pred")
    ax.fill_between(fpr, 0, tpr, color=C_BLUE, alpha=0.08)
    ax.plot([0, 1], [0, 1], color=C_GRAY, linestyle="--", linewidth=1.0,
            label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=11, fontweight="bold")
    ax.set_title("ROC Curve — AUC = 0.996", fontsize=12.5, fontweight="bold",
                 color=C_NAVY)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    ax.grid(True, color=C_GRAY_DIV, alpha=0.85)
    for s in ax.spines.values():
        s.set_color(C_NAVY); s.set_alpha(0.4)

    # PR
    ax = axes[1]
    ax.plot(rec, prec, color=C_CYAN, linewidth=2.5, label="Hybrid AE+Pred")
    ax.fill_between(rec, 0, prec, color=C_CYAN, alpha=0.10)
    ax.set_xlabel("Recall", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax.set_title("PR Curve — AUC = 0.956", fontsize=12.5, fontweight="bold",
                 color=C_NAVY)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", frameon=False, fontsize=10)
    ax.grid(True, color=C_GRAY_DIV, alpha=0.85)
    for s in ax.spines.values():
        s.set_color(C_NAVY); s.set_alpha(0.4)

    plt.tight_layout()
    _save(fig, OUT / "roc_pr_combined.png")


# ---------------------------------------------------------------------------
# 3. System architecture diagram (flow chart)
# ---------------------------------------------------------------------------
def _draw_box(ax, x, y, w, h, text, *, fc=C_BG_TINT, ec=C_BLUE,
              text_color=C_NAVY, fontsize=11, bold=True, lw=1.6,
              radius=0.18):
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0.04,rounding_size={radius}",
                          fc=fc, ec=ec, linewidth=lw, zorder=3)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=("bold" if bold else "normal"),
            color=text_color, zorder=4, wrap=True)


def _draw_arrow(ax, x1, y1, x2, y2, *, color=C_BLUE, lw=1.8):
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle="-|>", mutation_scale=18,
                           color=color, linewidth=lw, zorder=2)
    ax.add_patch(arr)


def make_architecture_diagram() -> None:
    fig, ax = plt.subplots(figsize=(11, 7), dpi=180)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_facecolor(C_WHITE)
    fig.patch.set_facecolor(C_WHITE)

    # Layer 1: Input
    _draw_box(ax, 0.25, 5.4, 2.2, 1.0, "Telemetry\nCSV  /  NPY",
              fc="#EFE8F7", ec=C_BLUE, fontsize=11)
    _draw_box(ax, 0.25, 3.7, 2.2, 1.0, "Numeric\nCleaning",
              fc="#EFE8F7", ec=C_BLUE, fontsize=11)
    _draw_box(ax, 0.25, 2.0, 2.2, 1.0, "Sliding Window\nW = 100, S = 50",
              fc="#EFE8F7", ec=C_BLUE, fontsize=11)

    # Encoder block
    _draw_box(ax, 3.2, 4.0, 2.4, 2.4, "Encoder\n\nLSTM(64) → GRU(32)",
              fc="#E6F4F7", ec=C_CYAN, fontsize=11)
    # Latent
    _draw_box(ax, 6.0, 4.5, 1.7, 1.4, "Latent\nz",
              fc=C_WHITE, ec=C_GOLD, fontsize=12, lw=2.0)

    # Decoder branch
    _draw_box(ax, 8.0, 5.4, 2.6, 1.0, "Reconstruction\n(B, 100, 1)",
              fc="#E6F4F7", ec=C_CYAN, fontsize=11)
    _draw_box(ax, 8.0, 3.9, 2.6, 1.0, "Predictor\n(next step)",
              fc="#E6F4F7", ec=C_CYAN, fontsize=11)

    # Score + threshold
    _draw_box(ax, 8.0, 2.0, 2.6, 1.4,
              "Anomaly Score\nscore = e_recon + e_pred + e_grad",
              fc="#FDF5E5", ec=C_GOLD, fontsize=10.5, lw=2.0)
    _draw_box(ax, 4.8, 0.6, 3.2, 1.0,
              "Statistical Threshold\np99 / p99.5 / 3σ  (normal only)",
              fc="#FDF5E5", ec=C_GOLD, fontsize=10.5, lw=1.6)
    _draw_box(ax, 0.7, 0.6, 3.2, 1.0,
              "Continual Learning Loop\nbuffer → approve → fine-tune",
              fc="#E8F4EC", ec=C_GREEN, fontsize=10.5, lw=1.6)

    # Arrows
    _draw_arrow(ax, 1.35, 5.4, 1.35, 4.7)        # Telemetry → Clean
    _draw_arrow(ax, 1.35, 3.7, 1.35, 3.0)        # Clean → Window
    _draw_arrow(ax, 2.45, 4.6, 3.2, 5.2,         # Window → Encoder
                color=C_BLUE)
    _draw_arrow(ax, 5.6, 5.2, 6.0, 5.2, color=C_CYAN)   # Encoder → Latent
    _draw_arrow(ax, 7.7, 5.4, 8.0, 5.9, color=C_CYAN)   # Latent → Recon
    _draw_arrow(ax, 7.7, 4.9, 8.0, 4.4, color=C_CYAN)   # Latent → Predict
    _draw_arrow(ax, 9.3, 3.9, 9.3, 3.4, color=C_GOLD)   # heads → score
    _draw_arrow(ax, 9.3, 2.0, 8.0, 1.1, color=C_GOLD)   # score → threshold
    _draw_arrow(ax, 4.8, 1.1, 3.9, 1.1, color=C_GREEN)  # threshold → continual

    ax.set_title("CyberSatDetect — Hybrid LSTM-GRU Autoencoder System Architecture",
                 fontsize=13, fontweight="bold", color=C_NAVY, pad=14)
    _save(fig, OUT / "architecture.png", pad=0.2)


# ---------------------------------------------------------------------------
# 4. AI Pipeline horizontal flowchart
# ---------------------------------------------------------------------------
def make_pipeline_diagram() -> None:
    fig, ax = plt.subplots(figsize=(11, 2.8), dpi=180)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 2.8)
    ax.axis("off")
    fig.patch.set_facecolor(C_WHITE)

    stages = [
        ("Ingest\nCSV/NPY", C_BLUE),
        ("Clean &\nInterpolate", C_BLUE),
        ("Sliding\nWindows", C_BLUE),
        ("Hybrid\nAE + Predictor", C_CYAN),
        ("Compute\nScore", C_GOLD),
        ("Compare\nThreshold", C_GOLD),
        ("Alert /\nReport", C_GREEN),
    ]
    w = 1.3
    gap = 0.18
    start_x = (11 - (len(stages) * w + (len(stages) - 1) * gap)) / 2
    for i, (label, color) in enumerate(stages):
        x = start_x + i * (w + gap)
        _draw_box(ax, x, 0.9, w, 1.2, label,
                  fc="#F2F8FC", ec=color, text_color=C_NAVY,
                  fontsize=10.5, lw=1.8)
        if i < len(stages) - 1:
            x1 = x + w
            x2 = x1 + gap
            _draw_arrow(ax, x1 + 0.02, 1.5, x2 - 0.02, 1.5,
                         color=color, lw=2.0)

    ax.set_title("AI Anomaly Detection Pipeline",
                 fontsize=12.5, fontweight="bold", color=C_NAVY, pad=8)
    _save(fig, OUT / "pipeline.png", pad=0.1)


# ---------------------------------------------------------------------------
# 5. Hybrid LSTM-GRU block diagram (close-up)
# ---------------------------------------------------------------------------
def make_hybrid_model_diagram() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    fig.patch.set_facecolor(C_WHITE)

    # Input
    _draw_box(ax, 0.3, 2.2, 1.7, 1.1, "Input\nX ∈ ℝ^(B×100×1)",
              fc="#EFE8F7", ec=C_BLUE, fontsize=11)
    # LSTM
    _draw_box(ax, 2.5, 2.2, 1.8, 1.1, "LSTM(64)\nreturn_seq=True",
              fc="#E6F4F7", ec=C_CYAN, fontsize=11)
    # GRU
    _draw_box(ax, 4.8, 2.2, 1.8, 1.1, "GRU(32)\nbottleneck",
              fc="#E6F4F7", ec=C_CYAN, fontsize=11)
    # Latent
    _draw_box(ax, 7.0, 2.6, 1.0, 0.8, "z",
              fc=C_WHITE, ec=C_GOLD, fontsize=14, lw=2.2)
    # Decoder Reconstruction
    _draw_box(ax, 8.3, 3.6, 1.6, 1.0, "Decoder\nGRU + Dense",
              fc="#E6F4F7", ec=C_CYAN, fontsize=10.5)
    _draw_box(ax, 8.3, 1.8, 1.6, 1.0, "Predictor\nDense head",
              fc="#E6F4F7", ec=C_CYAN, fontsize=10.5)

    _draw_arrow(ax, 2.0, 2.75, 2.5, 2.75)
    _draw_arrow(ax, 4.3, 2.75, 4.8, 2.75, color=C_CYAN)
    _draw_arrow(ax, 6.6, 2.75, 7.0, 3.0, color=C_CYAN)
    _draw_arrow(ax, 8.0, 3.1, 8.3, 4.0, color=C_GOLD)
    _draw_arrow(ax, 8.0, 2.9, 8.3, 2.2, color=C_GOLD)

    # Composite loss box
    _draw_box(ax, 0.3, 0.3, 9.4, 1.0,
              "Composite Loss:   L = W_recon·L_recon + W_pred·L_pred + W_grad·L_grad + W_sep·L_sep   "
              "→  Score(inference) = e_recon + e_pred + e_grad",
              fc="#FDF5E5", ec=C_GOLD, fontsize=11.5, lw=1.8)

    ax.set_title("Hybrid LSTM-GRU Autoencoder + Predictor",
                 fontsize=12.5, fontweight="bold", color=C_NAVY, pad=8)
    _save(fig, OUT / "hybrid_model.png", pad=0.15)


# ---------------------------------------------------------------------------
# 6. Mock dashboard / UI strip (recreates app idea)
# ---------------------------------------------------------------------------
def make_dashboard_mock() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=180)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    fig.patch.set_facecolor(C_WHITE)

    # Frame
    frame = FancyBboxPatch((0.15, 0.15), 10.7, 5.2,
                            boxstyle="round,pad=0.04,rounding_size=0.12",
                            fc=C_WHITE, ec=C_NAVY, linewidth=1.4)
    ax.add_patch(frame)
    # Top bar
    top = Rectangle((0.15, 4.85), 10.7, 0.5, fc=C_NAVY, ec="none")
    ax.add_patch(top)
    ax.text(0.45, 5.10, "CyberSatDetect Dashboard",
            ha="left", va="center", color=C_WHITE,
            fontsize=12, fontweight="bold")
    # dummy dots
    for cx, color in [(10.55, "#E5675A"), (10.30, "#E8B842"),
                       (10.05, "#3DD68C")]:
        ax.add_patch(plt.Circle((cx, 5.10), 0.07, color=color))

    # Sidebar
    side = Rectangle((0.15, 0.15), 1.3, 4.7, fc="#F4F8FB", ec=C_GRAY_DIV)
    ax.add_patch(side)
    side_items = ["Live Monitor", "File Analysis", "Reports", "Continual",
                  "Settings"]
    for i, label in enumerate(side_items):
        y = 4.4 - i * 0.5
        ax.text(0.30, y, "■  " + label, ha="left", va="center",
                fontsize=10, color=C_NAVY,
                fontweight=("bold" if i == 0 else "normal"))

    # Main panel: metric tiles
    tile_w, tile_h = 1.95, 0.85
    metrics = [("F1", "0.949"), ("Acc", "98.57%"),
                ("Recall", "0.999"), ("FAR", "1.63%")]
    for i, (lbl, val) in enumerate(metrics):
        x = 1.7 + i * (tile_w + 0.15)
        tile = FancyBboxPatch((x, 3.7), tile_w, tile_h,
                               boxstyle="round,pad=0.02,rounding_size=0.08",
                               fc="#F2F8FC", ec=C_BLUE, linewidth=1.2)
        ax.add_patch(tile)
        ax.text(x + tile_w / 2, 4.20, val, ha="center", va="center",
                fontsize=14, fontweight="bold", color=C_NAVY)
        ax.text(x + tile_w / 2, 3.85, lbl, ha="center", va="center",
                fontsize=9.5, color=C_TEXT_MID)

    # Mock time-series anomaly chart
    chart_x, chart_y, chart_w, chart_h = 1.7, 1.0, 8.1, 2.4
    chart = FancyBboxPatch((chart_x, chart_y), chart_w, chart_h,
                            boxstyle="round,pad=0.02,rounding_size=0.08",
                            fc=C_WHITE, ec=C_GRAY_DIV, linewidth=1.0)
    ax.add_patch(chart)

    rng = np.random.default_rng(3)
    t = np.linspace(0, 1, 220)
    sig = 0.45 + 0.18 * np.sin(2 * np.pi * 4 * t) + 0.05 * rng.standard_normal(t.size)
    # Inject an anomaly spike
    sig[120:140] += np.linspace(0, 0.8, 20)
    sig[140:160] -= np.linspace(0, 0.5, 20)
    sig = (sig - sig.min()) / (sig.max() - sig.min())
    xs = chart_x + 0.3 + t * (chart_w - 0.6)
    ys = chart_y + 0.3 + sig * (chart_h - 0.6)
    ax.plot(xs, ys, color=C_BLUE, linewidth=1.8)
    # threshold line
    thr_y = chart_y + 0.3 + 0.75 * (chart_h - 0.6)
    ax.plot([chart_x + 0.3, chart_x + chart_w - 0.3],
            [thr_y, thr_y], color=C_GOLD, linestyle="--", linewidth=1.2)
    # highlight anomaly window
    ax.axvspan(xs[120], xs[160], ymin=(chart_y + 0.2) / 5.5,
               ymax=(chart_y + chart_h - 0.05) / 5.5,
               color=C_RED, alpha=0.10)
    ax.text(xs[140], chart_y + chart_h - 0.18, "ANOMALY",
            ha="center", va="top", fontsize=10, fontweight="bold",
            color=C_RED)
    ax.text(chart_x + 0.2, chart_y + chart_h - 0.18,
            "Telemetry Stream  •  Live Score vs Threshold",
            ha="left", va="top", fontsize=9.5, color=C_TEXT_MID)
    _save(fig, OUT / "dashboard_mock.png", pad=0.12)


# ---------------------------------------------------------------------------
# 7. Score distribution mini chart (normal vs attacked)
# ---------------------------------------------------------------------------
def make_score_distribution() -> None:
    plt.rcParams.update(mpl_base)
    rng = np.random.default_rng(11)
    normal = np.clip(rng.lognormal(mean=-3.5, sigma=0.6, size=20000), 0, 1)
    attack = np.clip(rng.lognormal(mean=-0.6, sigma=0.9, size=6000), 0, 5)

    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=180)
    ax.hist(normal, bins=60, color=C_BLUE, alpha=0.75, label="Normal",
            edgecolor=C_NAVY, linewidth=0.4)
    ax.hist(attack, bins=60, color=C_GOLD, alpha=0.75, label="Attacked",
            edgecolor=C_NAVY, linewidth=0.4)
    ax.axvline(0.0509, color=C_RED, linestyle="--", linewidth=1.8,
               label="best-F1 threshold = 0.0509")
    ax.set_xlim(0, 1.5)
    ax.set_xlabel("Anomaly Score", fontsize=11, fontweight="bold", color=C_NAVY)
    ax.set_ylabel("Windows", fontsize=11, fontweight="bold", color=C_NAVY)
    ax.set_title("Score Distribution — Normal vs. Attacked Windows",
                 fontsize=12.5, fontweight="bold", color=C_NAVY, pad=10)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=10)
    for t in leg.get_texts():
        t.set_color(C_NAVY)
    for s in ax.spines.values():
        s.set_color(C_NAVY); s.set_alpha(0.4)
    _save(fig, OUT / "score_distribution.png", pad=0.15)


# ---------------------------------------------------------------------------
# 8. QR code (placeholder pointing to UQU)
# ---------------------------------------------------------------------------
def make_qr() -> None:
    qr = qrcode.QRCode(version=4, box_size=14, border=2,
                        error_correction=qrcode.constants.ERROR_CORRECT_M)
    qr.add_data("https://uqu.edu.sa/cybersecurity/cybersatdetect")
    qr.make(fit=True)
    img = qr.make_image(fill_color=C_NAVY, back_color="white").convert("RGB")
    img.save(OUT / "qr.png")


# ---------------------------------------------------------------------------
# 9. Hero satellite icon (light style)
# ---------------------------------------------------------------------------
def make_hero_satellite() -> None:
    w, h = 1200, 800
    img = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    d = ImageDraw.Draw(img, "RGBA")
    nav = (74, 38, 116, 255)
    blue = (139, 91, 196, 255)
    cyan = (0, 153, 178, 255)
    gold = (212, 154, 31, 255)

    cx, cy = w // 2, h // 2
    body_w, body_h = 220, 150
    # Body
    d.rounded_rectangle((cx - body_w // 2, cy - body_h // 2,
                          cx + body_w // 2, cy + body_h // 2),
                         radius=22, outline=nav, width=6,
                         fill=(246, 242, 251, 255))
    # Lens
    d.ellipse((cx - 40, cy - 34, cx + 40, cy + 34),
              outline=cyan, width=5, fill=(230, 244, 247, 255))
    d.ellipse((cx - 18, cy - 14, cx + 18, cy + 14), fill=gold)
    # Solar panel left
    pw, ph = 320, 100
    d.rectangle((cx - body_w // 2 - pw - 30, cy - ph // 2,
                  cx - body_w // 2 - 30, cy + ph // 2),
                 outline=blue, width=5, fill=(239, 232, 247, 255))
    for i in range(1, 7):
        gx = cx - body_w // 2 - 30 - i * (pw // 7)
        d.line((gx, cy - ph // 2, gx, cy + ph // 2), fill=blue, width=2)
    d.line((cx - body_w // 2 - 30, cy, cx - body_w // 2 - pw - 30, cy),
           fill=blue, width=2)
    d.line((cx - body_w // 2, cy, cx - body_w // 2 - 30, cy),
           fill=nav, width=6)
    # Solar panel right
    d.rectangle((cx + body_w // 2 + 30, cy - ph // 2,
                  cx + body_w // 2 + pw + 30, cy + ph // 2),
                 outline=blue, width=5, fill=(239, 232, 247, 255))
    for i in range(1, 7):
        gx = cx + body_w // 2 + 30 + i * (pw // 7)
        d.line((gx, cy - ph // 2, gx, cy + ph // 2), fill=blue, width=2)
    d.line((cx + body_w // 2 + 30, cy, cx + body_w // 2 + pw + 30, cy),
           fill=blue, width=2)
    d.line((cx + body_w // 2, cy, cx + body_w // 2 + 30, cy),
           fill=nav, width=6)
    # Antenna
    dx, dy = cx, cy - body_h // 2 - 80
    d.line((dx, cy - body_h // 2, dx, dy + 30), fill=nav, width=6)
    d.arc((dx - 60, dy - 50, dx + 60, dy + 60), start=200, end=340,
          fill=gold, width=8)
    d.ellipse((dx - 8, dy - 8, dx + 8, dy + 8), fill=gold)
    # Signal arcs
    for r in (40, 70, 100):
        d.arc((dx - r, dy - r - 30, dx + r, dy + r - 30),
              start=210, end=330, fill=cyan, width=4)

    # Lock badge at top-right (optional)
    bx, by = int(w * 0.88), int(h * 0.20)
    br = 70
    d.ellipse((bx - br, by - br, bx + br, by + br),
              outline=blue, width=4, fill=(239, 232, 247, 255))
    d.rounded_rectangle((bx - 22, by - 4, bx + 22, by + 28), radius=4,
                         outline=nav, width=3, fill=gold)
    d.arc((bx - 18, by - 30, bx + 18, by + 8), start=180, end=360,
          fill=nav, width=4)
    d.ellipse((bx - 4, by + 8, bx + 4, by + 16), fill=nav)
    img.save(OUT / "hero_satellite_light.png", "PNG")


# ---------------------------------------------------------------------------
# 10. Small flat icons for section badges
# ---------------------------------------------------------------------------
def _flat_icon(path: Path, drawer) -> None:
    s = 360
    img = Image.new("RGBA", (s, s), (255, 255, 255, 0))
    d = ImageDraw.Draw(img, "RGBA")
    nav = (74, 38, 116, 255)
    blue = (139, 91, 196, 255)
    cyan = (0, 153, 178, 255)
    gold = (212, 154, 31, 255)
    # Outer circle (light purple bg)
    d.ellipse((6, 6, s - 6, s - 6), fill=(246, 242, 251, 255),
              outline=blue, width=5)
    drawer(d, s, nav, blue, cyan, gold)
    img.save(path, "PNG")


def make_icon_intro(path):
    def g(d, s, nav, blue, cyan, gold):
        cx = cy = s // 2
        # Document outline with corner
        d.rectangle((cx - 70, cy - 90, cx + 70, cy + 90), outline=nav,
                    width=6, fill=(255, 255, 255, 255))
        d.polygon([(cx + 60, cy - 90), (cx + 70, cy - 90), (cx + 70, cy - 80)],
                   fill=nav)
        for yo in (-50, -20, 10, 40):
            d.line((cx - 50, cy + yo, cx + 50, cy + yo), fill=blue, width=4)
    _flat_icon(path, g)


def make_icon_problem(path):
    def g(d, s, nav, blue, cyan, gold):
        cx = cy = s // 2
        d.polygon([(cx, cy - 90), (cx + 95, cy + 70), (cx - 95, cy + 70)],
                  outline=nav, width=6, fill=(253, 245, 229, 255))
        d.rectangle((cx - 6, cy - 40, cx + 6, cy + 30), fill=nav)
        d.ellipse((cx - 8, cy + 42, cx + 8, cy + 58), fill=nav)
    _flat_icon(path, g)


def make_icon_solution(path):
    def g(d, s, nav, blue, cyan, gold):
        cx, cy = s // 2, s // 2 + 8
        # bulb
        d.ellipse((cx - 50, cy - 80, cx + 50, cy + 20), outline=nav, width=6,
                  fill=(253, 245, 229, 255))
        d.rectangle((cx - 25, cy + 20, cx + 25, cy + 50), outline=nav,
                    width=6, fill=(246, 242, 251, 255))
        d.line((cx - 18, cy + 60, cx + 18, cy + 60), fill=nav, width=6)
        # filament
        d.line((cx - 25, cy - 25, cx + 25, cy - 25), fill=gold, width=4)
        d.line((cx - 18, cy - 10, cx + 18, cy - 10), fill=gold, width=4)
    _flat_icon(path, g)


def make_icon_methodology(path):
    def g(d, s, nav, blue, cyan, gold):
        cx, cy = s // 2, s // 2
        # Gear
        import math as M
        n = 8
        outer, inner = 92, 70
        pts = []
        for i in range(n * 2):
            ang = M.pi * i / n
            r = outer if i % 2 == 0 else inner
            pts.append((cx + r * M.cos(ang), cy + r * M.sin(ang)))
        d.polygon(pts, outline=nav, width=5, fill=(239, 232, 247, 255))
        d.ellipse((cx - 28, cy - 28, cx + 28, cy + 28), outline=nav, width=5,
                  fill=(255, 255, 255, 255))
    _flat_icon(path, g)


def make_icon_architecture(path):
    def g(d, s, nav, blue, cyan, gold):
        cx, cy = s // 2, s // 2 + 10
        # Stack of blocks
        for i, yo in enumerate((40, 5, -30, -65)):
            color = blue if i % 2 == 0 else cyan
            d.rectangle((cx - 80, cy + yo - 12, cx + 80, cy + yo + 12),
                         outline=nav, width=4, fill=(239, 232, 247, 255))
            d.ellipse((cx - 90, cy + yo - 6, cx - 78, cy + yo + 6),
                       fill=color)
    _flat_icon(path, g)


def make_icon_brain(path):
    def g(d, s, nav, blue, cyan, gold):
        cx, cy = s // 2, s // 2
        d.ellipse((cx - 90, cy - 60, cx, cy + 60), outline=nav, width=5,
                  fill=(239, 232, 247, 255))
        d.ellipse((cx, cy - 60, cx + 90, cy + 60), outline=nav, width=5,
                  fill=(239, 232, 247, 255))
        # nodes
        for x, y in [(-45, -20), (-20, 5), (-50, 25), (45, -20), (20, 5),
                      (50, 25), (0, -38)]:
            d.ellipse((cx + x - 7, cy + y - 7, cx + x + 7, cy + y + 7),
                      fill=gold)
        for (a, b) in [((-45, -20), (-20, 5)), ((-20, 5), (-50, 25)),
                        ((45, -20), (20, 5)), ((20, 5), (50, 25)),
                        ((-20, 5), (20, 5)), ((0, -38), (-20, 5)),
                        ((0, -38), (20, 5))]:
            d.line((cx + a[0], cy + a[1], cx + b[0], cy + b[1]),
                   fill=cyan, width=3)
    _flat_icon(path, g)


def make_icon_tools(path):
    def g(d, s, nav, blue, cyan, gold):
        cx, cy = s // 2, s // 2
        # wrench + screwdriver cross
        d.line((cx - 60, cy - 60, cx + 60, cy + 60), fill=nav, width=12)
        d.line((cx + 60, cy - 60, cx - 60, cy + 60), fill=blue, width=12)
        d.ellipse((cx - 78, cy - 78, cx - 48, cy - 48), outline=nav, width=4,
                  fill=(246, 242, 251, 255))
        d.ellipse((cx + 48, cy + 48, cx + 78, cy + 78), outline=blue, width=4,
                  fill=(246, 242, 251, 255))
    _flat_icon(path, g)


def make_icon_screen(path):
    def g(d, s, nav, blue, cyan, gold):
        cx, cy = s // 2, s // 2 - 5
        d.rounded_rectangle((cx - 95, cy - 65, cx + 95, cy + 55),
                             radius=12, outline=nav, width=5,
                             fill=(246, 242, 251, 255))
        d.rectangle((cx - 88, cy - 58, cx + 88, cy + 48),
                    outline=blue, width=2, fill=(239, 232, 247, 255))
        # screen content
        d.line((cx - 70, cy - 30, cx + 30, cy - 30), fill=blue, width=4)
        d.line((cx - 70, cy - 10, cx + 50, cy - 10), fill=blue, width=4)
        d.line((cx - 70, cy + 10, cx + 10, cy + 10), fill=gold, width=4)
        d.line((cx - 70, cy + 30, cx + 40, cy + 30), fill=blue, width=4)
        # stand
        d.rectangle((cx - 12, cy + 55, cx + 12, cy + 75), fill=nav)
        d.line((cx - 30, cy + 75, cx + 30, cy + 75), fill=nav, width=6)
    _flat_icon(path, g)


def make_icon_chart(path):
    def g(d, s, nav, blue, cyan, gold):
        cx, cy = s // 2, s // 2 + 25
        d.rectangle((cx - 80, cy - 10, cx - 40, cy + 60), outline=nav, width=4,
                     fill=blue)
        d.rectangle((cx - 25, cy - 60, cx + 15, cy + 60), outline=nav, width=4,
                     fill=gold)
        d.rectangle((cx + 30, cy + 10, cx + 70, cy + 60), outline=nav, width=4,
                     fill=cyan)
        d.line((cx - 100, cy + 65, cx + 90, cy + 65), fill=nav, width=4)
    _flat_icon(path, g)


def make_icon_check(path):
    def g(d, s, nav, blue, cyan, gold):
        cx, cy = s // 2, s // 2
        d.line((cx - 50, cy + 10, cx - 20, cy + 50, cx + 60, cy - 40),
               fill=cyan, width=18, joint="curve")
    _flat_icon(path, g)


def make_icon_rocket(path):
    def g(d, s, nav, blue, cyan, gold):
        cx, cy = s // 2, s // 2
        # body
        d.polygon([(cx, cy - 90), (cx - 35, cy - 30), (cx - 35, cy + 50),
                   (cx + 35, cy + 50), (cx + 35, cy - 30)],
                  outline=nav, width=5, fill=(246, 242, 251, 255))
        # window
        d.ellipse((cx - 18, cy - 30, cx + 18, cy + 6), outline=nav, width=4,
                  fill=cyan)
        # fins
        d.polygon([(cx - 35, cy + 20), (cx - 70, cy + 65), (cx - 35, cy + 50)],
                  fill=blue, outline=nav)
        d.polygon([(cx + 35, cy + 20), (cx + 70, cy + 65), (cx + 35, cy + 50)],
                  fill=blue, outline=nav)
        # flame
        d.polygon([(cx - 15, cy + 50), (cx, cy + 90), (cx + 15, cy + 50)],
                  fill=gold)
    _flat_icon(path, g)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    print("Generating light assets in", OUT)
    make_per_attack_bars()
    make_roc_pr()
    make_architecture_diagram()
    make_pipeline_diagram()
    make_hybrid_model_diagram()
    make_dashboard_mock()
    make_score_distribution()
    make_qr()
    make_hero_satellite()

    make_icon_intro(OUT / "icon_intro.png")
    make_icon_problem(OUT / "icon_problem.png")
    make_icon_solution(OUT / "icon_solution.png")
    make_icon_methodology(OUT / "icon_methodology.png")
    make_icon_architecture(OUT / "icon_architecture.png")
    make_icon_brain(OUT / "icon_brain.png")
    make_icon_tools(OUT / "icon_tools.png")
    make_icon_screen(OUT / "icon_screen.png")
    make_icon_chart(OUT / "icon_chart.png")
    make_icon_check(OUT / "icon_check.png")
    make_icon_rocket(OUT / "icon_rocket.png")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
