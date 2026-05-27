# -*- coding: utf-8 -*-
"""
Build a poster-ready system architecture diagram for CyberSatDetect.

Output:
    docs/images/system_architecture_poster.png  (horizontal, 3600x2400 @ 300 DPI)
    docs/images/system_architecture_poster_ar.png  (Arabic labels variant)

The diagram shows the end-to-end pipeline:
    Satellite telemetry  ->  Frontend  ->  API/Security  ->  Processing
    ->  Hybrid LSTM-GRU AE  ->  Anomaly Scoring  ->  Storage  ->  Outputs
    ->  Continual Learning loop back to model.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D
import numpy as np

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_AR = True
except Exception:
    HAS_AR = False


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Palette (matches build_light_poster.py academic purple palette)
# ---------------------------------------------------------------------------
C_BG          = "#FFFFFF"
C_BG_TINT     = "#F6F2FB"
C_PRIMARY     = "#4A2674"   # deep purple
C_PRIMARY_DK  = "#351851"
C_PURPLE_MID  = "#6B3A9C"
C_PURPLE_LT   = "#8B5BC4"
C_CYAN        = "#0099B2"
C_GOLD        = "#D49A1F"
C_GREEN       = "#1F9D55"
C_RED         = "#C0392B"
C_GRAY        = "#9E91B0"
C_GRAY_LT     = "#EAE2F1"
C_TEXT        = "#1F1830"
C_TEXT_MID    = "#574B66"


def ar(text: str) -> str:
    """Shape Arabic for matplotlib (returns the same text if no AR libs)."""
    if not HAS_AR:
        return text
    return get_display(arabic_reshaper.reshape(text))


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def box(ax, x, y, w, h, *, fc=C_BG, ec=C_PRIMARY, lw=2.0, radius=0.02, zorder=2):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=zorder,
    )
    ax.add_patch(p)
    return p


def text(ax, x, y, s, *, color=C_TEXT, size=13, weight="normal",
         ha="center", va="center", zorder=5, font="DejaVu Sans"):
    ax.text(x, y, s, color=color, fontsize=size, fontweight=weight,
            ha=ha, va=va, zorder=zorder, family=font)


def arrow(ax, x1, y1, x2, y2, *, color=C_PURPLE_MID, lw=2.5, style="-|>",
          curved=False, rad=0.18, zorder=3):
    cs = f"arc3,rad={rad}" if curved else "arc3,rad=0.0"
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=22,
        linewidth=lw, color=color, zorder=zorder,
        connectionstyle=cs,
    )
    ax.add_patch(a)
    return a


def layer_band(ax, x, y, w, h, title, subtitle, color):
    """Translucent band; label sits as a tab OUTSIDE the band on the right edge."""
    r = Rectangle((x, y), w, h, facecolor=color, alpha=0.07,
                  edgecolor="none", zorder=1)
    ax.add_patch(r)
    # tab/badge on the far-right margin
    tab_w, tab_h = 1.05, 0.42
    tab = FancyBboxPatch(
        (x + w - tab_w - 0.10, y + h - tab_h - 0.08), tab_w, tab_h,
        boxstyle="round,pad=0.0,rounding_size=0.06",
        facecolor=color, edgecolor="none", zorder=4,
    )
    ax.add_patch(tab)
    text(ax, x + w - tab_w / 2 - 0.10, y + h - tab_h / 2 - 0.08,
         title, color="white", size=10, weight="bold")
    # subtitle: small line just below the tab, right-aligned inside band
    text(ax, x + w - 0.15, y + h - tab_h - 0.30,
         subtitle, color=C_TEXT_MID, size=9, ha="right", va="center", weight="bold")


def stage_box(ax, x, y, w, h, title, sub, *, icon=None, accent=C_PRIMARY,
              fc=C_BG, title_size=13, sub_size=9):
    box(ax, x, y, w, h, fc=fc, ec=accent, lw=2.0, radius=0.10)
    # thin accent stripe at top (decorative, no text)
    stripe = Rectangle((x + 0.02, y + h - 0.10), w - 0.04, 0.08,
                       facecolor=accent, edgecolor="none", zorder=3)
    ax.add_patch(stripe)
    # title at the top inside the card
    text(ax, x + w / 2, y + h - 0.32, title, color=accent,
         size=title_size, weight="bold")
    if sub:
        for i, line in enumerate(sub):
            text(ax, x + w / 2, y + h - 0.62 - i * 0.30, line,
                 color=C_TEXT_MID, size=sub_size)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------
def build(lang: str = "en") -> Path:
    """lang in {'en','ar'}.  In AR mode descriptive text is Arabic but
    technology/brand names (FastAPI, JWT, …) stay in Latin script."""
    is_ar = lang == "ar"

    def L(en: str, ar_text: str) -> str:
        """Pick the right label."""
        return ar(ar_text) if is_ar else en

    # Page titles for the title strip
    title    = L("CyberSatDetect — System Architecture",
                 "معمارية نظام CyberSatDetect")
    subtitle = L("Hybrid LSTM-GRU Autoencoder for Cyber-Attack Detection on Satellite Telemetry",
                 "كشف الهجمات السيبرانية على بيانات الأقمار الاصطناعية باستخدام مُشفِّر هجين LSTM-GRU")

    fig = plt.figure(figsize=(20, 13), dpi=180, facecolor=C_BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 13)
    ax.set_axis_off()

    # ------------------ Background flourish ------------------
    # soft top band
    ax.add_patch(Rectangle((0, 12.0), 20, 1.0, facecolor=C_BG_TINT,
                           edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0, 0.0), 20, 0.55, facecolor=C_BG_TINT,
                           edgecolor="none", zorder=0))

    # ------------------ Title ------------------
    text(ax, 10, 12.55, title, color=C_PRIMARY, size=26, weight="bold")
    text(ax, 10, 12.18, subtitle, color=C_TEXT_MID, size=12)

    # ------------------ Layer tinted bands (subtle, no inline label) ------------------
    def tinted_band(ax, y, h, color):
        ax.add_patch(Rectangle((0.20, y), 19.6, h, facecolor=color,
                               alpha=0.07, edgecolor="none", zorder=1))

    layer_specs = [
        (9.85, 1.45, "01", "FRONTEND",        C_PURPLE_LT),
        (8.10, 1.45, "02", "API / SECURITY",  C_CYAN),
        (5.55, 2.25, "03", "AI CORE",         C_PRIMARY),
        (3.45, 1.80, "04", "STORAGE",         C_GOLD),
        (1.30, 1.85, "05", "OUTPUTS  + CL",   C_GREEN),
    ]
    for y, h, idx, name, color in layer_specs:
        tinted_band(ax, y, h, color)

    # Numbered badges in a thin column on the left margin (outside content area)
    badge_x = 0.10
    for y, h, idx, name, color in layer_specs:
        # vertical pill / stripe
        stripe = FancyBboxPatch(
            (badge_x, y + 0.10), 0.32, h - 0.20,
            boxstyle="round,pad=0.0,rounding_size=0.10",
            facecolor=color, edgecolor="none", zorder=4,
        )
        ax.add_patch(stripe)
        text(ax, badge_x + 0.16, y + h / 2, idx,
             color="white", size=12, weight="bold")

    # tiny layer-name labels above the figure title strip is overkill;
    # we omit them — the colored stripe + boxed group names are self-explanatory.

    # ===============================================================
    # LAYER 0 — Data Sources (top-left external)
    # ===============================================================
    # Satellite icon (stylised)
    sat_x, sat_y = 1.2, 10.55
    ax.add_patch(FancyBboxPatch((sat_x - 0.20, sat_y - 0.18), 0.40, 0.32,
                                boxstyle="round,pad=0.0,rounding_size=0.04",
                                facecolor=C_PRIMARY, edgecolor=C_PRIMARY_DK, lw=1.2, zorder=4))
    # panels
    for dx in (-0.55, 0.55):
        ax.add_patch(Rectangle((sat_x + dx - 0.20, sat_y - 0.10), 0.40, 0.16,
                               facecolor=C_PRIMARY_DK, edgecolor=C_PURPLE_LT, lw=0.8, zorder=4))
    ax.add_patch(Circle((sat_x, sat_y), 0.05, facecolor=C_GOLD,
                        edgecolor=C_PRIMARY_DK, lw=1, zorder=5))
    # signal arcs
    for r, a in [(0.35, 0.40), (0.55, 0.30), (0.75, 0.20)]:
        ax.add_patch(mpatches.Arc((sat_x, sat_y + 0.22), r * 2, r * 1.3,
                                   theta1=20, theta2=160, color=C_CYAN,
                                   linewidth=2.0, alpha=a, zorder=4))
    text(ax, sat_x, sat_y + 0.95, L("Satellite Telemetry", "بيانات القمر"),
         color=C_PRIMARY, size=10, weight="bold")
    text(ax, sat_x, sat_y - 0.65, "CSV / NPY", color=C_TEXT_MID, size=9)

    # ===============================================================
    # LAYER 1 — Presentation (frontend pages)
    # ===============================================================
    pages = [
        (L("Upload",     "الرفع"),       L("CSV · NPY", "CSV · NPY"),                 C_PURPLE_LT, ""),
        (L("Detection",  "الكشف"),       L("Run inference", "تشغيل النموذج"),         C_PURPLE_LT, ""),
        (L("Reports",    "التقارير"),    L("PDF · Excel", "PDF · Excel"),             C_PURPLE_LT, ""),
        (L("Dashboard",  "لوحة المتابعة"), L("Charts · Stats", "رسوم · إحصائيات"),   C_PURPLE_LT, ""),
    ]
    page_w, page_h = 2.20, 1.05
    page_gap = 0.30
    page_total = len(pages) * page_w + (len(pages) - 1) * page_gap
    px0 = 10 - page_total / 2
    for i, (t1, t2, c, ico) in enumerate(pages):
        x = px0 + i * (page_w + page_gap)
        stage_box(ax, x, 10.0, page_w, page_h, t1, [t2],
                  icon=ico, accent=c)
    # tech chip in the GAP between band 1 and band 2 (band1 ends at 9.85, band2 starts at 9.55)
    chips = "HTML5  ·  CSS3  ·  Vanilla JS  ·  Chart.js  ·  Three.js  ·  Node / Express"
    text(ax, 10, 9.70, chips, color=C_PURPLE_MID, size=10, weight="bold")

    # ===============================================================
    # LAYER 2 — Edge / Security
    # ===============================================================
    # central FastAPI box
    api_w, api_h = 4.6, 1.05
    api_x, api_y = 10 - api_w / 2, 8.20
    stage_box(ax, api_x, api_y, api_w, api_h,
              "FastAPI Gateway",
              [L("REST API · Pydantic · Uvicorn (ASGI)",
                 "REST API · Pydantic · خادم Uvicorn")],
              icon="API", accent=C_CYAN, title_size=15)

    # Security pills (left & right)
    sec_left = [
        ("JWT",    L("Auth tokens",  "رموز المصادقة")),
        ("bcrypt", L("Password hash", "تجزئة كلمات المرور")),
    ]
    sec_right = [
        ("CORS / Helmet", L("Headers",   "رؤوس HTTP آمنة")),
        ("SlowAPI",       L("Rate-limit", "تقييد الطلبات")),
    ]
    for i, (n, d) in enumerate(sec_left):
        x = 1.0 + i * 2.10
        stage_box(ax, x, 8.20, 1.90, 1.05, n, [d], icon="LOCK", accent=C_CYAN)
    for i, (n, d) in enumerate(sec_right):
        x = 14.80 + i * 2.10
        stage_box(ax, x, 8.20, 1.90, 1.05, n, [d], icon="SHIELD", accent=C_CYAN)

    # ===============================================================
    # LAYER 3 — Processing & AI (the heart)
    # ===============================================================
    proc_y = 5.95
    proc_h = 1.55

    # 1) Cleaning
    stage_box(ax, 0.8, proc_y, 2.6, proc_h,
              L("Data Cleaning", "تنظيف البيانات"),
              [L("float32 · NaN→interp", "float32 · معالجة NaN"),
               L("drop non-finite",       "إزالة القيم اللانهائية")],
              icon="CLEAN", accent=C_PRIMARY)
    # 2) Windowing
    stage_box(ax, 3.7, proc_y, 2.6, proc_h,
              L("Sliding Windows", "نوافذ منزلقة"),
              ["W = 100  ·  S = 50",
               "X ∈ ℝᴮˣ¹⁰⁰ˣ¹"],
              icon="WIN", accent=C_PRIMARY)

    # 3) Hybrid LSTM-GRU AE (large central card)
    ae_x, ae_w = 6.6, 6.8
    box(ax, ae_x, proc_y, ae_w, proc_h, fc=C_BG, ec=C_PRIMARY, lw=2.4, radius=0.12)
    stripe = Rectangle((ae_x + 0.02, proc_y + proc_h - 0.20), ae_w - 0.04, 0.16,
                       facecolor=C_PRIMARY, edgecolor="none", zorder=3)
    ax.add_patch(stripe)
    text(ax, ae_x + ae_w / 2, proc_y + proc_h - 0.12,
         L("Hybrid LSTM-GRU Autoencoder", "النموذج الهجين Hybrid LSTM-GRU Autoencoder"),
         color="white", size=13, weight="bold")

    # mini network visual
    enc_x = ae_x + 0.55
    lat_x = ae_x + ae_w / 2
    dec_x = ae_x + ae_w - 0.55
    layer_y = proc_y + 0.62

    def neuron(cx, cy, r=0.10, fc=C_PURPLE_MID):
        ax.add_patch(Circle((cx, cy), r, facecolor=fc, edgecolor=C_PRIMARY_DK,
                            lw=0.8, zorder=4))

    # encoder column
    for i, dy in enumerate([0.55, 0.20, -0.15, -0.50]):
        neuron(enc_x, layer_y + dy)
    # bottleneck
    for dy in [0.20, -0.15]:
        neuron(lat_x, layer_y + dy, r=0.12, fc=C_GOLD)
    # decoder column
    for dy in [0.55, 0.20, -0.15, -0.50]:
        neuron(dec_x, layer_y + dy)
    # connection lines
    for y1 in [0.55, 0.20, -0.15, -0.50]:
        for y2 in [0.20, -0.15]:
            ax.plot([enc_x, lat_x], [layer_y + y1, layer_y + y2],
                    color=C_PURPLE_LT, lw=0.7, alpha=0.55, zorder=3)
            ax.plot([lat_x, dec_x], [layer_y + y2, layer_y + y1],
                    color=C_PURPLE_LT, lw=0.7, alpha=0.55, zorder=3)
    # labels for encoder/latent/decoder
    text(ax, enc_x, layer_y - 0.95, L("Encoder\n(LSTM)", "المُشفِّر\nLSTM"),
         color=C_PRIMARY, size=9, weight="bold")
    text(ax, lat_x, layer_y - 0.95, L("Latent\n(GRU)", "الفضاء الكامن\nGRU"),
         color=C_GOLD, size=9, weight="bold")
    text(ax, dec_x, layer_y - 0.95, L("Decoder\n+ Predictor", "فاكّ التشفير\n+ التنبؤ"),
         color=C_PRIMARY, size=9, weight="bold")

    # 4) Anomaly Scoring
    stage_box(ax, 13.7, proc_y, 2.55, proc_h,
              L("Anomaly Score", "حساب درجة الشذوذ"),
              ["e_recon + e_pred + e_grad",
               L("thresholds (p99 · 3σ)", "العتبات (p99 · 3σ)")],
              icon="SCORE", accent=C_PRIMARY)

    # 5) Decision (alert vs normal)
    stage_box(ax, 16.55, proc_y, 2.65, proc_h,
              L("Decision", "اتخاذ القرار"),
              [L("score > τ  →  Attack", "score > τ  ←  هجوم"),
               L("score ≤ τ  →  Normal", "score ≤ τ  ←  طبيعي")],
              icon="DECIDE", accent=C_PRIMARY)

    # arrows between processing blocks
    arrow_y = proc_y + proc_h / 2
    for x1, x2 in [(3.4, 3.7), (6.3, 6.6), (13.4, 13.7), (16.25, 16.55)]:
        arrow(ax, x1, arrow_y, x2, arrow_y, color=C_PRIMARY_DK, lw=2.2)

    # ===============================================================
    # LAYER 4 — Storage
    # ===============================================================
    db_specs = [
        ("telemetry",      L("uploaded windows",   "النوافذ المرفوعة"),     C_GOLD),
        ("anomaly_store",  L("scores + flags",     "الدرجات والإشارات"),    C_GOLD),
        ("model_registry", L("PENDING→APPROVED",   "PENDING→APPROVED"),     C_GOLD),
        ("users / logs",   L("auth · audit",       "المستخدمون · السجلات"), C_GOLD),
    ]
    db_w, db_h = 3.30, 1.30
    db_gap = 0.45
    db_total = len(db_specs) * db_w + (len(db_specs) - 1) * db_gap
    dx0 = 10 - db_total / 2
    for i, (n, d, c) in enumerate(db_specs):
        x = dx0 + i * (db_w + db_gap)
        # cylinder-ish style: rounded box with little top ellipse
        stage_box(ax, x, 3.65, db_w, db_h, n, [d], icon="DB", accent=c)
    text(ax, 10, 3.50,
         L("PostgreSQL 16  ·  SQLite (dev)  ·  SQLAlchemy ORM",
           "PostgreSQL 16  ·  SQLite للتطوير  ·  طبقة SQLAlchemy ORM"),
         color=C_TEXT_MID, size=10, weight="bold")

    # ===============================================================
    # LAYER 5 — Outputs & Continual Learning
    # ===============================================================
    out_y, out_h = 1.45, 1.40

    # Outputs (left side)
    outs = [
        (L("Alerts",    "تنبيهات"),       L("real-time",   "زمن حقيقي"),  C_RED),
        (L("Reports",   "تقارير"),        L("PDF · Excel", "PDF · Excel"),C_GREEN),
        (L("Dashboard", "لوحة المتابعة"), L("interactive", "تفاعلية"),    C_GREEN),
    ]
    ow = 2.45
    ox0 = 0.8
    for i, (n, d, c) in enumerate(outs):
        x = ox0 + i * (ow + 0.30)
        stage_box(ax, x, out_y, ow, out_h, n, [d], icon="OUT", accent=c)

    # Continual learning loop (right side)
    cl_x = 9.50
    cl_w = 9.70
    box(ax, cl_x, out_y, cl_w, out_h, fc="#F1FBF4", ec=C_GREEN, lw=2.0, radius=0.12)
    stripe = Rectangle((cl_x + 0.02, out_y + out_h - 0.18), cl_w - 0.04, 0.14,
                       facecolor=C_GREEN, edgecolor="none", zorder=3)
    ax.add_patch(stripe)
    text(ax, cl_x + cl_w / 2, out_y + out_h - 0.11,
         L("Continual Learning Loop", "حلقة التعلّم المستمر"),
         color="white", size=12, weight="bold")

    cl_steps = [
        ("buffer",         L("safe windows",        "نوافذ آمنة")),
        ("normal_pool",    L("low-score",           "درجات منخفضة")),
        ("admin",          L("approval",            "موافقة المشرف")),
        ("build_dataset",  L("merged.npz",          "merged.npz")),
        ("fine-tune",      L("Hybrid AE",           "ضبط النموذج")),
        ("registry",       L("PENDING→APPROVED",    "PENDING→APPROVED")),
    ]
    cw = (cl_w - 0.6) / len(cl_steps)
    for i, (n, d) in enumerate(cl_steps):
        x = cl_x + 0.30 + i * cw
        # mini chips
        box(ax, x, out_y + 0.25, cw - 0.12, 0.78, fc="white", ec=C_GREEN,
            lw=1.4, radius=0.06)
        text(ax, x + (cw - 0.12) / 2, out_y + 0.78, n,
             color=C_PRIMARY, size=10, weight="bold")
        text(ax, x + (cw - 0.12) / 2, out_y + 0.48, d,
             color=C_TEXT_MID, size=8)
        if i < len(cl_steps) - 1:
            arrow(ax, x + cw - 0.12, out_y + 0.64,
                  x + cw, out_y + 0.64, color=C_GREEN, lw=1.6)

    # ===============================================================
    # CROSS-LAYER FLOW ARROWS
    # ===============================================================
    # Satellite -> Upload page  (first page is at x = px0 = 5.15)
    arrow(ax, 1.95, 10.55, px0 - 0.05, 10.55, color=C_PRIMARY_DK, lw=2.2)
    # Pages -> FastAPI (down)
    arrow(ax, 10, 9.95, 10, 9.30, color=C_CYAN, lw=2.6)
    # FastAPI -> Processing
    arrow(ax, 10, 8.15, 10, 7.55, color=C_PRIMARY, lw=2.6)
    # Decision -> Storage  (down then left into row)
    arrow(ax, 17.85, 5.92, 17.85, 5.05, color=C_PRIMARY, lw=2.4)
    arrow(ax, 17.85, 5.05, 16.20, 5.05, color=C_PRIMARY, lw=2.4)
    # Storage -> Outputs/CL
    arrow(ax, 10, 3.62, 10, 2.92, color=C_GOLD, lw=2.4)
    # Continual loop -> AE  (curved back upward)
    cl_back = FancyArrowPatch(
        (cl_x + cl_w - 0.30, out_y + out_h - 0.05),
        (ae_x + ae_w / 2 + 0.20, proc_y + 0.05),
        arrowstyle="-|>", mutation_scale=22,
        linewidth=2.2, color=C_GREEN, zorder=3,
        connectionstyle="arc3,rad=-0.30",
        linestyle="--",
    )
    ax.add_patch(cl_back)
    # label sits at the upper right where the curved arrow exits storage row
    text(ax, 18.40, 5.05,
         L("Replay → re-train", "إعادة تدريب  ←"),
         color=C_GREEN, size=9.5, weight="bold", ha="right",
         font="DejaVu Sans")

    # ===============================================================
    # Footer: Legend (LEFT)  +  Key metrics (RIGHT) on a single bottom strip
    # ===============================================================
    # Legend on the LEFT half
    legend_items = [
        (C_PURPLE_LT, L("Frontend",       "الواجهة")),
        (C_CYAN,      L("API / Security", "الخادم / الأمن")),
        (C_PRIMARY,   L("AI Processing",  "المعالجة الذكية")),
        (C_GOLD,      L("Storage",        "التخزين")),
        (C_GREEN,     L("Outputs / CL",   "المخرجات والتعلّم")),
    ]
    legend_y = 0.27
    for i, (c, label) in enumerate(legend_items):
        lx = 0.55 + i * 1.95
        ax.add_patch(Circle((lx, legend_y), 0.09, facecolor=c, edgecolor="none", zorder=4))
        text(ax, lx + 0.18, legend_y, label, color=C_TEXT_MID, size=9.5,
             ha="left", weight="bold")

    # Key metrics on the RIGHT half — short compact tokens
    metrics_text = "ROC-AUC 0.889   ·   F1 0.78   ·   FAR 3.3 %   ·   W=100 / S=50"
    text(ax, 11.20, 0.27, metrics_text,
         color=C_TEXT_MID, size=9.5, ha="left", weight="bold")

    text(ax, 19.85, 0.27, "©  CyberSatDetect",
         color=C_PRIMARY, size=9.5, ha="right", weight="bold")

    # Output
    suffix = "_ar" if is_ar else ""
    out_path = OUT_DIR / f"system_architecture_poster{suffix}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor(), pad_inches=0.10)
    plt.close(fig)
    return out_path


def main():
    paths = []
    for lang in ("en", "ar"):
        p = build(lang)
        paths.append(p)
        print(f"[OK] wrote {p.relative_to(ROOT)}  ({p.stat().st_size/1024:.1f} KB)")
    return paths


if __name__ == "__main__":
    main()
