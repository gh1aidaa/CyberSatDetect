# -*- coding: utf-8 -*-
"""
Build a COMPACT system architecture figure for CyberSatDetect — in the style of
the GP2_Final.pdf "Figure 3.3" reference (light academic theme, dashed navy
border, white cards, subtle blue iconography).

This is the poster-ready compact replacement (~5 main blocks + CL loop).

Outputs:
    docs/images/system_diagram_compact.png      (English)
    docs/images/system_diagram_compact_ar.png   (Arabic, technical names kept Latin)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (
    FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon, Wedge,
)

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
# Academic light palette (matches Figure 3.3 reference)
# ---------------------------------------------------------------------------
C_BG          = "#FFFFFF"
C_FRAME       = "#1F3A5F"   # dashed border
C_CARD_BG     = "#FFFFFF"
C_CARD_EDGE   = "#C5D2E2"
C_HEADER_BAR  = "#E8EEF7"
C_ICON_BG     = "#E8EEF7"
C_ICON_FG     = "#1F3A5F"
C_TITLE       = "#1F3A5F"
C_TEXT        = "#33384A"
C_TEXT_DIM    = "#6C7691"
C_ARROW       = "#7A8AA8"
C_ARROW_HI    = "#1F3A5F"
C_ACCENT      = "#2E7DC3"
C_GREEN       = "#1F9D55"
C_RED         = "#C0392B"
C_GOLD        = "#D49A1F"


def ar(s: str) -> str:
    if not HAS_AR:
        return s
    return get_display(arabic_reshaper.reshape(s))


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def text(ax, x, y, s, *, color=C_TEXT, size=10, weight="normal",
         ha="center", va="center", zorder=6, rotation=0):
    return ax.text(x, y, s, color=color, fontsize=size, fontweight=weight,
                   ha=ha, va=va, zorder=zorder, rotation=rotation,
                   family="DejaVu Sans")


def card(ax, x, y, w, h, title, bullets=None, *,
         icon_fn=None, accent=C_TITLE):
    """White card with a header strip, optional icon and bullets."""
    # main card
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.10",
        linewidth=1.4, edgecolor=C_CARD_EDGE, facecolor=C_CARD_BG, zorder=3,
    )
    ax.add_patch(box)
    # header tint
    hdr = FancyBboxPatch(
        (x + 0.04, y + h - 0.46), w - 0.08, 0.40,
        boxstyle="round,pad=0.0,rounding_size=0.06",
        linewidth=0, facecolor=C_HEADER_BAR, zorder=4,
    )
    ax.add_patch(hdr)
    # icon circle (left in header)
    icon_r = 0.16
    icon_cx = x + 0.30
    icon_cy = y + h - 0.26
    ax.add_patch(Circle((icon_cx, icon_cy), icon_r,
                        facecolor="white", edgecolor=accent, lw=1.2, zorder=5))
    if icon_fn is not None:
        icon_fn(ax, icon_cx, icon_cy, icon_r, accent)
    # title (right of icon)
    text(ax, icon_cx + icon_r + 0.10, icon_cy, title,
         color=accent, size=10, weight="bold", ha="left", va="center")
    # bullets
    if bullets:
        for i, b in enumerate(bullets):
            by = y + h - 0.62 - i * 0.25
            # tiny dot
            ax.add_patch(Circle((x + 0.20, by), 0.025,
                                facecolor=C_ACCENT, edgecolor="none", zorder=5))
            text(ax, x + 0.30, by, b, color=C_TEXT_DIM, size=8.5,
                 ha="left", va="center")


def arrow_between(ax, x1, y1, x2, y2, *, label=None, color=C_ARROW,
                  lw=1.6, curved=False, rad=0.18, label_dy=0.18,
                  style="-|>"):
    cs = f"arc3,rad={rad}" if curved else "arc3,rad=0.0"
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=14,
        linewidth=lw, color=color, zorder=2,
        connectionstyle=cs,
    )
    ax.add_patch(a)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + label_dy
        text(ax, mx, my, label, color=C_TEXT_DIM, size=8.5,
             weight="bold", ha="center", va="center")


# Simple inline icon renderers (drawn on the card header circle)
def ic_user(ax, cx, cy, r, color):
    ax.add_patch(Circle((cx, cy + r * 0.25), r * 0.30, facecolor=color,
                        edgecolor="none", zorder=6))
    ax.add_patch(Wedge((cx, cy - r * 0.15), r * 0.55, 0, 180,
                       facecolor=color, edgecolor="none", zorder=6))

def ic_screen(ax, cx, cy, r, color):
    w = r * 1.05; h = r * 0.75
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2 + r * 0.05), w, h,
                           facecolor="none", edgecolor=color, lw=1.4, zorder=6))
    ax.plot([cx - r * 0.30, cx + r * 0.30],
            [cy - h / 2 - r * 0.10, cy - h / 2 - r * 0.10],
            color=color, lw=1.4, zorder=6)

def ic_brain(ax, cx, cy, r, color):
    ax.add_patch(Circle((cx - r * 0.20, cy), r * 0.45,
                        facecolor="none", edgecolor=color, lw=1.4, zorder=6))
    ax.add_patch(Circle((cx + r * 0.20, cy), r * 0.45,
                        facecolor="none", edgecolor=color, lw=1.4, zorder=6))
    ax.plot([cx - r * 0.20, cx + r * 0.20], [cy, cy], color=color, lw=1.2,
            zorder=6)

def ic_gauge(ax, cx, cy, r, color):
    ax.add_patch(Wedge((cx, cy - r * 0.10), r * 0.60, 20, 160,
                       facecolor="none", edgecolor=color, lw=1.6, zorder=6))
    ax.plot([cx, cx + r * 0.40], [cy - r * 0.10, cy + r * 0.30],
            color=color, lw=1.4, zorder=6)
    ax.add_patch(Circle((cx, cy - r * 0.10), r * 0.08,
                        facecolor=color, edgecolor="none", zorder=7))

def ic_doc(ax, cx, cy, r, color):
    w = r * 0.85; h = r * 1.15
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h,
                           facecolor="none", edgecolor=color, lw=1.4, zorder=6))
    for i, dy in enumerate([0.30, 0.05, -0.20]):
        ax.plot([cx - w / 2 + 0.04, cx + w / 2 - 0.04],
                [cy + dy * r, cy + dy * r], color=color, lw=1.0, zorder=6)

def ic_loop(ax, cx, cy, r, color):
    ax.add_patch(Wedge((cx, cy), r * 0.60, 30, 330,
                       width=r * 0.14, facecolor=color, edgecolor="none",
                       zorder=6))
    tri = [(cx + r * 0.50, cy + r * 0.30),
           (cx + r * 0.78, cy + r * 0.10),
           (cx + r * 0.55, cy - r * 0.05)]
    ax.add_patch(Polygon(tri, facecolor=color, edgecolor="none", zorder=7))

def ic_admin(ax, cx, cy, r, color):
    ic_user(ax, cx, cy + r * 0.10, r * 0.85, color)
    ax.add_patch(Rectangle((cx - r * 0.30, cy - r * 0.55), r * 0.60, r * 0.18,
                           facecolor=color, edgecolor="none", zorder=7))


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------
def build(lang: str = "en") -> Path:
    is_ar = lang == "ar"
    def L(en, ar_): return ar(ar_) if is_ar else en

    fig = plt.figure(figsize=(18, 10), dpi=180, facecolor=C_BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.set_axis_off()

    # title
    text(ax, 9, 9.55,
         L("CyberSatDetect — System Architecture",
           "معمارية نظام CyberSatDetect"),
         color=C_TITLE, size=18, weight="bold")
    text(ax, 9, 9.15,
         L("Hybrid LSTM-GRU Autoencoder with Continual Learning",
           "مُشفِّر هجين LSTM-GRU مع تعلّم مستمر"),
         color=C_TEXT_DIM, size=11)

    # dashed border around the whole system
    frame = FancyBboxPatch(
        (0.50, 0.80), 17.0, 7.85,
        boxstyle="round,pad=0.0,rounding_size=0.20",
        linewidth=1.8, edgecolor=C_FRAME, facecolor="none",
        linestyle=(0, (5, 4)), zorder=1,
    )
    ax.add_patch(frame)

    # ------------------------------------------------------------------
    # Top row: User --> Web App --> Hybrid AE --> Score --> Analysis --> Report
    # 6 stages in a clean horizontal flow.
    # ------------------------------------------------------------------
    card_w, card_h = 2.45, 1.85
    row_y = 5.40
    # 6 cards, equally spaced
    n = 6
    margin = 0.90
    avail = 18 - 2 * margin
    pitch = (avail - card_w) / (n - 1)

    def slot(i):
        return margin + i * pitch

    # 1) User
    x = slot(0)
    card(ax, x, row_y, card_w, card_h,
         L("User", "المستخدم"),
         [L("Upload telemetry", "رفع البيانات"),
          L("CSV · NPY files",  "ملفات CSV / NPY")],
         icon_fn=ic_user, accent=C_TITLE)

    # 2) Web App
    x = slot(1)
    card(ax, x, row_y, card_w, card_h,
         L("Web Application", "تطبيق الويب"),
         [L("HTML · CSS · JS",   "HTML · CSS · JS"),
          L("FastAPI · JWT auth", "FastAPI · مصادقة JWT")],
         icon_fn=ic_screen, accent=C_TITLE)

    # 3) Hybrid AE (most prominent)
    x = slot(2)
    card(ax, x, row_y, card_w, card_h,
         L("Hybrid LSTM-GRU", "النموذج الهجين"),
         [L("LSTM: long-term",  "LSTM: نمط طويل"),
          L("GRU: short-term",  "GRU: نمط قصير"),
          L("Autoencoder",      "Autoencoder")],
         icon_fn=ic_brain, accent=C_ACCENT)

    # 4) Anomaly Score Fusion
    x = slot(3)
    card(ax, x, row_y, card_w, card_h,
         L("Score Fusion", "دمج الدرجات"),
         [L("e_recon + e_pred",  "e_recon + e_pred"),
          L("+ gradient error",  "+ خطأ التدرّج"),
          L("unified score",     "درجة موحّدة")],
         icon_fn=ic_gauge, accent=C_TITLE)

    # 5) Anomaly Analysis (decision)
    x = slot(4)
    card(ax, x, row_y, card_w, card_h,
         L("Anomaly Analysis", "تحليل الشذوذ"),
         [L("threshold τ (p99)", "العتبة τ (p99)"),
          L("Normal / Attack",   "طبيعي / هجوم")],
         icon_fn=ic_gauge, accent=C_TITLE)

    # 6) Report
    x = slot(5)
    card(ax, x, row_y, card_w, card_h,
         L("Report Generation", "توليد التقرير"),
         [L("PDF · Excel · JSON", "PDF · Excel · JSON"),
          L("Sent to user",       "تُرسل للمستخدم")],
         icon_fn=ic_doc, accent=C_TITLE)

    # ------------------------------------------------------------------
    # Arrows on the top row (left -> right)
    # ------------------------------------------------------------------
    arrow_y = row_y + card_h / 2
    labels_top = [
        L("upload",  "رفع"),
        L("send",    "إرسال"),
        L("score",   "حساب"),
        L("classify","تصنيف"),
        L("report",  "تقرير"),
    ]
    for i in range(n - 1):
        x1 = slot(i) + card_w
        x2 = slot(i + 1)
        # arrow with label slightly above
        arrow_between(ax, x1 + 0.04, arrow_y, x2 - 0.04, arrow_y,
                      label=labels_top[i], color=C_ARROW, lw=1.8,
                      label_dy=0.30)

    # ------------------------------------------------------------------
    # Bottom row: Continual Learning loop + Admin approval
    # ------------------------------------------------------------------
    cl_w, cl_h = 4.20, 1.60
    cl_x = 5.10
    cl_y = 2.10
    card(ax, cl_x, cl_y, cl_w, cl_h,
         L("Continual Learning", "التعلّم المستمر"),
         [L("buffer → normal_pool → fine-tune",
            "buffer → normal_pool → ضبط"),
          L("incremental, no forgetting",
            "تحديث تدريجي · بدون نسيان")],
         icon_fn=ic_loop, accent=C_GREEN)

    # Admin card to the right of CL
    ad_w, ad_h = 2.40, 1.60
    ad_x = cl_x + cl_w + 0.90
    ad_y = cl_y
    card(ax, ad_x, ad_y, ad_w, ad_h,
         L("Admin", "المشرف"),
         [L("approve / discard", "موافقة / رفض"),
          L("model registry",    "سجلّ النماذج")],
         icon_fn=ic_admin, accent=C_GOLD)

    # 'New telemetry' card on the left of CL (input to CL)
    nt_w, nt_h = 2.40, 1.60
    nt_x = cl_x - nt_w - 0.90
    nt_y = cl_y
    card(ax, nt_x, nt_y, nt_w, nt_h,
         L("New Data", "بيانات جديدة"),
         [L("low-score windows", "نوافذ منخفضة"),
          L("safe candidates",   "مرشّحات آمنة")],
         icon_fn=ic_screen, accent=C_TITLE)

    # ------------------------------------------------------------------
    # Connecting arrows: top -> bottom loop
    # ------------------------------------------------------------------
    # Analysis -> New Data  (down from card 5)
    x_an = slot(4) + card_w / 2
    arrow_between(ax, x_an, row_y, nt_x + nt_w / 2, nt_y + nt_h,
                  label=L("new data", "بيانات"),
                  color=C_ARROW, lw=1.6, curved=True, rad=0.25,
                  label_dy=0.0)
    # New Data -> Continual Learning
    arrow_between(ax, nt_x + nt_w, nt_y + nt_h / 2,
                  cl_x, cl_y + cl_h / 2,
                  color=C_ARROW, lw=1.6,
                  label=L("feed", "إدخال"), label_dy=0.20)
    # Admin -> Continual Learning  (approval)
    arrow_between(ax, ad_x, ad_y + ad_h / 2,
                  cl_x + cl_w, cl_y + cl_h / 2,
                  color=C_GOLD, lw=1.6,
                  label=L("approve", "موافقة"), label_dy=0.20)
    # Continual Learning -> Hybrid AE  (fine-tune back to model)
    x_ae = slot(2) + card_w / 2
    arrow_between(ax, cl_x + cl_w / 2 - 0.30, cl_y + cl_h,
                  x_ae - 0.30, row_y,
                  label=L("fine-tune", "ضبط النموذج"),
                  color=C_GREEN, lw=1.8, curved=True, rad=-0.25,
                  label_dy=0.0, style="-|>")
    # Report -> User (return path, curved bottom-around)
    x_rep = slot(5) + card_w / 2
    x_user = slot(0) + card_w / 2
    # use a single long curved arrow going below the dashed border? Better, draw
    # an arc within the inside area so it stays inside the dashed frame.
    return_arr = FancyArrowPatch(
        (x_rep, row_y),
        (x_user, row_y),
        arrowstyle="-|>", mutation_scale=14,
        linewidth=1.5, color=C_ARROW,
        connectionstyle="arc3,rad=0.45",
        zorder=2,
        linestyle=(0, (4, 3)),
    )
    ax.add_patch(return_arr)
    # label on the return arc
    text(ax, 9, 1.10,
         L("display final report", "عرض التقرير النهائي"),
         color=C_TEXT_DIM, size=9, weight="bold")

    # ------------------------------------------------------------------
    # Footer caption (Figure-3.3 style)
    # ------------------------------------------------------------------
    text(ax, 9, 0.32,
         L("Figure 1: CyberSatDetect — Compact System Architecture",
           ar("الشكل 1: المعمارية المختصرة لنظام CyberSatDetect")),
         color=C_TITLE, size=11, weight="bold")

    # save
    suffix = "_ar" if is_ar else ""
    out = OUT_DIR / f"system_diagram_compact{suffix}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor(), pad_inches=0.20)
    plt.close(fig)
    return out


def main():
    for lang in ("en", "ar"):
        p = build(lang)
        print(f"[OK] wrote {p.relative_to(ROOT)}  ({p.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
