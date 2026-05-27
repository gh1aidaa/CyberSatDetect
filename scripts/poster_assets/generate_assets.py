# -*- coding: utf-8 -*-
"""Generate every PNG asset used by the professional CyberSatDetect poster.

All visuals are produced from project data:
  - Bar chart    -> thesis_official_evaluation_figures/overall_threshold_metrics_4attacks.csv
  - Per-attack   -> thesis_official_evaluation_figures/per_attack_full_metrics_4attacks.csv

The look (dark cyber theme) mirrors the reference infographic the user
provided. Output PNGs are written next to this script.
"""
from __future__ import annotations

import io
import math
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from PIL import Image, ImageDraw, ImageFilter, ImageFont

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ASSETS = Path(__file__).resolve().parent
ASSETS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
C_BG_TOP      = "#08131F"
C_BG_MID      = "#0F2638"
C_BG_BOTTOM   = "#0A1A28"
C_PANEL       = "#0F2A3A"
C_PANEL_EDGE  = "#1F4A60"
C_PANEL_HDR   = "#163B52"
C_ACCENT      = "#3FB7C2"  # bright teal
C_ACCENT2     = "#5DE0E6"
C_ACCENT3     = "#86F2F0"
C_GOLD        = "#E8B842"
C_WHITE       = "#FFFFFF"
C_TEXT        = "#D7E6EC"
C_MUTED       = "#8FA6B0"
C_GREEN       = "#3DD68C"
C_RED         = "#E5675A"
C_ORANGE      = "#F39A4B"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def linear_gradient(width: int, height: int, top_hex: str, mid_hex: str,
                    bottom_hex: str) -> Image.Image:
    """3-stop vertical gradient."""
    img = Image.new("RGB", (width, height))
    px = img.load()
    top = np.array(hex_to_rgb(top_hex))
    mid = np.array(hex_to_rgb(mid_hex))
    bot = np.array(hex_to_rgb(bottom_hex))
    for y in range(height):
        t = y / max(height - 1, 1)
        if t < 0.5:
            f = t / 0.5
            c = top * (1 - f) + mid * f
        else:
            f = (t - 0.5) / 0.5
            c = mid * (1 - f) + bot * f
        rgb = tuple(int(round(v)) for v in c)
        for x in range(width):
            px[x, y] = rgb
    return img


def rounded_rect(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int],
                 radius: int, fill=None, outline=None, width: int = 1) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline,
                           width=width)


def _try_load_font(name_candidates: list[str], size: int) -> ImageFont.FreeTypeFont:
    for name in name_candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_BOLD   = lambda s: _try_load_font(["seguibl.ttf", "segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"], s)
FONT_SEMI   = lambda s: _try_load_font(["seguisb.ttf", "segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"], s)
FONT_REG    = lambda s: _try_load_font(["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"], s)


# ---------------------------------------------------------------------------
# 1. Background (full poster) - dark gradient with subtle circuit dots
# ---------------------------------------------------------------------------
def make_background(path: Path, width: int = 2480, height: int = 3508) -> None:
    bg = linear_gradient(width, height, C_BG_TOP, C_BG_MID, C_BG_BOTTOM)
    draw = ImageDraw.Draw(bg, "RGBA")

    rng = np.random.default_rng(42)
    n_dots = 1400
    for _ in range(n_dots):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        r = float(rng.uniform(0.6, 2.2))
        alpha = int(rng.integers(28, 90))
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(80, 180, 200, alpha))

    n_lines = 32
    for _ in range(n_lines):
        x1 = int(rng.integers(0, width))
        y1 = int(rng.integers(0, height))
        x2 = x1 + int(rng.integers(-220, 220))
        y2 = y1 + int(rng.integers(-220, 220))
        draw.line((x1, y1, x2, y2), fill=(50, 120, 145, 25), width=1)

    bg = bg.filter(ImageFilter.SMOOTH)
    bg.save(path, "PNG", optimize=True)


# ---------------------------------------------------------------------------
# 2. Satellite illustration (top-right hero)
# ---------------------------------------------------------------------------
def make_satellite(path: Path, w: int = 1200, h: int = 900) -> None:
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img, "RGBA")

    cx, cy = int(w * 0.55), int(h * 0.55)

    # Satellite body
    body_w, body_h = 220, 150
    d.rounded_rectangle((cx - body_w // 2, cy - body_h // 2,
                         cx + body_w // 2, cy + body_h // 2),
                        radius=24, fill=(30, 70, 95, 255),
                        outline=hex_to_rgb(C_ACCENT2) + (255,), width=5)
    # Window/lens on body
    d.ellipse((cx - 38, cy - 32, cx + 38, cy + 32),
              fill=(15, 30, 50, 255),
              outline=hex_to_rgb(C_GOLD) + (255,), width=4)
    d.ellipse((cx - 18, cy - 14, cx + 14, cy + 14),
              fill=hex_to_rgb(C_ACCENT3) + (255,))

    # Solar panel - left
    pl_w, pl_h = 320, 110
    d.rectangle((cx - body_w // 2 - pl_w - 30, cy - pl_h // 2,
                 cx - body_w // 2 - 30, cy + pl_h // 2),
                fill=(15, 50, 80, 255),
                outline=hex_to_rgb(C_ACCENT) + (255,), width=4)
    # Solar grid lines
    for i in range(1, 6):
        gx = cx - body_w // 2 - 30 - i * (pl_w // 6)
        d.line((gx, cy - pl_h // 2, gx, cy + pl_h // 2),
               fill=hex_to_rgb(C_ACCENT) + (180,), width=2)
    d.line((cx - body_w // 2 - 30, cy, cx - body_w // 2 - pl_w - 30, cy),
           fill=hex_to_rgb(C_ACCENT) + (140,), width=2)
    # Connector to body
    d.line((cx - body_w // 2, cy, cx - body_w // 2 - 30, cy),
           fill=hex_to_rgb(C_ACCENT2) + (255,), width=6)

    # Solar panel - right
    d.rectangle((cx + body_w // 2 + 30, cy - pl_h // 2,
                 cx + body_w // 2 + pl_w + 30, cy + pl_h // 2),
                fill=(15, 50, 80, 255),
                outline=hex_to_rgb(C_ACCENT) + (255,), width=4)
    for i in range(1, 6):
        gx = cx + body_w // 2 + 30 + i * (pl_w // 6)
        d.line((gx, cy - pl_h // 2, gx, cy + pl_h // 2),
               fill=hex_to_rgb(C_ACCENT) + (180,), width=2)
    d.line((cx + body_w // 2 + 30, cy, cx + body_w // 2 + pl_w + 30, cy),
           fill=hex_to_rgb(C_ACCENT) + (140,), width=2)
    d.line((cx + body_w // 2, cy, cx + body_w // 2 + 30, cy),
           fill=hex_to_rgb(C_ACCENT2) + (255,), width=6)

    # Antenna dish on top
    dish_cx, dish_cy = cx, cy - body_h // 2 - 80
    d.line((dish_cx, cy - body_h // 2, dish_cx, dish_cy + 30),
           fill=hex_to_rgb(C_ACCENT2) + (255,), width=6)
    d.arc((dish_cx - 60, dish_cy - 50, dish_cx + 60, dish_cy + 60),
          start=200, end=340, fill=hex_to_rgb(C_GOLD) + (255,), width=8)
    d.ellipse((dish_cx - 8, dish_cy - 8, dish_cx + 8, dish_cy + 8),
              fill=hex_to_rgb(C_GOLD) + (255,))

    # Signal beams emitting from dish
    for k, ang in enumerate((-25, 0, 25)):
        rad = math.radians(ang - 90)
        for ring in range(3):
            d.arc(
                (dish_cx - (40 + ring * 28), dish_cy - 75 - ring * 28,
                 dish_cx + (40 + ring * 28), dish_cy + 5 - ring * 28),
                start=210 + ang, end=330 + ang,
                fill=hex_to_rgb(C_ACCENT2) + (180 - ring * 50,),
                width=3,
            )

    # Lock badge floating top-right (security symbol)
    bx, by = int(w * 0.85), int(h * 0.18)
    br = 90
    d.ellipse((bx - br, by - br, bx + br, by + br),
              fill=hex_to_rgb(C_ACCENT) + (60,))
    d.ellipse((bx - br + 18, by - br + 18, bx + br - 18, by + br - 18),
              fill=(15, 35, 55, 230),
              outline=hex_to_rgb(C_ACCENT2) + (255,), width=4)
    # Lock body
    d.rounded_rectangle((bx - 30, by - 5, bx + 30, by + 35), radius=6,
                        fill=hex_to_rgb(C_GOLD) + (255,))
    # Lock shackle
    d.arc((bx - 24, by - 38, bx + 24, by + 12), start=180, end=360,
          fill=hex_to_rgb(C_GOLD) + (255,), width=6)
    # Keyhole
    d.ellipse((bx - 6, by + 8, bx + 6, by + 20), fill=(30, 30, 30, 255))
    d.rectangle((bx - 2, by + 16, bx + 2, by + 28), fill=(30, 30, 30, 255))

    img = img.filter(ImageFilter.SMOOTH)
    img.save(path, "PNG")


# ---------------------------------------------------------------------------
# 3. Ground-station / satellite dish illustration (footer of methodology)
# ---------------------------------------------------------------------------
def make_dish(path: Path, w: int = 700, h: int = 700) -> None:
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img, "RGBA")
    cx, cy = w // 2, int(h * 0.55)
    # Dish bowl
    d.arc((cx - 230, cy - 220, cx + 230, cy + 120),
          start=200, end=340, fill=hex_to_rgb(C_ACCENT2) + (255,), width=14)
    d.chord((cx - 220, cy - 200, cx + 220, cy + 100),
            start=200, end=340, fill=(20, 60, 90, 220))
    # Feed horn
    d.line((cx, cy - 80, cx, cy + 60), fill=hex_to_rgb(C_GOLD) + (255,), width=10)
    d.ellipse((cx - 14, cy - 95, cx + 14, cy - 65),
              fill=hex_to_rgb(C_GOLD) + (255,))
    # Mount post
    d.rectangle((cx - 10, cy + 50, cx + 10, cy + 220),
                fill=hex_to_rgb(C_ACCENT) + (255,))
    # Base
    d.polygon([(cx - 100, h - 80), (cx + 100, h - 80),
               (cx + 60, h - 30), (cx - 60, h - 30)],
              fill=hex_to_rgb(C_ACCENT) + (255,))
    # Signal arcs
    for r in (260, 310, 360):
        d.arc((cx - r, cy - r - 80, cx + r, cy + r - 80),
              start=235, end=305,
              fill=hex_to_rgb(C_ACCENT2) + (160 - (r - 260),), width=6)
    img.save(path, "PNG")


# ---------------------------------------------------------------------------
# 4. Tagline icons (small white icons inside a teal circle)
# ---------------------------------------------------------------------------
def _make_round_icon(path: Path, draw_glyph) -> None:
    s = 380
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img, "RGBA")
    d.ellipse((4, 4, s - 4, s - 4),
              fill=(15, 60, 90, 255),
              outline=hex_to_rgb(C_ACCENT2) + (255,), width=6)
    d.ellipse((30, 30, s - 30, s - 30),
              fill=(15, 35, 55, 255))
    draw_glyph(d, s)
    img.save(path, "PNG")


def make_icon_target(path: Path) -> None:
    def glyph(d, s):
        cx = cy = s // 2
        for r in (110, 78, 46):
            d.ellipse((cx - r, cy - r, cx + r, cy + r),
                      outline=hex_to_rgb(C_ACCENT2) + (255,), width=6)
        d.ellipse((cx - 14, cy - 14, cx + 14, cy + 14),
                  fill=hex_to_rgb(C_GOLD) + (255,))
        d.line((cx - 130, cy, cx - 90, cy),
               fill=hex_to_rgb(C_ACCENT2) + (255,), width=6)
        d.line((cx + 90, cy, cx + 130, cy),
               fill=hex_to_rgb(C_ACCENT2) + (255,), width=6)
        d.line((cx, cy - 130, cx, cy - 90),
               fill=hex_to_rgb(C_ACCENT2) + (255,), width=6)
        d.line((cx, cy + 90, cx, cy + 130),
               fill=hex_to_rgb(C_ACCENT2) + (255,), width=6)
    _make_round_icon(path, glyph)


def make_icon_dish(path: Path) -> None:
    def glyph(d, s):
        cx, cy = s // 2, s // 2 + 10
        d.arc((cx - 95, cy - 95, cx + 95, cy + 60),
              start=200, end=340, fill=hex_to_rgb(C_ACCENT2) + (255,), width=7)
        d.chord((cx - 88, cy - 88, cx + 88, cy + 55),
                start=200, end=340, fill=(35, 90, 115, 255))
        d.line((cx, cy - 40, cx, cy + 60), fill=hex_to_rgb(C_GOLD) + (255,),
               width=6)
        d.ellipse((cx - 8, cy - 50, cx + 8, cy - 32),
                  fill=hex_to_rgb(C_GOLD) + (255,))
        d.rectangle((cx - 6, cy + 50, cx + 6, cy + 100),
                    fill=hex_to_rgb(C_ACCENT) + (255,))
        for r in (115, 140):
            d.arc((cx - r, cy - r - 40, cx + r, cy + r - 40),
                  start=235, end=305,
                  fill=hex_to_rgb(C_ACCENT2) + (170,), width=5)
    _make_round_icon(path, glyph)


def make_icon_brain(path: Path) -> None:
    def glyph(d, s):
        cx, cy = s // 2, s // 2
        # Two ellipses for brain halves
        d.ellipse((cx - 95, cy - 70, cx - 5, cy + 60),
                  fill=(35, 90, 115, 255),
                  outline=hex_to_rgb(C_ACCENT2) + (255,), width=5)
        d.ellipse((cx + 5, cy - 70, cx + 95, cy + 60),
                  fill=(35, 90, 115, 255),
                  outline=hex_to_rgb(C_ACCENT2) + (255,), width=5)
        # Node dots
        for px, py in [(-50, -25), (-25, 5), (-55, 25), (50, -25),
                       (25, 5), (55, 25), (0, -45), (0, 35)]:
            d.ellipse((cx + px - 7, cy + py - 7, cx + px + 7, cy + py + 7),
                      fill=hex_to_rgb(C_GOLD) + (255,))
        # connecting lines
        for (a, b) in [((-50, -25), (-25, 5)), ((-25, 5), (-55, 25)),
                        ((50, -25), (25, 5)), ((25, 5), (55, 25)),
                        ((-25, 5), (25, 5)), ((0, -45), (-25, 5)),
                        ((0, -45), (25, 5)), ((0, 35), (-25, 5)),
                        ((0, 35), (25, 5))]:
            d.line((cx + a[0], cy + a[1], cx + b[0], cy + b[1]),
                   fill=hex_to_rgb(C_ACCENT3) + (180,), width=3)
    _make_round_icon(path, glyph)


def make_icon_pulse(path: Path) -> None:
    def glyph(d, s):
        cx, cy = s // 2, s // 2
        points = [
            (cx - 130, cy), (cx - 80, cy), (cx - 60, cy - 50),
            (cx - 30, cy + 60), (cx, cy - 70), (cx + 30, cy + 30),
            (cx + 55, cy), (cx + 130, cy),
        ]
        d.line(points, fill=hex_to_rgb(C_ACCENT2) + (255,), width=8,
               joint="curve")
        d.line(points, fill=hex_to_rgb(C_GOLD) + (180,), width=3,
               joint="curve")
    _make_round_icon(path, glyph)


def make_icon_shield(path: Path) -> None:
    def glyph(d, s):
        cx, cy = s // 2, s // 2 + 10
        pts = [
            (cx, cy - 100),
            (cx + 85, cy - 60),
            (cx + 85, cy + 30),
            (cx, cy + 110),
            (cx - 85, cy + 30),
            (cx - 85, cy - 60),
        ]
        d.polygon(pts, fill=(35, 90, 115, 255),
                  outline=hex_to_rgb(C_ACCENT2) + (255,))
        # Check mark
        d.line((cx - 35, cy, cx - 10, cy + 30, cx + 40, cy - 30),
               fill=hex_to_rgb(C_GREEN) + (255,), width=12, joint="curve")
    _make_round_icon(path, glyph)


def make_icon_chart(path: Path) -> None:
    def glyph(d, s):
        cx, cy = s // 2, s // 2 + 20
        d.rectangle((cx - 60, cy - 5, cx - 25, cy + 70),
                    fill=hex_to_rgb(C_ACCENT2) + (255,))
        d.rectangle((cx - 18, cy - 50, cx + 17, cy + 70),
                    fill=hex_to_rgb(C_GOLD) + (255,))
        d.rectangle((cx + 25, cy + 20, cx + 60, cy + 70),
                    fill=hex_to_rgb(C_ACCENT3) + (255,))
        d.line((cx - 80, cy + 75, cx + 80, cy + 75),
               fill=hex_to_rgb(C_TEXT) + (220,), width=4)
    _make_round_icon(path, glyph)


def make_icon_gear(path: Path) -> None:
    def glyph(d, s):
        cx, cy = s // 2, s // 2
        n = 8
        outer_r, inner_r = 100, 75
        pts = []
        for i in range(n * 2):
            ang = math.pi * i / n
            r = outer_r if i % 2 == 0 else inner_r
            pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
        d.polygon(pts, fill=hex_to_rgb(C_ACCENT2) + (255,))
        d.ellipse((cx - 38, cy - 38, cx + 38, cy + 38),
                  fill=(15, 35, 55, 255))
    _make_round_icon(path, glyph)


def make_icon_database(path: Path) -> None:
    def glyph(d, s):
        cx, cy = s // 2, s // 2
        rx, ry = 65, 18
        for i, yo in enumerate((-50, -10, 30)):
            color = hex_to_rgb(C_ACCENT2) + (255,)
            d.ellipse((cx - rx, cy + yo - ry, cx + rx, cy + yo + ry),
                      fill=hex_to_rgb(C_ACCENT) + (255,),
                      outline=color, width=4)
        d.rectangle((cx - rx, cy - 50, cx + rx, cy + 30),
                    fill=hex_to_rgb(C_ACCENT) + (255,))
        for yo in (-50, -10, 30):
            d.ellipse((cx - rx, cy + yo - ry, cx + rx, cy + yo + ry),
                      outline=hex_to_rgb(C_ACCENT2) + (255,), width=4)
    _make_round_icon(path, glyph)


def make_icon_loop(path: Path) -> None:
    def glyph(d, s):
        cx, cy = s // 2, s // 2
        d.arc((cx - 90, cy - 90, cx + 90, cy + 90),
              start=20, end=340, fill=hex_to_rgb(C_ACCENT2) + (255,), width=10)
        # arrow head
        d.polygon([(cx + 95, cy - 10), (cx + 60, cy - 30), (cx + 60, cy + 10)],
                  fill=hex_to_rgb(C_ACCENT2) + (255,))
        d.line((cx - 55, cy - 5, cx, cy - 5),
               fill=hex_to_rgb(C_GOLD) + (255,), width=6)
        d.line((cx - 55, cy + 15, cx + 25, cy + 15),
               fill=hex_to_rgb(C_GOLD) + (255,), width=6)
    _make_round_icon(path, glyph)


def make_icon_alert(path: Path) -> None:
    def glyph(d, s):
        cx, cy = s // 2, s // 2 + 10
        d.polygon([(cx, cy - 100), (cx + 100, cy + 80), (cx - 100, cy + 80)],
                  fill=hex_to_rgb(C_GOLD) + (255,),
                  outline=hex_to_rgb(C_ACCENT2) + (255,))
        d.rectangle((cx - 8, cy - 50, cx + 8, cy + 25),
                    fill=(20, 40, 55, 255))
        d.ellipse((cx - 9, cy + 40, cx + 9, cy + 58),
                  fill=(20, 40, 55, 255))
    _make_round_icon(path, glyph)


def make_icon_users(path: Path) -> None:
    def glyph(d, s):
        cx, cy = s // 2, s // 2 + 15
        # head
        d.ellipse((cx - 30, cy - 80, cx + 30, cy - 20),
                  fill=hex_to_rgb(C_ACCENT2) + (255,))
        # body
        d.chord((cx - 70, cy - 15, cx + 70, cy + 105),
                start=180, end=360,
                fill=hex_to_rgb(C_ACCENT2) + (255,))
    _make_round_icon(path, glyph)


def make_icon_supervisor(path: Path) -> None:
    def glyph(d, s):
        cx, cy = s // 2, s // 2 + 15
        # mortarboard cap
        d.polygon([(cx - 90, cy - 30), (cx, cy - 75), (cx + 90, cy - 30),
                   (cx, cy + 15)], fill=hex_to_rgb(C_GOLD) + (255,))
        d.line((cx + 65, cy - 35, cx + 65, cy + 25),
               fill=hex_to_rgb(C_GOLD) + (255,), width=6)
        d.ellipse((cx + 60, cy + 25, cx + 70, cy + 35),
                  fill=hex_to_rgb(C_GOLD) + (255,))
        # head silhouette
        d.ellipse((cx - 35, cy + 20, cx + 35, cy + 80),
                  fill=hex_to_rgb(C_ACCENT2) + (255,))
        # shoulders
        d.chord((cx - 80, cy + 60, cx + 80, cy + 180),
                start=180, end=360,
                fill=hex_to_rgb(C_ACCENT2) + (255,))
    _make_round_icon(path, glyph)


def make_icon_college(path: Path) -> None:
    def glyph(d, s):
        cx, cy = s // 2, s // 2 + 15
        d.polygon([(cx - 110, cy - 10), (cx, cy - 75), (cx + 110, cy - 10)],
                  fill=hex_to_rgb(C_GOLD) + (255,))
        d.rectangle((cx - 100, cy - 10, cx + 100, cy - 4),
                    fill=hex_to_rgb(C_ACCENT2) + (255,))
        for px in (-70, -25, 25, 70):
            d.rectangle((cx + px - 12, cy, cx + px + 12, cy + 90),
                        fill=hex_to_rgb(C_ACCENT2) + (255,))
        d.rectangle((cx - 110, cy + 95, cx + 110, cy + 105),
                    fill=hex_to_rgb(C_ACCENT2) + (255,))
    _make_round_icon(path, glyph)


# ---------------------------------------------------------------------------
# 5. Results bar chart (from project CSV)
# ---------------------------------------------------------------------------
def make_results_chart(path: Path) -> None:
    panel_bg = "#0E283A"

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#FFFFFF",
        "axes.labelcolor": "#FFFFFF",
        "xtick.color": "#FFFFFF",
        "ytick.color": "#FFFFFF",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.facecolor": panel_bg,
        "figure.facecolor": panel_bg,
        "text.color": "#FFFFFF",
    })

    attacks = ["Drift", "Freeze", "Noise", "Spike"]
    f1_vals       = [0.7498, 0.8969, 0.9306, 0.9394]
    recall_vals   = [1.0000, 0.9969, 0.9998, 0.9997]
    bal_acc_vals  = [0.9948, 0.9928, 0.9947, 0.9953]

    x = np.arange(len(attacks))
    width = 0.26

    fig, ax = plt.subplots(figsize=(11, 6.8), dpi=180)
    fig.patch.set_facecolor(panel_bg)
    ax.set_facecolor(panel_bg)

    b1 = ax.bar(x - width, f1_vals, width, label="F1-Score",
                color="#3FB7C2", edgecolor="#0E283A", linewidth=1.2)
    b2 = ax.bar(x, recall_vals, width, label="Recall",
                color="#E8B842", edgecolor="#0E283A", linewidth=1.2)
    b3 = ax.bar(x + width, bal_acc_vals, width, label="Balanced Acc.",
                color="#86F2F0", edgecolor="#0E283A", linewidth=1.2)

    for bars in (b1, b2, b3):
        for rect in bars:
            v = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.018,
                    f"{v*100:.1f}%", ha="center", va="bottom",
                    color="white", fontsize=11.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(attacks, fontsize=14, fontweight="bold", color="white")
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["50%", "60%", "70%", "80%", "90%", "100%"],
                       fontsize=12, color="white")
    ax.set_ylabel("Score", fontsize=13, fontweight="bold", color="white")
    ax.grid(axis="y", color="white", alpha=0.18, linewidth=0.8)
    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.20), ncol=3,
                    frameon=False, fontsize=13)
    for txt in leg.get_texts():
        txt.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_alpha(0.45)
    plt.tight_layout()
    fig.savefig(path, dpi=180, facecolor=panel_bg, bbox_inches="tight",
                pad_inches=0.18)
    plt.close(fig)


def make_threshold_chart(path: Path) -> None:
    """Threshold trade-off line chart: Recall vs FAR for project thresholds."""
    panel_bg = "#0E283A"
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.facecolor": panel_bg,
        "figure.facecolor": panel_bg,
        "text.color": "#FFFFFF",
    })

    # From overall_threshold_metrics_4attacks.csv (project file)
    names = ["best_F1", "p95", "p97", "p99", "p99.5", "p99.7", "3sigma"]
    recall = [0.9991, 1.0000, 1.0000, 0.7539, 0.6543, 0.3938, 0.7071]
    far    = [0.0163, 0.0573, 0.0323, 0.0047, 0.0033, 0.0017, 0.0044]

    fig, ax = plt.subplots(figsize=(11, 6.8), dpi=180)
    fig.patch.set_facecolor(panel_bg)
    ax.set_facecolor(panel_bg)

    sc = ax.scatter([f * 100 for f in far], [r * 100 for r in recall],
                    s=320, c="#E8B842", edgecolor="#86F2F0", linewidth=2.5,
                    zorder=5)
    for n, x, y in zip(names, far, recall):
        ax.annotate(n, (x * 100, y * 100), textcoords="offset points",
                    xytext=(10, 10), color="white", fontsize=11,
                    fontweight="bold")

    ax.set_xlabel("FAR (%)", fontsize=13, fontweight="bold", color="white")
    ax.set_ylabel("Recall (%)", fontsize=13, fontweight="bold", color="white")
    ax.tick_params(colors="white", labelsize=11)
    ax.grid(True, color="white", alpha=0.18, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_alpha(0.45)
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(35, 105)
    plt.tight_layout()
    fig.savefig(path, dpi=180, facecolor=panel_bg, bbox_inches="tight",
                pad_inches=0.18)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    print("Asset directory:", ASSETS)
    make_background(ASSETS / "bg.png")
    make_satellite(ASSETS / "satellite.png")
    make_dish(ASSETS / "dish.png")

    make_icon_target(ASSETS / "icon_target.png")
    make_icon_dish(ASSETS / "icon_dish.png")
    make_icon_brain(ASSETS / "icon_brain.png")
    make_icon_pulse(ASSETS / "icon_pulse.png")
    make_icon_shield(ASSETS / "icon_shield.png")
    make_icon_chart(ASSETS / "icon_chart.png")
    make_icon_gear(ASSETS / "icon_gear.png")
    make_icon_database(ASSETS / "icon_database.png")
    make_icon_loop(ASSETS / "icon_loop.png")
    make_icon_alert(ASSETS / "icon_alert.png")
    make_icon_users(ASSETS / "icon_users.png")
    make_icon_supervisor(ASSETS / "icon_supervisor.png")
    make_icon_college(ASSETS / "icon_college.png")

    make_results_chart(ASSETS / "bar_chart.png")
    make_threshold_chart(ASSETS / "threshold_chart.png")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
