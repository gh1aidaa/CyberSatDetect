"""Render only the Evaluation section of 3.pptx to a PNG composite by
compositing the real chart images on top of the metric table sketch."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "scripts" / "poster_eval_assets"
OUT = ROOT / "docs" / "poster_eval_preview.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Eval section bounding box (cm) -- from the inspect output
EVAL = dict(left=5.0, top=79.42, width=74.10, height=15.50)

# After 2-column table re-layout: only Picture 36 (the big bar chart) remains.
PICS = [
    dict(name="Picture 36", left=47.84, top=82.77, w=27.57, h=11.91,
         file=ASSETS / "poster_per_attack_bars.png"),
]

# Render to a figure whose units mirror cm
fig_w_in = 30
scale = fig_w_in / EVAL["width"]
fig_h_in = (EVAL["height"] + 1.5) * scale
fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=140)

ax.set_xlim(EVAL["left"] - 1, EVAL["left"] + EVAL["width"] + 1)
ax.set_ylim(EVAL["top"] + EVAL["height"] + 0.5, EVAL["top"] - 1)  # invert Y
ax.set_aspect("equal")
ax.axis("off")

# Eval panel background
panel = FancyBboxPatch(
    (EVAL["left"], EVAL["top"]), EVAL["width"], EVAL["height"],
    boxstyle="round,pad=0.05,rounding_size=0.4",
    fc="#FFFFFF", ec="#8B5BC4", linewidth=2.0,
)
ax.add_patch(panel)

# Header strip
header = Rectangle((EVAL["left"], EVAL["top"]), EVAL["width"], 3.0,
                   fc="#8B5BC4", ec="none")
ax.add_patch(header)
ax.text(EVAL["left"] + EVAL["width"] / 2, EVAL["top"] + 1.5,
        "EVALUATION & RESULTS",
        ha="center", va="center", fontsize=22, fontweight="bold",
        color="#FFFFFF")

# Metrics table sketch -- MAXIMUM size, fills entire eval-panel vertical space
tbl_x, tbl_y = 5.70, 82.45
tbl_w = 41.80
half_w = tbl_w / 2
ax.text(tbl_x + 0.5, tbl_y + 1.05, "Performance Metrics  (best-F1)",
        fontsize=28, fontweight="bold", color="#4A2674")

pairs = [
    [("F1-Score", "0.9493"),         ("Recall", "0.9991")],
    [("Accuracy", "98.57%"),          ("Precision", "0.9043")],
    [("Balanced Accuracy", "99.14%"), ("FAR", "1.63%")],
    [("ROC-AUC", "0.9958"),           ("PR-AUC", "0.9561")],
]
header_h, row_h = 2.10, 2.55
for r, (left_pair, right_pair) in enumerate(pairs):
    y = tbl_y + header_h + r * row_h
    bg = "#F6F2FB" if r % 2 == 0 else "#FFFFFF"
    ax.add_patch(Rectangle((tbl_x, y), tbl_w, row_h, fc=bg, ec="#D5CCE0"))
    lbl, val = left_pair
    ax.text(tbl_x + 1.0, y + row_h / 2, lbl, fontsize=22, color="#1F1830",
            va="center")
    ax.text(tbl_x + half_w - 0.8, y + row_h / 2, val, fontsize=28,
            color="#4A2674", fontweight="bold", va="center", ha="right")
    lbl, val = right_pair
    ax.text(tbl_x + half_w + 1.0, y + row_h / 2, lbl, fontsize=22,
            color="#1F1830", va="center")
    ax.text(tbl_x + tbl_w - 0.8, y + row_h / 2, val, fontsize=28,
            color="#4A2674", fontweight="bold", va="center", ha="right")

# Add the three real chart images at their PPTX positions
for p in PICS:
    img = imread(str(p["file"]))
    ax.imshow(img, extent=(p["left"], p["left"] + p["w"],
                            p["top"] + p["h"], p["top"]), aspect="auto",
              zorder=3)

# (Test-set caption was deleted by the user; no longer rendered.)

plt.tight_layout()
plt.savefig(OUT, dpi=140, bbox_inches="tight", pad_inches=0.2,
            facecolor="#FFFFFF")
print(f"Saved preview to: {OUT}")
