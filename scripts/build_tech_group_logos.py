"""Build per-category tech-stack logo strips with TRANSPARENT background.

Output:  docs/images/tech_groups/<category>.png

Approach:
  1) For each category, write a tiny standalone HTML page that lays out the
     official logos in a single horizontal row.
  2) Render the page with Microsoft Edge in headless mode using the flag
     ``--default-background-color=00000000`` so the screenshot keeps an
     ALPHA channel (true transparent background).
  3) Auto-crop the transparent margins around the actual logos with Pillow.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


# ----- paths --------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "images" / "tech_groups"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EDGE_CANDIDATES = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]


def find_edge() -> Path:
    for p in EDGE_CANDIDATES:
        if p.exists():
            return p
    # fall back to chrome if available
    chrome = shutil.which("chrome") or shutil.which("chrome.exe")
    if chrome:
        return Path(chrome)
    raise SystemExit("Microsoft Edge not found. Please install Edge or update EDGE_CANDIDATES.")


# ----- logo registry ------------------------------------------------------
# Each tool: (label, image_url).  We prefer SVGs from devicon (full color with
# gradients) because Edge renders them faithfully.  For tools missing in
# devicon we fall back to simpleicons (flat brand color) or to the official
# site / Wikimedia.

D_ICON = "https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/{}/{}-original.svg"
D_ICON_W = "https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/{}/{}-original-wordmark.svg"
S_ICON = "https://cdn.simpleicons.org/{}/{}"


GROUPS: Dict[str, List[Tuple[str, str]]] = {
    "01_ai_data_processing": [
        ("Python",        D_ICON.format("python", "python")),
        ("TensorFlow",    D_ICON.format("tensorflow", "tensorflow")),
        ("Keras",         S_ICON.format("keras", "D00000")),
        ("NumPy",         D_ICON.format("numpy", "numpy")),
        ("Pandas",        D_ICON.format("pandas", "pandas")),
    ],
    "02_backend_api": [
        ("FastAPI",       D_ICON.format("fastapi", "fastapi")),
        ("Pydantic",      S_ICON.format("pydantic", "E92063")),
    ],
    "03_frontend_visualization": [
        ("JavaScript",    D_ICON.format("javascript", "javascript")),
        ("HTML5",         D_ICON.format("html5", "html5")),
        ("CSS3",          D_ICON.format("css3", "css3")),
    ],
    "04_database_storage": [
        ("SQLite",        D_ICON.format("sqlite", "sqlite")),
        ("DBeaver",       "https://avatars.githubusercontent.com/u/34743864?s=300&v=4"),
    ],
    "05_security_authentication": [
        ("JWT",           S_ICON.format("jsonwebtokens", "000000")),
        ("bcrypt",        S_ICON.format("keepassxc", "558B2F")),  # closest lock-style icon
    ],
    "06_dev_tools_version_control": [
        ("Cursor",        "https://avatars.githubusercontent.com/u/126759922?s=200&v=4"),
        ("VS Code",       D_ICON.format("vscode", "vscode")),
        ("Git",           D_ICON.format("git", "git")),
        ("GitHub",        S_ICON.format("github", "181717")),
    ],
}


# ----- HTML template ------------------------------------------------------

HTML_TMPL = """<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<style>
  html, body {{
    margin: 0; padding: 0;
    background: transparent;
  }}
  body {{
    display: inline-block;
    padding: 40px 50px;
  }}
  .row {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 70px;
    height: 180px;
  }}
  .item {{
    display: flex;
    align-items: center;
    justify-content: center;
    height: 180px;
    min-width: 80px;
  }}
  .item img {{
    height: 150px;
    width: auto;
    max-width: 260px;
    object-fit: contain;
    image-rendering: -webkit-optimize-contrast;
  }}
</style>
</head><body>
  <div class="row">
    {items}
  </div>
</body></html>
"""


def build_html(items: List[Tuple[str, str]]) -> str:
    inner = "\n    ".join(
        f'<div class="item"><img src="{url}" alt="{label}"/></div>'
        for label, url in items
    )
    return HTML_TMPL.format(items=inner)


# ----- rendering ----------------------------------------------------------

def screenshot_html(edge: Path, html_path: Path, png_path: Path,
                    width: int, height: int) -> None:
    """Render the HTML at <html_path> with Edge headless, transparent bg."""
    cmd = [
        str(edge),
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--force-device-scale-factor=2",
        "--default-background-color=00000000",
        "--virtual-time-budget=20000",
        "--run-all-compositor-stages-before-draw",
        f"--window-size={width},{height}",
        f"--screenshot={png_path}",
        html_path.as_uri(),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if not png_path.exists() or png_path.stat().st_size == 0:
        raise RuntimeError(
            f"Edge failed to produce {png_path.name}\n"
            f"stdout={res.stdout!r}\nstderr={res.stderr!r}"
        )


def crop_transparent(png_path: Path) -> None:
    """Trim fully-transparent rows/columns around the content."""
    img = Image.open(png_path).convert("RGBA")
    bbox = img.getbbox()
    if bbox:
        # add a small breathing padding back
        pad = 12
        l, t, r, b = bbox
        l = max(0, l - pad); t = max(0, t - pad)
        r = min(img.width, r + pad); b = min(img.height, b + pad)
        img = img.crop((l, t, r, b))
    img.save(png_path, optimize=True)


def estimate_window(num_logos: int) -> Tuple[int, int]:
    # generous estimate: each item up to ~280px wide (wordmarks), 70px gaps
    width = 100 + num_logos * 280 + (num_logos - 1) * 70 + 100
    height = 320
    return width, height


# ----- main ---------------------------------------------------------------

def main() -> None:
    edge = find_edge()
    print(f"Using Edge: {edge}")
    print(f"Output dir: {OUT_DIR}")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for name, items in GROUPS.items():
            html_file = td_path / f"{name}.html"
            png_file = OUT_DIR / f"{name}.png"
            html_file.write_text(build_html(items), encoding="utf-8")
            w, h = estimate_window(len(items))
            print(f"  · {name:38s} ({len(items)} logos)  ->  {png_file.name}")
            screenshot_html(edge, html_file, png_file, w, h)
            crop_transparent(png_file)
            print(f"      size: {png_file.stat().st_size//1024} KB")

    print("\nDone. Files in:", OUT_DIR)


if __name__ == "__main__":
    main()
