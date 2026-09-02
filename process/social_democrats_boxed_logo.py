#!/usr/bin/env python3
"""Generate the selected Social Democrats boxed logo treatment.

The selected asset uses the clean vector identity, removes the original square
background to isolate the foreground mark, then recreates the official white boxed
treatment on the vector's own dominant purple. No manifesto pixels are used and
nothing is written to S3.
"""

from __future__ import annotations

import io
import json
from collections import Counter
from pathlib import Path

import cairosvg
import requests
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "review/party_assets_v1/social_democrats"
SOURCE_URL = "https://commons.wikimedia.org/wiki/Special:Redirect/file/Social%20Democrats%20%28Ireland%29%20logo.svg"
CANVAS = 1600
BOX = (190, 360, 1410, 1240)
BOX_RADIUS = 58
BOX_WIDTH = 24
CONTENT_MARGIN_X = 120
CONTENT_MARGIN_Y = 110
BACKGROUND_DISTANCE = 36


def download_svg() -> bytes:
    response = requests.get(
        SOURCE_URL,
        timeout=35,
        allow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 EirePolitic-SocialDemocrats-Boxed/1.2"},
    )
    response.raise_for_status()
    return response.content


def _dominant_opaque_colour(image: Image.Image) -> tuple[int, int, int]:
    sample = image.convert("RGBA").resize((160, 160), Image.Resampling.NEAREST)
    colours = Counter((r, g, b) for r, g, b, a in sample.getdata() if a > 240)
    if not colours:
        raise ValueError("source vector has no opaque pixels")
    return colours.most_common(1)[0][0]


def extract_foreground_logo(svg_bytes: bytes) -> tuple[Image.Image, tuple[int, int, int]]:
    png = cairosvg.svg2png(bytestring=svg_bytes, output_width=1800, output_height=1800)
    with Image.open(io.BytesIO(png)) as source:
        rgba = source.convert("RGBA")

    background = _dominant_opaque_colour(rgba)
    bg_r, bg_g, bg_b = background
    pixels = rgba.load()
    mask = Image.new("L", rgba.size, 0)
    mask_pixels = mask.load()

    for y in range(rgba.height):
        for x in range(rgba.width):
            r, g, b, a = pixels[x, y]
            if a == 0:
                continue
            distance = ((r - bg_r) ** 2 + (g - bg_g) ** 2 + (b - bg_b) ** 2) ** 0.5
            if distance > BACKGROUND_DISTANCE:
                mask_pixels[x, y] = a

    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError("could not isolate Social Democrats foreground logo shapes")

    mask = mask.crop(bbox)
    white = Image.new("RGBA", mask.size, (255, 255, 255, 0))
    white.putalpha(mask)
    return white, background


def generate() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    source, purple = extract_foreground_logo(download_svg())

    inner_w = (BOX[2] - BOX[0]) - 2 * CONTENT_MARGIN_X
    inner_h = (BOX[3] - BOX[1]) - 2 * CONTENT_MARGIN_Y
    source.thumbnail((inner_w, inner_h), Image.Resampling.LANCZOS)

    transparent = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    draw = ImageDraw.Draw(transparent)
    draw.rounded_rectangle(BOX, radius=BOX_RADIUS, outline=(255, 255, 255, 255), width=BOX_WIDTH)

    x = BOX[0] + ((BOX[2] - BOX[0]) - source.width) // 2
    y = BOX[1] + ((BOX[3] - BOX[1]) - source.height) // 2
    transparent.alpha_composite(source, (x, y))

    transparent_path = OUT / "boxed_clean.png"
    transparent.save(transparent_path, "PNG", optimize=True)

    selected = Image.new("RGBA", (CANVAS, CANVAS), (*purple, 255))
    selected.alpha_composite(transparent)
    selected_path = OUT / "boxed_purple_selected.png"
    selected.save(selected_path, "PNG", optimize=True)

    preview = Image.new("RGB", (1000, 1000), purple)
    scaled = transparent.copy()
    scaled.thumbnail((820, 820), Image.Resampling.LANCZOS)
    preview.paste(scaled, ((1000 - scaled.width) // 2, (1000 - scaled.height) // 2), scaled)
    preview_path = OUT / "boxed_purple_selected_preview.png"
    preview.save(preview_path, "PNG", optimize=True)

    report = {
        "party_key": "social-democrats",
        "candidate_id": "sd_boxed_purple_selected_v1",
        "label": "Selected purple-backed boxed Social Democrats treatment",
        "source_url": SOURCE_URL,
        "source_type": "user_selected_derived_official_treatment",
        "purple_rgb": list(purple),
        "transparent_asset": str(transparent_path.relative_to(REPO_ROOT)),
        "selected_asset": str(selected_path.relative_to(REPO_ROOT)),
        "preview": str(preview_path.relative_to(REPO_ROOT)),
        "canonical": False,
        "notes": "User selected the boxed treatment specifically on purple. Purple is taken from the dominant background colour of the clean source vector; no manifesto pixels retained.",
    }
    (OUT / "boxed_clean_review.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2))
