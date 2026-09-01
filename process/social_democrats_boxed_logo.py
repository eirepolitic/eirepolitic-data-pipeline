#!/usr/bin/env python3
"""Generate a clean standalone Social Democrats boxed logo treatment.

The treatment is derived from the clean square vector identity and reproduces the
white rounded-box treatment used in official Social Democrats publications. The
purple square background from the source logo is removed by colour-keying the
rendered vector's dominant background colour; only the foreground logo shapes are
retained and recoloured white. No manifesto pixels are used and nothing is written
to S3.
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
        headers={"User-Agent": "Mozilla/5.0 EirePolitic-SocialDemocrats-Boxed/1.1"},
    )
    response.raise_for_status()
    return response.content


def _dominant_opaque_colour(image: Image.Image) -> tuple[int, int, int]:
    rgba = image.convert("RGBA")
    # Sample down for speed but preserve the dominant full-canvas background.
    sample = rgba.resize((160, 160), Image.Resampling.NEAREST)
    colours = Counter((r, g, b) for r, g, b, a in sample.getdata() if a > 240)
    if not colours:
        raise ValueError("source vector has no opaque pixels")
    return colours.most_common(1)[0][0]


def extract_foreground_logo(svg_bytes: bytes) -> Image.Image:
    png = cairosvg.svg2png(bytestring=svg_bytes, output_width=1800, output_height=1800)
    with Image.open(io.BytesIO(png)) as source:
        rgba = source.convert("RGBA")

    bg_r, bg_g, bg_b = _dominant_opaque_colour(rgba)
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
    return white


def generate() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    source = extract_foreground_logo(download_svg())

    inner_w = (BOX[2] - BOX[0]) - 2 * CONTENT_MARGIN_X
    inner_h = (BOX[3] - BOX[1]) - 2 * CONTENT_MARGIN_Y
    source.thumbnail((inner_w, inner_h), Image.Resampling.LANCZOS)

    asset = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    draw = ImageDraw.Draw(asset)
    draw.rounded_rectangle(BOX, radius=BOX_RADIUS, outline=(255, 255, 255, 255), width=BOX_WIDTH)

    x = BOX[0] + ((BOX[2] - BOX[0]) - source.width) // 2
    y = BOX[1] + ((BOX[3] - BOX[1]) - source.height) // 2
    asset.alpha_composite(source, (x, y))

    asset_path = OUT / "boxed_clean.png"
    asset.save(asset_path, "PNG", optimize=True)

    preview = Image.new("RGB", (1600, 900), "white")
    drawp = ImageDraw.Draw(preview)
    drawp.rectangle((0, 0, 800, 900), fill="#752f8b")
    drawp.rectangle((800, 0, 1600, 900), fill="#263238")
    scaled = asset.copy()
    scaled.thumbnail((720, 720), Image.Resampling.LANCZOS)
    preview.paste(scaled, (400 - scaled.width // 2, 450 - scaled.height // 2), scaled)
    preview.paste(scaled, (1200 - scaled.width // 2, 450 - scaled.height // 2), scaled)

    preview_path = OUT / "boxed_clean_preview.png"
    preview.save(preview_path, "PNG", optimize=True)

    report = {
        "party_key": "social-democrats",
        "candidate_id": "sd_boxed_clean_derived_v2",
        "label": "Clean boxed white treatment — corrected foreground extraction",
        "source_url": SOURCE_URL,
        "source_type": "derived_from_clean_vector_to_match_official_publication_treatment",
        "official_publication_evidence": [
            "Social Democrats General Election Manifesto 2024",
            "Social Democrats Code of Conduct",
            "Social Democrats Emergency Winter Payment 2026",
        ],
        "asset": str(asset_path.relative_to(REPO_ROOT)),
        "preview": str(preview_path.relative_to(REPO_ROOT)),
        "canonical": False,
        "notes": "Corrected standalone reconstruction: purple source background removed; foreground Social Democrats logo shapes retained and recoloured white; no publication pixels retained.",
    }
    (OUT / "boxed_clean_review.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2))
