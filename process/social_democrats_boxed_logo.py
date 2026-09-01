#!/usr/bin/env python3
"""Generate a clean standalone Social Democrats boxed logo treatment.

The treatment is derived from the clean vector identity and reproduces the white
outlined box treatment used in official Social Democrats publications. It does not
crop pixels from a manifesto or campaign graphic and does not write to S3.
"""

from __future__ import annotations

import io
import json
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


def download_svg() -> bytes:
    response = requests.get(
        SOURCE_URL,
        timeout=35,
        allow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 EirePolitic-SocialDemocrats-Boxed/1.0"},
    )
    response.raise_for_status()
    return response.content


def whiten_logo(svg_bytes: bytes) -> Image.Image:
    png = cairosvg.svg2png(bytestring=svg_bytes, output_width=1200, output_height=1200)
    with Image.open(io.BytesIO(png)) as image:
        rgba = image.convert("RGBA")
    bbox = rgba.getbbox()
    if bbox is None:
        raise ValueError("clean vector rendered empty")
    rgba = rgba.crop(bbox)

    alpha = rgba.getchannel("A")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 0))
    white.putalpha(alpha)
    return white


def generate() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    source = whiten_logo(download_svg())

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
    drawp.rectangle((0, 0, 800, 900), fill="#6f2c91")
    drawp.rectangle((800, 0, 1600, 900), fill="#263238")
    scaled = asset.copy()
    scaled.thumbnail((720, 720), Image.Resampling.LANCZOS)
    preview.paste(scaled, (400 - scaled.width // 2, 450 - scaled.height // 2), scaled)
    preview.paste(scaled, (1200 - scaled.width // 2, 450 - scaled.height // 2), scaled)

    preview_path = OUT / "boxed_clean_preview.png"
    preview.save(preview_path, "PNG", optimize=True)

    report = {
        "party_key": "social-democrats",
        "candidate_id": "sd_boxed_clean_derived",
        "label": "Clean boxed white treatment",
        "source_url": SOURCE_URL,
        "source_type": "derived_from_clean_vector_to_match_official_publication_treatment",
        "official_publication_evidence": [
            "Social Democrats General Election Manifesto 2024",
            "Social Democrats Homes for Ireland Savings Account 2025",
        ],
        "asset": str(asset_path.relative_to(REPO_ROOT)),
        "preview": str(preview_path.relative_to(REPO_ROOT)),
        "canonical": False,
        "notes": "Standalone reconstruction of the boxed treatment; no manifesto pixels retained.",
    }
    (OUT / "boxed_clean_review.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2))
