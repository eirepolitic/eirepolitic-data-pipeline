#!/usr/bin/env python3
"""Build the reviewed v1 party asset set for S3 publication.

Source artwork is retained under assets/{party_key}/source.*. Consumer assets are
1600x1600 white-square PNGs. Branded colour blocks/badges that were explicitly
approved in review are rendered once during this build and stored as static images;
consumers never render party-specific treatments at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import shutil
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import cairosvg
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont

from process.party_assets import load_registry, validate_registry

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "configs/reference/party_assets_v1.csv"
CANVAS = 1600
DEFAULT_SAFE = 1120
POST_GREEN = "#0f2f24"
TIMEOUT = 45
MAX_BYTES = 15 * 1024 * 1024

SPECS = {
    "100-rdr": {
        "source_url": "https://www.100percentredressparty.ie/wp-content/uploads/2026/04/100percentredressarty-Logo.webp",
        "source_ext": ".webp",
        "mode": "plain",
        "safe": 1440,
    },
    "aontu": {
        "source_url": "https://aontu.ie/wp-content/uploads/2024/12/aontu-logo.webp",
        "source_ext": ".webp",
        "mode": "plain",
        "safe": DEFAULT_SAFE,
    },
    "fianna-fail": {
        "source_url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Fianna%20F%C3%A1il%20logo%20%282024%29.svg",
        "source_ext": ".svg",
        "mode": "plain",
        "safe": DEFAULT_SAFE,
    },
    "fine-gael": {
        "source_url": "https://www.finegael.ie/app/uploads/2024/10/FG-Logo-white-text-2.png",
        "source_ext": ".png",
        "mode": "fine_gael_blue",
        "safe": DEFAULT_SAFE,
    },
    "green-party": {
        "source_url": "https://www.greenparty.ie/themes/custom/misti/logo.svg",
        "source_ext": ".svg",
        "mode": "green_party_green",
        "safe": 1510,
    },
    "independent": {
        "source_url": "",
        "source_ext": ".png",
        "mode": "independent_generated",
        "safe": DEFAULT_SAFE,
    },
    "independent-ireland": {
        "source_url": "https://www.electoralcommission.ie/app/uploads/2023/11/Independent-Ireland.jpg",
        "source_ext": ".jpg",
        "mode": "plain",
        "safe": DEFAULT_SAFE,
    },
    "labour-party": {
        "source_url": "https://labour.ie/wp-content/uploads/2021/11/Labour_RGB_Mark_Col.svg",
        "source_ext": ".svg",
        "mode": "plain",
        "safe": DEFAULT_SAFE,
    },
    "people-before-profit-solidarity": {
        "source_url": "https://www.electoralcommission.ie/app/uploads/2024/10/Solidarity.png",
        "source_ext": ".png",
        "mode": "plain",
        "safe": DEFAULT_SAFE,
    },
    "sinn-fein": {
        "source_url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Sinn%20F%C3%A9in%20wordmark.svg",
        "source_ext": ".svg",
        "mode": "plain",
        "safe": DEFAULT_SAFE,
    },
    "social-democrats": {
        "source_url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Social%20Democrats%20%28Ireland%29%20logo.svg",
        "source_ext": ".svg",
        "mode": "social_democrats_badge",
        "safe": 1510,
    },
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _download(url: str) -> bytes:
    response = requests.get(
        url,
        timeout=TIMEOUT,
        stream=True,
        allow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 EirePolitic-PartyAssets/1.0"},
    )
    response.raise_for_status()
    if urlparse(response.url).scheme != "https":
        raise ValueError(f"source redirected to non-HTTPS URL: {response.url}")
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(128 * 1024):
        if not chunk:
            continue
        total += len(chunk)
        if total > MAX_BYTES:
            raise ValueError(f"source exceeds {MAX_BYTES} bytes: {url}")
        chunks.append(chunk)
    if total == 0:
        raise ValueError(f"empty source response: {url}")
    return b"".join(chunks)


def _source_to_rgba(data: bytes, suffix: str) -> Image.Image:
    if suffix == ".svg":
        png = cairosvg.svg2png(bytestring=data, output_width=2000)
        return Image.open(io.BytesIO(png)).convert("RGBA")
    return Image.open(io.BytesIO(data)).convert("RGBA")


def _crop_alpha(image: Image.Image) -> Image.Image:
    bbox = image.getbbox()
    return image.crop(bbox) if bbox else image


def _font(size: int, bold: bool = False):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(filename, size)
    except OSError:
        return ImageFont.load_default()


def _generated_independent() -> Image.Image:
    """Approved neutral Independent v3: centered person, circle frame, no arc."""
    size = 1200
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    ink = (44, 52, 57, 255)
    cx = size // 2
    # Circular civic frame.
    draw.ellipse((250, 120, 950, 820), outline=ink, width=34)
    # Person silhouette centered in circle.
    draw.ellipse((510, 270, 690, 450), fill=ink)
    draw.rounded_rectangle((425, 470, 775, 710), radius=115, fill=ink)
    # Label, centered; no extra arc beneath the figure.
    draw.text((cx, 990), "INDEPENDENT", font=_font(92, True), fill=ink, anchor="mm")
    return _crop_alpha(image)


def _render_fine_gael(source: Image.Image) -> Image.Image:
    """Reviewed blue Fine Gael treatment using the official white logo artwork."""
    mark = _crop_alpha(source.convert("RGBA"))
    width, height = 1200, 900
    background = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    pixels = background.load()
    top = (2, 150, 200, 255)
    bottom = (0, 118, 157, 255)
    for y in range(height):
        t = y / max(1, height - 1)
        colour = tuple(round(top[i] * (1 - t) + bottom[i] * t) for i in range(4))
        for x in range(width):
            pixels[x, y] = colour
    mark.thumbnail((760, 420), Image.Resampling.LANCZOS)
    background.alpha_composite(mark, ((width - mark.width) // 2, (height - mark.height) // 2))
    return background


def _render_green_party(source: Image.Image) -> Image.Image:
    """Reviewed Green Party treatment: official white artwork on party green."""
    mark = _crop_alpha(source.convert("RGBA"))
    width, height = 1400, 760
    background = Image.new("RGBA", (width, height), (47, 182, 106, 255))
    mark.thumbnail((1280, 650), Image.Resampling.LANCZOS)
    background.alpha_composite(mark, ((width - mark.width) // 2, (height - mark.height) // 2))
    return background


def _white_art_mask(source: Image.Image) -> Image.Image:
    """Extract white wordmark pixels from a rasterised source logo."""
    source = source.convert("RGBA")
    alpha = Image.new("L", source.size, 0)
    src = source.load()
    dst = alpha.load()
    for y in range(source.height):
        for x in range(source.width):
            r, g, b, a = src[x, y]
            brightness = min(r, g, b)
            if a and brightness >= 170:
                dst[x, y] = min(a, max(0, (brightness - 155) * 3))
    return alpha


def _render_social_democrats(source: Image.Image) -> Image.Image:
    """Reviewed Social Democrats badge: purple badge, white border, no black canvas."""
    purple = (93, 39, 133, 255)
    white = (255, 255, 255, 255)
    width, height = 1500, 720
    badge = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(badge)
    border = 12
    radius = 85
    # Outer silhouette with rounded top-left and bottom-right only.
    outer = Image.new("L", (width, height), 0)
    od = ImageDraw.Draw(outer)
    od.rectangle((0, 0, width - 1, height - 1), fill=255)
    od.rectangle((0, 0, radius * 2, radius * 2), fill=0)
    od.pieslice((0, 0, radius * 2, radius * 2), 180, 270, fill=255)
    od.rectangle((width - radius * 2 - 1, height - radius * 2 - 1, width - 1, height - 1), fill=0)
    od.pieslice((width - radius * 2 - 1, height - radius * 2 - 1, width - 1, height - 1), 0, 90, fill=255)
    white_layer = Image.new("RGBA", (width, height), white)
    badge.alpha_composite(Image.composite(white_layer, Image.new("RGBA", (width, height)), outer))
    # Purple inset follows the same asymmetric geometry.
    inner_w, inner_h = width - border * 2, height - border * 2
    inner = Image.new("L", (inner_w, inner_h), 0)
    idraw = ImageDraw.Draw(inner)
    inner_r = radius - border
    idraw.rectangle((0, 0, inner_w - 1, inner_h - 1), fill=255)
    idraw.rectangle((0, 0, inner_r * 2, inner_r * 2), fill=0)
    idraw.pieslice((0, 0, inner_r * 2, inner_r * 2), 180, 270, fill=255)
    idraw.rectangle((inner_w - inner_r * 2 - 1, inner_h - inner_r * 2 - 1, inner_w - 1, inner_h - 1), fill=0)
    idraw.pieslice((inner_w - inner_r * 2 - 1, inner_h - inner_r * 2 - 1, inner_w - 1, inner_h - 1), 0, 90, fill=255)
    purple_layer = Image.new("RGBA", (inner_w, inner_h), purple)
    inset = Image.composite(purple_layer, Image.new("RGBA", (inner_w, inner_h)), inner)
    badge.alpha_composite(inset, (border, border))
    # Prefer the official source wordmark; fall back to text only if extraction is empty.
    mask = _white_art_mask(source)
    bbox = mask.getbbox()
    if bbox:
        mask = mask.crop(bbox)
        scale = min(1320 / mask.width, 560 / mask.height)
        mask = mask.resize((max(1, round(mask.width * scale)), max(1, round(mask.height * scale))), Image.Resampling.LANCZOS)
        text_layer = Image.new("RGBA", mask.size, white)
        badge.alpha_composite(Image.composite(text_layer, Image.new("RGBA", mask.size), mask), (75, (height - mask.height) // 2))
    else:
        draw.text((80, 210), "Social", font=_font(180, True), fill=white)
        draw.text((80, 440), "Democrats", font=_font(180, True), fill=white)
    return badge


def _render_mode(source: Image.Image | None, mode: str) -> Image.Image:
    if mode == "independent_generated":
        return _generated_independent()
    if source is None:
        raise ValueError(f"mode {mode} requires a source image")
    if mode == "fine_gael_blue":
        return _render_fine_gael(source)
    if mode == "green_party_green":
        return _render_green_party(source)
    if mode == "social_democrats_badge":
        return _render_social_democrats(source)
    return _crop_alpha(source.convert("RGBA"))


def _square(cleaned: Image.Image, safe: int) -> Image.Image:
    cleaned = _crop_alpha(cleaned.convert("RGBA"))
    scale = min(safe / cleaned.width, safe / cleaned.height)
    size = (max(1, round(cleaned.width * scale)), max(1, round(cleaned.height * scale)))
    resized = cleaned.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (CANVAS, CANVAS), "white")
    canvas.alpha_composite(resized, ((CANVAS - resized.width) // 2, (CANVAS - resized.height) // 2))
    return canvas.convert("RGB")


def _contact_sheet(rows, build_root: Path, green: bool = False) -> Image.Image:
    cols = 4
    cell_w, cell_h = 430, 450
    rows_n = math.ceil(len(rows) / cols)
    background = POST_GREEN if green else "#f2f2f2"
    sheet = Image.new("RGB", (cols * cell_w, rows_n * cell_h + 55), background)
    draw = ImageDraw.Draw(sheet)
    ink = "white" if green else "#111111"
    title = "Party assets v1 — circle comparison" if green else "Party assets v1 — white squares"
    draw.text((24, 14), title, font=_font(26, True), fill=ink)
    for idx, row in enumerate(rows):
        r, c = divmod(idx, cols)
        left = c * cell_w
        top = 55 + r * cell_h
        square = Image.open(build_root / "assets" / row.party_key / "display_square.png").convert("RGB")
        if green:
            circle_d = 300
            mask = Image.new("L", (circle_d, circle_d), 0)
            ImageDraw.Draw(mask).ellipse((0, 0, circle_d - 1, circle_d - 1), fill=255)
            badge = Image.new("RGB", (circle_d, circle_d), "white")
            square = square.resize((234, 234), Image.Resampling.LANCZOS)
            badge.paste(square, ((circle_d - 234) // 2, (circle_d - 234) // 2))
            sheet.paste(badge, (left + (cell_w - circle_d) // 2, top + 18), mask)
        else:
            square = square.resize((320, 320), Image.Resampling.LANCZOS)
            sheet.paste(square, (left + (cell_w - 320) // 2, top + 18))
        bbox = draw.textbbox((0, 0), row.party_name, font=_font(19, True))
        draw.text((left + (cell_w - (bbox[2] - bbox[0])) // 2, top + 350), row.party_name, font=_font(19, True), fill=ink)
    return sheet


def build(output_root: Path) -> dict:
    registry_rows = load_registry(REGISTRY)
    errors = validate_registry(registry_rows)
    if errors:
        raise ValueError("registry validation failed: " + "; ".join(errors))
    if {row.party_key for row in registry_rows} != set(SPECS):
        raise ValueError("registry party keys do not exactly match finalizer specs")
    if output_root.exists():
        shutil.rmtree(output_root)
    (output_root / "assets").mkdir(parents=True)

    manifest_rows = []
    for row in registry_rows:
        spec = SPECS[row.party_key]
        party_dir = output_root / "assets" / row.party_key
        party_dir.mkdir(parents=True)
        if spec["mode"] == "independent_generated":
            cleaned = _generated_independent()
            source_io = io.BytesIO()
            cleaned.save(source_io, "PNG")
            data = source_io.getvalue()
            source = cleaned
        else:
            data = _download(spec["source_url"])
            source = _source_to_rgba(data, spec["source_ext"])
            cleaned = _render_mode(source, spec["mode"])
        source_path = party_dir / f"source{spec['source_ext']}"
        source_path.write_bytes(data)
        clean_path = party_dir / "logo_clean.png"
        cleaned.save(clean_path, "PNG", optimize=True)
        square = _square(cleaned, spec["safe"])
        display_path = party_dir / "display_square.png"
        logo_path = party_dir / "logo.png"
        square.save(display_path, "PNG", optimize=True)
        square.save(logo_path, "PNG", optimize=True)
        manifest_rows.append({
            "party_key": row.party_key,
            "party_name": row.party_name,
            "source_url": spec.get("source_url", ""),
            "source_type": row.source_type,
            "source_file": source_path.relative_to(output_root).as_posix(),
            "source_sha256": sha256_bytes(data),
            "logo_clean_file": clean_path.relative_to(output_root).as_posix(),
            "logo_clean_sha256": sha256_bytes(clean_path.read_bytes()),
            "display_square_file": display_path.relative_to(output_root).as_posix(),
            "display_square_sha256": sha256_bytes(display_path.read_bytes()),
            "consumer_logo_file": logo_path.relative_to(output_root).as_posix(),
            "consumer_logo_sha256": sha256_bytes(logo_path.read_bytes()),
            "canvas": [CANVAS, CANVAS],
            "safe_dimension": spec["safe"],
            "render_mode": spec["mode"],
        })

    shutil.copy2(REGISTRY, output_root / "party_assets.csv")
    pd.read_csv(REGISTRY).to_parquet(output_root / "party_assets.parquet", index=False)
    _contact_sheet(registry_rows, output_root, green=False).save(output_root / "contact_sheet.png", "PNG", optimize=True)
    _contact_sheet(registry_rows, output_root, green=True).save(output_root / "contact_sheet_green.png", "PNG", optimize=True)
    manifest = {
        "version": "v1",
        "generated_on": date.today().isoformat(),
        "party_count": len(manifest_rows),
        "success": len(manifest_rows) == 11,
        "storage_contract": {
            "source": "assets/{party_key}/source.*",
            "clean_master": "assets/{party_key}/logo_clean.png",
            "consumer_logo": "assets/{party_key}/logo.png",
            "display_square": "assets/{party_key}/display_square.png",
            "consumer_canvas": [CANVAS, CANVAS],
            "consumer_background": "white",
        },
        "review_green": POST_GREEN,
        "parties": manifest_rows,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the reviewed party asset v1 publication set")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    manifest = build(Path(args.output_root))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0 if manifest["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
