#!/usr/bin/env python3
"""Build the reviewed v1 party asset set for S3 publication.

The build preserves selected sources, creates cleaned masters, and writes a common
1600x1600 white-square PNG used by consumers. The green-circle sheet is review-only.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import math
import shutil
from collections import deque
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
EMBEDDED_ROOT = REPO_ROOT / "configs/reference/party_asset_sources_v1"
CANVAS = 1600
DEFAULT_SAFE = 1120
POST_GREEN = "#0f2f24"
TIMEOUT = 45
MAX_BYTES = 15 * 1024 * 1024

# Exact reviewed source choices. Four user-supplied/generated files are staged from
# base64 text so GitHub Actions can reproduce the exact binaries selected in chat.
SPECS = {
    "100-rdr": {
        "source_url": "https://www.100percentredressparty.ie/wp-content/uploads/2026/04/100percentredressarty-Logo.webp",
        "source_ext": ".webp",
        "cleanup": "keep",
        "safe": 1440,
    },
    "aontu": {
        "source_url": "https://aontu.ie/wp-content/uploads/2024/12/aontu-logo.webp",
        "source_ext": ".webp",
        "cleanup": "keep",
        "safe": DEFAULT_SAFE,
    },
    "fianna-fail": {
        "source_url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Fianna%20F%C3%A1il%20logo%20%282024%29.svg",
        "source_ext": ".svg",
        "cleanup": "keep",
        "safe": DEFAULT_SAFE,
    },
    "fine-gael": {
        "embedded": "fine-gael.jpg.b64",
        "source_ext": ".jpg",
        "sha256": "d8e337db36b901e7e631c7c296c8af928ddea9c7f18cff62cf379aed95d1793b",
        "cleanup": "keep",
        "safe": DEFAULT_SAFE,
    },
    "green-party": {
        "embedded": "green-party.jpg.b64",
        "source_ext": ".jpg",
        "sha256": "3fef584615d1572494bdf52247115c1d2932c1f73473afaa6f71efa95bb15a3d",
        "cleanup": "keep",
        "safe": 1510,
    },
    "independent": {
        "embedded": "independent.png.b64",
        "source_ext": ".png",
        "sha256": "0c9416fa6b97341a4cfae55edbf91d219835fdb95bef006fe54913ea0ec1a77f",
        "cleanup": "keep",
        "safe": DEFAULT_SAFE,
    },
    "independent-ireland": {
        "source_url": "https://www.electoralcommission.ie/app/uploads/2023/11/Independent-Ireland.jpg",
        "source_ext": ".jpg",
        "cleanup": "keep",
        "safe": DEFAULT_SAFE,
    },
    "labour-party": {
        "source_url": "https://labour.ie/wp-content/uploads/2021/11/Labour_RGB_Mark_Col.svg",
        "source_ext": ".svg",
        "cleanup": "keep",
        "safe": DEFAULT_SAFE,
    },
    "people-before-profit-solidarity": {
        "source_url": "https://www.electoralcommission.ie/app/uploads/2024/10/Solidarity.png",
        "source_ext": ".png",
        "cleanup": "keep",
        "safe": DEFAULT_SAFE,
    },
    "sinn-fein": {
        "source_url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Sinn%20F%C3%A9in%20wordmark.svg",
        "source_ext": ".svg",
        "cleanup": "keep",
        "safe": DEFAULT_SAFE,
    },
    "social-democrats": {
        "embedded": "social-democrats.png.b64",
        "source_ext": ".png",
        "sha256": "09c82b36f4174fccbd725e38ccf9b67100657698c611a8693c39a07fcd1b8354",
        "cleanup": "edge_black_white",
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


def _load_source_bytes(party_key: str, spec: dict) -> bytes:
    if spec.get("embedded"):
        path = EMBEDDED_ROOT / spec["embedded"]
        encoded = "".join(path.read_text(encoding="ascii").split())
        data = base64.b64decode(encoded, validate=True)
    else:
        data = _download(spec["source_url"])
    expected = spec.get("sha256")
    actual = sha256_bytes(data)
    if expected and actual != expected:
        raise ValueError(f"{party_key}: embedded source checksum mismatch: {actual}")
    return data


def _source_to_rgba(data: bytes, suffix: str) -> Image.Image:
    if suffix == ".svg":
        png = cairosvg.svg2png(bytestring=data, output_width=2000, output_height=2000)
        return Image.open(io.BytesIO(png)).convert("RGBA")
    return Image.open(io.BytesIO(data)).convert("RGBA")


def _crop_alpha(image: Image.Image) -> Image.Image:
    bbox = image.getbbox()
    return image.crop(bbox) if bbox else image


def _remove_edge_black_white(image: Image.Image, white_threshold: int = 244, black_threshold: int = 24) -> Image.Image:
    """Remove only near-white/near-black pixels connected to the outer canvas.

    This preserves intrinsic white artwork and the Social Democrats purple badge while
    removing the black/white publication background surrounding the approved badge.
    """
    image = image.convert("RGBA")
    width, height = image.size
    pixels = image.load()
    visited = bytearray(width * height)
    queue: deque[tuple[int, int]] = deque()

    def index(x: int, y: int) -> int:
        return y * width + x

    def is_background(x: int, y: int) -> bool:
        r, g, b, a = pixels[x, y]
        if a == 0:
            return True
        near_white = r >= white_threshold and g >= white_threshold and b >= white_threshold
        near_black = r <= black_threshold and g <= black_threshold and b <= black_threshold
        return near_white or near_black

    def seed(x: int, y: int) -> None:
        idx = index(x, y)
        if not visited[idx] and is_background(x, y):
            visited[idx] = 1
            queue.append((x, y))

    for x in range(width):
        seed(x, 0)
        seed(x, height - 1)
    for y in range(height):
        seed(0, y)
        seed(width - 1, y)

    for_x_y = ((1, 0), (-1, 0), (0, 1), (0, -1))
    while queue:
        x, y = queue.popleft()
        for dx, dy in for_x_y:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                idx = index(nx, ny)
                if not visited[idx] and is_background(nx, ny):
                    visited[idx] = 1
                    queue.append((nx, ny))

    cleaned = image.copy()
    output = cleaned.load()
    for y in range(height):
        row_offset = y * width
        for x in range(width):
            if visited[row_offset + x]:
                output[x, y] = (0, 0, 0, 0)
    return _crop_alpha(cleaned)


def _clean(image: Image.Image, method: str) -> Image.Image:
    if method == "edge_black_white":
        return _remove_edge_black_white(image)
    return _crop_alpha(image)


def _square(cleaned: Image.Image, safe: int) -> Image.Image:
    cleaned = _crop_alpha(cleaned.convert("RGBA"))
    scale = min(safe / cleaned.width, safe / cleaned.height)
    size = (max(1, round(cleaned.width * scale)), max(1, round(cleaned.height * scale)))
    resized = cleaned.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (CANVAS, CANVAS), "white")
    x = (CANVAS - resized.width) // 2
    y = (CANVAS - resized.height) // 2
    canvas.alpha_composite(resized, (x, y))
    return canvas.convert("RGB")


def _font(size: int, bold: bool = False):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(filename, size)
    except OSError:
        return ImageFont.load_default()


def _contact_sheet(rows, build_root: Path, green: bool = False) -> Image.Image:
    cols = 4
    cell_w, cell_h = 430, 470
    rows_n = math.ceil(len(rows) / cols)
    background = POST_GREEN if green else "#f2f2f2"
    sheet = Image.new("RGB", (cols * cell_w, rows_n * cell_h + 55), background)
    draw = ImageDraw.Draw(sheet)
    title_fill = "white" if green else "#111111"
    label_fill = title_fill
    draw.text((24, 14), "Party assets v1" + (" — circle comparison" if green else " — white squares"), font=_font(26, True), fill=title_fill)

    for idx, row in enumerate(rows):
        r, c = divmod(idx, cols)
        left = c * cell_w
        top = 55 + r * cell_h
        square = Image.open(build_root / "assets" / row.party_key / "display_square.png").convert("RGB")
        if green:
            circle_d = 300
            circle = Image.new("RGB", (circle_d, circle_d), "white")
            mask = Image.new("L", (circle_d, circle_d), 0)
            ImageDraw.Draw(mask).ellipse((0, 0, circle_d - 1, circle_d - 1), fill=255)
            square = square.resize((234, 234), Image.Resampling.LANCZOS)
            badge = Image.new("RGB", (circle_d, circle_d), "white")
            badge.paste(square, ((circle_d - 234) // 2, (circle_d - 234) // 2))
            x = left + (cell_w - circle_d) // 2
            y = top + 18
            sheet.paste(badge, (x, y), mask)
        else:
            square = square.resize((320, 320), Image.Resampling.LANCZOS)
            x = left + (cell_w - 320) // 2
            y = top + 18
            sheet.paste(square, (x, y))
        label = row.party_name
        bbox = draw.textbbox((0, 0), label, font=_font(20, True))
        draw.text((left + (cell_w - (bbox[2] - bbox[0])) // 2, top + 350), label, font=_font(20, True), fill=label_fill)
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
        data = _load_source_bytes(row.party_key, spec)
        party_dir = output_root / "assets" / row.party_key
        party_dir.mkdir(parents=True)
        source_path = party_dir / f"source{spec['source_ext']}"
        source_path.write_bytes(data)

        original = _source_to_rgba(data, spec["source_ext"])
        cleaned = _clean(original, spec["cleanup"])
        clean_path = party_dir / "logo_clean.png"
        cleaned.save(clean_path, "PNG")

        square = _square(cleaned, spec["safe"])
        display_path = party_dir / "display_square.png"
        logo_path = party_dir / "logo.png"
        square.save(display_path, "PNG", optimize=True)
        square.save(logo_path, "PNG", optimize=True)

        manifest_rows.append({
            "party_key": row.party_key,
            "party_name": row.party_name,
            "source_url": spec.get("source_url", row.source_url),
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
        })

    shutil.copy2(REGISTRY, output_root / "party_assets.csv")
    frame = pd.read_csv(REGISTRY)
    frame.to_parquet(output_root / "party_assets.parquet", index=False)

    white_sheet = _contact_sheet(registry_rows, output_root, green=False)
    white_sheet.save(output_root / "contact_sheet.png", "PNG", optimize=True)
    green_sheet = _contact_sheet(registry_rows, output_root, green=True)
    green_sheet.save(output_root / "contact_sheet_green.png", "PNG", optimize=True)

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
