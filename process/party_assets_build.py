#!/usr/bin/env python3
"""Build normalized party logo assets from reviewed local source files.

This tool does not download from external sites and does not write to S3. It converts
reviewed source artwork into deterministic PNGs, validates technical properties, and
produces a manifest plus contact sheet for human review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from process.party_assets import DEFAULT_REGISTRY, PartyAsset, load_registry

CANVAS_SIZE = 1600
SAFE_MARGIN = 160
CONTACT_CELL_W = 420
CONTACT_CELL_H = 500
CONTACT_COLUMNS = 3
SUPPORTED_RASTER = {".png", ".jpg", ".jpeg", ".webp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _source_for_party(source_root: Path, row: PartyAsset) -> Path | None:
    party_dir = source_root / row.party_key
    if not party_dir.exists():
        return None
    candidates = sorted(
        path for path in party_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_RASTER
    )
    if len(candidates) > 1:
        raise ValueError(f"{row.party_key}: expected one reviewed raster source, found {len(candidates)}")
    return candidates[0] if candidates else None


def normalize_logo(source: Path, output: Path) -> dict[str, Any]:
    with Image.open(source) as image:
        image.load()
        original_format = image.format or source.suffix.lstrip(".").upper()
        original_size = image.size
        rgba = image.convert("RGBA")

    bbox = rgba.getbbox()
    if bbox is None:
        raise ValueError(f"{source}: source image is fully transparent/empty")
    cropped = rgba.crop(bbox)

    max_side = CANVAS_SIZE - (SAFE_MARGIN * 2)
    ratio = min(max_side / cropped.width, max_side / cropped.height, 1.0 if max(cropped.size) >= max_side else max_side / max(cropped.size))
    resized = cropped.resize(
        (max(1, round(cropped.width * ratio)), max(1, round(cropped.height * ratio))),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
    x = (CANVAS_SIZE - resized.width) // 2
    y = (CANVAS_SIZE - resized.height) // 2
    canvas.alpha_composite(resized, (x, y))

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)

    alpha = canvas.getchannel("A")
    return {
        "source_format": original_format,
        "source_width": original_size[0],
        "source_height": original_size[1],
        "output_width": CANVAS_SIZE,
        "output_height": CANVAS_SIZE,
        "has_transparency": alpha.getextrema()[0] < 255,
        "content_bbox": [x, y, x + resized.width, y + resized.height],
        "sha256": sha256_file(output),
    }


def validate_normalized(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return ["missing normalized PNG"]
    try:
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG":
                errors.append(f"expected PNG, got {image.format}")
            if image.size != (CANVAS_SIZE, CANVAS_SIZE):
                errors.append(f"expected {CANVAS_SIZE}x{CANVAS_SIZE}, got {image.width}x{image.height}")
            if image.mode != "RGBA":
                errors.append(f"expected RGBA, got {image.mode}")
            if image.convert("RGBA").getbbox() is None:
                errors.append("image is empty/fully transparent")
    except Exception as exc:  # Pillow validation should report a usable error
        errors.append(f"cannot open image: {exc}")
    return errors


def build_contact_sheet(entries: list[dict[str, Any]], output: Path) -> None:
    rows = max(1, (len(entries) + CONTACT_COLUMNS - 1) // CONTACT_COLUMNS)
    sheet = Image.new("RGB", (CONTACT_COLUMNS * CONTACT_CELL_W, rows * CONTACT_CELL_H), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = _font(24, bold=True)
    meta_font = _font(18)

    for index, entry in enumerate(entries):
        col = index % CONTACT_COLUMNS
        row_idx = index // CONTACT_COLUMNS
        left = col * CONTACT_CELL_W
        top = row_idx * CONTACT_CELL_H
        draw.rectangle((left, top, left + CONTACT_CELL_W - 1, top + CONTACT_CELL_H - 1), outline="black", width=1)

        preview_box = (left + 30, top + 30, left + CONTACT_CELL_W - 30, top + 355)
        if entry.get("normalized_path") and Path(entry["normalized_path"]).is_file():
            with Image.open(entry["normalized_path"]) as logo:
                logo = logo.convert("RGBA")
                logo.thumbnail((preview_box[2] - preview_box[0], preview_box[3] - preview_box[1]), Image.Resampling.LANCZOS)
                x = preview_box[0] + ((preview_box[2] - preview_box[0]) - logo.width) // 2
                y = preview_box[1] + ((preview_box[3] - preview_box[1]) - logo.height) // 2
                sheet.paste(logo, (x, y), logo)
        else:
            draw.text((left + CONTACT_CELL_W // 2, top + 190), "NO PARTY LOGO", font=title_font, anchor="mm", fill="black")

        draw.text((left + 20, top + 382), entry["party_name"], font=title_font, fill="black")
        draw.text((left + 20, top + 420), entry["party_key"], font=meta_font, fill="black")
        draw.text((left + 20, top + 450), entry["asset_status"], font=meta_font, fill="black")

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG")


def build_assets(source_root: Path, output_root: Path, registry_path: Path) -> dict[str, Any]:
    rows = load_registry(registry_path)
    entries: list[dict[str, Any]] = []
    errors: list[str] = []

    for row in rows:
        entry: dict[str, Any] = {
            "party_key": row.party_key,
            "party_name": row.party_name,
            "asset_status": row.asset_status,
            "fallback_type": row.fallback_type,
            "source_url": row.source_url,
            "source_type": row.source_type,
            "logo_s3_uri": row.logo_s3_uri,
        }

        if row.asset_status == "approved_fallback":
            entry["build_status"] = "fallback"
            entries.append(entry)
            continue

        source = _source_for_party(source_root, row)
        if source is None:
            entry["build_status"] = "missing_source"
            errors.append(f"{row.party_key}: reviewed source file missing")
            entries.append(entry)
            continue

        party_output = output_root / "assets" / row.party_key
        source_copy = party_output / f"source{source.suffix.lower()}"
        logo_output = party_output / "logo.png"
        party_output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, source_copy)
        try:
            technical = normalize_logo(source, logo_output)
            validation_errors = validate_normalized(logo_output)
        except Exception as exc:
            entry["build_status"] = "invalid"
            entry["validation_errors"] = [str(exc)]
            errors.append(f"{row.party_key}: {exc}")
            entries.append(entry)
            continue

        entry.update({
            "build_status": "built" if not validation_errors else "invalid",
            "source_file": str(source_copy),
            "normalized_path": str(logo_output),
            "technical": technical,
            "validation_errors": validation_errors,
        })
        if validation_errors:
            errors.extend(f"{row.party_key}: {error}" for error in validation_errors)
        entries.append(entry)

    contact_sheet = output_root / "contact_sheet.png"
    build_contact_sheet(entries, contact_sheet)
    manifest = {
        "asset_spec_version": 1,
        "canvas_size": [CANVAS_SIZE, CANVAS_SIZE],
        "safe_margin_px": SAFE_MARGIN,
        "registry": str(registry_path),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "success": not errors,
        "errors": errors,
        "entries": entries,
        "contact_sheet": str(contact_sheet),
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize reviewed party logo sources and build a contact sheet")
    parser.add_argument("--source-root", required=True, help="directory containing {party_key}/<reviewed-source-image>")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    args = parser.parse_args()

    manifest = build_assets(Path(args.source_root), Path(args.output_root), Path(args.registry))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0 if manifest["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
