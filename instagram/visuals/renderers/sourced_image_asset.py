from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from instagram.renderer.constants import FONT_CANDIDATES
from .common import utc_now, write_json


def _font(kind: str, size: int) -> ImageFont.ImageFont:
    key = "bold" if kind == "bold" else "regular"
    for candidate in FONT_CANDIDATES.get(key, []):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _shorten(draw: ImageDraw.ImageDraw, text: Any, font: ImageFont.ImageFont, max_width: int) -> str:
    value = str(text or "").strip()
    if draw.textbbox((0, 0), value, font=font)[2] <= max_width:
        return value
    suffix = "…"
    while value and draw.textbbox((0, 0), value + suffix, font=font)[2] > max_width:
        value = value[:-1]
    return (value.rstrip() or "") + suffix


def _first_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return rows[0] if rows else {}


def _validate(row: dict[str, Any], required_fields: list[str]) -> list[str]:
    warnings: list[str] = []
    for field in required_fields:
        if not str(row.get(field, "")).strip():
            warnings.append(f"missing_required_field:{field}")
    source_url = str(row.get("source_url", "")).strip()
    if source_url and not source_url.startswith(("https://", "http://")):
        warnings.append("invalid_source_url")
    license_text = str(row.get("license", "")).strip().lower()
    if license_text in {"unknown", "", "n/a"}:
        warnings.append("license_not_confirmed")
    return warnings


def render(
    template: dict[str, Any],
    sample: dict[str, Any],
    rows: list[dict[str, Any]],
    output_png: str | Path,
    metadata_path: str | Path,
    manifest_path: str | Path,
    input_metadata: dict[str, Any],
) -> dict[str, Any]:
    visual_id = str(sample.get("visual_id") or template.get("template_id") or "sourced_image_asset_draft_v1")
    params = template.get("params", {}) or {}
    palette = template.get("palette", {}) or {}
    width = int(params.get("width", 1080))
    height = int(params.get("height", 860))
    required_fields = list(params.get("required_image_fields", []))
    allow_remote = bool(params.get("allow_remote_downloads", False))

    background = str(palette.get("background", "#0f2f24"))
    panel = str(palette.get("panel", "#173d30"))
    panel_alt = str(palette.get("panel_alt", "#214a3b"))
    text = str(palette.get("text", "#f4ead7"))
    muted = str(palette.get("muted", "#cbbf9f"))
    accent = str(palette.get("accent", "#d8b45f"))
    warning = str(palette.get("warning", "#b55b5b"))

    row = _first_row(rows)
    warnings = _validate(row, required_fields)
    if not rows:
        warnings.append("empty_rows")
    if allow_remote:
        warnings.append("remote_downloads_not_implemented")

    image_ref = str(row.get("image_ref", "placeholder-image")).strip() or "placeholder-image"
    source_name = str(row.get("source_name", "Unknown source")).strip() or "Unknown source"
    source_url = str(row.get("source_url", "")).strip()
    license_text = str(row.get("license", "Unknown license")).strip() or "Unknown license"
    retrieved_at = str(row.get("retrieved_at", "")).strip()
    alt_text = str(row.get("alt_text", "")).strip()
    asset_kind = str(row.get("asset_kind", "image")).strip() or "image"

    canvas = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(canvas)
    margin = 48
    panel_box = (margin, margin, width - margin, height - margin)
    draw.rounded_rectangle(panel_box, radius=28, fill=panel)

    image_box = (panel_box[0] + 34, panel_box[1] + 34, panel_box[2] - 34, panel_box[3] - 154)
    draw.rounded_rectangle(image_box, radius=22, fill=panel_alt, outline=accent, width=3)

    title_font = _font("bold", 46)
    label_font = _font("bold", 24)
    note_font = _font("regular", 18)
    small_font = _font("regular", 15)

    title = _shorten(draw, image_ref, title_font, image_box[2] - image_box[0] - 100)
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(
        (image_box[0] + ((image_box[2] - image_box[0]) - (title_box[2] - title_box[0])) / 2, image_box[1] + 210),
        title,
        font=title_font,
        fill=text,
    )
    kind_text = f"{asset_kind} asset placeholder"
    kind_box = draw.textbbox((0, 0), kind_text, font=label_font)
    draw.text(
        (image_box[0] + ((image_box[2] - image_box[0]) - (kind_box[2] - kind_box[0])) / 2, image_box[1] + 270),
        kind_text,
        font=label_font,
        fill=muted,
    )
    if alt_text:
        alt = _shorten(draw, alt_text, note_font, image_box[2] - image_box[0] - 100)
        alt_box = draw.textbbox((0, 0), alt, font=note_font)
        draw.text(
            (image_box[0] + ((image_box[2] - image_box[0]) - (alt_box[2] - alt_box[0])) / 2, image_box[1] + 315),
            alt,
            font=note_font,
            fill=muted,
        )

    metadata_y = panel_box[3] - 120
    fields = [
        ("Source", source_name),
        ("License", license_text),
        ("Retrieved", retrieved_at or "missing"),
        ("URL", source_url or "missing"),
    ]
    col_w = (panel_box[2] - panel_box[0] - 80) / 2
    for idx, (label, value) in enumerate(fields):
        col = idx % 2
        row_idx = idx // 2
        x = panel_box[0] + 40 + col * col_w
        y = metadata_y + row_idx * 42
        draw.text((x, y), f"{label}: ", font=small_font, fill=accent)
        draw.text((x + 84, y), _shorten(draw, value, small_font, int(col_w - 100)), font=small_font, fill=muted)

    if warnings:
        badge_text = f"{len(warnings)} metadata warning{'s' if len(warnings) != 1 else ''}"
        badge_box = draw.textbbox((0, 0), badge_text, font=note_font)
        bx1 = panel_box[2] - 40
        bx0 = bx1 - (badge_box[2] - badge_box[0]) - 28
        by0 = panel_box[1] + 34
        draw.rounded_rectangle((bx0, by0, bx1, by0 + 42), radius=14, fill=warning)
        draw.text((bx0 + 14, by0 + 10), badge_text, font=note_font, fill=text)

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_png, format="PNG")

    created_at = utc_now()
    metadata = {
        "visual_id": visual_id,
        "template_id": template.get("template_id"),
        "renderer": "sourced_image_asset",
        "created_at": created_at,
        "input": input_metadata,
        "bindings": sample.get("bindings", {}),
        "filters": sample.get("filters", []),
        "grouping": sample.get("grouping", {}),
        "source_note": sample.get("source_note", ""),
        "attribution": sample.get("attribution", {}),
        "asset": {
            "image_ref": image_ref,
            "asset_kind": asset_kind,
            "source_name": source_name,
            "source_url": source_url,
            "license": license_text,
            "retrieved_at": retrieved_at,
            "alt_text": alt_text,
            "remote_downloaded": False,
        },
        "rows_rendered": rows,
        "warnings": warnings,
    }
    manifest = {
        "success": True,
        "visual_id": visual_id,
        "template_id": template.get("template_id"),
        "renderer": "sourced_image_asset",
        "output_png": str(output_png),
        "metadata_path": str(metadata_path),
        "width": width,
        "height": height,
        "warnings": warnings,
        "created_at": created_at,
    }
    write_json(metadata_path, metadata)
    write_json(manifest_path, manifest)
    return manifest
