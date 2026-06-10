from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from instagram.renderer.constants import FONT_CANDIDATES
from .common import load_palette, utc_now, write_json


def _font(kind: str, size: int) -> ImageFont.ImageFont:
    key = "bold" if kind == "bold" else "regular"
    for candidate in FONT_CANDIDATES.get(key, []):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _fit_text(draw: ImageDraw.ImageDraw, text: str, kind: str, max_size: int, min_size: int, max_width: int) -> ImageFont.ImageFont:
    for size in range(max_size, min_size - 1, -1):
        font = _font(kind, size)
        if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
            return font
    return _font(kind, min_size)


def _shorten(draw: ImageDraw.ImageDraw, text: Any, font: ImageFont.ImageFont, max_width: int) -> str:
    value = str(text or "").strip()
    if draw.textbbox((0, 0), value, font=font)[2] <= max_width:
        return value
    suffix = "…"
    while value and draw.textbbox((0, 0), value + suffix, font=font)[2] > max_width:
        value = value[:-1]
    return (value.rstrip() or "") + suffix


def _as_float(value: Any) -> tuple[float | None, bool]:
    try:
        if value is None or value == "":
            return None, False
        return float(value), True
    except Exception:
        return None, False


def _format_value(raw: Any, value_format: str) -> str:
    number, ok = _as_float(raw)
    if not ok or number is None:
        return str(raw or "—")
    if value_format == "percent":
        return f"{number:g}%"
    if value_format == "decimal":
        return f"{number:g}"
    if value_format == "currency":
        return f"€{number:,.0f}"
    if value_format == "auto":
        if abs(number) >= 1_000_000:
            return f"{number / 1_000_000:.1f}m"
        if abs(number) >= 1_000:
            return f"{number / 1_000:.1f}k"
        return f"{number:g}"
    return f"{number:,.0f}" if math.isfinite(number) else "0"


def _clean_rows(rows: list[dict[str, Any]], template: dict[str, Any], sample: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    bindings = sample.get("bindings", {}) or {}
    label_field = str(bindings.get("label", "label"))
    value_field = str(bindings.get("value", "value"))
    sublabel_field = str(bindings.get("sublabel", "sublabel"))
    params = template.get("params", {}) or {}
    max_items = int(params.get("max_items", 8))

    warnings: list[str] = []
    clean: list[dict[str, Any]] = []
    for row in rows:
        label = str(row.get(label_field, "")).strip() or "Missing label"
        value = row.get(value_field, "")
        sublabel = str(row.get(sublabel_field, "")).strip()
        if len(label) > 34:
            warnings.append(f"long_label:{label[:34]}")
        if len(sublabel) > 48:
            warnings.append(f"long_sublabel:{label[:24]}")
        _, numeric_ok = _as_float(value)
        if value not in (None, "") and not numeric_ok:
            warnings.append(f"non_numeric_value:{label[:24]}")
        clean.append({"label": label, "value": value, "sublabel": sublabel})

    if len(clean) > max_items:
        warnings.append(f"truncated_rows:{len(clean)}->{max_items}")
        clean = clean[:max_items]
    return clean, warnings


def render(
    template: dict[str, Any],
    sample: dict[str, Any],
    rows: list[dict[str, Any]],
    output_png: str | Path,
    metadata_path: str | Path,
    manifest_path: str | Path,
    input_metadata: dict[str, Any],
) -> dict[str, Any]:
    visual_id = str(sample.get("visual_id") or template.get("template_id") or "table_card_draft_v1")
    params = template.get("params", {}) or {}
    width = int(params.get("width", 1080))
    height = int(params.get("height", 860))
    columns = max(1, int(params.get("columns", 2)))
    value_format = str(params.get("value_format", "auto"))
    show_sublabel = bool(params.get("show_sublabel", True))
    palette = load_palette(template)

    clean_rows, warnings = _clean_rows(rows, template, sample)
    image = Image.new("RGB", (width, height), palette["background"])
    draw = ImageDraw.Draw(image)

    margin = 48
    gap = 22
    item_count = max(len(clean_rows), 1)
    rows_count = max(1, math.ceil(item_count / columns))
    card_w = int((width - margin * 2 - gap * (columns - 1)) / columns)
    card_h = int((height - margin * 2 - gap * (rows_count - 1)) / rows_count)

    for index, item in enumerate(clean_rows):
        row_index = index // columns
        col_index = index % columns
        x0 = margin + col_index * (card_w + gap)
        y0 = margin + row_index * (card_h + gap)
        x1 = x0 + card_w
        y1 = y0 + card_h
        fill = palette["panel_alt"] if index % 2 else palette["panel"]
        draw.rounded_rectangle((x0, y0, x1, y1), radius=24, fill=fill)

        inner_w = card_w - 52
        label_font = _fit_text(draw, item["label"], "bold", 28, 18, inner_w)
        value_text = _format_value(item["value"], value_format)
        value_font = _fit_text(draw, value_text, "bold", 54, 28, inner_w)
        sub_font = _font("regular", 19)

        label = _shorten(draw, item["label"], label_font, inner_w)
        draw.text((x0 + 26, y0 + 24), label, font=label_font, fill=palette["muted"])

        value_box = draw.textbbox((0, 0), value_text, font=value_font)
        value_y = y0 + max(72, int((card_h - (value_box[3] - value_box[1])) / 2) - 4)
        draw.text((x0 + 26, value_y), value_text, font=value_font, fill=palette["accent"])

        if show_sublabel and item["sublabel"]:
            sublabel = _shorten(draw, item["sublabel"], sub_font, inner_w)
            draw.text((x0 + 26, y1 - 46), sublabel, font=sub_font, fill=palette["text"])

    if not clean_rows:
        warnings.append("empty_rows")
        font = _font("bold", 34)
        msg = "No metrics available"
        box = draw.textbbox((0, 0), msg, font=font)
        draw.text(((width - box[2]) // 2, (height - box[3]) // 2), msg, font=font, fill=palette["muted"])

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_png, format="PNG")

    created_at = utc_now()
    metadata = {
        "visual_id": visual_id,
        "template_id": template.get("template_id"),
        "renderer": "table_card",
        "created_at": created_at,
        "input": input_metadata,
        "bindings": sample.get("bindings", {}),
        "filters": sample.get("filters", []),
        "grouping": sample.get("grouping", {}),
        "source_note": sample.get("source_note", ""),
        "attribution": sample.get("attribution", {}),
        "rows_rendered": clean_rows,
        "warnings": warnings,
    }
    manifest = {
        "success": True,
        "visual_id": visual_id,
        "template_id": template.get("template_id"),
        "renderer": "table_card",
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
