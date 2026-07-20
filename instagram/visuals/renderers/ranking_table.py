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


def _shorten(draw: ImageDraw.ImageDraw, text: Any, fnt: ImageFont.ImageFont, max_width: int) -> str:
    value = str(text or "").strip()
    if draw.textbbox((0, 0), value, font=fnt)[2] <= max_width:
        return value
    suffix = "…"
    while value and draw.textbbox((0, 0), value + suffix, font=fnt)[2] > max_width:
        value = value[:-1]
    return (value.rstrip() or "") + suffix


def _as_float(value: Any) -> tuple[float, bool]:
    try:
        if value is None or value == "":
            return 0.0, False
        return float(value), True
    except Exception:
        return 0.0, False


def _format_value(value: float, value_format: str) -> str:
    if value_format == "percent":
        return f"{value:g}%"
    if value_format == "decimal":
        return f"{value:g}"
    return f"{value:,.0f}" if math.isfinite(value) else "0"


def _clean_rows(rows: list[dict[str, Any]], template: dict[str, Any], sample: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    bindings = sample.get("bindings", {}) or {}
    name_field = str(bindings.get("name", "name"))
    sublabel_field = str(bindings.get("sublabel", "sublabel"))
    value_field = str(bindings.get("value", "value"))
    rank_field = str(bindings.get("rank", "rank"))
    params = template.get("params", {}) or {}
    max_items = int(params.get("max_items", 8))
    sort = str(params.get("sort", "descending"))

    warnings: list[str] = []
    clean: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        name = str(row.get(name_field, "")).strip() or "Missing name"
        sublabel = str(row.get(sublabel_field, "")).strip()
        value, ok = _as_float(row.get(value_field))
        rank = str(row.get(rank_field, idx)).strip() or str(idx)
        if not ok:
            warnings.append(f"non_numeric_value:{name[:30]}")
        if value < 0:
            warnings.append("negative_values_present")
        if len(name) > 44:
            warnings.append(f"long_name:{name[:44]}")
        if len(sublabel) > 42:
            warnings.append(f"long_sublabel:{name[:30]}")
        clean.append({"rank": rank, "name": name, "sublabel": sublabel, "value": value})

    if sort in {"ascending", "descending"}:
        reverse = sort != "ascending"
        clean = sorted(clean, key=lambda row: row["value"], reverse=reverse)
        for idx, row in enumerate(clean, start=1):
            row["rank"] = str(idx)

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
    visual_id = str(sample.get("visual_id") or template.get("template_id") or "ranking_table_draft_v1")
    params = template.get("params", {}) or {}
    width = int(params.get("width", 1080))
    height = int(params.get("height", 860))
    max_items = int(params.get("max_items", 8))
    value_format = str(params.get("value_format", "integer"))
    show_header = bool(params.get("show_header", True))
    palette = load_palette(template)

    clean_rows, warnings = _clean_rows(rows, template, sample)
    img = Image.new("RGB", (width, height), palette["background"])
    draw = ImageDraw.Draw(img)

    margin_x = 58
    margin_y = 58
    panel_x0 = margin_x
    panel_y0 = margin_y
    panel_x1 = width - margin_x
    panel_y1 = height - margin_y
    draw.rounded_rectangle((panel_x0, panel_y0, panel_x1, panel_y1), radius=28, fill=palette["panel"])

    header_font = _font("bold", 24)
    rank_font = _font("bold", 28)
    name_font = _font("bold", 28)
    sub_font = _font("regular", 20)
    value_font = _font("bold", 28)

    inner_x0 = panel_x0 + 36
    inner_x1 = panel_x1 - 36
    top = panel_y0 + 32
    bottom = panel_y1 - 32
    header_h = 34 if show_header else 0
    available_h = bottom - top - header_h
    row_count = max(len(clean_rows), 1)
    row_h = min(78, max(54, available_h // min(max_items, row_count)))

    rank_x = inner_x0
    name_x = inner_x0 + 86
    value_right = inner_x1
    value_col_w = 170
    name_max_w = value_right - value_col_w - name_x - 22

    y = top
    if show_header:
        draw.text((rank_x, y), str(params.get("rank_header", "#")), font=header_font, fill=palette["accent"])
        draw.text((name_x, y), str(params.get("name_header", "Name")), font=header_font, fill=palette["accent"])
        value_header = str(params.get("value_header", "Value"))
        value_header_w = draw.textbbox((0, 0), value_header, font=header_font)[2]
        draw.text((value_right - value_header_w, y), value_header, font=header_font, fill=palette["accent"])
        y += header_h

    if clean_rows:
        for idx, row in enumerate(clean_rows):
            row_y0 = y + idx * row_h
            row_y1 = row_y0 + row_h - 8
            fill = palette["panel_alt"] if idx % 2 == 0 else palette["panel"]
            draw.rounded_rectangle((inner_x0, row_y0, inner_x1, row_y1), radius=16, fill=fill)
            draw.text((rank_x + 8, row_y0 + 18), str(row["rank"]), font=rank_font, fill=palette["accent"])
            draw.text((name_x, row_y0 + 10), _shorten(draw, row["name"], name_font, name_max_w), font=name_font, fill=palette["text"])
            if row["sublabel"] and row_h >= 66:
                draw.text((name_x, row_y0 + 43), _shorten(draw, row["sublabel"], sub_font, name_max_w), font=sub_font, fill=palette["muted"])
            value_text = _format_value(float(row["value"]), value_format)
            value_w = draw.textbbox((0, 0), value_text, font=value_font)[2]
            draw.text((value_right - value_w - 8, row_y0 + 18), value_text, font=value_font, fill=palette["text"])
    else:
        warnings.append("empty_rows")
        no_data_font = _font("bold", 34)
        message = "No data available"
        box = draw.textbbox((0, 0), message, font=no_data_font)
        draw.text(((width - box[2]) // 2, (height - box[3]) // 2), message, font=no_data_font, fill=palette["muted"])

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_png, format="PNG")

    created_at = utc_now()
    metadata = {
        "visual_id": visual_id,
        "template_id": template.get("template_id"),
        "renderer": "ranking_table",
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
        "renderer": "ranking_table",
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
