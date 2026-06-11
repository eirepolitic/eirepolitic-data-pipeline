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


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _shorten(draw: ImageDraw.ImageDraw, text: Any, font: ImageFont.ImageFont, max_width: int) -> str:
    value = str(text or "").strip()
    if draw.textbbox((0, 0), value, font=font)[2] <= max_width:
        return value
    suffix = "…"
    while value and draw.textbbox((0, 0), value + suffix, font=font)[2] > max_width:
        value = value[:-1]
    return (value.rstrip() or "") + suffix


def _format_value(value: float, value_format: str) -> str:
    if value_format == "percent":
        return f"{value:g}%"
    if value_format == "decimal":
        return f"{value:g}"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}m"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:,.0f}" if math.isfinite(value) else "0"


def _category_color(category: str, order: list[str], palette: dict[str, str]) -> str:
    colors = [palette["accent"], palette["accent_2"], palette["text"], palette["warning"], palette["muted"]]
    try:
        return colors[order.index(category) % len(colors)]
    except ValueError:
        return palette["accent"]


def _clean_rows(rows: list[dict[str, Any]], template: dict[str, Any], sample: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    bindings = sample.get("bindings", {}) or {}
    label_field = str(bindings.get("label", "label"))
    value_field = str(bindings.get("value", "value"))
    category_field = str(bindings.get("category", "category"))
    params = template.get("params", {}) or {}
    max_items = int(params.get("max_items", 12))
    sort_descending = bool(params.get("sort_descending", True))

    warnings: list[str] = []
    clean: list[dict[str, Any]] = []
    for row in rows:
        label = str(row.get(label_field, "")).strip() or "Missing label"
        category = str(row.get(category_field, "Default")).strip() or "Default"
        value = _as_float(row.get(value_field))
        if value is None:
            warnings.append(f"missing_or_invalid_value:{label[:24]}")
            continue
        if value < 0:
            warnings.append("negative_values_present")
        if len(label) > 34:
            warnings.append(f"long_label:{label[:34]}")
        clean.append({"label": label, "value": value, "category": category})

    clean = sorted(clean, key=lambda item: item["value"], reverse=sort_descending)
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
    visual_id = str(sample.get("visual_id") or template.get("template_id") or "dot_plot_draft_v1")
    params = template.get("params", {}) or {}
    width = int(params.get("width", 1080))
    height = int(params.get("height", 860))
    min_radius = float(params.get("min_radius", 10))
    max_radius = float(params.get("max_radius", 18))
    show_values = bool(params.get("show_values", True))
    show_legend = bool(params.get("show_legend", True))
    value_format = str(params.get("value_format", "integer"))
    palette = load_palette(template)

    clean_rows, warnings = _clean_rows(rows, template, sample)
    values = [float(item["value"]) for item in clean_rows]
    min_value = min(values + [0.0])
    max_value = max(values + [1.0])
    span = max(max_value - min_value, 1.0)
    category_order: list[str] = []
    for item in clean_rows:
        if item["category"] not in category_order:
            category_order.append(item["category"])

    image = Image.new("RGB", (width, height), palette["background"])
    draw = ImageDraw.Draw(image)
    margin = 48
    legend_h = 58 if show_legend and len(category_order) > 1 else 0
    panel_box = (margin, margin, width - margin, height - margin - legend_h)
    draw.rounded_rectangle(panel_box, radius=28, fill=palette["panel"])

    label_font = _font("bold", 24)
    value_font = _font("regular", 18)
    legend_font = _font("regular", 16)
    axis_font = _font("regular", 16)

    if clean_rows:
        label_w = 330
        plot_x0 = panel_box[0] + label_w + 36
        plot_x1 = panel_box[2] - 110
        plot_w = max(1, plot_x1 - plot_x0)
        top = panel_box[1] + 42
        bottom = panel_box[3] - 42
        row_gap = (bottom - top) / max(len(clean_rows) - 1, 1)

        for grid_index in range(5):
            ratio = grid_index / 4
            x = plot_x0 + ratio * plot_w
            draw.line((x, top - 18, x, bottom + 18), fill=palette["grid"], width=1)
        draw.text((plot_x0, bottom + 24), _format_value(min_value, value_format), font=axis_font, fill=palette["muted"])
        max_label = _format_value(max_value, value_format)
        max_box = draw.textbbox((0, 0), max_label, font=axis_font)
        draw.text((plot_x1 - (max_box[2] - max_box[0]), bottom + 24), max_label, font=axis_font, fill=palette["muted"])

        for index, item in enumerate(clean_rows):
            y = top + index * row_gap
            value = float(item["value"])
            ratio = (value - min_value) / span
            x = plot_x0 + ratio * plot_w
            radius = min_radius + ratio * (max_radius - min_radius)
            label = _shorten(draw, item["label"], label_font, label_w - 28)
            draw.text((panel_box[0] + 24, y - 14), label, font=label_font, fill=palette["text"])
            draw.line((plot_x0, y, x, y), fill=palette["grid"], width=2)
            color = _category_color(str(item["category"]), category_order, palette)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=palette["text"], width=2)
            if show_values:
                value_text = _format_value(value, value_format)
                draw.text((x + radius + 10, y - 10), value_text, font=value_font, fill=palette["muted"])
    else:
        warnings.append("empty_rows")
        font = _font("bold", 34)
        msg = "No values available"
        box = draw.textbbox((0, 0), msg, font=font)
        draw.text(((width - box[2]) // 2, (height - box[3]) // 2), msg, font=font, fill=palette["muted"])

    if show_legend and len(category_order) > 1:
        x = margin + 70
        y = height - margin - 35
        for category in category_order[:5]:
            color = _category_color(category, category_order, palette)
            draw.ellipse((x, y, x + 16, y + 16), fill=color, outline=palette["text"], width=1)
            draw.text((x + 24, y - 2), category, font=legend_font, fill=palette["muted"])
            x += 190

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_png, format="PNG")

    created_at = utc_now()
    metadata = {
        "visual_id": visual_id,
        "template_id": template.get("template_id"),
        "renderer": "dot_plot",
        "created_at": created_at,
        "input": input_metadata,
        "bindings": sample.get("bindings", {}),
        "filters": sample.get("filters", []),
        "grouping": sample.get("grouping", {}),
        "source_note": sample.get("source_note", ""),
        "attribution": sample.get("attribution", {}),
        "item_count": len(clean_rows),
        "category_count": len(category_order),
        "value_range": {"min": min_value, "max": max_value},
        "rows_rendered": clean_rows,
        "warnings": warnings,
    }
    manifest = {
        "success": True,
        "visual_id": visual_id,
        "template_id": template.get("template_id"),
        "renderer": "dot_plot",
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
