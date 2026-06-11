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


def _colors(palette: dict[str, str]) -> list[str]:
    return [
        palette["accent"],
        palette["accent_2"],
        palette["text"],
        palette["warning"],
        palette["muted"],
        "#6fa88d",
        "#b99b52",
        "#e2d5b8",
        "#7c5c43",
        "#4f7f68",
    ]


def _clean_rows(rows: list[dict[str, Any]], template: dict[str, Any], sample: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    bindings = sample.get("bindings", {}) or {}
    label_field = str(bindings.get("label", "label"))
    value_field = str(bindings.get("value", "value"))
    max_slices = int((template.get("params", {}) or {}).get("max_slices", 10))

    warnings: list[str] = []
    grouped: dict[str, float] = {}
    for row in rows:
        label = str(row.get(label_field, "")).strip() or "Missing label"
        value = _as_float(row.get(value_field))
        if value is None:
            warnings.append(f"missing_or_invalid_value:{label[:24]}")
            continue
        if value < 0:
            warnings.append("negative_values_present")
            continue
        if len(label) > 34:
            warnings.append(f"long_label:{label[:34]}")
        grouped[label] = grouped.get(label, 0.0) + value

    clean = [{"label": label, "value": value} for label, value in grouped.items() if value > 0]
    clean = sorted(clean, key=lambda item: item["value"], reverse=True)
    if len(clean) > max_slices:
        kept = clean[: max_slices - 1]
        other_value = sum(float(item["value"]) for item in clean[max_slices - 1 :])
        warnings.append(f"aggregated_other:{len(clean)}->{max_slices}")
        clean = kept + [{"label": "Other", "value": other_value}]
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
    visual_id = str(sample.get("visual_id") or template.get("template_id") or "donut_chart_draft_v1")
    params = template.get("params", {}) or {}
    width = int(params.get("width", 1080))
    height = int(params.get("height", 860))
    donut_width_ratio = float(params.get("donut_width_ratio", 0.34))
    show_percentages = bool(params.get("show_percentages", True))
    show_values = bool(params.get("show_values", True))
    value_format = str(params.get("value_format", "integer"))
    palette = load_palette(template)

    clean_rows, warnings = _clean_rows(rows, template, sample)
    total = sum(float(item["value"]) for item in clean_rows)

    image = Image.new("RGB", (width, height), palette["background"])
    draw = ImageDraw.Draw(image)
    margin = 48
    panel_box = (margin, margin, width - margin, height - margin)
    draw.rounded_rectangle(panel_box, radius=28, fill=palette["panel"])

    label_font = _font("bold", 22)
    detail_font = _font("regular", 18)
    center_font = _font("bold", 42)
    center_small = _font("regular", 18)

    cx = panel_box[0] + 330
    cy = (panel_box[1] + panel_box[3]) // 2
    outer_r = 250
    inner_r = int(outer_r * (1 - donut_width_ratio))
    box = (cx - outer_r, cy - outer_r, cx + outer_r, cy + outer_r)

    if clean_rows and total > 0:
        start = -90.0
        palette_colors = _colors(palette)
        rendered_rows: list[dict[str, Any]] = []
        for idx, item in enumerate(clean_rows):
            value = float(item["value"])
            percent = value / total
            extent = percent * 360.0
            color = palette_colors[idx % len(palette_colors)]
            draw.pieslice(box, start=start, end=start + extent, fill=color, outline=palette["background"], width=2)
            rendered_rows.append({**item, "percent": percent})
            start += extent

        draw.ellipse((cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r), fill=palette["panel"])
        total_text = _format_value(total, value_format)
        total_box = draw.textbbox((0, 0), total_text, font=center_font)
        draw.text((cx - (total_box[2] - total_box[0]) / 2, cy - 34), total_text, font=center_font, fill=palette["text"])
        sub = "total"
        sub_box = draw.textbbox((0, 0), sub, font=center_small)
        draw.text((cx - (sub_box[2] - sub_box[0]) / 2, cy + 14), sub, font=center_small, fill=palette["muted"])

        legend_x = panel_box[0] + 625
        legend_y = panel_box[1] + 72
        row_h = 58
        for idx, item in enumerate(rendered_rows):
            y = legend_y + idx * row_h
            color = palette_colors[idx % len(palette_colors)]
            draw.rounded_rectangle((legend_x, y + 6, legend_x + 26, y + 32), radius=6, fill=color)
            label = _shorten(draw, item["label"], label_font, 250)
            draw.text((legend_x + 40, y), label, font=label_font, fill=palette["text"])
            detail_parts = []
            if show_percentages:
                detail_parts.append(f"{item['percent'] * 100:.1f}%")
            if show_values:
                detail_parts.append(_format_value(float(item["value"]), value_format))
            draw.text((legend_x + 40, y + 28), " · ".join(detail_parts), font=detail_font, fill=palette["muted"])
    else:
        warnings.append("empty_rows")
        font = _font("bold", 34)
        msg = "No positive values available"
        msg_box = draw.textbbox((0, 0), msg, font=font)
        draw.text(((width - msg_box[2]) // 2, (height - msg_box[3]) // 2), msg, font=font, fill=palette["muted"])
        rendered_rows = []

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_png, format="PNG")

    created_at = utc_now()
    metadata = {
        "visual_id": visual_id,
        "template_id": template.get("template_id"),
        "renderer": "donut_chart",
        "created_at": created_at,
        "input": input_metadata,
        "bindings": sample.get("bindings", {}),
        "filters": sample.get("filters", []),
        "grouping": sample.get("grouping", {}),
        "source_note": sample.get("source_note", ""),
        "attribution": sample.get("attribution", {}),
        "slice_count": len(clean_rows),
        "total": total,
        "rows_rendered": clean_rows,
        "warnings": warnings,
    }
    manifest = {
        "success": True,
        "visual_id": visual_id,
        "template_id": template.get("template_id"),
        "renderer": "donut_chart",
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
