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


def _change_text(change: float, value_format: str) -> str:
    prefix = "+" if change > 0 else ""
    return f"{prefix}{_format_value(change, value_format)}"


def _clean_rows(rows: list[dict[str, Any]], template: dict[str, Any], sample: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str], tuple[str, str]]:
    bindings = sample.get("bindings", {}) or {}
    label_field = str(bindings.get("label", "label"))
    period_field = str(bindings.get("period", "period"))
    value_field = str(bindings.get("value", "value"))
    params = template.get("params", {}) or {}
    max_items = int(params.get("max_items", 10))
    sort_by_change_abs = bool(params.get("sort_by_change_abs", True))

    warnings: list[str] = []
    periods: list[str] = []
    by_label: dict[str, dict[str, float]] = {}
    for row in rows:
        label = str(row.get(label_field, "")).strip() or "Missing label"
        period = str(row.get(period_field, "")).strip()
        value = _as_float(row.get(value_field))
        if not period or value is None:
            warnings.append(f"missing_or_invalid_point:{label[:24]}")
            continue
        if len(label) > 34:
            warnings.append(f"long_label:{label[:34]}")
        if value < 0:
            warnings.append("negative_values_present")
        if period not in periods:
            periods.append(period)
        by_label.setdefault(label, {})[period] = value

    if len(periods) < 2:
        warnings.append("fewer_than_two_periods")
        periods = (periods + ["Start", "End"])[:2]
    if len(periods) > 2:
        warnings.append(f"truncated_periods:{len(periods)}->2")
        periods = periods[:2]
    start_period, end_period = periods[0], periods[1]

    clean: list[dict[str, Any]] = []
    for label, values in by_label.items():
        if start_period not in values or end_period not in values:
            warnings.append(f"missing_endpoint:{label[:24]}")
            continue
        start = float(values[start_period])
        end = float(values[end_period])
        clean.append({"label": label, "start": start, "end": end, "change": end - start})

    if sort_by_change_abs:
        clean = sorted(clean, key=lambda item: abs(float(item["change"])), reverse=True)
    if len(clean) > max_items:
        warnings.append(f"truncated_rows:{len(clean)}->{max_items}")
        clean = clean[:max_items]
    return clean, warnings, (start_period, end_period)


def render(
    template: dict[str, Any],
    sample: dict[str, Any],
    rows: list[dict[str, Any]],
    output_png: str | Path,
    metadata_path: str | Path,
    manifest_path: str | Path,
    input_metadata: dict[str, Any],
) -> dict[str, Any]:
    visual_id = str(sample.get("visual_id") or template.get("template_id") or "slope_chart_draft_v1")
    params = template.get("params", {}) or {}
    width = int(params.get("width", 1080))
    height = int(params.get("height", 860))
    show_values = bool(params.get("show_values", True))
    show_change = bool(params.get("show_change", True))
    value_format = str(params.get("value_format", "integer"))
    palette = load_palette(template)

    clean_rows, warnings, periods = _clean_rows(rows, template, sample)
    all_values = [float(item["start"]) for item in clean_rows] + [float(item["end"]) for item in clean_rows]
    min_value = min(all_values + [0.0])
    max_value = max(all_values + [1.0])
    span = max(max_value - min_value, 1.0)

    image = Image.new("RGB", (width, height), palette["background"])
    draw = ImageDraw.Draw(image)
    margin = 48
    panel_box = (margin, margin, width - margin, height - margin)
    draw.rounded_rectangle(panel_box, radius=28, fill=palette["panel"])

    label_font = _font("bold", 22)
    value_font = _font("regular", 17)
    period_font = _font("bold", 24)
    change_font = _font("bold", 18)

    left_x = panel_box[0] + 260
    right_x = panel_box[2] - 260
    top = panel_box[1] + 88
    bottom = panel_box[3] - 58
    plot_h = max(1, bottom - top)

    def y_for(value: float) -> float:
        return bottom - ((value - min_value) / span) * plot_h

    draw.text((left_x - 42, panel_box[1] + 30), periods[0], font=period_font, fill=palette["muted"])
    right_period_box = draw.textbbox((0, 0), periods[1], font=period_font)
    draw.text((right_x - (right_period_box[2] - right_period_box[0]) + 42, panel_box[1] + 30), periods[1], font=period_font, fill=palette["muted"])

    for grid_index in range(5):
        ratio = grid_index / 4
        y = top + ratio * plot_h
        draw.line((left_x - 36, y, right_x + 36, y), fill=palette["grid"], width=1)

    if clean_rows:
        for item in clean_rows:
            start = float(item["start"])
            end = float(item["end"])
            change = float(item["change"])
            y0 = y_for(start)
            y1 = y_for(end)
            color = palette["accent"] if change >= 0 else palette["warning"]
            draw.line((left_x, y0, right_x, y1), fill=color, width=4)
            draw.ellipse((left_x - 9, y0 - 9, left_x + 9, y0 + 9), fill=color, outline=palette["text"], width=2)
            draw.ellipse((right_x - 11, y1 - 11, right_x + 11, y1 + 11), fill=color, outline=palette["text"], width=2)

            label = _shorten(draw, item["label"], label_font, 220)
            draw.text((panel_box[0] + 24, y0 - 13), label, font=label_font, fill=palette["text"])
            if show_values:
                draw.text((left_x + 16, y0 - 10), _format_value(start, value_format), font=value_font, fill=palette["muted"])
                end_text = _format_value(end, value_format)
                end_box = draw.textbbox((0, 0), end_text, font=value_font)
                draw.text((right_x - 16 - (end_box[2] - end_box[0]), y1 - 10), end_text, font=value_font, fill=palette["muted"])
            if show_change:
                ch_text = _change_text(change, value_format)
                ch_box = draw.textbbox((0, 0), ch_text, font=change_font)
                draw.text((panel_box[2] - 24 - (ch_box[2] - ch_box[0]), y1 - 10), ch_text, font=change_font, fill=color)
    else:
        warnings.append("empty_rows")
        font = _font("bold", 34)
        msg = "No two-period values available"
        box = draw.textbbox((0, 0), msg, font=font)
        draw.text(((width - box[2]) // 2, (height - box[3]) // 2), msg, font=font, fill=palette["muted"])

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_png, format="PNG")

    created_at = utc_now()
    metadata = {
        "visual_id": visual_id,
        "template_id": template.get("template_id"),
        "renderer": "slope_chart",
        "created_at": created_at,
        "input": input_metadata,
        "bindings": sample.get("bindings", {}),
        "filters": sample.get("filters", []),
        "grouping": sample.get("grouping", {}),
        "source_note": sample.get("source_note", ""),
        "attribution": sample.get("attribution", {}),
        "item_count": len(clean_rows),
        "periods": list(periods),
        "value_range": {"min": min_value, "max": max_value},
        "rows_rendered": clean_rows,
        "warnings": warnings,
    }
    manifest = {
        "success": True,
        "visual_id": visual_id,
        "template_id": template.get("template_id"),
        "renderer": "slope_chart",
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
