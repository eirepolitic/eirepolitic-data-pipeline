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
    return f"{value:,.0f}" if math.isfinite(value) else "0"


def _clean_rows(rows: list[dict[str, Any]], template: dict[str, Any], sample: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    bindings = sample.get("bindings", {}) or {}
    group_field = str(bindings.get("group", "group"))
    x_field = str(bindings.get("x", "period"))
    value_field = str(bindings.get("value", "value"))
    params = template.get("params", {}) or {}
    max_groups = int(params.get("max_groups", 6))
    max_points = int(params.get("max_points_per_group", 12))

    warnings: list[str] = []
    clean: list[dict[str, Any]] = []
    for row in rows:
        group = str(row.get(group_field, "")).strip() or "Missing group"
        x = str(row.get(x_field, "")).strip()
        value = _as_float(row.get(value_field))
        if not x or value is None:
            warnings.append(f"missing_or_invalid_point:{group[:24]}")
            continue
        if len(group) > 32:
            warnings.append(f"long_group_label:{group[:32]}")
        if value < 0:
            warnings.append("negative_values_present")
        clean.append({"group": group, "x": x, "value": value})

    group_order: list[str] = []
    for item in clean:
        if item["group"] not in group_order:
            group_order.append(item["group"])
    if len(group_order) > max_groups:
        warnings.append(f"truncated_groups:{len(group_order)}->{max_groups}")
        allowed = set(group_order[:max_groups])
        clean = [item for item in clean if item["group"] in allowed]
        group_order = group_order[:max_groups]

    trimmed: list[dict[str, Any]] = []
    for group in group_order:
        group_items = [item for item in clean if item["group"] == group]
        if len(group_items) > max_points:
            warnings.append(f"truncated_points:{group[:24]}:{len(group_items)}->{max_points}")
            group_items = group_items[:max_points]
        trimmed.extend(group_items)
    return trimmed, warnings


def render(
    template: dict[str, Any],
    sample: dict[str, Any],
    rows: list[dict[str, Any]],
    output_png: str | Path,
    metadata_path: str | Path,
    manifest_path: str | Path,
    input_metadata: dict[str, Any],
) -> dict[str, Any]:
    visual_id = str(sample.get("visual_id") or template.get("template_id") or "small_multiples_draft_v1")
    params = template.get("params", {}) or {}
    width = int(params.get("width", 1080))
    height = int(params.get("height", 860))
    columns = max(1, int(params.get("columns", 2)))
    show_markers = bool(params.get("show_markers", True))
    value_format = str(params.get("value_format", "integer"))
    palette = load_palette(template)

    clean_rows, warnings = _clean_rows(rows, template, sample)
    groups: list[str] = []
    x_order_by_group: dict[str, list[str]] = {}
    values_by_group: dict[str, list[float]] = {}
    for item in clean_rows:
        group = item["group"]
        if group not in groups:
            groups.append(group)
            x_order_by_group[group] = []
            values_by_group[group] = []
        x_order_by_group[group].append(item["x"])
        values_by_group[group].append(float(item["value"]))

    all_values = [float(item["value"]) for item in clean_rows]
    min_value = min(all_values + [0.0])
    max_value = max(all_values + [1.0])
    span = max(max_value - min_value, 1.0)
    y_min = min_value - span * 0.12
    y_max = max_value + span * 0.16
    y_span = max(y_max - y_min, 1.0)

    image = Image.new("RGB", (width, height), palette["background"])
    draw = ImageDraw.Draw(image)
    title_font = _font("bold", 23)
    small_font = _font("regular", 15)
    value_font = _font("bold", 17)

    margin = 44
    gap = 22
    group_count = max(len(groups), 1)
    rows_count = max(1, math.ceil(group_count / columns))
    tile_w = int((width - margin * 2 - gap * (columns - 1)) / columns)
    tile_h = int((height - margin * 2 - gap * (rows_count - 1)) / rows_count)

    def tx(tile_x0: int, plot_x0: int, plot_w: int, index: int, count: int) -> float:
        if count <= 1:
            return plot_x0 + plot_w / 2
        return plot_x0 + (index / (count - 1)) * plot_w

    def ty(value: float, plot_y0: int, plot_h: int) -> float:
        return plot_y0 + plot_h - ((value - y_min) / y_span) * plot_h

    for index, group in enumerate(groups):
        row_index = index // columns
        col_index = index % columns
        x0 = margin + col_index * (tile_w + gap)
        y0 = margin + row_index * (tile_h + gap)
        x1 = x0 + tile_w
        y1 = y0 + tile_h
        fill = palette["panel_alt"] if index % 2 else palette["panel"]
        draw.rounded_rectangle((x0, y0, x1, y1), radius=22, fill=fill)

        group_label = _shorten(draw, group, title_font, tile_w - 42)
        draw.text((x0 + 20, y0 + 16), group_label, font=title_font, fill=palette["text"])

        plot_x0 = x0 + 34
        plot_y0 = y0 + 64
        plot_w = tile_w - 68
        plot_h = tile_h - 104
        if plot_h < 40:
            plot_h = max(30, tile_h - 86)

        for grid_index in range(3):
            gy = plot_y0 + grid_index * plot_h / 2
            draw.line((plot_x0, gy, plot_x0 + plot_w, gy), fill=palette["grid"], width=1)

        values = values_by_group.get(group, [])
        points: list[tuple[float, float]] = []
        for point_index, value in enumerate(values):
            points.append((tx(x0, plot_x0, plot_w, point_index, len(values)), ty(value, plot_y0, plot_h)))
        if len(points) >= 2:
            draw.line(points, fill=palette["accent"], width=4, joint="curve")
        for px, py in points:
            if show_markers:
                draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=palette["accent"], outline=palette["text"], width=1)

        if values:
            first = _format_value(values[0], value_format)
            last = _format_value(values[-1], value_format)
            draw.text((plot_x0, y1 - 32), first, font=small_font, fill=palette["muted"])
            last_box = draw.textbbox((0, 0), last, font=value_font)
            draw.text((plot_x0 + plot_w - (last_box[2] - last_box[0]), y1 - 34), last, font=value_font, fill=palette["accent"])

    if not groups:
        warnings.append("empty_rows")
        font = _font("bold", 34)
        msg = "No series available"
        box = draw.textbbox((0, 0), msg, font=font)
        draw.text(((width - box[2]) // 2, (height - box[3]) // 2), msg, font=font, fill=palette["muted"])

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_png, format="PNG")

    created_at = utc_now()
    metadata = {
        "visual_id": visual_id,
        "template_id": template.get("template_id"),
        "renderer": "small_multiples",
        "created_at": created_at,
        "input": input_metadata,
        "bindings": sample.get("bindings", {}),
        "filters": sample.get("filters", []),
        "grouping": sample.get("grouping", {}),
        "source_note": sample.get("source_note", ""),
        "attribution": sample.get("attribution", {}),
        "group_count": len(groups),
        "point_count": len(clean_rows),
        "value_range": {"min": min_value, "max": max_value},
        "rows_rendered": clean_rows,
        "warnings": warnings,
    }
    manifest = {
        "success": True,
        "visual_id": visual_id,
        "template_id": template.get("template_id"),
        "renderer": "small_multiples",
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
