from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from instagram.renderer.constants import FONT_CANDIDATES
from .common import resolve_repo_path, utc_now, write_json


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


def _format_value(value: float, value_format: str) -> str:
    if value_format == "percent":
        return f"{value:g}%"
    if value_format == "decimal":
        return f"{value:g}"
    return f"{value:,.0f}" if math.isfinite(value) else "0"


def _load_geography(sample: dict[str, Any]) -> tuple[dict[str, Any] | None, Path | None]:
    geography_cfg = sample.get("geography", {}) or {}
    path = geography_cfg.get("path")
    if not path:
        return None, None
    resolved = resolve_repo_path(str(path))
    return json.loads(resolved.read_text(encoding="utf-8")), resolved


def _centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    if not points:
        return 0.0, 0.0
    return sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)


def _draw_basemap(
    draw: ImageDraw.ImageDraw,
    geography: dict[str, Any] | None,
    map_box: tuple[int, int, int, int],
    palette: dict[str, str],
) -> tuple[float, float, float, float, float]:
    if not geography:
        return 0.0, 0.0, 1000.0, 760.0, 1.0
    coordinate_space = geography.get("coordinate_space", {}) or {}
    source_w = float(coordinate_space.get("width", 1000))
    source_h = float(coordinate_space.get("height", 760))
    inner_margin = 30
    usable_w = (map_box[2] - map_box[0]) - inner_margin * 2
    usable_h = (map_box[3] - map_box[1]) - inner_margin * 2
    scale = min(usable_w / source_w, usable_h / source_h)
    x_offset = map_box[0] + ((map_box[2] - map_box[0]) - source_w * scale) / 2
    y_offset = map_box[1] + ((map_box[3] - map_box[1]) - source_h * scale) / 2
    for feature in geography.get("features", []) or []:
        geometry = feature.get("geometry", {}) or {}
        coordinates = geometry.get("coordinates", []) or []
        if geometry.get("type") != "Polygon" or not coordinates:
            continue
        points = [(x_offset + float(p[0]) * scale, y_offset + float(p[1]) * scale) for p in coordinates[0]]
        draw.polygon(points, fill=palette["map_fill"], outline=palette["map_outline"])
    return x_offset, y_offset, source_w, source_h, scale


def _color_for_category(category: str, order: list[str], palette: dict[str, str]) -> str:
    colors = [palette["accent"], palette["accent_2"], palette["text"], palette["warning"], palette["muted"]]
    try:
        return colors[order.index(category) % len(colors)]
    except ValueError:
        return palette["accent"]


def render(
    template: dict[str, Any],
    sample: dict[str, Any],
    rows: list[dict[str, Any]],
    output_png: str | Path,
    metadata_path: str | Path,
    manifest_path: str | Path,
    input_metadata: dict[str, Any],
) -> dict[str, Any]:
    visual_id = str(sample.get("visual_id") or template.get("template_id") or "point_map_draft_v1")
    params = template.get("params", {}) or {}
    raw_palette = template.get("palette", {}) or {}
    palette = {
        "background": str(raw_palette.get("background", "#0f2f24")),
        "panel": str(raw_palette.get("panel", "#173d30")),
        "text": str(raw_palette.get("text", "#f4ead7")),
        "muted": str(raw_palette.get("muted", "#cbbf9f")),
        "accent": str(raw_palette.get("accent", "#d8b45f")),
        "accent_2": str(raw_palette.get("accent_2", "#9ec5a2")),
        "warning": str(raw_palette.get("warning", "#b55b5b")),
        "map_fill": str(raw_palette.get("map_fill", "#24483b")),
        "map_outline": str(raw_palette.get("map_outline", "#f4ead7")),
    }
    width = int(params.get("width", 1080))
    height = int(params.get("height", 860))
    max_points = int(params.get("max_points", 60))
    min_radius = float(params.get("min_radius", 8))
    max_radius = float(params.get("max_radius", 30))
    show_labels = bool(params.get("show_labels", True))
    show_legend = bool(params.get("show_legend", True))
    value_format = str(params.get("value_format", "integer"))

    bindings = sample.get("bindings", {}) or {}
    x_field = str(bindings.get("x", "x"))
    y_field = str(bindings.get("y", "y"))
    label_field = str(bindings.get("label", "label"))
    value_field = str(bindings.get("value", "value"))
    category_field = str(bindings.get("category", "category"))

    warnings: list[str] = []
    points: list[dict[str, Any]] = []
    for row in rows:
        x = _as_float(row.get(x_field))
        y = _as_float(row.get(y_field))
        value = _as_float(row.get(value_field))
        label = str(row.get(label_field, "")).strip() or "Point"
        category = str(row.get(category_field, "Default")).strip() or "Default"
        if x is None or y is None:
            warnings.append(f"missing_or_invalid_coordinates:{label[:24]}")
            continue
        if x < 0 or y < 0 or x > 1000 or y > 760:
            warnings.append(f"out_of_bounds_point:{label[:24]}")
        if value is None:
            warnings.append(f"missing_or_invalid_value:{label[:24]}")
            value = 1.0
        if value < 0:
            warnings.append("negative_values_present")
        if len(label) > 30:
            warnings.append(f"long_label:{label[:30]}")
        points.append({"x": x, "y": y, "label": label, "category": category, "value": value})

    if len(points) > max_points:
        warnings.append(f"truncated_points:{len(points)}->{max_points}")
        points = points[:max_points]

    values = [max(float(point["value"]), 0.0) for point in points]
    min_value = min(values) if values else 0.0
    max_value = max(values) if values else 1.0
    value_span = max(max_value - min_value, 1.0)
    category_order: list[str] = []
    for point in points:
        if point["category"] not in category_order:
            category_order.append(point["category"])

    image = Image.new("RGB", (width, height), palette["background"])
    draw = ImageDraw.Draw(image)
    margin = 42
    legend_h = 84 if show_legend else 0
    map_box = (margin, margin, width - margin, height - margin - legend_h)
    draw.rounded_rectangle(map_box, radius=26, fill=palette["panel"])

    geography, geography_path = _load_geography(sample)
    x_offset, y_offset, _, _, scale = _draw_basemap(draw, geography, map_box, palette)

    label_font = _font("bold", 17)
    value_font = _font("regular", 14)
    for point in sorted(points, key=lambda item: float(item["value"]), reverse=True):
        x = x_offset + float(point["x"]) * scale
        y = y_offset + float(point["y"]) * scale
        value = max(float(point["value"]), 0.0)
        radius = min_radius + ((value - min_value) / value_span) * (max_radius - min_radius)
        color = _color_for_category(str(point["category"]), category_order, palette)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=palette["text"], width=2)
        if show_labels:
            label = str(point["label"])
            label_text = label if len(label) <= 18 else label[:17] + "…"
            draw.text((x + radius + 5, y - radius), label_text, font=label_font, fill=palette["text"])
            draw.text((x + radius + 5, y - radius + 20), _format_value(float(point["value"]), value_format), font=value_font, fill=palette["muted"])

    if show_legend:
        legend_font = _font("regular", 16)
        x = margin + 72
        y = height - margin - 44
        for category in category_order[:5]:
            color = _color_for_category(category, category_order, palette)
            draw.ellipse((x, y, x + 16, y + 16), fill=color, outline=palette["text"], width=1)
            draw.text((x + 24, y - 2), category, font=legend_font, fill=palette["muted"])
            x += 190

    if not points:
        warnings.append("empty_points")
        no_data_font = _font("bold", 34)
        msg = "No points available"
        box = draw.textbbox((0, 0), msg, font=no_data_font)
        draw.text(((width - box[2]) // 2, (height - box[3]) // 2), msg, font=no_data_font, fill=palette["muted"])

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_png, format="PNG")

    created_at = utc_now()
    metadata = {
        "visual_id": visual_id,
        "template_id": template.get("template_id"),
        "renderer": "point_map",
        "created_at": created_at,
        "input": input_metadata,
        "geography": {
            "source": str(sample.get("geography", {}).get("path", "")),
            "resolved_source": str(geography_path) if geography_path else "",
            "has_basemap": geography is not None,
        },
        "bindings": sample.get("bindings", {}),
        "filters": sample.get("filters", []),
        "grouping": sample.get("grouping", {}),
        "source_note": sample.get("source_note", ""),
        "attribution": sample.get("attribution", {}),
        "point_count": len(points),
        "value_range": {"min": min_value, "max": max_value},
        "rows_rendered": points,
        "warnings": warnings,
    }
    manifest = {
        "success": True,
        "visual_id": visual_id,
        "template_id": template.get("template_id"),
        "renderer": "point_map",
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
