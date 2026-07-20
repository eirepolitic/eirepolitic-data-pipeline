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


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def _mix(low: str, high: str, ratio: float) -> str:
    low_rgb = _hex_to_rgb(low)
    high_rgb = _hex_to_rgb(high)
    ratio = max(0.0, min(1.0, ratio))
    mixed = tuple(round(a + (b - a) * ratio) for a, b in zip(low_rgb, high_rgb))
    return "#%02x%02x%02x" % mixed


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


def _load_layout(sample: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    geography_cfg = sample.get("geography", {}) or {}
    path = geography_cfg.get("path")
    if not path:
        raise ValueError("Tile map sample requires geography.path")
    resolved = resolve_repo_path(str(path))
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    return payload, resolved


def _shorten(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    value = str(text or "").strip()
    if draw.textbbox((0, 0), value, font=font)[2] <= max_width:
        return value
    suffix = "…"
    while value and draw.textbbox((0, 0), value + suffix, font=font)[2] > max_width:
        value = value[:-1]
    return (value.rstrip() or "") + suffix


def render(
    template: dict[str, Any],
    sample: dict[str, Any],
    rows: list[dict[str, Any]],
    output_png: str | Path,
    metadata_path: str | Path,
    manifest_path: str | Path,
    input_metadata: dict[str, Any],
) -> dict[str, Any]:
    visual_id = str(sample.get("visual_id") or template.get("template_id") or "tile_map_draft_v1")
    params = template.get("params", {}) or {}
    palette = template.get("palette", {}) or {}
    width = int(params.get("width", 1080))
    height = int(params.get("height", 860))
    max_tiles = int(params.get("max_tiles", 80))
    show_labels = bool(params.get("show_labels", True))
    show_values = bool(params.get("show_values", True))
    show_legend = bool(params.get("show_legend", True))
    value_format = str(params.get("value_format", "integer"))

    background = str(palette.get("background", "#0f2f24"))
    panel = str(palette.get("panel", "#173d30"))
    text = str(palette.get("text", "#f4ead7"))
    muted = str(palette.get("muted", "#cbbf9f"))
    no_data = str(palette.get("map_no_data", "#2a463b"))
    outline = str(palette.get("map_outline", "#f4ead7"))
    low_color = str(palette.get("map_low", "#275346"))
    high_color = str(palette.get("map_high", "#d8b45f"))

    bindings = sample.get("bindings", {}) or {}
    feature_field = str(bindings.get("feature_id", "feature_id"))
    value_field = str(bindings.get("value", "value"))

    warnings: list[str] = []
    value_by_feature: dict[str, float] = {}
    for row in rows:
        feature_id = str(row.get(feature_field, "")).strip()
        value = _as_float(row.get(value_field))
        if not feature_id:
            warnings.append("missing_feature_id")
            continue
        if value is None:
            warnings.append(f"missing_or_invalid_value:{feature_id}")
            continue
        value_by_feature[feature_id] = value

    layout, layout_path = _load_layout(sample)
    tiles = list(layout.get("tiles", []) or [])
    if len(tiles) > max_tiles:
        warnings.append(f"truncated_tiles:{len(tiles)}->{max_tiles}")
        tiles = tiles[:max_tiles]

    tile_ids = {str(tile.get("feature_id", "")).strip() for tile in tiles}
    unmatched_data_ids = sorted(set(value_by_feature) - tile_ids)
    if unmatched_data_ids:
        warnings.append(f"unmatched_data_features:{','.join(unmatched_data_ids)}")

    joined_values = [value_by_feature[feature_id] for feature_id in tile_ids if feature_id in value_by_feature]
    min_value = min(joined_values) if joined_values else 0.0
    max_value = max(joined_values) if joined_values else 0.0
    value_span = max(max_value - min_value, 1.0)

    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    margin = 42
    legend_h = 90 if show_legend else 0
    map_box = (margin, margin, width - margin, height - margin - legend_h)
    draw.rounded_rectangle(map_box, radius=26, fill=panel)

    max_col = max([int(tile.get("col", 0)) for tile in tiles] + [0])
    max_row = max([int(tile.get("row", 0)) for tile in tiles] + [0])
    cols = max_col + 1
    rows_count = max_row + 1
    gap = 14
    usable_w = (map_box[2] - map_box[0]) - 60
    usable_h = (map_box[3] - map_box[1]) - 60
    tile_size = int(min((usable_w - gap * (cols - 1)) / max(cols, 1), (usable_h - gap * (rows_count - 1)) / max(rows_count, 1)))
    tile_size = max(42, tile_size)
    grid_w = cols * tile_size + (cols - 1) * gap
    grid_h = rows_count * tile_size + (rows_count - 1) * gap
    start_x = map_box[0] + ((map_box[2] - map_box[0]) - grid_w) / 2
    start_y = map_box[1] + ((map_box[3] - map_box[1]) - grid_h) / 2

    label_font = _font("bold", max(14, int(tile_size * 0.20)))
    value_font = _font("regular", max(12, int(tile_size * 0.17)))
    missing_feature_ids: list[str] = []
    joined_tile_count = 0

    for tile in tiles:
        feature_id = str(tile.get("feature_id", "")).strip()
        label = str(tile.get("label", feature_id)).strip()
        short_label = str(tile.get("short_label", label[:3])).strip()
        col = int(tile.get("col", 0))
        row = int(tile.get("row", 0))
        x0 = start_x + col * (tile_size + gap)
        y0 = start_y + row * (tile_size + gap)
        x1 = x0 + tile_size
        y1 = y0 + tile_size

        value = value_by_feature.get(feature_id)
        if value is None:
            fill = no_data
            missing_feature_ids.append(feature_id)
        else:
            joined_tile_count += 1
            ratio = (value - min_value) / value_span if value_span else 0.0
            fill = _mix(low_color, high_color, ratio)

        draw.rounded_rectangle((x0, y0, x1, y1), radius=12, fill=fill, outline=outline, width=2)
        if show_labels:
            label_text = _shorten(draw, short_label or label, label_font, tile_size - 12)
            box = draw.textbbox((0, 0), label_text, font=label_font)
            draw.text((x0 + (tile_size - (box[2] - box[0])) / 2, y0 + tile_size * 0.28), label_text, font=label_font, fill=text)
        if show_values and value is not None:
            value_text = _format_value(value, value_format)
            vbox = draw.textbbox((0, 0), value_text, font=value_font)
            draw.text((x0 + (tile_size - (vbox[2] - vbox[0])) / 2, y0 + tile_size * 0.57), value_text, font=value_font, fill=text)

    if missing_feature_ids:
        warnings.append(f"no_data_features:{','.join(sorted(missing_feature_ids))}")

    if show_legend:
        legend_x0 = margin + 80
        legend_y0 = height - margin - 48
        legend_w = width - margin * 2 - 160
        bins = 5
        step_w = legend_w / bins
        legend_font = _font("regular", 16)
        for index in range(bins):
            ratio = index / (bins - 1)
            x0 = legend_x0 + index * step_w
            x1 = legend_x0 + (index + 1) * step_w
            draw.rectangle((x0, legend_y0, x1, legend_y0 + 18), fill=_mix(low_color, high_color, ratio))
        draw.text((legend_x0, legend_y0 + 24), _format_value(min_value, value_format), font=legend_font, fill=muted)
        max_text = _format_value(max_value, value_format)
        max_box = draw.textbbox((0, 0), max_text, font=legend_font)
        draw.text((legend_x0 + legend_w - (max_box[2] - max_box[0]), legend_y0 + 24), max_text, font=legend_font, fill=muted)

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_png, format="PNG")

    created_at = utc_now()
    metadata = {
        "visual_id": visual_id,
        "template_id": template.get("template_id"),
        "renderer": "tile_map",
        "created_at": created_at,
        "input": input_metadata,
        "geography": {
            "source": str(sample.get("geography", {}).get("path", "")),
            "resolved_source": str(layout_path),
            "tile_count": len(tiles),
            "joined_tile_count": joined_tile_count,
            "missing_feature_ids": sorted(missing_feature_ids),
            "unmatched_data_ids": unmatched_data_ids,
        },
        "bindings": sample.get("bindings", {}),
        "filters": sample.get("filters", []),
        "grouping": sample.get("grouping", {}),
        "source_note": sample.get("source_note", ""),
        "attribution": sample.get("attribution", {}),
        "value_range": {"min": min_value, "max": max_value},
        "rows_rendered": rows,
        "warnings": warnings,
    }
    manifest = {
        "success": True,
        "visual_id": visual_id,
        "template_id": template.get("template_id"),
        "renderer": "tile_map",
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
