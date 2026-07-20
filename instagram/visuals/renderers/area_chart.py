from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import load_palette, utc_now, write_json


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


def _clean_rows(rows: list[dict[str, Any]], template: dict[str, Any], sample: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    bindings = sample.get("bindings", {}) or {}
    x_field = str(bindings.get("x", "period"))
    value_field = str(bindings.get("value", "value"))
    series_field = str(bindings.get("series", "series"))
    params = template.get("params", {}) or {}
    max_points = int(params.get("max_points", 18))
    max_series = int(params.get("max_series", 4))

    warnings: list[str] = []
    clean: list[dict[str, Any]] = []
    for row in rows:
        x = str(row.get(x_field, "")).strip()
        series = str(row.get(series_field, "Series")).strip() or "Series"
        value = _as_float(row.get(value_field))
        if not x or value is None:
            warnings.append(f"missing_or_invalid_point:{series[:24]}")
            continue
        if value < 0:
            warnings.append("negative_values_present")
        if len(series) > 30:
            warnings.append(f"long_series_label:{series[:30]}")
        clean.append({"x": x, "series": series, "value": value})

    series_order: list[str] = []
    for item in clean:
        if item["series"] not in series_order:
            series_order.append(item["series"])
    if len(series_order) > max_series:
        warnings.append(f"truncated_series:{len(series_order)}->{max_series}")
        allowed = set(series_order[:max_series])
        clean = [item for item in clean if item["series"] in allowed]
        series_order = series_order[:max_series]

    trimmed: list[dict[str, Any]] = []
    for series in series_order:
        series_items = [item for item in clean if item["series"] == series]
        if len(series_items) > max_points:
            warnings.append(f"truncated_points:{series[:24]}:{len(series_items)}->{max_points}")
            series_items = series_items[:max_points]
        trimmed.extend(series_items)
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
    visual_id = str(sample.get("visual_id") or template.get("template_id") or "area_chart_draft_v1")
    params = template.get("params", {}) or {}
    width = int(params.get("width", 1080))
    height = int(params.get("height", 860))
    show_markers = bool(params.get("show_markers", True))
    show_legend = bool(params.get("show_legend", True))
    value_format = str(params.get("value_format", "integer"))
    palette = load_palette(template)

    clean_rows, warnings = _clean_rows(rows, template, sample)
    series_order: list[str] = []
    x_order: list[str] = []
    for item in clean_rows:
        if item["series"] not in series_order:
            series_order.append(item["series"])
        if item["x"] not in x_order:
            x_order.append(item["x"])

    values_by_key = {(item["series"], item["x"]): float(item["value"]) for item in clean_rows}
    all_values = [float(item["value"]) for item in clean_rows]
    min_value = min(all_values + [0.0])
    max_value = max(all_values + [1.0])
    span = max(max_value - min_value, 1.0)
    y_min = min(0.0, min_value - span * 0.10)
    y_max = max_value + span * 0.16

    fig = plt.figure(figsize=(width / 150, height / 150), dpi=150)
    fig.patch.set_facecolor(palette["background"])
    ax = fig.add_axes([0.10, 0.17, 0.84, 0.72])
    ax.set_facecolor(palette["panel"])

    colors = [palette["accent"], palette["accent_2"], palette["text"], palette["warning"]]
    x_positions = list(range(len(x_order)))
    if series_order and x_order:
        for idx, series in enumerate(series_order):
            y_values = [values_by_key.get((series, x), math.nan) for x in x_order]
            valid_x = [x for x, y in zip(x_positions, y_values) if math.isfinite(y)]
            valid_y = [y for y in y_values if math.isfinite(y)]
            if not valid_x:
                continue
            color = colors[idx % len(colors)]
            ax.fill_between(valid_x, valid_y, y_min, color=color, alpha=0.33, linewidth=0)
            ax.plot(valid_x, valid_y, color=color, linewidth=3.0, label=series)
            if show_markers:
                ax.scatter(valid_x, valid_y, color=color, edgecolor=palette["text"], linewidth=0.8, s=28, zorder=5)
            last_x = valid_x[-1]
            last_y = valid_y[-1]
            ax.text(last_x, last_y, f"  {_format_value(last_y, value_format)}", color=palette["text"], fontsize=10, fontweight="bold", va="center")
        ax.set_xlim(-0.2, max(len(x_order) - 0.8, 0.8))
        ax.set_ylim(y_min, y_max)
        step = max(1, math.ceil(len(x_order) / 6))
        shown_ticks = x_positions[::step]
        ax.set_xticks(shown_ticks)
        ax.set_xticklabels([x_order[i] for i in shown_ticks], color=palette["muted"], fontsize=10, rotation=0)
        if show_legend and len(series_order) > 1:
            legend = ax.legend(loc="upper left", frameon=False, fontsize=10, ncol=2)
            for text in legend.get_texts():
                text.set_color(palette["text"])
    else:
        warnings.append("empty_rows")
        ax.text(0.5, 0.5, "No data available", color=palette["muted"], fontsize=16, ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

    ax.yaxis.grid(True, color=palette["grid"], alpha=0.22)
    ax.tick_params(axis="y", colors=palette["muted"], labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)

    created_at = utc_now()
    metadata = {
        "visual_id": visual_id,
        "template_id": template.get("template_id"),
        "renderer": "area_chart",
        "created_at": created_at,
        "input": input_metadata,
        "bindings": sample.get("bindings", {}),
        "filters": sample.get("filters", []),
        "grouping": sample.get("grouping", {}),
        "source_note": sample.get("source_note", ""),
        "attribution": sample.get("attribution", {}),
        "series_count": len(series_order),
        "point_count": len(clean_rows),
        "value_range": {"min": min_value, "max": max_value},
        "rows_rendered": clean_rows,
        "warnings": warnings,
    }
    manifest = {
        "success": True,
        "visual_id": visual_id,
        "template_id": template.get("template_id"),
        "renderer": "area_chart",
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
