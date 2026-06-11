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
    x_field = str(bindings.get("x", "x"))
    y_field = str(bindings.get("y", "y"))
    size_field = str(bindings.get("size", "size"))
    label_field = str(bindings.get("label", "label"))
    category_field = str(bindings.get("category", "category"))
    max_points = int((template.get("params", {}) or {}).get("max_points", 80))

    warnings: list[str] = []
    clean: list[dict[str, Any]] = []
    for row in rows:
        label = str(row.get(label_field, "")).strip() or "Point"
        x = _as_float(row.get(x_field))
        y = _as_float(row.get(y_field))
        size = _as_float(row.get(size_field))
        category = str(row.get(category_field, "Default")).strip() or "Default"
        if x is None or y is None:
            warnings.append(f"missing_or_invalid_xy:{label[:24]}")
            continue
        if size is None:
            warnings.append(f"missing_or_invalid_size:{label[:24]}")
            size = 1.0
        if x < 0 or y < 0 or size < 0:
            warnings.append("negative_values_present")
        if len(label) > 30:
            warnings.append(f"long_label:{label[:30]}")
        clean.append({"x": x, "y": y, "size": size, "label": label, "category": category})

    if len(clean) > max_points:
        warnings.append(f"truncated_points:{len(clean)}->{max_points}")
        clean = clean[:max_points]
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
    visual_id = str(sample.get("visual_id") or template.get("template_id") or "scatter_plot_draft_v1")
    params = template.get("params", {}) or {}
    width = int(params.get("width", 1080))
    height = int(params.get("height", 860))
    min_radius = float(params.get("min_radius", 7))
    max_radius = float(params.get("max_radius", 24))
    show_labels = bool(params.get("show_labels", True))
    show_legend = bool(params.get("show_legend", True))
    value_format = str(params.get("value_format", "integer"))
    palette = load_palette(template)

    clean_rows, warnings = _clean_rows(rows, template, sample)
    categories: list[str] = []
    for item in clean_rows:
        if item["category"] not in categories:
            categories.append(item["category"])

    x_values = [float(item["x"]) for item in clean_rows]
    y_values = [float(item["y"]) for item in clean_rows]
    size_values = [max(float(item["size"]), 0.0) for item in clean_rows]
    x_min = min(x_values + [0.0])
    x_max = max(x_values + [1.0])
    y_min = min(y_values + [0.0])
    y_max = max(y_values + [1.0])
    s_min = min(size_values + [0.0])
    s_max = max(size_values + [1.0])
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)
    s_span = max(s_max - s_min, 1.0)

    fig = plt.figure(figsize=(width / 150, height / 150), dpi=150)
    fig.patch.set_facecolor(palette["background"])
    ax = fig.add_axes([0.10, 0.15, 0.84, 0.75])
    ax.set_facecolor(palette["panel"])
    colors = [palette["accent"], palette["accent_2"], palette["text"], palette["warning"], palette["muted"]]

    if clean_rows:
        for index, category in enumerate(categories):
            category_rows = [item for item in clean_rows if item["category"] == category]
            xs = [float(item["x"]) for item in category_rows]
            ys = [float(item["y"]) for item in category_rows]
            radii = [min_radius + ((max(float(item["size"]), 0.0) - s_min) / s_span) * (max_radius - min_radius) for item in category_rows]
            areas = [(radius ** 2) * 3.14 for radius in radii]
            ax.scatter(xs, ys, s=areas, color=colors[index % len(colors)], edgecolor=palette["text"], linewidth=0.9, alpha=0.88, label=category)
            if show_labels:
                for item in category_rows:
                    label = str(item["label"])
                    label = label if len(label) <= 16 else label[:15] + "…"
                    ax.text(float(item["x"]), float(item["y"]), f"  {label}", color=palette["text"], fontsize=9, va="center")
        ax.set_xlim(x_min - x_span * 0.10, x_max + x_span * 0.16)
        ax.set_ylim(y_min - y_span * 0.12, y_max + y_span * 0.14)
        if show_legend and len(categories) > 1:
            legend = ax.legend(loc="upper left", frameon=False, fontsize=10, ncol=2)
            for text in legend.get_texts():
                text.set_color(palette["text"])
    else:
        warnings.append("empty_rows")
        ax.text(0.5, 0.5, "No data available", color=palette["muted"], fontsize=16, ha="center", va="center", transform=ax.transAxes)

    ax.xaxis.grid(True, color=palette["grid"], alpha=0.20)
    ax.yaxis.grid(True, color=palette["grid"], alpha=0.20)
    ax.tick_params(axis="x", colors=palette["muted"], labelsize=10)
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
        "renderer": "scatter_plot",
        "created_at": created_at,
        "input": input_metadata,
        "bindings": sample.get("bindings", {}),
        "filters": sample.get("filters", []),
        "grouping": sample.get("grouping", {}),
        "source_note": sample.get("source_note", ""),
        "attribution": sample.get("attribution", {}),
        "point_count": len(clean_rows),
        "category_count": len(categories),
        "x_range": {"min": x_min, "max": x_max},
        "y_range": {"min": y_min, "max": y_max},
        "size_range": {"min": s_min, "max": s_max},
        "rows_rendered": clean_rows,
        "warnings": warnings,
    }
    manifest = {
        "success": True,
        "visual_id": visual_id,
        "template_id": template.get("template_id"),
        "renderer": "scatter_plot",
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
