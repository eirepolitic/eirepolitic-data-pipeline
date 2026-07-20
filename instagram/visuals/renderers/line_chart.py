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
        x_value = str(row.get(x_field, "")).strip()
        series = str(row.get(series_field, "Series")).strip() or "Series"
        value = _as_float(row.get(value_field))
        if not x_value or value is None:
            warnings.append("missing_or_invalid_point")
            continue
        clean.append({"x": x_value, "series": series, "value": value})

    series_order = []
    for item in clean:
        if item["series"] not in series_order:
            series_order.append(item["series"])
    if len(series_order) > max_series:
        warnings.append(f"truncated_series:{len(series_order)}->{max_series}")
        allowed = set(series_order[:max_series])
        clean = [item for item in clean if item["series"] in allowed]

    x_order = []
    for item in clean:
        if item["x"] not in x_order:
            x_order.append(item["x"])
    if len(x_order) > max_points:
        warnings.append(f"truncated_points:{len(x_order)}->{max_points}")
        allowed_x = set(x_order[:max_points])
        clean = [item for item in clean if item["x"] in allowed_x]

    if any(item["value"] < 0 for item in clean):
        warnings.append("negative_values_present")
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
    visual_id = str(sample.get("visual_id") or template.get("template_id") or "line_chart_draft_v1")
    params = template.get("params", {}) or {}
    width = int(params.get("width", 1080))
    height = int(params.get("height", 860))
    palette = load_palette(template)
    value_format = str(params.get("value_format", "integer"))
    show_markers = bool(params.get("show_markers", True))

    clean_rows, warnings = _clean_rows(rows, template, sample)
    x_order: list[str] = []
    series_order: list[str] = []
    for item in clean_rows:
        if item["x"] not in x_order:
            x_order.append(item["x"])
        if item["series"] not in series_order:
            series_order.append(item["series"])

    colors = [palette["accent"], palette["accent_2"], palette["text"], palette["warning"]]
    fig = plt.figure(figsize=(width / 150, height / 150), dpi=150)
    fig.patch.set_facecolor(palette["background"])
    ax = fig.add_axes([0.10, 0.20, 0.84, 0.70])
    ax.set_facecolor(palette["panel"])

    if clean_rows:
        x_positions = list(range(len(x_order)))
        lookup = {(item["series"], item["x"]): item["value"] for item in clean_rows}
        for index, series in enumerate(series_order):
            values = [lookup.get((series, x), float("nan")) for x in x_order]
            ax.plot(
                x_positions,
                values,
                color=colors[index % len(colors)],
                linewidth=3.2,
                marker="o" if show_markers else None,
                markersize=7,
                label=series,
            )
            valid = [(pos, value) for pos, value in zip(x_positions, values) if not math.isnan(value)]
            if valid:
                last_x, last_value = valid[-1]
                ax.text(
                    last_x,
                    last_value,
                    f"  {_format_value(last_value, value_format)}",
                    color=colors[index % len(colors)],
                    fontsize=10,
                    fontweight="bold",
                    va="center",
                )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_order, color=palette["text"], fontsize=10, rotation=35, ha="right")
        if len(series_order) > 1:
            legend = ax.legend(loc="upper left", frameon=False, fontsize=10)
            for text in legend.get_texts():
                text.set_color(palette["text"])
    else:
        warnings.append("empty_rows")
        ax.text(0.5, 0.5, "No data available", color=palette["muted"], fontsize=16, ha="center", va="center", transform=ax.transAxes)

    ax.yaxis.grid(True, color=palette["grid"], alpha=0.22)
    ax.xaxis.grid(False)
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
        "renderer": "line_chart",
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
        "renderer": "line_chart",
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
