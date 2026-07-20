from __future__ import annotations

import math
import textwrap
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


def _wrap_label(label: str, width: int = 12) -> str:
    return "\n".join(textwrap.wrap(label, width=width, max_lines=3, placeholder="…"))


def _clean_rows(rows: list[dict[str, Any]], template: dict[str, Any], sample: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    bindings = sample.get("bindings", {}) or {}
    category_field = str(bindings.get("category", "category"))
    segment_field = str(bindings.get("segment", "segment"))
    value_field = str(bindings.get("value", "value"))
    params = template.get("params", {}) or {}
    max_categories = int(params.get("max_categories", 8))
    max_segments = int(params.get("max_segments", 5))

    warnings: list[str] = []
    clean: list[dict[str, Any]] = []
    for row in rows:
        category = str(row.get(category_field, "")).strip() or "Missing category"
        segment = str(row.get(segment_field, "")).strip() or "Missing segment"
        value = _as_float(row.get(value_field))
        if value is None:
            warnings.append("missing_or_invalid_value")
            continue
        if len(category) > 24:
            warnings.append(f"long_category:{category[:24]}")
        if len(segment) > 24:
            warnings.append(f"long_segment:{segment[:24]}")
        clean.append({"category": category, "segment": segment, "value": value})

    if any(item["value"] < 0 for item in clean):
        warnings.append("negative_values_present")

    category_order: list[str] = []
    for item in clean:
        if item["category"] not in category_order:
            category_order.append(item["category"])
    if len(category_order) > max_categories:
        warnings.append(f"truncated_categories:{len(category_order)}->{max_categories}")
        allowed = set(category_order[:max_categories])
        clean = [item for item in clean if item["category"] in allowed]

    segment_order: list[str] = []
    for item in clean:
        if item["segment"] not in segment_order:
            segment_order.append(item["segment"])
    if len(segment_order) > max_segments:
        warnings.append(f"truncated_segments:{len(segment_order)}->{max_segments}")
        allowed = set(segment_order[:max_segments])
        clean = [item for item in clean if item["segment"] in allowed]

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
    visual_id = str(sample.get("visual_id") or template.get("template_id") or "stacked_bar_draft_v1")
    params = template.get("params", {}) or {}
    width = int(params.get("width", 1080))
    height = int(params.get("height", 860))
    value_format = str(params.get("value_format", "integer"))
    normalize_to_percent = bool(params.get("normalize_to_percent", False))
    palette = load_palette(template)

    clean_rows, warnings = _clean_rows(rows, template, sample)
    categories: list[str] = []
    segments: list[str] = []
    for item in clean_rows:
        if item["category"] not in categories:
            categories.append(item["category"])
        if item["segment"] not in segments:
            segments.append(item["segment"])

    colors = [
        palette["accent"],
        palette["accent_2"],
        palette["text"],
        palette["muted"],
        palette["warning"],
    ]

    values_by_key = {(item["category"], item["segment"]): item["value"] for item in clean_rows}
    if normalize_to_percent:
        totals = {
            category: sum(max(values_by_key.get((category, segment), 0.0), 0.0) for segment in segments)
            for category in categories
        }
        normalized: dict[tuple[str, str], float] = {}
        for category in categories:
            total = totals.get(category, 0.0)
            for segment in segments:
                raw = max(values_by_key.get((category, segment), 0.0), 0.0)
                normalized[(category, segment)] = (raw / total * 100.0) if total else 0.0
        values_by_key = normalized
        value_format = "percent"

    fig = plt.figure(figsize=(width / 150, height / 150), dpi=150)
    fig.patch.set_facecolor(palette["background"])
    ax = fig.add_axes([0.10, 0.24, 0.84, 0.62])
    ax.set_facecolor(palette["panel"])

    if categories and segments:
        x_positions = list(range(len(categories)))
        bottoms = [0.0 for _ in categories]
        for index, segment in enumerate(segments):
            values = [max(values_by_key.get((category, segment), 0.0), 0.0) for category in categories]
            ax.bar(
                x_positions,
                values,
                bottom=bottoms,
                color=colors[index % len(colors)],
                width=0.62,
                label=segment,
            )
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

        for x, total in zip(x_positions, bottoms):
            if total > 0:
                ax.text(
                    x,
                    total + max(bottoms) * 0.025,
                    _format_value(total, value_format),
                    color=palette["text"],
                    fontsize=10,
                    fontweight="bold",
                    ha="center",
                    va="bottom",
                )
        ax.set_xticks(x_positions)
        ax.set_xticklabels([_wrap_label(category) for category in categories], color=palette["text"], fontsize=10)
        y_max = max(bottoms) * 1.18 if max(bottoms) > 0 else 1
        ax.set_ylim(0, y_max)
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
        "renderer": "stacked_bar",
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
        "renderer": "stacked_bar",
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
