from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from .common import load_palette, utc_now, write_json

PLOT_BOTTOM = 0.045
PLOT_RIGHT = 0.97
PLOT_HEIGHT = 0.91
MIN_PLOT_LEFT = 0.28
MAX_PLOT_LEFT = 0.40
MAX_CATEGORY_FONT_SIZE = 18
MIN_CATEGORY_FONT_SIZE = 14
MAX_VALUE_FONT_SIZE = 16
MIN_VALUE_FONT_SIZE = 14
AXIS_FONT_SIZE = 12
MAX_LABEL_LINES = 2
CLIP_TOLERANCE_PX = 3.0
VALUE_LABEL_TARGET_X_RATIO = 0.965


def _as_float(value: Any, fallback: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return fallback
        return float(value)
    except Exception:
        return fallback


def _clean_rows(
    rows: list[dict[str, Any]],
    template: dict[str, Any],
    sample: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    bindings = sample.get("bindings", {}) or {}
    label_field = str(bindings.get("label", "label"))
    value_field = str(bindings.get("value", "value"))
    group_field = bindings.get("group")
    params = template.get("params", {}) or {}
    max_items = int(params.get("max_items", 8))
    sort = str(params.get("sort", "descending"))
    warnings: list[str] = []
    clean: list[dict[str, Any]] = []
    for row in rows:
        label = str(row.get(label_field, "")).strip() or "Missing label"
        value = _as_float(row.get(value_field), 0.0)
        group_value = str(row.get(group_field, "")).strip() if group_field else ""
        clean.append({"label": label, "value": value, "group": group_value})
    reverse = sort != "ascending"
    clean = sorted(clean, key=lambda item: item["value"], reverse=reverse)
    if len(clean) > max_items:
        warnings.append(f"truncated_rows:{len(clean)}->{max_items}")
        clean = clean[:max_items]
    if any(item["value"] < 0 for item in clean):
        warnings.append("negative_values_present")
    return clean, warnings


def _text_width(renderer: Any, text: str, font_size: int) -> float:
    properties = FontProperties(size=font_size)
    width, _, _ = renderer.get_text_width_height_descent(text, properties, ismath=False)
    return float(width)


def _ellipsize(renderer: Any, text: str, font_size: int, max_width_px: float) -> str:
    suffix = "…"
    candidate = str(text)
    while candidate and _text_width(renderer, candidate + suffix, font_size) > max_width_px:
        candidate = candidate[:-1]
    return (candidate.rstrip() or "") + suffix


def _wrap_two_lines(
    renderer: Any,
    label: str,
    *,
    font_size: int,
    max_width_px: float,
    allow_truncate: bool,
) -> tuple[str, bool, float]:
    label = " ".join(str(label).split())
    if _text_width(renderer, label, font_size) <= max_width_px:
        return label, False, _text_width(renderer, label, font_size)

    # Only wrap at word boundaries. If a label cannot fit at a valid word
    # boundary, let the existing truncation/validation path reject it rather
    # than silently splitting a word for visual balance.
    split_points = [index for index, character in enumerate(label) if character == " "]
    best: tuple[float, str, str] | None = None
    for index in split_points:
        left = label[:index].rstrip()
        right = label[index:].lstrip()
        if not left or not right:
            continue
        left_width = _text_width(renderer, left, font_size)
        right_width = _text_width(renderer, right, font_size)
        widest = max(left_width, right_width)
        if widest <= max_width_px:
            score = widest + abs(left_width - right_width) * 0.15
            if best is None or score < best[0]:
                best = (score, left, right)
    if best is not None:
        _, left, right = best
        return f"{left}\n{right}", False, max(
            _text_width(renderer, left, font_size),
            _text_width(renderer, right, font_size),
        )

    if not allow_truncate:
        return label, True, _text_width(renderer, label, font_size)

    midpoint = max(1, len(label) // 2)
    split = min(range(1, len(label)), key=lambda value: abs(value - midpoint))
    left = _ellipsize(renderer, label[:split].rstrip(), font_size, max_width_px)
    right = _ellipsize(renderer, label[split:].lstrip(), font_size, max_width_px)
    return f"{left}\n{right}", True, max(
        _text_width(renderer, left, font_size),
        _text_width(renderer, right, font_size),
    )


def _select_label_layout(
    renderer: Any,
    raw_labels: list[str],
    *,
    width: int,
) -> tuple[int, list[str], list[bool], float, float]:
    max_label_width_px = width * (MAX_PLOT_LEFT - 0.035)
    for font_size in range(MAX_CATEGORY_FONT_SIZE, MIN_CATEGORY_FONT_SIZE - 1, -1):
        wrapped: list[str] = []
        truncated: list[bool] = []
        widths: list[float] = []
        for label in raw_labels:
            rendered, did_truncate, rendered_width = _wrap_two_lines(
                renderer,
                label,
                font_size=font_size,
                max_width_px=max_label_width_px,
                allow_truncate=False,
            )
            wrapped.append(rendered)
            truncated.append(did_truncate)
            widths.append(rendered_width)
        if not any(truncated):
            max_width = max(widths, default=0.0)
            plot_left = min(
                MAX_PLOT_LEFT,
                max(MIN_PLOT_LEFT, (max_width + 42.0) / width),
            )
            return font_size, wrapped, truncated, max_width, plot_left

    wrapped = []
    truncated = []
    widths = []
    for label in raw_labels:
        rendered, did_truncate, rendered_width = _wrap_two_lines(
            renderer,
            label,
            font_size=MIN_CATEGORY_FONT_SIZE,
            max_width_px=max_label_width_px,
            allow_truncate=True,
        )
        wrapped.append(rendered)
        truncated.append(did_truncate)
        widths.append(rendered_width)
    max_width = max(widths, default=0.0)
    plot_left = min(MAX_PLOT_LEFT, max(MIN_PLOT_LEFT, (max_width + 42.0) / width))
    return MIN_CATEGORY_FONT_SIZE, wrapped, truncated, max_width, plot_left


def _bbox_payload(bbox: Any) -> list[float]:
    return [round(float(bbox.x0), 2), round(float(bbox.y0), 2), round(float(bbox.x1), 2), round(float(bbox.y1), 2)]


def _outside(bbox: Any, container: Any, tolerance: float = CLIP_TOLERANCE_PX) -> bool:
    return bool(
        bbox.x0 < container.x0 - tolerance
        or bbox.y0 < container.y0 - tolerance
        or bbox.x1 > container.x1 + tolerance
        or bbox.y1 > container.y1 + tolerance
    )


def render(
    template: dict[str, Any],
    sample: dict[str, Any],
    rows: list[dict[str, Any]],
    output_png: str | Path,
    metadata_path: str | Path,
    manifest_path: str | Path,
    input_metadata: dict[str, Any],
) -> dict[str, Any]:
    visual_id = str(sample.get("visual_id") or template.get("template_id") or "horizontal_bar_draft_v1")
    params = template.get("params", {}) or {}
    width = int(params.get("width", 1032))
    height = int(params.get("height", 1210))
    palette = load_palette(template)
    clean_rows, warnings = _clean_rows(rows, template, sample)
    raw_labels = [str(item["label"]) for item in clean_rows]
    values = [item["value"] for item in clean_rows]

    fig = plt.figure(figsize=(width / 150, height / 150), dpi=150)
    fig.patch.set_facecolor(palette["background"])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    category_font_size, labels, truncated_flags, max_label_width_px, plot_left = _select_label_layout(
        renderer,
        raw_labels,
        width=width,
    )
    plot_bounds = [plot_left, PLOT_BOTTOM, PLOT_RIGHT - plot_left, PLOT_HEIGHT]
    ax = fig.add_axes(plot_bounds)
    ax.set_facecolor(palette["panel"])

    value_font_size = MAX_VALUE_FONT_SIZE
    if values and max(len(f"{value:,.0f}") for value in values) >= 7:
        value_font_size = MIN_VALUE_FONT_SIZE
    elif len(clean_rows) >= 6:
        value_font_size = 15

    bar_height = 0.0
    value_texts: list[Any] = []
    if clean_rows and max(values) > 0:
        bar_height = 0.72 if len(clean_rows) <= 4 else 0.62
        ax.barh(range(len(clean_rows)), values, color=palette["accent"], height=bar_height)
        ax.set_yticks(range(len(clean_rows)))
        ax.set_yticklabels(labels, color=palette["text"], fontsize=category_font_size)
        ax.invert_yaxis()
        max_value = max(values)
        x_limit = max_value * 1.16 if max_value else 1
        ax.set_xlim(0, x_limit)
        value_format = str(params.get("value_format", "integer"))
        for idx, value in enumerate(values):
            value_label = f"{value:g}%" if value_format == "percent" else (f"{value:,.0f}" if math.isfinite(value) else "0")
            value_texts.append(
                ax.annotate(
                    value_label,
                    xy=(value, idx),
                    xytext=(8, 0),
                    textcoords="offset points",
                    color=palette["text"],
                    fontsize=value_font_size,
                    fontweight="bold",
                    va="center",
                )
            )

        for _ in range(8):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            axes_bbox = ax.get_window_extent(renderer)
            max_ratio = max(
                (
                    (text.get_window_extent(renderer).x1 - axes_bbox.x0) / axes_bbox.width
                    for text in value_texts
                ),
                default=0.0,
            )
            if max_ratio <= VALUE_LABEL_TARGET_X_RATIO:
                break
            expansion = max(1.02, max_ratio / VALUE_LABEL_TARGET_X_RATIO)
            x_limit *= expansion
            ax.set_xlim(0, x_limit)
    else:
        warnings.append("empty_or_zero_rows")
        ax.text(0.5, 0.5, "No data available", color=palette["muted"], fontsize=20, ha="center", va="center", transform=ax.transAxes)
        ax.set_yticks([])
        ax.set_xticks([])

    ax.xaxis.grid(True, color=palette["grid"], alpha=0.22)
    ax.tick_params(axis="x", colors=palette["muted"], labelsize=AXIS_FONT_SIZE)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_bbox = fig.bbox
    axes_bbox = ax.get_window_extent(renderer)

    category_bounds: list[dict[str, Any]] = []
    for raw, rendered_label, truncated, text in zip(raw_labels, labels, truncated_flags, ax.get_yticklabels()):
        bbox = text.get_window_extent(renderer)
        category_bounds.append({
            "raw_label": raw,
            "rendered_label": rendered_label,
            "font_size": category_font_size,
            "line_count": rendered_label.count("\n") + 1,
            "truncated": bool(truncated),
            "bbox_px": _bbox_payload(bbox),
            "clipped_to_figure": _outside(bbox, figure_bbox),
        })

    value_bounds: list[dict[str, Any]] = []
    for value, text in zip(values, value_texts):
        bbox = text.get_window_extent(renderer)
        value_bounds.append({
            "value": value,
            "text": text.get_text(),
            "font_size": value_font_size,
            "bbox_px": _bbox_payload(bbox),
            "clipped_to_axes": _outside(bbox, axes_bbox),
            "clipped_to_figure": _outside(bbox, figure_bbox),
        })

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)

    created_at = utc_now()
    plot_area_ratio = round(plot_bounds[2] * plot_bounds[3], 4)
    plot_height_px = height * plot_bounds[3]
    bar_thickness_px = round((plot_height_px / max(len(clean_rows), 1)) * bar_height, 2) if clean_rows else 0.0
    max_value_label_x_ratio = max(
        ((item["bbox_px"][2] - axes_bbox.x0) / axes_bbox.width for item in value_bounds),
        default=0.0,
    )
    category_clipped_count = sum(1 for item in category_bounds if item["clipped_to_figure"])
    value_clipped_count = sum(
        1
        for item in value_bounds
        if item["clipped_to_axes"] or item["clipped_to_figure"]
    )
    truncated_label_count = sum(1 for item in category_bounds if item["truncated"])
    readability = {
        "category_label_font_size": category_font_size,
        "category_label_font_size_requested": MAX_CATEGORY_FONT_SIZE,
        "category_label_font_size_minimum": MIN_CATEGORY_FONT_SIZE,
        "category_label_font_shrunk": category_font_size < MAX_CATEGORY_FONT_SIZE,
        "value_label_font_size": value_font_size,
        "value_label_font_size_requested": MAX_VALUE_FONT_SIZE,
        "value_label_font_size_minimum": MIN_VALUE_FONT_SIZE,
        "value_label_font_shrunk": value_font_size < MAX_VALUE_FONT_SIZE,
        "axis_font_size": AXIS_FONT_SIZE,
        "bar_thickness_px": bar_thickness_px,
        "max_wrapped_label_lines": max((item["line_count"] for item in category_bounds), default=0),
        "max_value_label_x_ratio": round(max_value_label_x_ratio, 4),
        "displayed_item_count": len(clean_rows),
        "max_category_label_width_px": round(max_label_width_px, 2),
        "plot_left_ratio": round(plot_left, 4),
        "category_text_clipped_count": category_clipped_count,
        "value_text_clipped_count": value_clipped_count,
        "truncated_label_count": truncated_label_count,
        "category_text_bounds": category_bounds,
        "value_text_bounds": value_bounds,
    }
    if category_clipped_count:
        warnings.append(f"category_text_clipped:{category_clipped_count}")
    if value_clipped_count:
        warnings.append(f"value_text_clipped:{value_clipped_count}")
    if truncated_label_count:
        warnings.append(f"category_labels_truncated:{truncated_label_count}")

    metadata = {
        "visual_id": visual_id,
        "template_id": template.get("template_id"),
        "renderer": "horizontal_bar",
        "created_at": created_at,
        "input": input_metadata,
        "bindings": sample.get("bindings", {}),
        "filters": sample.get("filters", []),
        "grouping": sample.get("grouping", {}),
        "source_note": sample.get("source_note", ""),
        "attribution": sample.get("attribution", {}),
        "rows_rendered": clean_rows,
        "plot_bounds": plot_bounds,
        "plot_vertical_fill_ratio": plot_bounds[3],
        "plot_area_ratio": plot_area_ratio,
        "readability": readability,
        "warnings": warnings,
    }
    manifest = {
        "success": True,
        "visual_id": visual_id,
        "template_id": template.get("template_id"),
        "renderer": "horizontal_bar",
        "output_png": str(output_png),
        "metadata_path": str(metadata_path),
        "width": width,
        "height": height,
        "plot_bounds": plot_bounds,
        "plot_vertical_fill_ratio": plot_bounds[3],
        "plot_area_ratio": plot_area_ratio,
        "readability": readability,
        "warnings": warnings,
        "created_at": created_at,
    }
    write_json(metadata_path, metadata)
    write_json(manifest_path, manifest)
    return manifest
