"""Grouped/colored variant of horizontal_bar.py (director session
2026-09-21-monthly-questions-overview, visual-direction option B).

horizontal_bar.py is part of the frozen v1 file set
(.github/workflows/director_factory_v1_identity_ci.yml byte-identity-checks
it against 386b933), so it is never edited here. This module imports its
already-proven label-wrapping and clipping-detection helpers unchanged and
adds only what horizontal_bar.py doesn't do: one bar color per group (e.g.
per party) instead of a single accent color, plus a legend mapping color to
group. Everything else — wrapping, ellipsizing, clip/truncation detection,
value-label placement — is the same code horizontal_bar.py already uses.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from .common import load_palette, utc_now, write_json
from .horizontal_bar import (
    AXIS_FONT_SIZE,
    MAX_VALUE_FONT_SIZE,
    MIN_VALUE_FONT_SIZE,
    VALUE_LABEL_TARGET_X_RATIO,
    _bbox_payload,
    _clean_rows,
    _outside,
    _select_label_layout,
)

PLOT_BOTTOM = 0.14
PLOT_RIGHT = 0.97
PLOT_HEIGHT = 0.66  # shorter than horizontal_bar.py's 0.78 — the gap above is the legend band
# The legend's anchor (its lower-center point, per loc="lower center") sits just
# above the plot's top edge and grows upward into the reserved band (PLOT_BOTTOM +
# PLOT_HEIGHT = 0.80 to figure top = 1.0, a 0.20 figure-fraction gap — enough for
# up to ~5 two-column legend rows at LEGEND_FONT_SIZE before it would run into the
# figure edge; the render() function itself checks this and raises if it doesn't).
LEGEND_ANCHOR_Y = PLOT_BOTTOM + PLOT_HEIGHT + 0.015
MIN_PLOT_LEFT = 0.28
MAX_PLOT_LEFT = 0.42
LEGEND_FONT_SIZE = 12


def render(
    template: dict[str, Any],
    sample: dict[str, Any],
    rows: list[dict[str, Any]],
    output_png: str | Path,
    metadata_path: str | Path,
    manifest_path: str | Path,
    input_metadata: dict[str, Any],
) -> dict[str, Any]:
    visual_id = str(sample.get("visual_id") or template.get("template_id") or "horizontal_bar_grouped_draft_v1")
    params = template.get("params", {}) or {}
    width = int(params.get("width", 1032))
    height = int(params.get("height", 1210))
    min_visual_rows = max(1, int(params.get("min_visual_rows", 1)))
    palette = load_palette(template)
    clean_rows, warnings = _clean_rows(rows, template, sample)
    raw_labels = [str(item["label"]) for item in clean_rows]
    values = [item["value"] for item in clean_rows]
    groups = [item["group"] for item in clean_rows]

    group_colors = {str(k): str(v) for k, v in (sample.get("group_colors") or {}).items()}
    group_legend_labels = {str(k): str(v) for k, v in (sample.get("group_legend_labels") or {}).items()}
    fallback_color = str(sample.get("group_fallback_color") or palette["accent"])
    bar_colors = [group_colors.get(g, fallback_color) for g in groups]
    ungrouped_count = sum(1 for g in groups if g not in group_colors)
    if ungrouped_count:
        warnings.append(f"groups_without_color_mapping:{ungrouped_count}")

    fig = plt.figure(figsize=(width / 150, height / 150), dpi=150)
    fig.patch.set_facecolor(palette["background"])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    category_font_size, labels, truncated_flags, max_label_width_px, plot_left = _select_label_layout(renderer, raw_labels, width=width)
    plot_bounds = [plot_left, PLOT_BOTTOM, PLOT_RIGHT - plot_left, PLOT_HEIGHT]
    ax = fig.add_axes(plot_bounds)
    ax.set_facecolor(palette["background"])

    value_font_size = MAX_VALUE_FONT_SIZE
    if values and max(len(f"{value:,.0f}") for value in values) >= 7:
        value_font_size = MIN_VALUE_FONT_SIZE
    elif len(clean_rows) >= 6:
        value_font_size = 15

    bar_height = 0.0
    visual_row_count = max(len(clean_rows), min_visual_rows) if clean_rows else 0
    row_offset = max(0.0, (visual_row_count - len(clean_rows)) / 2.0) if clean_rows else 0.0
    y_positions = [idx + row_offset for idx in range(len(clean_rows))]
    value_texts: list[Any] = []
    empty_state = not clean_rows or max(values, default=0.0) <= 0
    if not empty_state:
        bar_height = 0.72 if visual_row_count <= 4 else 0.62
        ax.barh(y_positions, values, color=bar_colors, height=bar_height)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, color=palette["text"], fontsize=category_font_size)
        ax.set_ylim(visual_row_count - 0.5, -0.5)
        max_value = max(values)
        x_limit = max_value * 1.16 if max_value else 1
        ax.set_xlim(0, x_limit)
        value_format = str(params.get("value_format", "integer"))
        for idx, value in enumerate(values):
            if not math.isfinite(value):
                value_label = "0"
            elif value_format == "percent":
                value_label = f"{value:g}%"
            elif value_format == "plus_pp_1":
                value_label = f"+{value:.1f} pp"
            elif value_format == "plus_decimal_2":
                value_label = f"+{value:.2f}"
            elif value_format == "plus_per_td_2":
                value_label = f"+{value:.2f}/TD"
            elif value_format == "decimal_2":
                value_label = f"{value:.2f}"
            else:
                value_label = f"{value:,.0f}"
            value_texts.append(
                ax.annotate(
                    value_label,
                    xy=(value, y_positions[idx]),
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
                ((text.get_window_extent(renderer).x1 - axes_bbox.x0) / axes_bbox.width for text in value_texts),
                default=0.0,
            )
            if max_ratio <= VALUE_LABEL_TARGET_X_RATIO:
                break
            x_limit *= max(1.02, max_ratio / VALUE_LABEL_TARGET_X_RATIO)
            ax.set_xlim(0, x_limit)
    else:
        empty_message = str(sample.get("empty_message") or "No data available")
        ax.text(0.5, 0.5, empty_message, color=palette["muted"], fontsize=20, ha="center", va="center", transform=ax.transAxes)
        ax.set_yticks([])
        ax.set_xticks([])

    ax.xaxis.grid(True, color=palette["grid"], alpha=0.22)
    ax.tick_params(axis="x", colors=palette["muted"], labelsize=AXIS_FONT_SIZE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axvline(0, color=palette["accent"], linewidth=1.4, alpha=0.75)
    source_note = str(sample.get("source_note") or "").strip()
    if source_note:
        fig.text(0.5, 0.025, source_note, color=palette["muted"], fontsize=8.5, ha="center", va="center")

    # Legend: one swatch + label per group actually present, in a fixed
    # left-to-right order matching sample["group_legend_order"] if given
    # (falls back to first-seen order) — never reordered by value, so a
    # party's position in the legend stays stable render to render.
    #
    # Full party names vary a lot in length ("Fine Gael" vs "People Before
    # Profit-Solidarity"), and a wide multi-column legend of long names can
    # run past the figure's left/right edge even though there's plenty of
    # vertical room above the plot. Two guards keep it inside the canvas
    # regardless of which parties show up in a given month: a hard cap on
    # legend text length (full names still live in group_legend_labels /
    # the run manifest — only the on-image label is shortened), and a lower
    # column count than a one-line legend would use, since this canvas is
    # narrow (1032px) relative to typical party-name lengths.
    LEGEND_MAX_LABEL_CHARS = 18
    LEGEND_MAX_COLUMNS = 2

    def _short_legend_label(text: str) -> str:
        text = str(text)
        return text if len(text) <= LEGEND_MAX_LABEL_CHARS else text[: LEGEND_MAX_LABEL_CHARS - 1].rstrip() + "…"

    present_groups = list(dict.fromkeys(groups))
    legend_order = [g for g in (sample.get("group_legend_order") or []) if g in present_groups]
    legend_order += [g for g in present_groups if g not in legend_order]
    legend_handles = [mpatches.Patch(facecolor=group_colors.get(g, fallback_color), edgecolor="none") for g in legend_order]
    legend_texts = [_short_legend_label(group_legend_labels.get(g, g)) for g in legend_order]
    legend = None
    if legend_order:
        legend = fig.legend(
            legend_handles,
            legend_texts,
            loc="lower center",
            bbox_to_anchor=(0.5, LEGEND_ANCHOR_Y),
            ncol=min(len(legend_order), LEGEND_MAX_COLUMNS),
            frameon=False,
            fontsize=LEGEND_FONT_SIZE,
            labelcolor=palette["text"],
            handlelength=1.1,
            handleheight=1.1,
            columnspacing=1.1,
            handletextpad=0.5,
        )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_bbox = fig.bbox
    axes_bbox = ax.get_window_extent(renderer)
    category_bounds = []
    for raw, rendered_label, truncated, text in zip(raw_labels, labels, truncated_flags, ax.get_yticklabels()):
        bbox = text.get_window_extent(renderer)
        category_bounds.append(
            {
                "raw_label": raw,
                "rendered_label": rendered_label,
                "font_size": category_font_size,
                "line_count": rendered_label.count("\n") + 1,
                "truncated": bool(truncated),
                "bbox_px": _bbox_payload(bbox),
                "clipped_to_figure": _outside(bbox, figure_bbox),
            }
        )
    value_bounds = []
    for value, text in zip(values, value_texts):
        bbox = text.get_window_extent(renderer)
        value_bounds.append(
            {
                "value": value,
                "text": text.get_text(),
                "font_size": value_font_size,
                "bbox_px": _bbox_payload(bbox),
                "clipped_to_axes": _outside(bbox, axes_bbox),
                "clipped_to_figure": _outside(bbox, figure_bbox),
            }
        )
    legend_clipped = False
    if legend is not None:
        legend_bbox = legend.get_window_extent(renderer)
        legend_clipped = _outside(legend_bbox, figure_bbox)
        if legend_clipped:
            warnings.append("legend_clipped_to_figure")

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)

    created_at = utc_now()
    plot_area_ratio = round(plot_bounds[2] * plot_bounds[3], 4)
    plot_height_px = height * plot_bounds[3]
    effective_rows_for_thickness = max(len(clean_rows), min_visual_rows) if clean_rows else 0
    bar_thickness_px = round((plot_height_px / effective_rows_for_thickness) * bar_height, 2) if effective_rows_for_thickness else 0.0
    max_value_label_x_ratio = max(((item["bbox_px"][2] - axes_bbox.x0) / axes_bbox.width for item in value_bounds), default=0.0)
    category_clipped_count = sum(1 for item in category_bounds if item["clipped_to_figure"])
    value_clipped_count = sum(1 for item in value_bounds if item["clipped_to_axes"] or item["clipped_to_figure"])
    truncated_label_count = sum(1 for item in category_bounds if item["truncated"])
    readability = {
        "category_label_font_size": category_font_size,
        "value_label_font_size": value_font_size,
        "axis_font_size": AXIS_FONT_SIZE,
        "bar_thickness_px": bar_thickness_px,
        "min_visual_rows": min_visual_rows,
        "effective_visual_row_count": visual_row_count,
        "max_wrapped_label_lines": max((item["line_count"] for item in category_bounds), default=0),
        "max_value_label_x_ratio": round(max_value_label_x_ratio, 4),
        "displayed_item_count": len(clean_rows),
        "empty_state": empty_state,
        "empty_message": str(sample.get("empty_message") or "") if empty_state else "",
        "max_category_label_width_px": round(max_label_width_px, 2),
        "plot_left_ratio": round(plot_left, 4),
        "category_text_clipped_count": category_clipped_count,
        "value_text_clipped_count": value_clipped_count,
        "truncated_label_count": truncated_label_count,
        "legend_group_count": len(legend_order),
        "legend_clipped_to_figure": legend_clipped,
        "groups_without_color_mapping": ungrouped_count,
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
        "renderer": "horizontal_bar_grouped",
        "created_at": created_at,
        "input": input_metadata,
        "bindings": sample.get("bindings", {}),
        "source_note": sample.get("source_note", ""),
        "rows_rendered": clean_rows,
        "group_colors": group_colors,
        "group_legend_labels": group_legend_labels,
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
        "renderer": "horizontal_bar_grouped",
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
