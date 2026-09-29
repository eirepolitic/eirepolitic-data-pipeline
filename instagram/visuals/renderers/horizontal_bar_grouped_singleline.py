"""Review-only grouped horizontal bars for pq_monthly_overview_v1.

Changes requested during final visual review:
- category/name labels are forced to one line at a slightly smaller font size;
- the legend uses the full party names and is expanded across almost the full
  figure width instead of applying horizontal_bar_grouped's 13-character cap.

All chart rendering/QA otherwise delegates to horizontal_bar_grouped.
"""

from __future__ import annotations

from typing import Any

from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

from . import horizontal_bar_grouped as grouped

MAX_SINGLELINE_FONT_SIZE = 13
MIN_SINGLELINE_FONT_SIZE = 10
WIDE_LEGEND_FONT_SIZE = 10.5
WIDE_LEGEND_COLUMNS = 3


def _text_width(renderer: Any, text: str, font_size: int) -> float:
    props = FontProperties(size=font_size)
    width, _, _ = renderer.get_text_width_height_descent(str(text), props, ismath=False)
    return float(width)


def _select_label_layout_singleline(
    renderer: Any,
    raw_labels: list[str],
    *,
    width: int,
) -> tuple[int, list[str], list[bool], float, float]:
    max_label_width_px = width * (grouped.MAX_PLOT_LEFT - 0.035)

    selected_size = MIN_SINGLELINE_FONT_SIZE
    selected_widths = [_text_width(renderer, label, selected_size) for label in raw_labels]
    for font_size in range(MAX_SINGLELINE_FONT_SIZE, MIN_SINGLELINE_FONT_SIZE - 1, -1):
        widths = [_text_width(renderer, label, font_size) for label in raw_labels]
        if max(widths, default=0.0) <= max_label_width_px:
            selected_size = font_size
            selected_widths = widths
            break

    max_width = max(selected_widths, default=0.0)
    plot_left = min(
        grouped.MAX_PLOT_LEFT,
        max(grouped.MIN_PLOT_LEFT, (max_width + 42.0) / width),
    )
    return selected_size, list(raw_labels), [False] * len(raw_labels), max_width, plot_left


def render(
    template: dict[str, Any],
    sample: dict[str, Any],
    rows: list[dict[str, Any]],
    output_png,
    metadata_path,
    manifest_path,
    input_metadata: dict[str, Any],
):
    groups = [str(row.get("group") or "") for row in rows]
    present_groups = list(dict.fromkeys(groups))
    configured_order = [str(g) for g in (sample.get("group_legend_order") or [])]
    legend_order = [g for g in configured_order if g in present_groups]
    legend_order += [g for g in present_groups if g not in legend_order]
    legend_labels = {str(k): str(v) for k, v in (sample.get("group_legend_labels") or {}).items()}
    full_legend_texts = [legend_labels.get(g, g) for g in legend_order]

    original_layout = grouped._select_label_layout
    original_figure_legend = Figure.legend

    def _wide_figure_legend(self, handles=None, labels=None, *args, **kwargs):
        # grouped.render deliberately shortens legend labels to 13 characters.
        # Replace those display-only labels with the full configured party names
        # and distribute the three columns across almost the full figure width.
        if full_legend_texts and labels is not None and len(labels) == len(full_legend_texts):
            labels = full_legend_texts
            kwargs.update(
                {
                    "loc": "lower left",
                    "bbox_to_anchor": (0.035, grouped.LEGEND_ANCHOR_Y, 0.93, 0.11),
                    "ncol": min(len(full_legend_texts), WIDE_LEGEND_COLUMNS),
                    "mode": "expand",
                    "fontsize": WIDE_LEGEND_FONT_SIZE,
                    "columnspacing": 0.7,
                    "handletextpad": 0.35,
                    "borderaxespad": 0.0,
                }
            )
        return original_figure_legend(self, handles, labels, *args, **kwargs)

    grouped._select_label_layout = _select_label_layout_singleline
    Figure.legend = _wide_figure_legend
    try:
        return grouped.render(
            template,
            sample,
            rows,
            output_png,
            metadata_path,
            manifest_path,
            input_metadata,
        )
    finally:
        grouped._select_label_layout = original_layout
        Figure.legend = original_figure_legend
