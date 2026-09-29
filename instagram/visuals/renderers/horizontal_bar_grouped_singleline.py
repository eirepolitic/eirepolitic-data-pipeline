"""Review-only grouped horizontal bars with single-line category labels.

Used by pq_monthly_overview_v1 during final visual review. It delegates all
rendering/QA to horizontal_bar_grouped and only replaces its category-label
layout selector so names stay on one line at a slightly smaller font size.
"""

from __future__ import annotations

from typing import Any

from matplotlib.font_manager import FontProperties

from . import horizontal_bar_grouped as grouped

MAX_SINGLELINE_FONT_SIZE = 13
MIN_SINGLELINE_FONT_SIZE = 10


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


def render(*args, **kwargs):
    original = grouped._select_label_layout
    grouped._select_label_layout = _select_label_layout_singleline
    try:
        return grouped.render(*args, **kwargs)
    finally:
        grouped._select_label_layout = original
