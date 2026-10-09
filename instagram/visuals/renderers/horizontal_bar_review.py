"""Review-only wrapper around the frozen horizontal_bar renderer.

For pq_monthly_overview_v1's party-per-TD slide only:
- display value labels as whole numbers while preserving underlying bar values;
- render value labels at normal weight;
- shorten People Before Profit-Solidarity to People Before Profit for display.

All other horizontal-bar slides delegate unchanged to the frozen renderer.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from matplotlib.axes import Axes

from . import horizontal_bar as base


def _is_party_per_td(sample: dict[str, Any], input_metadata: dict[str, Any]) -> bool:
    visual_id = str(sample.get("visual_id") or "")
    metric_id = str(input_metadata.get("metric_id") or "")
    return "party_per_td" in visual_id or metric_id == "party_per_td"


def _short_party_label(label: Any) -> str:
    text = str(label)
    if text in {
        "People Before Profit-Solidarity",
        "People Before Profit–Solidarity",
        "People Before Profit — Solidarity",
        "People Before Profit-Solidarity Alliance",
    }:
        return "People Before Profit"
    return text


def render(
    template: dict[str, Any],
    sample: dict[str, Any],
    rows: list[dict[str, Any]],
    output_png,
    metadata_path,
    manifest_path,
    input_metadata: dict[str, Any],
):
    if not _is_party_per_td(sample, input_metadata):
        return base.render(
            template,
            sample,
            rows,
            output_png,
            metadata_path,
            manifest_path,
            input_metadata,
        )

    review_template = deepcopy(template)
    review_template.setdefault("params", {})["value_format"] = "integer"

    label_field = str((sample.get("bindings") or {}).get("label", "label"))
    review_rows = []
    for row in rows:
        copied = dict(row)
        copied[label_field] = _short_party_label(copied.get(label_field, ""))
        review_rows.append(copied)

    original_annotate = Axes.annotate

    def _normal_weight_annotate(self, *args, **kwargs):
        if kwargs.get("fontweight") == "bold":
            kwargs["fontweight"] = "normal"
        return original_annotate(self, *args, **kwargs)

    Axes.annotate = _normal_weight_annotate
    try:
        return base.render(
            review_template,
            sample,
            review_rows,
            output_png,
            metadata_path,
            manifest_path,
            input_metadata,
        )
    finally:
        Axes.annotate = original_annotate
