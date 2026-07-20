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


def _wrap_title(value: str) -> str:
    return "\n".join(textwrap.wrap(value, width=22, max_lines=2, placeholder="…"))


def _clean_rows(rows: list[dict[str, Any]], template: dict[str, Any], sample: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    bindings = sample.get("bindings", {}) or {}
    group_field = str(bindings.get("group", "group"))
    x_field = str(bindings.get("x", "period"))
    value_field = str(bindings.get("value", "value"))
    params = template.get("params", {}) or {}
    max_groups = int(params.get("max_groups", 6))
    max_points = int(params.get("max_points_per_group", 12))

    warnings: list[str] = []
    clean: list[dict[str, Any]] = []
    for row in rows:
        group = str(row.get(group_field, "")).strip() or "Missing group"
        x_value = str(row.get(x_field, "")).strip()
        value = _as_float(row.get(value_field))
        if not x_value:
            warnings.append(f"missing_x:{group[:24]}")
            continue
        if value is None:
            warnings.append(f"missing_or_invalid_value:{group[:24]}")
            continue
        if value < 0:
            warnings.append("negative_values_present")
        if len(group) > 32:
            warnings.append(f"long_group:{group[:32]}")
        clean.append({"group": group, "x": x_value, "value": value})

    group_order: list[str] = []
    for item in clean:
        if item["group"] not in group_order:
            group_order.append(item["group"])
    if len(group_order) > max_groups:
        warnings.append(f"truncated_groups:{len(group_order)}->{max_groups}")
        allowed = set(group_order[:max_groups])
        clean = [item for item in clean if item["group"] in allowed]

    trimmed: list[dict[str, Any]] = []
    for group in group_order[:max_groups]:
        group_rows = [item for item in clean if item["group"] == group]
        if len(group_rows) > max_points:
            warnings.append(f"truncated_points:{group}:{len(group_rows)}->{max_points}")
            group_rows = group_rows[:max_points]
        trimmed.extend(group_rows)
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
    visual_id = str(sample.get("visual_id") or template.get("template_id") or "small_multiples_draft_v1")
    params = template.get("params", {}) or {}
    width = int(params.get("width", 1080))
    height = int(params.get("height", 860))
    columns = max(1, int(params.get("columns", 2)))
    show_markers = bool(params.get("show_markers", True))
    palette = load_palette(template)

    clean_rows, warnings = _clean_rows(rows, template, sample)
    groups: list[str] = []
    for item in clean_rows:
        if item["group"] not in groups:
            groups.append(item["group"])

    rows_count = max(1, math.ceil(max(len(groups), 1) / columns))
    fig = plt.figure(figsize=(width / 150, height / 150), dpi=150)
    fig.patch.set_facecolor(palette["background"])

    left = 0.08
    right = 0.94
    bottom = 0.08
    top = 0.92
    h_gap = 0.06
    v_gap = 0.08
    cell_w = (right - left - h_gap * (columns - 1)) / columns
    cell_h = (top - bottom - v_gap * (rows_count - 1)) / rows_count

    if not groups:
        warnings.append("empty_rows")
        ax = fig.add_axes([left, bottom, right - left, top - bottom])
        ax.set_facecolor(palette["panel"])
        ax.text(0.5, 0.5, "No data available", color=palette["muted"], fontsize=16, ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    else:
        all_values = [float(item["value"]) for item in clean_rows]
        y_min = min(all_values)
        y_max = max(all_values)
        if y_min == y_max:
            y_min -= 1
            y_max += 1
        y_pad = (y_max - y_min) * 0.12
        for index, group in enumerate(groups):
            row_index = index // columns
            col_index = index % columns
            x0 = left + col_index * (cell_w + h_gap)
            y0 = top - (row_index + 1) * cell_h - row_index * v_gap
            ax = fig.add_axes([x0, y0, cell_w, cell_h])
            ax.set_facecolor(palette["panel"] if index % 2 == 0 else palette["panel_alt"])
            group_rows = [item for item in clean_rows if item["group"] == group]
            x_labels = [item["x"] for item in group_rows]
            y_values = [float(item["value"]) for item in group_rows]
            x_positions = list(range(len(x_labels)))
            ax.plot(
                x_positions,
                y_values,
                color=palette["accent"],
                linewidth=2.6,
                marker="o" if show_markers else None,
                markersize=5,
            )
            ax.set_title(_wrap_title(group), color=palette["text"], fontsize=10, fontweight="bold", pad=8)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.yaxis.grid(True, color=palette["grid"], alpha=0.18)
            ax.xaxis.grid(False)
            ax.tick_params(axis="y", colors=palette["muted"], labelsize=7)
            if len(x_labels) <= 6:
                ax.set_xticks(x_positions)
                ax.set_xticklabels(x_labels, color=palette["muted"], fontsize=7, rotation=35, ha="right")
            else:
                ax.set_xticks([0, len(x_labels) - 1])
                ax.set_xticklabels([x_labels[0], x_labels[-1]], color=palette["muted"], fontsize=7)
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
        "renderer": "small_multiples",
        "created_at": created_at,
        "input": input_metadata,
        "bindings": sample.get("bindings", {}),
        "filters": sample.get("filters", []),
        "grouping": sample.get("grouping", {}),
        "source_note": sample.get("source_note", ""),
        "attribution": sample.get("attribution", {}),
        "rows_rendered": clean_rows,
        "groups_rendered": groups,
        "warnings": warnings,
    }
    manifest = {
        "success": True,
        "visual_id": visual_id,
        "template_id": template.get("template_id"),
        "renderer": "small_multiples",
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
