from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import load_palette, utc_now, write_json


def _as_float(value: Any, fallback: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return fallback
        return float(value)
    except Exception:
        return fallback


def _clean_rows(rows: list[dict[str, Any]], template: dict[str, Any], sample: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
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
        if len(label) > 38:
            warnings.append(f"long_label:{label[:38]}")
        clean.append({"label": label, "value": value, "group": group_value})

    reverse = sort != "ascending"
    clean = sorted(clean, key=lambda item: item["value"], reverse=reverse)
    if len(clean) > max_items:
        warnings.append(f"truncated_rows:{len(clean)}->{max_items}")
        clean = clean[:max_items]
    if any(item["value"] < 0 for item in clean):
        warnings.append("negative_values_present")

    return clean, warnings


def _wrap_label(label: str, width: int = 24) -> str:
    return "\n".join(textwrap.wrap(label, width=width, max_lines=2, placeholder="…"))


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
    width = int(params.get("width", 1080))
    height = int(params.get("height", 860))
    palette = load_palette(template)

    clean_rows, warnings = _clean_rows(rows, template, sample)
    labels = [_wrap_label(item["label"]) for item in clean_rows]
    values = [item["value"] for item in clean_rows]

    fig = plt.figure(figsize=(width / 150, height / 150), dpi=150)
    fig.patch.set_facecolor(palette["background"])
    ax = fig.add_axes([0.22, 0.10, 0.71, 0.82])
    ax.set_facecolor(palette["panel"])

    if clean_rows and max(values) > 0:
        ax.barh(range(len(clean_rows)), values, color=palette["accent"], height=0.58)
        ax.set_yticks(range(len(clean_rows)))
        ax.set_yticklabels(labels, color=palette["text"], fontsize=12)
        ax.invert_yaxis()
        max_value = max(values)
        x_limit = max_value * 1.18 if max_value else 1
        ax.set_xlim(0, x_limit)
        value_format = str(params.get("value_format", "integer"))
        for idx, value in enumerate(values):
            if value_format == "percent":
                value_label = f"{value:g}%"
            else:
                value_label = f"{value:,.0f}" if math.isfinite(value) else "0"
            ax.text(
                value + x_limit * 0.018,
                idx,
                value_label,
                color=palette["text"],
                fontsize=12,
                fontweight="bold",
                va="center",
            )
    else:
        warnings.append("empty_or_zero_rows")
        ax.text(
            0.5,
            0.5,
            "No data available",
            color=palette["muted"],
            fontsize=16,
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_yticks([])
        ax.set_xticks([])

    ax.xaxis.grid(True, color=palette["grid"], alpha=0.22)
    ax.tick_params(axis="x", colors=palette["muted"], labelsize=10)
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
        "renderer": "horizontal_bar",
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
        "renderer": "horizontal_bar",
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
