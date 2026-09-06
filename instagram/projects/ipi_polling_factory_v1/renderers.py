from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw

from instagram.factory.render_primitives import BG, draw_glossary, draw_title
from instagram.visuals.renderers.common import load_palette

PLOT_BOTTOM = 0.14
PLOT_TOP = 0.92
PLOT_LEFT = 0.28
PLOT_RIGHT = 0.97
CATEGORY_FONT_SIZE = 16
VALUE_FONT_SIZE = 15
AXIS_FONT_SIZE = 12
DEFAULT_TREND_COLORS = ["#d8b45f", "#f4ead7", "#79b8a4", "#df8a72", "#91a8d0"]


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_diverging(
    template: dict[str, Any],
    sample: dict[str, Any],
    rows: list[dict[str, Any]],
    output_png: str | Path,
    metadata_path: str | Path,
    manifest_path: str | Path,
    input_metadata: dict[str, Any],
) -> dict[str, Any]:
    params = template.get("params", {}) or {}
    width = int(params.get("width", 1032))
    height = int(params.get("height", 1210))
    max_items = int(params.get("max_items", 8))
    palette = load_palette(template)
    clean = []
    warnings: list[str] = []
    for row in rows:
        label = str(row.get("label") or "Missing label").strip()
        try:
            value = float(row.get("value", 0) or 0)
        except Exception:
            value = 0.0
            warnings.append(f"non_numeric_value:{label}")
        clean.append({"label": label, "value": value})
    clean = sorted(clean, key=lambda item: item["value"], reverse=True)[:max_items]

    labels = [item["label"] for item in clean]
    values = [item["value"] for item in clean]
    fig = plt.figure(figsize=(width / 150, height / 150), dpi=150)
    fig.patch.set_facecolor(palette["background"])
    ax = fig.add_axes([PLOT_LEFT, PLOT_BOTTOM, PLOT_RIGHT - PLOT_LEFT, 0.72])
    ax.set_facecolor(palette["background"])

    comparison_label = str(sample.get("comparison_label") or "").strip()
    if comparison_label:
        fig.text(0.52, 0.91, comparison_label, color=palette["text"], fontsize=13.5, fontweight="bold", ha="center", va="center")

    y = list(range(len(clean)))
    colors = [palette["accent"] if value >= 0 else palette["muted"] for value in values]
    if clean:
        ax.barh(y, values, color=colors, height=0.62)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, color=palette["text"], fontsize=CATEGORY_FONT_SIZE)
        ax.invert_yaxis()
        max_abs = max(abs(value) for value in values) or 0.5
        limit = max_abs * 1.45
        ax.set_xlim(-limit, limit)
        for idx, value in enumerate(values):
            text = f"{value:+.1f}%"
            if value >= 0:
                ax.annotate(text, xy=(value, idx), xytext=(8, 0), textcoords="offset points", color=palette["text"], fontsize=VALUE_FONT_SIZE, fontweight="bold", va="center", ha="left")
            else:
                ax.annotate(text, xy=(value, idx), xytext=(-8, 0), textcoords="offset points", color=palette["text"], fontsize=VALUE_FONT_SIZE, fontweight="bold", va="center", ha="right")
    else:
        ax.text(0.5, 0.5, str(sample.get("empty_message") or "No change data available"), color=palette["muted"], fontsize=20, ha="center", va="center", transform=ax.transAxes)
        ax.set_yticks([])

    ax.axvline(0, color=palette["accent"], linewidth=1.4, alpha=0.8)
    ax.xaxis.grid(True, color=palette["grid"], alpha=0.22)
    ax.tick_params(axis="x", colors=palette["muted"], labelsize=AXIS_FONT_SIZE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    source_note = str(sample.get("source_note") or "").strip()
    if source_note:
        fig.text(0.5, 0.025, source_note, color=palette["muted"], fontsize=8.5, ha="center", va="center")

    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    metadata = {"created_at": _utc_now(), "input": input_metadata, "rows": clean, "renderer": "diverging_bar_factory_v2", "comparison_label": comparison_label}
    manifest = {"success": True, "renderer": "diverging_bar_factory_v2", "output_path": str(output_png), "warnings": warnings, "displayed_item_count": len(clean), "comparison_label": comparison_label}
    _write_json(metadata_path, metadata)
    _write_json(manifest_path, manifest)
    return manifest


def render_trend(
    template: dict[str, Any],
    sample: dict[str, Any],
    series: list[dict[str, Any]],
    output_png: str | Path,
    metadata_path: str | Path,
    manifest_path: str | Path,
    input_metadata: dict[str, Any],
) -> dict[str, Any]:
    params = template.get("params", {}) or {}
    width = int(params.get("width", 1032))
    height = int(params.get("height", 1210))
    legend_variant = str(params.get("legend_variant") or sample.get("legend_variant") or "single_row")
    palette = load_palette(template)
    raw_colors = template.get("series_colors") or sample.get("series_colors") or DEFAULT_TREND_COLORS
    colors = [str(value) for value in raw_colors]
    warnings: list[str] = []

    fig = plt.figure(figsize=(width / 150, height / 150), dpi=150)
    fig.patch.set_facecolor(palette["background"])
    ax = fig.add_axes([0.11, 0.14, 0.82, 0.68])
    ax.set_facecolor(palette["background"])
    rendered = 0
    for idx, item in enumerate(series):
        label = str(item.get("label") or f"Series {idx + 1}")
        points = item.get("points") or []
        dates: list[pd.Timestamp] = []
        values: list[float] = []
        for point in points:
            date = pd.to_datetime(point.get("date"), format="%Y-%m-%d", errors="coerce")
            try:
                value = float(point.get("value"))
            except Exception:
                warnings.append(f"non_numeric_point:{label}")
                continue
            if pd.isna(date):
                warnings.append(f"invalid_date_point:{label}")
                continue
            dates.append(date)
            values.append(value)
        if not dates:
            continue
        ordered = sorted(zip(dates, values), key=lambda pair: pair[0])
        dates = [pair[0] for pair in ordered]
        values = [pair[1] for pair in ordered]
        color = colors[idx % len(colors)]
        ax.plot(dates, values, linewidth=2.9, color=color, label=label, zorder=2)
        ax.scatter(dates, values, s=42, color=color, edgecolors=palette["background"], linewidths=1.1, zorder=3)
        rendered += 1

    legend_columns = 3 if legend_variant == "two_row" else max(1, rendered)
    if rendered == 0:
        ax.text(0.5, 0.5, str(sample.get("empty_message") or "No trend data available"), color=palette["muted"], fontsize=20, ha="center", va="center", transform=ax.transAxes)
    else:
        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(
            handles,
            labels,
            frameon=False,
            fontsize=12.5,
            labelcolor=palette["text"],
            loc="upper center",
            bbox_to_anchor=(0.52, 0.91),
            ncol=legend_columns,
            columnspacing=1.2 if legend_variant == "single_row" else 1.8,
            handlelength=2.2,
            handletextpad=0.55,
            borderaxespad=0.0,
        )
        for line in legend.get_lines():
            line.set_linewidth(3.2)

    range_label = str(sample.get("range_label") or "").strip()
    if range_label:
        fig.text(0.52, 0.845, range_label, color=palette["muted"], fontsize=11.5, ha="center", va="center")

    ax.grid(True, color=palette["grid"], alpha=0.18)
    ax.tick_params(axis="x", colors=palette["muted"], labelsize=AXIS_FONT_SIZE, rotation=25)
    ax.tick_params(axis="y", colors=palette["muted"], labelsize=AXIS_FONT_SIZE)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=2, maxticks=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    for spine in ax.spines.values():
        spine.set_visible(False)
    source_note = str(sample.get("source_note") or "").strip()
    if source_note:
        fig.text(0.5, 0.025, source_note, color=palette["muted"], fontsize=8.5, ha="center", va="center")

    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    metadata = {
        "created_at": _utc_now(),
        "input": input_metadata,
        "series": series,
        "renderer": "line_trend_factory_v3_raw_polls",
        "legend_variant": legend_variant,
        "series_colors": colors[:rendered],
        "range_label": range_label,
    }
    manifest = {
        "success": True,
        "renderer": "line_trend_factory_v3_raw_polls",
        "output_path": str(output_png),
        "warnings": warnings,
        "series_rendered": rendered,
        "legend_variant": legend_variant,
        "series_colors": colors[:rendered],
        "range_label": range_label,
    }
    _write_json(metadata_path, metadata)
    _write_json(manifest_path, manifest)
    return manifest


def render_methodology(entries: list[tuple[str, str]], output_png: str | Path, *, title: str) -> dict[str, Any]:
    output_png = Path(output_png)
    metrics = draw_glossary(entries, output_png)
    image = Image.open(output_png).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((112, 34, 968, 169), fill=BG)
    draw_title(image, [title])
    image.save(output_png)
    return {
        "success": True,
        "renderer": "approved_glossary_methodology_v1",
        "output_path": str(output_png),
        "warnings": [],
        "glossary_metrics": metrics,
    }
