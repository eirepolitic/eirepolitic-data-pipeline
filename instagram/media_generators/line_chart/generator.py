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

PALETTES = {
    "eirepolitic_dark": {
        "background": "#0f2f24",
        "panel": "#173d30",
        "text": "#f4ead7",
        "muted": "#cbbf9f",
        "grid": "#cbbf9f",
    },
    "eirepolitic_light": {
        "background": "#f4ead7",
        "panel": "#fff8eb",
        "text": "#102f25",
        "muted": "#5a5347",
        "grid": "#5a5347",
    },
}


def render(spec: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    params = spec.get("params", {})
    series = spec.get("input", {}).get("series", [])
    width = int(params.get("width", 920))
    height = int(params.get("height", 820))
    palette_id = str(params.get("palette", "eirepolitic_dark"))
    palette = PALETTES.get(palette_id, PALETTES["eirepolitic_dark"])
    value_suffix = str(params.get("value_suffix", ""))
    warnings: list[str] = []

    fig = plt.figure(figsize=(width / 150, height / 150), dpi=150)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor(palette["background"])
    ax.set_facecolor(palette["panel"])

    rendered = 0
    for item in series:
        label = str(item.get("label", "Series")).strip() or "Series"
        points = item.get("points", [])
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
        ax.plot(dates, values, marker=None, linewidth=2.0, label=label)
        ax.scatter([dates[-1]], [values[-1]], s=20)
        ax.text(dates[-1], values[-1], f" {label} {values[-1]:g}{value_suffix}", color=palette["text"], fontsize=8, va="center")
        rendered += 1

    if rendered == 0:
        ax.text(0.5, 0.5, "No trend data", ha="center", va="center", color=palette["text"], transform=ax.transAxes)
    else:
        ax.legend(frameon=False, fontsize=8, labelcolor=palette["text"], loc="upper left")

    ax.set_title(str(params.get("title", "Trend")), color=palette["text"], loc="left", fontsize=16, pad=14)
    subtitle = str(params.get("subtitle", ""))
    if subtitle:
        ax.text(0, 1.01, subtitle, transform=ax.transAxes, color=palette["muted"], fontsize=10)
    ax.grid(True, color=palette["grid"], alpha=0.2)
    ax.tick_params(axis="x", colors=palette["muted"], labelrotation=25)
    ax.tick_params(axis="y", colors=palette["muted"])
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "media.png"
    fig.savefig(out_path, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)

    manifest = {
        "success": True,
        "generator": "line_chart",
        "output_path": str(out_path),
        "width": width,
        "height": height,
        "warnings": warnings,
        "series_rendered": rendered,
    }
    source_values = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator": "line_chart",
        "input_series": series,
        "params": params,
        "warnings": warnings,
    }
    (output_dir / "source_values.json").write_text(json.dumps(source_values, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "render_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
