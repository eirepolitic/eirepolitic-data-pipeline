from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTES = {
    "eirepolitic_dark": {
        "background": "#0f2f24",
        "panel": "#173d30",
        "text": "#f4ead7",
        "muted": "#cbbf9f",
        "bar": "#d8b45f",
        "grid": "#cbbf9f",
    },
    "eirepolitic_light": {
        "background": "#f4ead7",
        "panel": "#fff8eb",
        "text": "#102f25",
        "muted": "#5a5347",
        "bar": "#9b7327",
        "grid": "#5a5347",
    },
}


def normalise_rows(rows: list[dict[str, Any]], max_items: int, sort: str) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    clean: list[dict[str, Any]] = []
    for row in rows:
        label = str(row.get("label", "")).strip() or "Missing label"
        try:
            value = float(row.get("value", 0) or 0)
        except Exception:
            value = 0
            warnings.append(f"non_numeric_value:{label}")
        if len(label) > 42:
            warnings.append(f"long_label:{label[:42]}")
        clean.append({"label": label, "value": value})
    reverse = sort != "ascending"
    clean = sorted(clean, key=lambda r: r["value"], reverse=reverse)[:max_items]
    if len(rows) > max_items:
        warnings.append(f"truncated_rows:{len(rows)}->{max_items}")
    if any(row["value"] < 0 for row in clean):
        warnings.append("negative_values_present")
    return clean, warnings


def render(spec: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    params = spec.get("params", {})
    output = spec.get("output", {})
    rows, warnings = normalise_rows(
        spec.get("input", {}).get("rows", []),
        int(params.get("max_items", 10)),
        str(params.get("sort", "descending")),
    )
    width = int(params.get("width", 920))
    height = int(params.get("height", 720))
    palette_id = str(params.get("palette", "eirepolitic_dark"))
    palette = PALETTES.get(palette_id, PALETTES["eirepolitic_dark"])

    labels = [r["label"] for r in rows]
    values = [r["value"] for r in rows]
    fig = plt.figure(figsize=(width / 150, height / 150), dpi=150)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor(palette["background"])
    ax.set_facecolor(palette["panel"])

    if rows:
        ax.barh(range(len(rows)), values, color=palette["bar"])
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(labels, color=palette["text"], fontsize=10)
        ax.invert_yaxis()
        for i, value in enumerate(values):
            ax.text(value, i, f" {value:g}", va="center", color=palette["text"], fontsize=10)
    else:
        ax.text(0.5, 0.5, "No rows", ha="center", va="center", color=palette["text"])

    ax.set_title(str(params.get("title", "Horizontal bar chart")), color=palette["text"], loc="left", fontsize=16, pad=14)
    subtitle = str(params.get("subtitle", ""))
    if subtitle:
        ax.text(0, 1.01, subtitle, transform=ax.transAxes, color=palette["muted"], fontsize=10)
    ax.xaxis.grid(True, color=palette["grid"], alpha=0.25)
    ax.tick_params(axis="x", colors=palette["muted"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "media.png"
    fig.savefig(out_path, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)

    source_values = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator": "horizontal_bar_chart",
        "input_rows": rows,
        "params": params,
        "warnings": warnings,
    }
    manifest = {
        "success": True,
        "generator": "horizontal_bar_chart",
        "output_path": str(out_path),
        "width": width,
        "height": height,
        "warnings": warnings,
    }
    (output_dir / "source_values.json").write_text(json.dumps(source_values, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "render_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
