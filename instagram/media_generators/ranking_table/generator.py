from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from instagram.renderer.constants import FONT_CANDIDATES

PALETTES = {
    "eirepolitic_dark": {
        "background": "#0f2f24",
        "panel": "#173d30",
        "panel_alt": "#214a3b",
        "text": "#f4ead7",
        "muted": "#cbbf9f",
        "accent": "#d8b45f",
        "border": "#f4ead7",
    },
    "eirepolitic_light": {
        "background": "#f4ead7",
        "panel": "#fff8eb",
        "panel_alt": "#eadfca",
        "text": "#0f2f24",
        "muted": "#4d6259",
        "accent": "#98733b",
        "border": "#0f2f24",
    },
}


def font_path(kind: str) -> str | None:
    key = "bold" if kind == "bold" else "regular"
    for path in FONT_CANDIDATES.get(key, []):
        if Path(path).exists():
            return path
    return None


def font(kind: str, size: int) -> ImageFont.ImageFont:
    path = font_path(kind)
    if path:
        return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def shorten(draw: ImageDraw.ImageDraw, text: Any, fnt: ImageFont.ImageFont, max_width: int) -> str:
    value = str(text or "").strip()
    if draw.textbbox((0, 0), value, font=fnt)[2] <= max_width:
        return value
    suffix = "…"
    while value and draw.textbbox((0, 0), value + suffix, font=fnt)[2] > max_width:
        value = value[:-1]
    return (value.rstrip() or "") + suffix


def normalise_rows(rows: list[dict[str, Any]], row_limit: int, sort: str) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    clean: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        name = str(row.get("name", row.get("label", ""))).strip() or "Missing name"
        sublabel = str(row.get("sublabel", row.get("party", ""))).strip()
        try:
            value = float(row.get("value", 0) or 0)
        except Exception:
            value = 0.0
            warnings.append(f"non_numeric_value:{name}")
        rank = row.get("rank", idx)
        if len(name) > 42:
            warnings.append(f"long_name:{name[:42]}")
        if len(sublabel) > 38:
            warnings.append(f"long_sublabel:{name}")
        clean.append({"rank": rank, "name": name, "sublabel": sublabel, "value": value})

    reverse = sort != "ascending"
    if sort in {"ascending", "descending"}:
        clean = sorted(clean, key=lambda r: r["value"], reverse=reverse)
        for idx, row in enumerate(clean, start=1):
            row["rank"] = idx
    if len(clean) > row_limit:
        warnings.append(f"truncated_rows:{len(clean)}->{row_limit}")
    return clean[:row_limit], warnings


def format_value(value: float, value_format: str) -> str:
    if value_format == "percent":
        return f"{value:g}%"
    if value_format == "integer":
        return f"{value:,.0f}"
    return f"{value:g}"


def render(spec: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    params = spec.get("params", {})
    width = int(params.get("width", 920))
    height = int(params.get("height", 720))
    row_limit = int(params.get("row_limit", params.get("max_items", 10)))
    row_height = int(params.get("row_height", 54))
    palette_id = str(params.get("palette", "eirepolitic_dark"))
    palette = PALETTES.get(palette_id, PALETTES["eirepolitic_dark"])
    rows, warnings = normalise_rows(spec.get("input", {}).get("rows", []), row_limit, str(params.get("sort", "descending")))

    img = Image.new("RGB", (width, height), palette["background"])
    draw = ImageDraw.Draw(img)
    margin = int(params.get("margin", 42))
    draw.rounded_rectangle((margin, margin, width - margin, height - margin), radius=26, fill=palette["panel"], outline=palette["border"], width=2)

    title_font = font("bold", int(params.get("title_font_size", 38)))
    subtitle_font = font("regular", int(params.get("subtitle_font_size", 22)))
    header_font = font("bold", 18)
    name_font = font("bold", int(params.get("name_font_size", 24)))
    sub_font = font("regular", int(params.get("sublabel_font_size", 18)))
    value_font = font("bold", int(params.get("value_font_size", 24)))

    y = margin + 30
    title = str(params.get("title", "Ranking"))
    draw.text((margin + 30, y), shorten(draw, title, title_font, width - margin * 2 - 60), font=title_font, fill=palette["text"])
    y += 48
    subtitle = str(params.get("subtitle", ""))
    if subtitle:
        draw.text((margin + 30, y), shorten(draw, subtitle, subtitle_font, width - margin * 2 - 60), font=subtitle_font, fill=palette["muted"])
        y += 38
    y += 10

    rank_x = margin + 34
    name_x = margin + 108
    value_x = width - margin - 185
    draw.text((rank_x, y), "#", font=header_font, fill=palette["accent"])
    draw.text((name_x, y), str(params.get("name_header", "Name")), font=header_font, fill=palette["accent"])
    draw.text((value_x, y), str(params.get("value_header", "Value")), font=header_font, fill=palette["accent"])
    y += 32

    value_format = str(params.get("value_format", "integer"))
    max_name_width = value_x - name_x - 24
    for idx, row in enumerate(rows):
        if y + row_height > height - margin - 28:
            warnings.append(f"rows_exceed_canvas:{len(rows)}")
            break
        fill = palette["panel_alt"] if idx % 2 == 0 else palette["panel"]
        draw.rounded_rectangle((margin + 22, y, width - margin - 22, y + row_height - 6), radius=14, fill=fill)
        draw.text((rank_x, y + 12), str(row["rank"]), font=value_font, fill=palette["accent"])
        draw.text((name_x, y + 7), shorten(draw, row["name"], name_font, max_name_width), font=name_font, fill=palette["text"])
        if row["sublabel"]:
            draw.text((name_x, y + 33), shorten(draw, row["sublabel"], sub_font, max_name_width), font=sub_font, fill=palette["muted"])
        value_text = format_value(float(row["value"]), value_format)
        value_width = draw.textbbox((0, 0), value_text, font=value_font)[2]
        draw.text((width - margin - 46 - value_width, y + 14), value_text, font=value_font, fill=palette["text"])
        y += row_height

    footer = str(params.get("footer", ""))
    if footer:
        footer_font = font("regular", 18)
        draw.text((margin + 30, height - margin - 34), shorten(draw, footer, footer_font, width - margin * 2 - 60), font=footer_font, fill=palette["muted"])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "media.png"
    img.save(out_path, format="PNG")

    source_values = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator": "ranking_table",
        "input_rows": rows,
        "params": params,
        "warnings": warnings,
    }
    manifest = {
        "success": True,
        "generator": "ranking_table",
        "output_path": str(out_path),
        "width": width,
        "height": height,
        "warnings": warnings,
    }
    (output_dir / "source_values.json").write_text(json.dumps(source_values, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "render_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
