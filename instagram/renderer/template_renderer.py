from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .constants import DEFAULT_PALETTE, DEFAULT_PALETTES, FONT_MAP


@dataclass
class RenderResult:
    output_path: Path
    warnings: list[str]
    element_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)


def _resolve_color(value: str, palette: Mapping[str, str]) -> str:
    if value.startswith("{palette.") and value.endswith("}"):
        key = value[len("{palette.") : -1]
        return palette.get(key, "#FFFFFF")
    return value


def _font_path(alias: str) -> Path:
    return FONT_MAP.get(alias, FONT_MAP["default_regular"])


def _load_font(alias: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(_font_path(alias)), size=size)


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _draw_text(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    element: Mapping[str, Any],
    text: str,
    palette: Mapping[str, str],
) -> tuple[list[str], dict[str, Any]]:
    style = element.get("style", {}) or {}
    x = int(element["x"])
    y = int(element["y"])
    width = int(element["w"])
    height = int(element["h"])
    requested_font_size = int(style.get("font_size", 40))
    font_size = requested_font_size
    min_font_size = int(style.get("min_font_size", 18))
    max_lines = int(style.get("max_lines", 4))
    family = str(style.get("font_family", "default_regular"))
    align = str(style.get("align", "left"))
    valign = str(style.get("valign", "top"))
    shrink = bool(style.get("shrink_to_fit", True))
    line_spacing = int(style.get("line_spacing", 4))
    warnings: list[str] = []
    was_truncated = False

    while True:
        font = _load_font(family, font_size)
        lines = _wrap_text(draw, text, font, width)
        if len(lines) <= max_lines:
            break
        if not shrink or font_size <= min_font_size:
            lines = lines[:max_lines]
            if lines:
                lines[-1] = lines[-1].rstrip(" .") + "…"
            was_truncated = True
            warnings.append(f"text_truncated:{element['id']}")
            break
        font_size -= 2

    line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1]
    block_height = line_height * len(lines) + max(0, len(lines) - 1) * line_spacing
    draw_y = y
    if valign == "middle":
        draw_y = y + max(0, (height - block_height) // 2)
    elif valign == "bottom":
        draw_y = y + max(0, height - block_height)

    content = "\n".join(lines)
    anchor = None
    draw_x = x
    if align == "center":
        draw_x = x + width // 2
        anchor = "ma"
    elif align == "right":
        draw_x = x + width
        anchor = "ra"

    fill = _resolve_color(str(style.get("color", "{palette.text_primary}")), palette)
    draw.multiline_text(
        (draw_x, draw_y),
        content,
        font=font,
        fill=fill,
        spacing=line_spacing,
        align=align,
        anchor=anchor,
    )
    bbox = draw.multiline_textbbox(
        (draw_x, draw_y),
        content,
        font=font,
        spacing=line_spacing,
        align=align,
        anchor=anchor,
    )
    slot_bbox = [x, y, x + width, y + height]
    clipped = bool(
        bbox[0] < slot_bbox[0]
        or bbox[1] < slot_bbox[1]
        or bbox[2] > slot_bbox[2]
        or bbox[3] > slot_bbox[3]
    )
    if clipped:
        warnings.append(f"text_clipped:{element['id']}")

    metrics = {
        "element_id": str(element["id"]),
        "type": "text",
        "requested_font_size": requested_font_size,
        "final_font_size": font_size,
        "min_font_size": min_font_size,
        "font_shrunk": font_size < requested_font_size,
        "line_count": len(lines),
        "max_lines": max_lines,
        "truncated": was_truncated,
        "clipped": clipped,
        "text_bbox": [int(value) for value in bbox],
        "slot_bbox": slot_bbox,
        "text": text,
        "rendered_text": content,
    }
    return warnings, metrics


def _draw_image(
    canvas: Image.Image,
    element: Mapping[str, Any],
    source: Path,
) -> list[str]:
    warnings: list[str] = []
    if not source.exists():
        warnings.append(f"missing_image:{element['id']}:{source}")
        return warnings

    with Image.open(source) as image:
        image = image.convert("RGB")
        target_size = (int(element["w"]), int(element["h"]))
        fit = str(element.get("fit", "cover"))
        if fit == "contain":
            resized = ImageOps.contain(image, target_size)
            background = Image.new(
                "RGB",
                target_size,
                str(element.get("background", "#FFFFFF")),
            )
            offset = (
                (target_size[0] - resized.width) // 2,
                (target_size[1] - resized.height) // 2,
            )
            background.paste(resized, offset)
            image = background
        else:
            image = ImageOps.fit(image, target_size, method=Image.Resampling.LANCZOS)
        canvas.paste(image, (int(element["x"]), int(element["y"])))
    return warnings


def render_template(
    template: Mapping[str, Any],
    bindings: Mapping[str, Any],
    output_path: str | Path,
    palettes: Mapping[str, Mapping[str, str]] | None = None,
) -> RenderResult:
    width = int(template.get("width", 1080))
    height = int(template.get("height", 1350))
    palette_name = str(template.get("palette", DEFAULT_PALETTE))
    palette_map = palettes or DEFAULT_PALETTES
    palette = palette_map.get(palette_name, palette_map[DEFAULT_PALETTE])
    background = template.get("background", {"type": "solid", "color": "{palette.background}"})
    background_color = _resolve_color(str(background.get("color", "#0F2F24")), palette)
    canvas = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(canvas)
    warnings: list[str] = []
    element_metrics: dict[str, dict[str, Any]] = {}

    for element in template.get("elements", []):
        element_type = str(element.get("type"))
        placeholder = element.get("placeholder")
        value = bindings.get(placeholder, "") if placeholder else element.get("value", "")
        if element_type == "text":
            text_warnings, metrics = _draw_text(canvas, draw, element, str(value), palette)
            warnings.extend(text_warnings)
            element_metrics[str(element["id"])] = metrics
        elif element_type == "image":
            warnings.extend(_draw_image(canvas, element, Path(str(value))))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG")
    return RenderResult(output_path=output, warnings=warnings, element_metrics=element_metrics)
