from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw, ImageFont

PALETTE_ROOT = Path("instagram/templates/palettes")
FONT_CANDIDATES = {
    "regular": ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"],
    "bold": ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"],
}


@dataclass
class RenderResult:
    output_path: Path
    warnings: list[str]
    text_metrics: list[dict[str, Any]]


def _palette(palette_id: str) -> dict[str, str]:
    payload = json.loads((PALETTE_ROOT / f"{palette_id}.json").read_text(encoding="utf-8"))
    return dict(payload.get("colors") or {})


def _resolve(value: Any, palette: Mapping[str, str]) -> Any:
    if not isinstance(value, str):
        return value
    return re.sub(r"\{palette\.([A-Za-z0-9_]+)\}", lambda m: palette.get(m.group(1), m.group(0)), value)


def _font(kind: str, size: int) -> ImageFont.ImageFont:
    key = "bold" if kind in {"default_bold", "bold"} else "regular"
    for candidate in FONT_CANDIDATES[key]:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = str(text or "").split()
    if not words:
        return []
    lines, current = [], words[0]
    for word in words[1:]:
        probe = f"{current} {word}"
        if draw.textbbox((0, 0), probe, font=font)[2] <= max_width:
            current = probe
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _fit_text(draw: ImageDraw.ImageDraw, text: str, style: Mapping[str, Any], width: int, height: int):
    requested = int(style.get("font_size", 32))
    minimum = int(style.get("min_font_size", 16))
    max_lines = int(style.get("max_lines", 999))
    spacing = int(style.get("line_spacing", 8))
    size = requested
    while True:
        font = _font(str(style.get("font_family", "default_regular")), size)
        lines = _wrap(draw, text, font, width)
        bbox = draw.multiline_textbbox((0, 0), "\n".join(lines), font=font, spacing=spacing) if lines else (0, 0, 0, 0)
        fits = len(lines) <= max_lines and (bbox[2] - bbox[0]) <= width and (bbox[3] - bbox[1]) <= height
        if fits or size <= minimum or not bool(style.get("shrink_to_fit", False)):
            return font, lines, {"requested_font_size": requested, "actual_font_size": int(getattr(font, "size", size)), "line_count": len(lines), "truncated": False}
        size -= 2


def _draw_text(draw, element, bindings, palette, warnings):
    placeholder = element.get("placeholder")
    text = str(bindings.get(placeholder, "") if placeholder else element.get("text", ""))
    x, y, w, h = [int(element.get(k, 0)) for k in ("x", "y", "w", "h")]
    style = dict(element.get("style") or {})
    font, lines, metrics = _fit_text(draw, text, style, w, h)
    if not lines:
        return metrics
    spacing = int(style.get("line_spacing", 8))
    boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    heights = [box[3] - box[1] for box in boxes]
    total_h = sum(heights) + spacing * max(0, len(lines) - 1)
    cursor_y = y + max(0, (h - total_h) // 2) if style.get("valign") == "middle" else y
    rendered = []
    for line, box, line_h in zip(lines, boxes, heights):
        line_w = box[2] - box[0]
        align = style.get("align", "left")
        cursor_x = x + max(0, (w - line_w) // 2) if align == "center" else (x + max(0, w - line_w) if align == "right" else x)
        draw_x, draw_y = cursor_x - box[0], cursor_y - box[1]
        draw.text((draw_x, draw_y), line, font=font, fill=_resolve(style.get("color", "#000000"), palette))
        rendered.append(draw.textbbox((draw_x, draw_y), line, font=font))
        cursor_y += line_h + spacing
    clipped = any(b[0] < x or b[1] < y or b[2] > x + w or b[3] > y + h for b in rendered)
    if clipped:
        warnings.append(f"text_clipped:{element.get('id')}")
    return {**metrics, "element_id": element.get("id"), "source_text": text, "rendered_lines": lines, "clipped": clipped}


def _draw_image(base: Image.Image, draw: ImageDraw.ImageDraw, element: Mapping[str, Any], bindings: Mapping[str, Any], palette: Mapping[str, str], warnings: list[str]) -> None:
    placeholder = element.get("placeholder")
    reference = str(bindings.get(placeholder, "") if placeholder else element.get("source", ""))
    x, y, w, h = [int(element.get(k, 0)) for k in ("x", "y", "w", "h")]
    background = _resolve(element.get("background"), palette)
    if background:
        draw.rectangle((x, y, x + w, y + h), fill=background)
    path = Path(reference)
    if not path.exists():
        warnings.append(f"image_not_found:{reference}")
        return
    image = Image.open(path).convert("RGBA")
    if element.get("fit", "cover") == "contain":
        image.thumbnail((w, h), Image.Resampling.LANCZOS)
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        layer.alpha_composite(image, ((w - image.width) // 2, (h - image.height) // 2))
        image = layer
    else:
        image = image.resize((w, h), Image.Resampling.LANCZOS)
    base.paste(image, (x, y), image.getchannel("A"))


def render_template(template: Mapping[str, Any], bindings: Mapping[str, Any], output_path: str | Path) -> RenderResult:
    palette = _palette(str(template.get("palette", "eirepolitic_dark")))
    width, height = int(template["width"]), int(template["height"])
    bg = _resolve((template.get("background") or {}).get("color", "#ffffff"), palette)
    image = Image.new("RGBA", (width, height), bg)
    draw = ImageDraw.Draw(image)
    warnings: list[str] = []
    text_metrics: list[dict[str, Any]] = []
    for element in template.get("elements", []):
        kind = element.get("type")
        if kind == "rectangle":
            x, y, w, h = [int(element.get(k, 0)) for k in ("x", "y", "w", "h")]
            draw.rectangle((x, y, x + w, y + h), fill=_resolve(element.get("fill", "#000000"), palette))
        elif kind == "text":
            text_metrics.append(_draw_text(draw, element, bindings, palette, warnings))
        elif kind == "image":
            _draw_image(image, draw, element, bindings, palette, warnings)
        else:
            warnings.append(f"unsupported_element:{element.get('id')}:{kind}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path, format="PNG")
    return RenderResult(output_path=output_path, warnings=warnings, text_metrics=text_metrics)
