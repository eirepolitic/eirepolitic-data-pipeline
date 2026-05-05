from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .constants import FONT_CANDIDATES
from .util import draw_wrapped_text, fit_cover, rounded_panel

PALETTE_DIR = Path("instagram/templates/palettes")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_palette(template: Mapping[str, Any], override: str | None = None) -> dict[str, str]:
    palette_id = override or str(template.get("palette") or "eirepolitic_dark")
    path = PALETTE_DIR / f"{palette_id}.json"
    palette = load_json(path)
    return dict(palette.get("colors", {}))


def resolve_value(value: Any, palette: Mapping[str, str]) -> Any:
    if not isinstance(value, str):
        return value
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key.startswith("palette."):
            return str(palette.get(key.split(".", 1)[1], ""))
        return match.group(0)
    return re.sub(r"\{([^{}]+)\}", repl, value)


def font_path(kind: str) -> str | None:
    family = "bold" if kind == "default_bold" else "regular"
    for candidate in FONT_CANDIDATES[family]:
        if Path(candidate).exists():
            return candidate
    return None


def load_font(size: int, family: str = "default") -> ImageFont.ImageFont:
    path = font_path(family)
    if path:
        return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def image_from_reference(reference: str | None, timeout: int = 20) -> Image.Image | None:
    if not reference:
        return None
    if reference.startswith(("http://", "https://")):
        response = requests.get(reference, timeout=timeout)
        response.raise_for_status()
        from io import BytesIO
        return Image.open(BytesIO(response.content)).convert("RGBA")
    path = Path(reference)
    if path.exists():
        return Image.open(path).convert("RGBA")
    return None


def contain_image(image: Image.Image, width: int, height: int, background: str | None = None) -> Image.Image:
    canvas = Image.new("RGBA", (width, height), background or (0, 0, 0, 0))
    image = ImageOps.contain(image.convert("RGBA"), (width, height), method=Image.Resampling.LANCZOS)
    canvas.alpha_composite(image, ((width - image.width) // 2, (height - image.height) // 2))
    return canvas


def mask_rounded(image: Image.Image, radius: int) -> Image.Image:
    if radius <= 0:
        return image
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, image.width, image.height), radius=radius, fill=255)
    out = image.convert("RGBA")
    out.putalpha(mask)
    return out


def text_for_element(element: Mapping[str, Any], bindings: Mapping[str, Any]) -> str:
    if "text" in element:
        return str(element.get("text") or "")
    placeholder = element.get("placeholder")
    return str(bindings.get(str(placeholder), "")) if placeholder else ""


def render_text(draw: ImageDraw.ImageDraw, element: Mapping[str, Any], bindings: Mapping[str, Any], palette: Mapping[str, str], warnings: list[str]) -> None:
    style = dict(element.get("style", {}))
    text = text_for_element(element, bindings)
    x, y, w, h = [int(element[k]) for k in ("x", "y", "w", "h")]
    size = int(style.get("font_size", 32))
    min_size = int(style.get("min_font_size", 16))
    max_lines = style.get("max_lines")
    family = str(style.get("font_family", "default"))
    color = resolve_value(style.get("color", "#000000"), palette)
    align = str(style.get("align", "left"))
    valign = str(style.get("valign", "top"))

    while size > min_size:
        font = load_font(size, family)
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=6)
        if not style.get("shrink_to_fit") or (bbox[2] - bbox[0] <= w and bbox[3] - bbox[1] <= h):
            break
        size -= 2
    if size <= min_size and style.get("shrink_to_fit"):
        warnings.append(f"text_shrunk:{element.get('id')}")
    font = load_font(size, family)

    lines = text.splitlines() or [text]
    text_h = draw.multiline_textbbox((0, 0), "\n".join(lines), font=font, spacing=6)[3]
    text_y = y + (h - text_h) // 2 if valign == "middle" else y
    if max_lines or " " in text:
        draw_wrapped_text(draw, text, font, color, (x, text_y, x + w, y + h), max_lines=int(max_lines) if max_lines else None)
        return
    text_w = draw.textbbox((0, 0), text, font=font)[2]
    text_x = x + (w - text_w) // 2 if align == "center" else x + w - text_w if align == "right" else x
    draw.text((text_x, text_y), text, font=font, fill=color)


def render_template(template: Mapping[str, Any], bindings: Mapping[str, Any], palette: Mapping[str, str]) -> tuple[Image.Image, list[str]]:
    warnings: list[str] = []
    width = int(template.get("width", 1080))
    height = int(template.get("height", 1350))
    bg = template.get("background", {}).get("color", "#ffffff")
    image = Image.new("RGBA", (width, height), resolve_value(bg, palette))
    draw = ImageDraw.Draw(image)

    for element in template.get("elements", []):
        etype = element.get("type")
        x, y, w, h = [int(element.get(k, 0)) for k in ("x", "y", "w", "h")]
        if etype == "rectangle":
            rounded_panel(draw, (x, y, x + w, y + h), int(element.get("radius", 0)), resolve_value(element.get("fill", "#ffffff"), palette), outline=resolve_value(element.get("stroke"), palette) if element.get("stroke") else None)
        elif etype == "line":
            draw.line((x, y, x + w, y + h), fill=resolve_value(element.get("stroke", "#000000"), palette), width=int(element.get("stroke_width", 2)))
        elif etype == "image":
            ph = element.get("placeholder")
            ref = str(bindings.get(str(ph), "")) if ph else ""
            bg_fill = resolve_value(element.get("background"), palette) if element.get("background") else None
            if bg_fill:
                rounded_panel(draw, (x, y, x + w, y + h), int(element.get("radius", 0)), bg_fill)
            img = image_from_reference(ref) if ref else None
            if img is None:
                warnings.append(f"missing_image:{element.get('id')}")
                continue
            fitted = fit_cover(img, w, h) if element.get("fit", "cover") == "cover" else contain_image(img, w, h, bg_fill)
            image.alpha_composite(mask_rounded(fitted, int(element.get("radius", 0))), (x, y))
        elif etype == "text":
            render_text(draw, element, bindings, palette, warnings)
    return image.convert("RGB"), warnings


def render_template_file(template_path: str | Path, bindings_path: str | Path, output_path: str | Path, palette_override: str | None = None) -> dict[str, Any]:
    import yaml
    template = load_json(template_path)
    bindings_doc = yaml.safe_load(Path(bindings_path).read_text(encoding="utf-8")) or {}
    bindings = bindings_doc.get("bindings", bindings_doc)
    palette = load_palette(template, palette_override)
    image, warnings = render_template(template, bindings, palette)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")

    meta_dir = output_path.parent.parent / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    source_values = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "template": str(template_path),
        "bindings": bindings,
        "warnings": warnings,
    }
    manifest = {
        "success": True,
        "output_path": str(output_path),
        "width": image.width,
        "height": image.height,
        "template_id": template.get("template_id"),
        "warnings": warnings,
    }
    stem = output_path.stem
    (meta_dir / f"{stem}_source_values.json").write_text(json.dumps(source_values, indent=2, ensure_ascii=False), encoding="utf-8")
    (meta_dir / f"{stem}_render_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
