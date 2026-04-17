from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from PIL import Image, ImageDraw, ImageOps

try:
    import cairosvg
except Exception:  # pragma: no cover
    cairosvg = None

import requests


def normalize_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\b(td|teachta d[aá]la|minister|deputy)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_constituency(value: Any) -> str:
    return normalize_name(value)


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    text = str(value).strip()
    return text == "" or text.lower() in {"none", "nan", "null"}


def coalesce_text(*values: Any) -> str | None:
    for value in values:
        if is_missing(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def safe_int(value: Any) -> int:
    if is_missing(value):
        return 0
    try:
        return int(float(value))
    except Exception:
        return 0


def clamp_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= max_chars:
        return text
    trimmed = text[:max_chars].rsplit(" ", 1)[0].strip()
    return (trimmed or text[:max_chars]).rstrip(" .,;:") + "…"


def rounded_panel(draw: ImageDraw.ImageDraw, box: Sequence[int], radius: int, fill: str, outline: str | None = None, width: int = 2) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font,
    fill: str,
    box: Sequence[int],
    line_spacing: int = 8,
    max_lines: int | None = None,
) -> int:
    x0, y0, x1, y1 = box
    max_width = x1 - x0
    words = (text or "").split()
    if not words:
        return y0

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        probe = f"{current} {word}"
        if draw.textbbox((0, 0), probe, font=font)[2] <= max_width:
            current = probe
        else:
            lines.append(current)
            current = word
    lines.append(current)

    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = clamp_text(lines[-1], max(12, len(lines[-1]) - 1))

    cursor_y = y0
    for line in lines:
        bbox = draw.textbbox((x0, cursor_y), line, font=font)
        line_height = bbox[3] - bbox[1]
        if cursor_y + line_height > y1:
            break
        draw.text((x0, cursor_y), line, font=font, fill=fill)
        cursor_y += line_height + line_spacing
    return cursor_y


def fit_cover(image: Image.Image, width: int, height: int) -> Image.Image:
    return ImageOps.fit(image.convert("RGBA"), (width, height), method=Image.Resampling.LANCZOS)


def load_image_from_reference(reference: str | None, width: int, height: int, timeout: int = 20) -> Image.Image | None:
    if not reference:
        return None

    if reference.startswith(("http://", "https://")):
        response = requests.get(reference, timeout=timeout)
        response.raise_for_status()
        content = response.content
        lower = reference.lower()
        if lower.endswith(".svg") or response.headers.get("content-type", "").startswith("image/svg"):
            if cairosvg is None:
                raise RuntimeError("cairosvg is required to convert SVG assets.")
            content = cairosvg.svg2png(bytestring=content, output_width=width, output_height=height)
        image = Image.open(io.BytesIO(content))
        return fit_cover(image, width, height)

    path = Path(reference)
    if path.exists():
        image = Image.open(path)
        return fit_cover(image, width, height)

    return None


def ordinal_rank(rank: int) -> str:
    if rank <= 0:
        return "N/A"
    if 10 <= rank % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(rank % 10, "th")
    return f"{rank}{suffix}"


def percent_string(value: Any) -> str:
    if is_missing(value):
        return "N/A"
    text = str(value).strip()
    if text.endswith("%"):
        return text
    try:
        num = float(text)
        if num.is_integer():
            return f"{int(num)}%"
        return f"{num:.1f}%"
    except Exception:
        return text
