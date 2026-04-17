from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict

from PIL import ImageFont

from .constants import FONT_CANDIDATES


@lru_cache(maxsize=None)
def resolve_font_path(kind: str) -> str | None:
    for candidate in FONT_CANDIDATES.get(kind, []):
        if Path(candidate).exists():
            return candidate
    return None


@lru_cache(maxsize=None)
def get_font(kind: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = resolve_font_path(kind)
    if path:
        return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def font_set() -> Dict[str, ImageFont.ImageFont]:
    return {
        "title": get_font("bold", 72),
        "headline": get_font("bold", 54),
        "subhead": get_font("bold", 38),
        "body": get_font("regular", 32),
        "body_small": get_font("regular", 28),
        "label": get_font("regular", 24),
        "small": get_font("regular", 22),
        "metric": get_font("bold", 50),
        "metric_label": get_font("regular", 24),
        "footer": get_font("regular", 22),
    }
