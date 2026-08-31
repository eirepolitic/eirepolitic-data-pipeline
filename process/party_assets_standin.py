#!/usr/bin/env python3
"""Generate EirePolitic-owned neutral stand-in imagery for non-party identities.

The generated mark is deliberately generic and must never be represented as official
party branding. It exists only so downstream consumers can render a consistent visual
for records whose party identity is Independent / Non-Party.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SOURCE_SIZE = 1200
DISC_MARGIN = 130
DISC_FILL = (96, 101, 105, 255)
DISC_OUTLINE = (210, 212, 214, 255)
TEXT_FILL = (255, 255, 255, 255)


def _font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def generate_independent_standin(output: Path) -> dict:
    """Create a neutral circular IND marker on a transparent square canvas."""
    output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (SOURCE_SIZE, SOURCE_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    disc = (
        DISC_MARGIN,
        DISC_MARGIN,
        SOURCE_SIZE - DISC_MARGIN,
        SOURCE_SIZE - DISC_MARGIN,
    )
    draw.ellipse(disc, fill=DISC_FILL, outline=DISC_OUTLINE, width=12)

    draw.text(
        (SOURCE_SIZE // 2, SOURCE_SIZE // 2 - 55),
        "IND",
        font=_font(255, bold=True),
        fill=TEXT_FILL,
        anchor="mm",
    )
    draw.text(
        (SOURCE_SIZE // 2, SOURCE_SIZE // 2 + 155),
        "INDEPENDENT",
        font=_font(64, bold=True),
        fill=TEXT_FILL,
        anchor="mm",
    )

    image.save(output, "PNG", optimize=True)
    return {
        "party_key": "independent",
        "path": str(output),
        "source_type": "eirepolitic_generated_standin",
        "official_branding": False,
        "usage_note": "Neutral EirePolitic stand-in for Independent / Non-Party records; not official branding.",
        "width": SOURCE_SIZE,
        "height": SOURCE_SIZE,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate neutral EirePolitic party stand-ins")
    parser.add_argument("--staging-root", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    staging_root = Path(args.staging_root)
    result = generate_independent_standin(staging_root / "independent/source.png")
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
