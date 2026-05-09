from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image


def remove_green_background(
    input_path: str | Path,
    output_path: str | Path,
    green_excess_transparent: int = 55,
    green_excess_opaque: int = 12,
    crop: bool = True,
    pad: int = 28,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input image: {input_path}")

    image = Image.open(input_path).convert("RGBA")
    pixels = image.load()
    width, height = image.size

    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            max_rb = max(r, b)
            green_excess = g - max_rb

            # Fully remove strongly green background pixels.
            if g > 120 and green_excess >= green_excess_transparent:
                pixels[x, y] = (0, 0, 0, 0)
                continue

            # Preserve cream/white ornament pixels.
            if r > 170 and g > 160 and b > 130 and (max(r, g, b) - min(r, g, b)) < 80:
                pixels[x, y] = (r, g, b, a)
                continue

            # Feather edge pixels between green and ornament.
            if green_excess > green_excess_opaque:
                alpha = int(max(0, min(255, ((green_excess_transparent - green_excess) / max(1, green_excess_transparent - green_excess_opaque)) * 255)))
                if alpha <= 2:
                    pixels[x, y] = (0, 0, 0, 0)
                else:
                    pixels[x, y] = (max(r, 235), max(g, 228), max(b, 205), alpha)

    if crop:
        alpha = image.getchannel("A")
        bbox = alpha.getbbox()
        if bbox:
            left, top, right, bottom = bbox
            left = max(0, left - pad)
            top = max(0, top - pad)
            right = min(width, right + pad)
            bottom = min(height, bottom + pad)
            image = image.crop((left, top, right, bottom))
            side = max(image.size)
            square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
            square.alpha_composite(image, (0, 0))
            image = square

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)
    alpha = image.getchannel("A")
    alpha_min, alpha_max = alpha.getextrema()
    return {
        "success": True,
        "input": str(input_path),
        "output": str(output_path),
        "size": list(image.size),
        "alpha_min": int(alpha_min),
        "alpha_max": int(alpha_max),
        "crop": crop,
        "pad": pad,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove flat green background from GPT-generated Celtic corner assets.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--green-excess-transparent", type=int, default=55)
    parser.add_argument("--green-excess-opaque", type=int, default=12)
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--pad", type=int, default=28)
    args = parser.parse_args()

    result = remove_green_background(
        args.input,
        args.output,
        green_excess_transparent=args.green_excess_transparent,
        green_excess_opaque=args.green_excess_opaque,
        crop=not args.no_crop,
        pad=args.pad,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
