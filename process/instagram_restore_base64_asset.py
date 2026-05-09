from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def _draw_variant_a(output_path: Path) -> None:
    scale = 4
    size = 96
    image = Image.new("RGBA", (size * scale, size * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    def line(points: list[tuple[float, float]], fill: tuple[int, int, int, int], width: float) -> None:
        draw.line([(int(x * scale), int(y * scale)) for x, y in points], fill=fill, width=int(width * scale), joint="curve")

    def arc(box: tuple[float, float, float, float], start: int, end: int, fill: tuple[int, int, int, int], width: float) -> None:
        draw.arc([int(v * scale) for v in box], start=start, end=end, fill=fill, width=int(width * scale))

    for offset, color, width in [(0.7, (0, 0, 0, 70), 3.2), (0, (255, 255, 255, 235), 2.2)]:
        arc((8 + offset, 8 + offset, 56 + offset, 56 + offset), 180, 360, color, width)
        arc((24 + offset, 24 + offset, 72 + offset, 72 + offset), 180, 360, color, width)
        arc((40 + offset, 40 + offset, 88 + offset, 88 + offset), 180, 360, color, width)
        line([(8 + offset, 32 + offset), (8 + offset, 10 + offset), (30 + offset, 10 + offset)], color, width)
        line([(24 + offset, 50 + offset), (24 + offset, 28 + offset), (46 + offset, 28 + offset)], color, width)
        line([(40 + offset, 68 + offset), (40 + offset, 46 + offset), (62 + offset, 46 + offset)], color, width)
        line([(15 + offset, 42 + offset), (42 + offset, 15 + offset)], color, width)
        line([(31 + offset, 58 + offset), (58 + offset, 31 + offset)], color, width)
        line([(47 + offset, 74 + offset), (74 + offset, 47 + offset)], color, width)

    image = image.resize((size, size), Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)


def _draw_variant_c(output_path: Path) -> None:
    scale = 4
    size = 96
    image = Image.new("RGBA", (size * scale, size * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    def line(points: list[tuple[float, float]], fill: tuple[int, int, int, int], width: float) -> None:
        draw.line([(int(x * scale), int(y * scale)) for x, y in points], fill=fill, width=int(width * scale), joint="curve")

    def arc(box: tuple[float, float, float, float], start: int, end: int, fill: tuple[int, int, int, int], width: float) -> None:
        draw.arc([int(v * scale) for v in box], start=start, end=end, fill=fill, width=int(width * scale))

    for offset, color, width in [(0.7, (0, 0, 0, 70), 3.4), (0, (255, 255, 255, 230), 2.4)]:
        line([(8 + offset, 70 + offset), (8 + offset, 12 + offset), (66 + offset, 12 + offset), (66 + offset, 28 + offset), (24 + offset, 28 + offset), (24 + offset, 54 + offset), (82 + offset, 54 + offset)], color, width)
        line([(22 + offset, 82 + offset), (22 + offset, 42 + offset), (52 + offset, 42 + offset), (52 + offset, 24 + offset), (82 + offset, 24 + offset)], color, width)
        line([(38 + offset, 86 + offset), (38 + offset, 68 + offset), (70 + offset, 68 + offset), (70 + offset, 40 + offset), (86 + offset, 40 + offset)], color, width)
        arc((2 + offset, 62 + offset, 18 + offset, 78 + offset), 90, 270, color, width)
        arc((58 + offset, 4 + offset, 74 + offset, 20 + offset), 0, 180, color, width)
        line([(14 + offset, 18 + offset), (46 + offset, 50 + offset), (78 + offset, 82 + offset)], color, width)
        line([(18 + offset, 64 + offset), (64 + offset, 18 + offset)], color, width)

    image = image.resize((size, size), Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)


def generate_fallback_corner(output_path: str | Path, variant: str) -> dict[str, Any]:
    output_path = Path(output_path)
    variant = variant.lower().strip()
    if variant == "c":
        _draw_variant_c(output_path)
    else:
        _draw_variant_a(output_path)

    image = Image.open(output_path).convert("RGBA")
    alpha_min, alpha_max = image.getchannel("A").getextrema()
    return {
        "fallback_used": True,
        "fallback_variant": variant,
        "output": str(output_path),
        "bytes": output_path.stat().st_size,
        "mode": image.mode,
        "size": list(image.size),
        "alpha_min": int(alpha_min),
        "alpha_max": int(alpha_max),
    }


def restore_base64_asset(
    input_path: str | Path,
    output_path: str | Path,
    verify_image: bool = False,
    fallback_variant: str | None = None,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        if fallback_variant:
            return generate_fallback_corner(output_path, fallback_variant) | {
                "success": True,
                "input": str(input_path),
                "restore_error": f"Missing base64 asset: {input_path}",
            }
        raise FileNotFoundError(f"Missing base64 asset: {input_path}")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        raw = base64.b64decode("".join(input_path.read_text(encoding="utf-8").split()))
        output_path.write_bytes(raw)

        result: dict[str, Any] = {
            "success": True,
            "fallback_used": False,
            "input": str(input_path),
            "output": str(output_path),
            "bytes": len(raw),
        }

        if verify_image:
            image = Image.open(output_path)
            image.verify()
            image = Image.open(output_path).convert("RGBA")
            alpha_min, alpha_max = image.getchannel("A").getextrema()
            result.update(
                {
                    "mode": image.mode,
                    "size": list(image.size),
                    "alpha_min": int(alpha_min),
                    "alpha_max": int(alpha_max),
                }
            )

        return result
    except Exception as exc:
        if not fallback_variant:
            raise
        result = generate_fallback_corner(output_path, fallback_variant)
        result.update(
            {
                "success": True,
                "input": str(input_path),
                "restore_error": f"{type(exc).__name__}: {exc}",
            }
        )
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore a binary asset from a committed .b64 text file.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify-image", action="store_true")
    parser.add_argument("--fallback-variant", choices=["a", "b", "c"])
    args = parser.parse_args()

    print(json.dumps(restore_base64_asset(args.input, args.output, args.verify_image, args.fallback_variant), indent=2))


if __name__ == "__main__":
    main()
