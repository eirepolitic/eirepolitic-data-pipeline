from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from PIL import Image, ImageDraw

CREAM_WHITE = (244, 234, 215, 255)


def _cubic(p0: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float], steps: int = 120) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for i in range(steps + 1):
        t = i / steps
        mt = 1 - t
        x = mt**3 * p0[0] + 3 * mt**2 * t * p1[0] + 3 * mt * t**2 * p2[0] + t**3 * p3[0]
        y = mt**3 * p0[1] + 3 * mt**2 * t * p1[1] + 3 * mt * t**2 * p2[1] + t**3 * p3[1]
        points.append((round(x), round(y)))
    return points


def _draw_curve(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], width: int) -> None:
    draw.line(points, fill=CREAM_WHITE, width=width, joint="curve")


def _draw_compact_fallback_asset(output_path: Path) -> None:
    """Create a compact transparent corner asset only when the committed PNG is absent.

    The preview workflow normally restores the committed image-derived transparent PNG.
    This fallback keeps the workflow review-only and avoids using the older full-frame
    procedural corner generator.
    """
    scale = 3
    size = 512
    canvas = Image.new("RGBA", (size * scale, size * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    def s(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return [(x * scale, y * scale) for x, y in points]

    def c(p0, p1, p2, p3, steps=120):
        return s(_cubic(p0, p1, p2, p3, steps))

    # Corner knot cluster.
    for box in [(26, 26, 130, 130), (82, 26, 186, 130), (26, 82, 130, 186), (72, 72, 176, 176)]:
        draw.rounded_rectangle(tuple(v * scale for v in box), radius=36 * scale, outline=CREAM_WHITE, width=11 * scale)

    # Shorter, image-asset-like flourishes: enough ornament without forming a frame.
    for pts in [
        c((116, 56), (190, 20), (285, 30), (340, 84)),
        c((162, 116), (242, 162), (318, 152), (392, 86)),
        c((110, 152), (182, 212), (286, 214), (354, 148)),
        c((56, 116), (20, 190), (30, 285), (84, 340)),
        c((116, 162), (162, 242), (152, 318), (86, 392)),
        c((152, 110), (212, 182), (214, 286), (148, 354)),
    ]:
        _draw_curve(draw, pts, 11 * scale)

    # Small terminal curls.
    for pts in [
        c((330, 83), (390, 38), (444, 42), (478, 82), 80),
        c((83, 330), (38, 390), (42, 444), (82, 478), 80),
        c((244, 72), (306, 44), (356, 48), (410, 92), 80),
        c((72, 244), (44, 306), (48, 356), (92, 410), 80),
    ]:
        _draw_curve(draw, pts, 6 * scale)

    image = canvas.resize((size, size), Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)


def restore_asset(input_path: str | Path, output_path: str | Path) -> dict[str, object]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source = "committed_b64"
    if input_path.exists():
        raw = base64.b64decode("".join(input_path.read_text(encoding="utf-8").split()))
        output_path.write_bytes(raw)
    else:
        source = "compact_fallback"
        _draw_compact_fallback_asset(output_path)

    image = Image.open(output_path).convert("RGBA")
    image.save(output_path, format="PNG", optimize=True)
    alpha = image.getchannel("A")
    alpha_min, alpha_max = alpha.getextrema()
    if alpha_min != 0 or alpha_max != 255:
        raise RuntimeError(f"Asset does not have expected transparent alpha range: {(alpha_min, alpha_max)}")
    return {
        "success": True,
        "source": source,
        "input": str(input_path),
        "output": str(output_path),
        "mode": image.mode,
        "size": list(image.size),
        "alpha_min": int(alpha_min),
        "alpha_max": int(alpha_max),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore a committed transparent Celtic corner PNG asset.")
    parser.add_argument("--input", default="instagram/assets/corners/celtic_corner_white_v1.png.b64")
    parser.add_argument("--output", default="instagram/assets/corners/celtic_corner_white_v1.png")
    args = parser.parse_args()
    print(json.dumps(restore_asset(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
