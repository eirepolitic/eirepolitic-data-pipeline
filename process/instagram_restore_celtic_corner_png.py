from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from PIL import Image, ImageDraw

CREAM_WHITE = (244, 234, 215, 255)


def _cubic(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    steps: int = 100,
) -> list[tuple[int, int]]:
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
    """Create a minimal transparent corner ornament.

    This intentionally avoids full-width top/bottom/side lines. The artwork stays
    inside the corner area so the final card reads as four small ornaments rather
    than a continuous Celtic frame.
    """
    size = 512
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    # Small knot cluster, inset from the actual edges.
    for box in [
        (42, 42, 142, 142),
        (98, 42, 198, 142),
        (42, 98, 142, 198),
    ]:
        draw.rounded_rectangle(box, radius=32, outline=CREAM_WHITE, width=10)

    # Short flourishes only; none run far enough to become frame rails.
    for pts in [
        _cubic((146, 72), (205, 42), (268, 54), (316, 104)),
        _cubic((146, 128), (202, 172), (265, 166), (324, 112)),
        _cubic((72, 146), (42, 205), (54, 268), (104, 316)),
        _cubic((128, 146), (172, 202), (166, 265), (112, 324)),
        _cubic((112, 92), (158, 120), (206, 112), (246, 82)),
        _cubic((92, 112), (120, 158), (112, 206), (82, 246)),
    ]:
        _draw_curve(draw, pts, 14)

    # Small terminal curls, kept away from the card edges.
    for pts in [
        _cubic((306, 102), (340, 72), (378, 74), (404, 106), 70),
        _cubic((102, 306), (72, 340), (74, 378), (106, 404), 70),
        _cubic((235, 84), (280, 58), (318, 62), (350, 92), 70),
        _cubic((84, 235), (58, 280), (62, 318), (92, 350), 70),
    ]:
        _draw_curve(draw, pts, 8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)


def restore_asset(
    input_path: str | Path,
    output_path: str | Path,
    prefer_generated: bool = False,
) -> dict[str, object]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if prefer_generated:
        source = "minimal_generated"
        _draw_compact_fallback_asset(output_path)
    elif input_path.exists():
        source = "committed_b64"
        raw = base64.b64decode("".join(input_path.read_text(encoding="utf-8").split()))
        output_path.write_bytes(raw)
    else:
        source = "minimal_generated_missing_b64"
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
    parser = argparse.ArgumentParser(description="Restore or generate a transparent Celtic corner PNG asset.")
    parser.add_argument("--input", default="instagram/assets/corners/celtic_corner_white_v1.png.b64")
    parser.add_argument("--output", default="instagram/assets/corners/celtic_corner_white_v1.png")
    parser.add_argument(
        "--prefer-generated",
        action="store_true",
        help="Generate the minimal no-frame corner asset even if the committed b64 asset exists.",
    )
    args = parser.parse_args()
    print(json.dumps(restore_asset(args.input, args.output, args.prefer_generated), indent=2))


if __name__ == "__main__":
    main()
