from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw


def cubic(p0, p1, p2, p3, steps=160):
    pts = []
    for i in range(steps + 1):
        t = i / steps
        mt = 1 - t
        x = mt**3 * p0[0] + 3 * mt**2 * t * p1[0] + 3 * mt * t**2 * p2[0] + t**3 * p3[0]
        y = mt**3 * p0[1] + 3 * mt**2 * t * p1[1] + 3 * mt * t**2 * p2[1] + t**3 * p3[1]
        pts.append((x, y))
    return pts


def draw_curve(draw: ImageDraw.ImageDraw, points, width: int, fill=(244, 234, 215, 255)):
    draw.line([(int(x), int(y)) for x, y in points], fill=fill, width=width, joint="curve")


def draw_loop(draw: ImageDraw.ImageDraw, box, width: int, fill=(244, 234, 215, 255)):
    draw.ellipse(box, outline=fill, width=width)


def draw_corner(size: int = 1254, scale: int = 3) -> Image.Image:
    s = scale
    canvas = Image.new("RGBA", (size * s, size * s), (0, 0, 0, 0))
    d = ImageDraw.Draw(canvas)
    f = (244, 234, 215, 255)

    def P(x, y):
        return (x * s, y * s)

    def C(p0, p1, p2, p3, steps=200):
        return [(x * s, y * s) for x, y in cubic(p0, p1, p2, p3, steps)]

    w_main = 34 * s
    w_mid = 22 * s
    w_thin = 13 * s

    # Top horizontal Celtic flourish.
    curves = [
        C((54, 126), (145, 22), (305, 30), (382, 124)),
        C((382, 124), (455, 214), (570, 210), (643, 122)),
        C((643, 122), (731, 18), (895, 30), (976, 130)),
        C((976, 130), (1046, 217), (1156, 190), (1200, 92)),
        C((138, 54), (263, 205), (520, 205), (641, 62)),
        C((641, 62), (765, 205), (1028, 203), (1132, 52)),
    ]
    for pts in curves:
        draw_curve(d, pts, w_mid, f)

    # Left vertical flourish.
    curves = [
        C((126, 54), (22, 145), (30, 305), (124, 382)),
        C((124, 382), (214, 455), (210, 570), (122, 643)),
        C((122, 643), (18, 731), (30, 895), (130, 976)),
        C((130, 976), (217, 1046), (190, 1156), (92, 1200)),
        C((54, 138), (205, 263), (205, 520), (62, 641)),
        C((62, 641), (205, 765), (203, 1028), (52, 1132)),
    ]
    for pts in curves:
        draw_curve(d, pts, w_mid, f)

    # Dense corner knot cluster.
    loops = [
        (38, 38, 238, 238),
        (132, 38, 332, 238),
        (38, 132, 238, 332),
        (160, 160, 344, 344),
        (88, 88, 286, 286),
    ]
    for box in loops:
        draw_loop(d, tuple(v * s for v in box), w_mid, f)

    # Interlace accent gaps: erase short crossings, then redraw over-strands.
    erase = (0, 0, 0, 0)
    eraser = ImageDraw.Draw(canvas)
    gap_w = 44 * s
    gap_segments = [
        (P(204, 112), P(255, 112)),
        (P(112, 204), P(112, 255)),
        (P(278, 190), P(315, 227)),
        (P(190, 278), P(227, 315)),
        (P(495, 135), P(555, 135)),
        (P(135, 495), P(135, 555)),
        (P(760, 135), P(822, 135)),
        (P(135, 760), P(135, 822)),
    ]
    for a, b in gap_segments:
        eraser.line([a, b], fill=erase, width=gap_w)

    over_curves = [
        C((42, 178), (110, 95), (238, 92), (308, 178), 120),
        C((178, 42), (95, 110), (92, 238), (178, 308), 120),
        C((438, 124), (518, 80), (594, 84), (654, 138), 120),
        C((124, 438), (80, 518), (84, 594), (138, 654), 120),
    ]
    for pts in over_curves:
        draw_curve(d, pts, w_main, f)

    # Fine decorative terminals and curls.
    curls = [
        C((1010, 102), (1100, 36), (1180, 34), (1228, 86), 100),
        C((102, 1010), (36, 1100), (34, 1180), (86, 1228), 100),
        C((730, 128), (790, 72), (870, 72), (922, 128), 100),
        C((128, 730), (72, 790), (72, 870), (128, 922), 100),
        C((280, 38), (390, 16), (482, 44), (548, 118), 120),
        C((38, 280), (16, 390), (44, 482), (118, 548), 120),
    ]
    for pts in curls:
        draw_curve(d, pts, w_thin, f)

    # Trim to smooth alpha by downsampling.
    return canvas.resize((size, size), Image.Resampling.LANCZOS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a transparent white Celtic corner PNG asset.")
    parser.add_argument("--output", default="instagram/assets/corners/celtic_corner_white_v1.png")
    parser.add_argument("--size", type=int, default=1254)
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    image = draw_corner(args.size)
    image.save(out, format="PNG")
    alpha = image.getchannel("A")
    result = {
        "success": True,
        "output": str(out),
        "mode": image.mode,
        "size": list(image.size),
        "alpha_min": int(alpha.getextrema()[0]),
        "alpha_max": int(alpha.getextrema()[1]),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
