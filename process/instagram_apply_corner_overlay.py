from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


def apply_overlay_to_image(
    image_path: Path,
    asset_path: Path,
    size: int,
    margin_x: int,
    margin_y: int,
    opacity: float,
) -> dict[str, Any]:
    base = Image.open(image_path).convert("RGBA")
    corner = Image.open(asset_path).convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)

    if opacity < 1.0:
        alpha = corner.getchannel("A").point(lambda a: int(a * opacity))
        corner.putalpha(alpha)

    width, height = base.size
    placements = [
        (corner, margin_x, margin_y),
        (ImageOps.mirror(corner), width - size - margin_x, margin_y),
        (ImageOps.flip(corner), margin_x, height - size - margin_y),
        (ImageOps.flip(ImageOps.mirror(corner)), width - size - margin_x, height - size - margin_y),
    ]
    for overlay, x, y in placements:
        base.alpha_composite(overlay, (x, y))

    base.convert("RGB").save(image_path, format="PNG")
    return {"image": str(image_path), "size": size, "margin_x": margin_x, "margin_y": margin_y, "opacity": opacity}


def apply_to_output_root(
    output_root: str | Path,
    asset: str | Path,
    size: int,
    margin_x: int,
    margin_y: int,
    opacity: float,
) -> dict[str, Any]:
    output_root = Path(output_root)
    asset = Path(asset)
    png_dir = output_root / "png"
    if not png_dir.exists():
        raise FileNotFoundError(f"Missing png directory: {png_dir}")
    if not asset.exists():
        raise FileNotFoundError(f"Missing corner asset: {asset}")

    items = []
    for image_path in sorted(png_dir.glob("*.png")):
        items.append(apply_overlay_to_image(image_path, asset, size, margin_x, margin_y, opacity))

    manifest = {
        "success": True,
        "output_root": str(output_root),
        "asset": str(asset),
        "processed": len(items),
        "items": items,
        "review_only": True,
    }
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "corner_overlay_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a transparent Celtic corner PNG to rendered Instagram PNGs.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--asset", default="instagram/assets/corners/celtic_corner_white_v1.png")
    parser.add_argument("--size", type=int, default=150)
    parser.add_argument("--margin-x", type=int, default=8)
    parser.add_argument("--margin-y", type=int, default=8)
    parser.add_argument("--opacity", type=float, default=0.82)
    args = parser.parse_args()

    result = apply_to_output_root(args.output_root, args.asset, args.size, args.margin_x, args.margin_y, args.opacity)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
