from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from PIL import Image


def restore_asset(input_path: str | Path, output_path: str | Path) -> dict[str, object]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing committed base64 asset: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw = base64.b64decode("".join(input_path.read_text(encoding="utf-8").split()))
    output_path.write_bytes(raw)

    image = Image.open(output_path).convert("RGBA")
    image.save(output_path, format="PNG", optimize=True)
    alpha = image.getchannel("A")
    alpha_min, alpha_max = alpha.getextrema()
    if alpha_min != 0 or alpha_max != 255:
        raise RuntimeError(f"Asset does not have expected transparent alpha range: {(alpha_min, alpha_max)}")
    return {
        "success": True,
        "input": str(input_path),
        "output": str(output_path),
        "mode": image.mode,
        "size": list(image.size),
        "alpha_min": int(alpha_min),
        "alpha_max": int(alpha_max),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore a committed base64 transparent Celtic corner PNG asset.")
    parser.add_argument("--input", default="instagram/assets/corners/celtic_corner_white_v1.png.b64")
    parser.add_argument("--output", default="instagram/assets/corners/celtic_corner_white_v1.png")
    args = parser.parse_args()
    print(json.dumps(restore_asset(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
