from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from PIL import Image


def restore_base64_asset(input_path: str | Path, output_path: str | Path, verify_image: bool = False) -> dict[str, object]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing base64 asset: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw = base64.b64decode("".join(input_path.read_text(encoding="utf-8").split()))
    output_path.write_bytes(raw)

    result: dict[str, object] = {
        "success": True,
        "input": str(input_path),
        "output": str(output_path),
        "bytes": len(raw),
    }

    if verify_image:
        image = Image.open(output_path)
        image.verify()
        image = Image.open(output_path).convert("RGBA")
        alpha = image.getchannel("A")
        alpha_min, alpha_max = alpha.getextrema()
        result.update(
            {
                "mode": image.mode,
                "size": list(image.size),
                "alpha_min": int(alpha_min),
                "alpha_max": int(alpha_max),
            }
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore a binary asset from a committed .b64 text file.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify-image", action="store_true")
    args = parser.parse_args()

    print(json.dumps(restore_base64_asset(args.input, args.output, args.verify_image), indent=2))


if __name__ == "__main__":
    main()
