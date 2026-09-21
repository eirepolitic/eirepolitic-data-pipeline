from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


class AssetFinalizationError(ValueError):
    pass


@dataclass(frozen=True)
class FinalizedAsset:
    path: Path
    mime_type: str
    width: int
    height: int
    size_bytes: int
    sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finalize_image_to_jpeg(source: str | Path, destination: str | Path, *, quality: int = 92) -> FinalizedAsset:
    """Create a deterministic Instagram delivery JPEG from one reviewed source image."""
    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_file():
        raise AssetFinalizationError(f"source image does not exist: {source_path}")
    if not (1 <= quality <= 100):
        raise AssetFinalizationError("JPEG quality must be between 1 and 100")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        image.load()
        if image.width <= 0 or image.height <= 0:
            raise AssetFinalizationError("source image has invalid dimensions")
        # JPEG has no alpha channel. Composite transparency over white rather than silently dropping alpha.
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.getchannel("A"))
            rgb = background
        else:
            rgb = image.convert("RGB")

        rgb.save(
            destination_path,
            format="JPEG",
            quality=quality,
            optimize=False,
            progressive=False,
            subsampling=0,
        )
        width, height = rgb.size

    return FinalizedAsset(
        path=destination_path,
        mime_type="image/jpeg",
        width=width,
        height=height,
        size_bytes=destination_path.stat().st_size,
        sha256=sha256_file(destination_path),
    )
