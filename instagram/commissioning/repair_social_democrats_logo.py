from __future__ import annotations

import hashlib
import io
import json

import boto3
from PIL import Image

BUCKET = "eirepolitic-data"
KEY = "processed/reference/party_assets/v1/assets/social-democrats/logo.png"
EXPECTED_SIZE = (1600, 1600)
MAX_NEUTRAL_SPREAD = 18
MAX_NEUTRAL_VALUE = 244
MIN_CHANGED_PIXELS = 1000


def _is_unwanted_neutral(pixel: tuple[int, int, int]) -> bool:
    low = min(pixel)
    high = max(pixel)
    return high <= MAX_NEUTRAL_VALUE and (high - low) <= MAX_NEUTRAL_SPREAD


def _clean(image: Image.Image) -> tuple[Image.Image, int]:
    rgb = image.convert("RGB")
    cleaned: list[tuple[int, int, int]] = []
    changed = 0
    for pixel in rgb.getdata():
        if _is_unwanted_neutral(pixel):
            replacement = (255, 255, 255)
            cleaned.append(replacement)
            if pixel != replacement:
                changed += 1
        else:
            cleaned.append(pixel)
    output = Image.new("RGB", rgb.size, "white")
    output.putdata(cleaned)
    return output, changed


def _validate_png(raw: bytes) -> Image.Image:
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        if image.format != "PNG":
            raise RuntimeError(f"Canonical Social Democrats asset is {image.format}, expected PNG")
        if image.size != EXPECTED_SIZE:
            raise RuntimeError(
                f"Canonical Social Democrats asset is {image.size}, expected {EXPECTED_SIZE}"
            )
        return image.convert("RGB")


def _remaining_unwanted_neutral(image: Image.Image) -> int:
    return sum(1 for pixel in image.convert("RGB").getdata() if _is_unwanted_neutral(pixel))


def main() -> None:
    s3 = boto3.client("s3", region_name="ca-central-1")
    current = s3.get_object(Bucket=BUCKET, Key=KEY)
    source_raw = current["Body"].read()
    source_sha256 = hashlib.sha256(source_raw).hexdigest()
    source_image = _validate_png(source_raw)

    cleaned, changed_pixels = _clean(source_image)
    if changed_pixels < MIN_CHANGED_PIXELS:
        raise RuntimeError(
            f"Expected a substantial removable black/gray fringe; only {changed_pixels} pixels matched"
        )

    remaining = _remaining_unwanted_neutral(cleaned)
    if remaining != 0:
        raise RuntimeError(f"Neutral fringe cleanup incomplete: {remaining} matching pixels remain")

    output = io.BytesIO()
    cleaned.save(output, format="PNG", compress_level=9)
    corrected_raw = output.getvalue()
    corrected_sha256 = hashlib.sha256(corrected_raw).hexdigest()
    if corrected_sha256 == source_sha256:
        raise RuntimeError("Corrected asset hash unexpectedly matches the source asset")

    metadata = dict(current.get("Metadata") or {})
    metadata.update(
        {
            "source-type": "user-supplied-approved-asset",
            "correction": "remove-neutral-black-gray-halo",
            "parent-sha256": source_sha256,
            "corrected-sha256": corrected_sha256,
        }
    )
    s3.put_object(
        Bucket=BUCKET,
        Key=KEY,
        Body=corrected_raw,
        ContentType="image/png",
        Metadata=metadata,
    )

    read_back = s3.get_object(Bucket=BUCKET, Key=KEY)
    promoted_raw = read_back["Body"].read()
    promoted_sha256 = hashlib.sha256(promoted_raw).hexdigest()
    if promoted_sha256 != corrected_sha256:
        raise RuntimeError(
            f"S3 read-back hash mismatch: expected {corrected_sha256}, got {promoted_sha256}"
        )
    promoted_image = _validate_png(promoted_raw)
    promoted_remaining = _remaining_unwanted_neutral(promoted_image)
    if promoted_remaining != 0:
        raise RuntimeError(
            f"S3 read-back still contains {promoted_remaining} black/gray fringe pixels"
        )

    print(
        json.dumps(
            {
                "status": "PASS",
                "s3_uri": f"s3://{BUCKET}/{KEY}",
                "dimensions": list(EXPECTED_SIZE),
                "changed_pixels": changed_pixels,
                "remaining_unwanted_neutral_pixels": promoted_remaining,
                "parent_sha256": source_sha256,
                "corrected_sha256": corrected_sha256,
                "etag": str(read_back.get("ETag") or "").strip('"'),
                "version_id": read_back.get("VersionId"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
