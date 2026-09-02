from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path

import boto3
import numpy as np
from PIL import Image

BUCKET = "eirepolitic-data"
KEY = "processed/reference/party_assets/v1/assets/social-democrats/logo.png"
INPUT_B64 = Path("instagram/commissioning/input/social-democrats-approved-logo.b64")
OUTPUT_ROOT = Path("instagram/commissioning/output/party-logo-cover-v2-review")
NORMALIZED_PREVIEW = OUTPUT_ROOT / "social-democrats-approved-source-logo.png"
PROMOTION_MANIFEST = OUTPUT_ROOT / "social-democrats-asset-promotion.json"
EXPECTED_INPUT_SHA256 = "09c82b36f4174fccbd725e38ccf9b67100657698c611a8693c39a07fcd1b8354"
CANVAS_SIZE = 1600
CONTENT_MAX_WIDTH = 1440


def _decode_source() -> bytes:
    raw = base64.b64decode(INPUT_B64.read_text(encoding="ascii"), validate=True)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_INPUT_SHA256:
        raise RuntimeError(f"Unexpected uploaded source SHA256: {digest}")
    return raw


def _standardize(raw: bytes) -> Image.Image:
    source = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.array(source)

    # Replace only the near-black exterior/background pixels with white.
    # The supplied badge itself is purple/white and remains unchanged.
    near_black = arr.max(axis=2) < 35
    arr[near_black] = [255, 255, 255]
    cleaned = Image.fromarray(arr, mode="RGB")

    # Crop to non-white content, then center the approved badge on the
    # shared 1600x1600 white-square consumer canvas.
    arr2 = np.array(cleaned)
    non_white = arr2.min(axis=2) < 248
    ys, xs = np.where(non_white)
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("Approved Social Democrats source contains no visible non-white content")
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    content = cleaned.crop(bbox)

    scale = min(CONTENT_MAX_WIDTH / content.width, CONTENT_MAX_WIDTH / content.height)
    size = (round(content.width * scale), round(content.height * scale))
    content = content.resize(size, Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")
    pos = ((CANVAS_SIZE - size[0]) // 2, (CANVAS_SIZE - size[1]) // 2)
    canvas.paste(content, pos)
    return canvas


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    raw = _decode_source()
    normalized = _standardize(raw)
    normalized.save(NORMALIZED_PREVIEW, format="PNG", optimize=True)

    png_bytes = NORMALIZED_PREVIEW.read_bytes()
    normalized_sha256 = hashlib.sha256(png_bytes).hexdigest()
    with Image.open(io.BytesIO(png_bytes)) as check:
        if check.format != "PNG" or check.size != (1600, 1600):
            raise RuntimeError(f"Normalized asset contract failed: {check.format} {check.size}")

    s3 = boto3.client("s3", region_name="ca-central-1")
    put = s3.put_object(
        Bucket=BUCKET,
        Key=KEY,
        Body=png_bytes,
        ContentType="image/png",
        Metadata={
            "approved-by-user": "2026-09-02",
            "source-sha256": EXPECTED_INPUT_SHA256,
            "normalized-sha256": normalized_sha256,
        },
    )

    fetched = s3.get_object(Bucket=BUCKET, Key=KEY)["Body"].read()
    fetched_sha256 = hashlib.sha256(fetched).hexdigest()
    if fetched_sha256 != normalized_sha256:
        raise RuntimeError(
            f"S3 verification mismatch: uploaded={normalized_sha256}, fetched={fetched_sha256}"
        )
    with Image.open(io.BytesIO(fetched)) as check:
        if check.format != "PNG" or check.size != (1600, 1600):
            raise RuntimeError(f"S3 asset contract failed: {check.format} {check.size}")

    manifest = {
        "party_key": "social-democrats",
        "source_type": "user_supplied_approved_asset",
        "source_sha256": EXPECTED_INPUT_SHA256,
        "normalized_sha256": normalized_sha256,
        "s3_bucket": BUCKET,
        "s3_key": KEY,
        "s3_uri": f"s3://{BUCKET}/{KEY}",
        "s3_version_id": put.get("VersionId"),
        "content_type": "image/png",
        "dimensions": [1600, 1600],
        "approved_date": "2026-09-02",
        "publication_enabled": False,
    }
    PROMOTION_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
