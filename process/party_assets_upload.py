#!/usr/bin/env python3
"""Upload an approved party asset build to the versioned S3 prefix.

Safe defaults:
- dry-run unless --apply is provided
- refuses to upload if manifest success is false
- refuses to overwrite existing S3 objects
- uploads only the deterministic v1 asset tree, manifest and contact sheet
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

DEFAULT_BUCKET = "eirepolitic-data"
DEFAULT_PREFIX = "processed/reference/party_assets/v1"


def object_exists(client, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def collect_uploads(build_root: Path, prefix: str) -> list[tuple[Path, str]]:
    allowed_roots = [build_root / "assets"]
    files: list[Path] = []
    for root in allowed_roots:
        if root.exists():
            files.extend(path for path in root.rglob("*") if path.is_file())
    for name in ("manifest.json", "contact_sheet.png"):
        path = build_root / name
        if path.is_file():
            files.append(path)

    uploads: list[tuple[Path, str]] = []
    for path in sorted(set(files)):
        relative = path.relative_to(build_root).as_posix()
        uploads.append((path, f"{prefix.rstrip('/')}/{relative}"))
    return uploads


def upload_build(build_root: Path, bucket: str, prefix: str, apply: bool, client=None) -> dict:
    manifest_path = build_root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing build manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("success"):
        raise ValueError("Refusing upload: build manifest is not successful")

    client = client or boto3.client("s3", region_name=os.getenv("AWS_REGION", "ca-central-1"))
    uploads = collect_uploads(build_root, prefix)
    if not uploads:
        raise ValueError("No party asset files found to upload")

    collisions = []
    for _, key in uploads:
        if object_exists(client, bucket, key):
            collisions.append(key)
    if collisions:
        raise ValueError(
            "Refusing upload because versioned target objects already exist: " + ", ".join(collisions)
        )

    report = {
        "bucket": bucket,
        "prefix": prefix,
        "apply": apply,
        "upload_count": len(uploads),
        "objects": [f"s3://{bucket}/{key}" for _, key in uploads],
    }
    if not apply:
        return report

    for path, key in uploads:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        client.upload_file(
            str(path),
            bucket,
            key,
            ExtraArgs={"ContentType": content_type},
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely upload a reviewed party asset build to S3")
    parser.add_argument("--build-root", required=True)
    parser.add_argument("--bucket", default=os.getenv("S3_BUCKET", DEFAULT_BUCKET))
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--apply", action="store_true", help="perform upload; default is dry-run")
    args = parser.parse_args()

    report = upload_build(Path(args.build_root), args.bucket, args.prefix, args.apply)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
