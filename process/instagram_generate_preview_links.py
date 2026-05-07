from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3

DEFAULT_REGION = "ca-central-1"
DEFAULT_BUCKET = "eirepolitic-data"
DEFAULT_ROOT_PREFIX = "instagram/previews"
DEFAULT_EXPIRES_IN = 604800
DEFAULT_OUTPUT = "preview_links.json"


def env_default(name: str, fallback: str) -> str:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return fallback
    return str(value).strip()


def int_arg(value: str | int | None, fallback: int) -> int:
    if value is None or str(value).strip() == "":
        return fallback
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"Expected a positive integer, got: {value}")
    return parsed


def public_url(bucket: str, region: str, key: str) -> str:
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"


def iter_local_files(output_root: Path) -> list[Path]:
    skip_names = {DEFAULT_OUTPUT}
    return sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file() and path.name not in skip_names
    )


def list_objects(bucket: str, prefix: str, region: str) -> list[dict[str, Any]]:
    s3 = boto3.client("s3", region_name=region)
    paginator = s3.get_paginator("list_objects_v2")
    objects: list[dict[str, Any]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for item in page.get("Contents", []):
            key = item.get("Key")
            if key and not key.endswith("/"):
                objects.append(item)
    return sorted(objects, key=lambda item: str(item.get("Key", "")))


def object_entries_from_local(output_root: Path, prefix: str) -> list[dict[str, Any]]:
    if not output_root.exists():
        raise FileNotFoundError(f"Output root does not exist: {output_root}")
    files = iter_local_files(output_root)
    if not files:
        raise RuntimeError(f"No local preview files found under: {output_root}")
    entries: list[dict[str, Any]] = []
    for path in files:
        rel = path.relative_to(output_root).as_posix()
        entries.append({
            "Key": f"{prefix.rstrip('/')}/{rel}",
            "RelativePath": rel,
            "Size": path.stat().st_size,
            "LastModified": None,
        })
    return entries


def object_entries_from_s3(bucket: str, prefix: str, region: str) -> list[dict[str, Any]]:
    objects = list_objects(bucket=bucket, prefix=prefix, region=region)
    entries: list[dict[str, Any]] = []
    for item in objects:
        key = str(item["Key"])
        entries.append({
            "Key": key,
            "RelativePath": key[len(prefix.rstrip("/")) + 1 :],
            "Size": int(item.get("Size", 0)),
            "LastModified": item.get("LastModified"),
        })
    return entries


def generate_links(
    campaign_slug: str,
    bucket: str,
    region: str,
    root_prefix: str,
    prefix: str | None,
    expires_in: int,
    output_root: str | None,
) -> dict[str, Any]:
    campaign_slug = campaign_slug.strip().strip("/")
    root_prefix = root_prefix.strip().strip("/")
    resolved_prefix = (prefix or f"{root_prefix}/{campaign_slug}/latest").strip().strip("/")
    if not campaign_slug:
        raise RuntimeError("campaign_slug is required.")
    if not bucket:
        raise RuntimeError("S3 bucket is required.")
    if not region:
        raise RuntimeError("AWS region is required.")

    if output_root:
        objects = object_entries_from_local(Path(output_root), resolved_prefix)
        source = "local_output_root"
    else:
        objects = object_entries_from_s3(bucket=bucket, prefix=resolved_prefix, region=region)
        source = "s3_list_objects_v2"

    if not objects:
        raise RuntimeError(f"No preview objects found for s3://{bucket}/{resolved_prefix}/")

    s3 = boto3.client("s3", region_name=region)
    files: list[dict[str, Any]] = []
    review_index_url: str | None = None
    png_urls: list[dict[str, str]] = []

    for item in objects:
        key = str(item["Key"])
        relative_path = str(item.get("RelativePath") or key[len(resolved_prefix.rstrip("/")) + 1 :])
        presigned_url = s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires_in,
        )
        last_modified = item.get("LastModified")
        entry = {
            "relative_path": relative_path,
            "s3_key": key,
            "s3_uri": f"s3://{bucket}/{key}",
            "public_style_url": public_url(bucket, region, key),
            "presigned_url": presigned_url,
            "size_bytes": int(item.get("Size", 0)),
            "last_modified": last_modified.isoformat() if hasattr(last_modified, "isoformat") else None,
        }
        files.append(entry)
        if relative_path == "review/review_index.html":
            review_index_url = presigned_url
        if relative_path.startswith("png/") and relative_path.lower().endswith(".png"):
            png_urls.append({"relative_path": relative_path, "presigned_url": presigned_url})

    return {
        "success": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bucket": bucket,
        "region": region,
        "campaign_slug": campaign_slug,
        "prefix": resolved_prefix,
        "source": source,
        "expires_in_seconds": expires_in,
        "expires_in_days": round(expires_in / 86400, 2),
        "review_index_url": review_index_url,
        "png_urls": png_urls,
        "files": files,
        "notes": [
            "These are temporary presigned GET URLs for review-only preview assets.",
            "This does not publish to Instagram, schedule posts, or approve content.",
            "Regenerate links after expiry or after a new preview upload overwrites latest/.",
            "The signing AWS principal still needs permission to read these S3 objects when the links are opened."
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate presigned links for Instagram preview assets in S3.")
    parser.add_argument("--campaign-slug", required=True)
    parser.add_argument("--bucket", default=env_default("S3_BUCKET", DEFAULT_BUCKET))
    parser.add_argument("--region", default=env_default("AWS_REGION", DEFAULT_REGION))
    parser.add_argument("--root-prefix", default=env_default("PREVIEW_ROOT_PREFIX", DEFAULT_ROOT_PREFIX))
    parser.add_argument("--prefix", default=None, help="Optional full S3 prefix. Defaults to <root-prefix>/<campaign-slug>/latest.")
    parser.add_argument("--output-root", default=None, help="Optional local generated output root. Avoids needing s3:ListBucket.")
    parser.add_argument("--expires-in", default=env_default("PREVIEW_LINK_EXPIRES_IN", str(DEFAULT_EXPIRES_IN)))
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expires_in = int_arg(args.expires_in, DEFAULT_EXPIRES_IN)
    result = generate_links(
        campaign_slug=args.campaign_slug,
        bucket=args.bucket,
        region=args.region,
        root_prefix=args.root_prefix,
        prefix=args.prefix,
        expires_in=expires_in,
        output_root=args.output_root,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "success": True,
        "output": str(output_path),
        "prefix": result["prefix"],
        "source": result["source"],
        "file_count": len(result["files"]),
        "review_index_url_present": bool(result.get("review_index_url")),
        "png_count": len(result.get("png_urls", [])),
        "expires_in_seconds": expires_in,
    }, indent=2))


if __name__ == "__main__":
    main()
