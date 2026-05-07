from __future__ import annotations

import argparse
import json
import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import boto3

DEFAULT_REGION = "ca-central-1"
DEFAULT_BUCKET = "eirepolitic-data"
DEFAULT_ROOT_PREFIX = "instagram/previews"


def env_default(name: str, fallback: str) -> str:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return fallback
    return str(value).strip()


def bool_arg(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def public_url(bucket: str, region: str, key: str) -> str:
    encoded_key = "/".join(quote(part) for part in key.split("/"))
    return f"https://{bucket}.s3.{region}.amazonaws.com/{encoded_key}"


def content_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed:
        return guessed
    if path.suffix.lower() == ".csv":
        return "text/csv"
    if path.suffix.lower() == ".json":
        return "application/json"
    if path.suffix.lower() == ".html":
        return "text/html"
    return "application/octet-stream"


def iter_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def upload_preview(
    output_root: str | Path,
    campaign_slug: str,
    run_label: str,
    bucket: str,
    region: str,
    root_prefix: str,
    public_read: bool,
) -> dict[str, Any]:
    output_root = Path(output_root)
    if not output_root.exists():
        raise FileNotFoundError(f"Output root does not exist: {output_root}")

    bucket = (bucket or DEFAULT_BUCKET).strip()
    region = (region or DEFAULT_REGION).strip()
    root_prefix = (root_prefix or DEFAULT_ROOT_PREFIX).strip()
    if not bucket:
        raise RuntimeError("S3 bucket is blank after applying defaults.")
    if not region:
        raise RuntimeError("AWS region is blank after applying defaults.")

    files = iter_files(output_root)
    if not files:
        raise RuntimeError(f"No files found to upload under: {output_root}")

    print(json.dumps({
        "event": "s3_preview_upload_start",
        "bucket": bucket,
        "region": region,
        "root_prefix": root_prefix,
        "campaign_slug": campaign_slug,
        "run_label": run_label,
        "output_root": str(output_root),
        "file_count": len(files),
        "public_read": public_read,
    }, indent=2))

    s3 = boto3.client("s3", region_name=region)
    preview_prefix = f"{root_prefix.strip('/')}/{campaign_slug.strip('/')}/{run_label.strip('/')}"
    latest_prefix = f"{root_prefix.strip('/')}/{campaign_slug.strip('/')}/latest"
    uploaded: list[dict[str, Any]] = []

    extra_args_base: dict[str, Any] = {}
    if public_read:
        extra_args_base["ACL"] = "public-read"

    for path in files:
        rel = path.relative_to(output_root).as_posix()
        for prefix in [preview_prefix, latest_prefix]:
            key = f"{prefix}/{rel}"
            extra_args = {**extra_args_base, "ContentType": content_type(path)}
            print(f"Uploading {rel} -> s3://{bucket}/{key}")
            s3.upload_file(str(path), bucket, key, ExtraArgs=extra_args)
            if prefix == preview_prefix:
                uploaded.append({
                    "relative_path": rel,
                    "s3_key": key,
                    "url": public_url(bucket, region, key),
                    "content_type": extra_args["ContentType"],
                    "size_bytes": path.stat().st_size,
                })

    manifest = {
        "success": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bucket": bucket,
        "region": region,
        "campaign_slug": campaign_slug,
        "run_label": run_label,
        "output_root": str(output_root),
        "preview_prefix": preview_prefix,
        "latest_prefix": latest_prefix,
        "preview_url": public_url(bucket, region, f"{preview_prefix}/review/review_index.html"),
        "latest_url": public_url(bucket, region, f"{latest_prefix}/review/review_index.html"),
        "public_read_requested": public_read,
        "uploaded_count": len(uploaded),
        "files": uploaded,
        "notes": [
            "This uploads generated review assets only; it does not publish to Instagram.",
            "If the bucket blocks public access, URLs may require AWS access even when public_read_requested is true.",
            "The latest prefix is overwritten on each preview upload for the same campaign.",
        ],
    }

    manifest_path = output_root / "preview_upload_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    for prefix in [preview_prefix, latest_prefix]:
        key = f"{prefix}/preview_upload_manifest.json"
        extra_args = {**extra_args_base, "ContentType": "application/json"}
        print(f"Uploading preview manifest -> s3://{bucket}/{key}")
        s3.upload_file(str(manifest_path), bucket, key, ExtraArgs=extra_args)

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload generated Instagram campaign outputs to S3 preview prefixes.")
    parser.add_argument("--output-root", required=True, help="Local generated_posts campaign output root.")
    parser.add_argument("--campaign-slug", required=True)
    parser.add_argument("--run-label", required=True, help="Preview version label, for example github run ID.")
    parser.add_argument("--bucket", default=env_default("S3_BUCKET", DEFAULT_BUCKET))
    parser.add_argument("--region", default=env_default("AWS_REGION", DEFAULT_REGION))
    parser.add_argument("--root-prefix", default=env_default("PREVIEW_ROOT_PREFIX", DEFAULT_ROOT_PREFIX))
    parser.add_argument("--public-read", default=env_default("PREVIEW_PUBLIC_READ", "0"), help="Set true/1/yes to request public-read ACL.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = upload_preview(
        output_root=args.output_root,
        campaign_slug=args.campaign_slug,
        run_label=args.run_label,
        bucket=args.bucket,
        region=args.region,
        root_prefix=args.root_prefix,
        public_read=bool_arg(args.public_read),
    )
    print(json.dumps({
        "success": True,
        "preview_url": result["preview_url"],
        "latest_url": result["latest_url"],
        "uploaded_count": result["uploaded_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
