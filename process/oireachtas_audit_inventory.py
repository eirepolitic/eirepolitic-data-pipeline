from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3


PREFIXES = (
    "processed/oireachtas_unified/latest/",
    "processed/oireachtas_unified/compat/",
    "processed/oireachtas_unified/silver/",
    "processed/oireachtas_unified/gold/",
    "processed/oireachtas_unified/control/",
    "processed/oireachtas_unified/review/",
)


def list_objects(s3: Any, *, bucket: str, prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rows.append(
                {
                    "prefix": prefix,
                    "key": obj["Key"],
                    "size": int(obj.get("Size", 0)),
                    "etag": str(obj.get("ETag", "")).strip('"'),
                    "last_modified_utc": obj["LastModified"].astimezone(timezone.utc).isoformat(),
                    "storage_class": obj.get("StorageClass", ""),
                }
            )
    return rows


def main() -> int:
    bucket = os.getenv("S3_BUCKET", "eirepolitic-data")
    region = os.getenv("AWS_REGION", "ca-central-1")
    output_dir = Path(os.getenv("AUDIT_OUTPUT_DIR", "artifacts/oireachtas-baseline-audit"))
    output_dir.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", region_name=region)

    rows: list[dict[str, Any]] = []
    for prefix in PREFIXES:
        rows.extend(list_objects(s3, bucket=bucket, prefix=prefix))

    rows.sort(key=lambda row: row["key"])
    with (output_dir / "s3_inventory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prefix", "key", "size", "etag", "last_modified_utc", "storage_class"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bucket": bucket,
        "region": region,
        "object_count": len(rows),
        "total_bytes": sum(int(row["size"]) for row in rows),
        "prefix_counts": {prefix: sum(1 for row in rows if row["prefix"] == prefix) for prefix in PREFIXES},
        "github_repository": os.getenv("GITHUB_REPOSITORY", ""),
        "github_ref": os.getenv("GITHUB_REF", ""),
        "github_sha": os.getenv("GITHUB_SHA", ""),
        "github_run_id": os.getenv("GITHUB_RUN_ID", ""),
        "github_run_attempt": os.getenv("GITHUB_RUN_ATTEMPT", ""),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
