from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from extract.oireachtas.batch import (
    BATCH_POINTER_MODE,
    PRODUCTION_POINTER_KEY,
    assemble_batch_manifest,
    batch_entry_key,
    batch_key_for_production_key,
    batch_manifest_key,
    read_json_required,
    validate_batch_id,
)
from extract.oireachtas.io_s3 import DEFAULT_BUCKET, DEFAULT_REGION, make_s3_client
from extract.oireachtas.normalize import stable_json_dumps, utc_now_iso


def _put_json(s3: Any, *, bucket: str, key: str, payload: dict[str, Any]) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=(stable_json_dumps(payload) + "\n").encode("utf-8"),
        ContentType="application/json",
    )


def seed_candidate(s3: Any, *, bucket: str, batch_id: str) -> dict[str, Any]:
    batch_id = validate_batch_id(batch_id)
    pointer = read_json_required(s3, bucket=bucket, key=PRODUCTION_POINTER_KEY)
    if str(pointer.get("mode") or BATCH_POINTER_MODE) != BATCH_POINTER_MODE:
        raise RuntimeError("Automatic candidate seeding requires a batch-mode production pointer")

    source_batch_id = validate_batch_id(str(pointer.get("batch_id") or ""))
    if source_batch_id == batch_id:
        raise ValueError("destination batch_id must differ from the production batch_id")

    source_manifest = read_json_required(s3, bucket=bucket, key=batch_manifest_key(source_batch_id))
    if source_manifest.get("status") != "validated":
        raise RuntimeError(f"Production batch {source_batch_id} is not validated")

    copied_objects = 0
    copied_entries = 0
    for source_entry in source_manifest.get("tables", []):
        entry = deepcopy(source_entry)
        table = str(entry.get("table") or "").strip()
        if not table:
            raise ValueError("Production manifest contains an entry without a table name")

        destination_objects: list[dict[str, Any]] = []
        for source_object in entry.get("objects", []):
            logical_key = str(source_object.get("logical_key") or "").strip()
            source_key = str(source_object.get("batch_key") or "").strip()
            if not logical_key or not source_key:
                raise ValueError(f"Table {table} contains an invalid object reference")
            destination_key = batch_key_for_production_key(logical_key, batch_id)
            copy_result = s3.copy_object(
                Bucket=bucket,
                Key=destination_key,
                CopySource={"Bucket": bucket, "Key": source_key},
                MetadataDirective="COPY",
            )
            head = s3.head_object(Bucket=bucket, Key=destination_key)
            destination_objects.append(
                {
                    "logical_key": logical_key,
                    "batch_key": destination_key,
                    "exists": True,
                    "size": int(head.get("ContentLength", 0)),
                    "etag": str(head.get("ETag", "")).strip('"'),
                    "version_id": copy_result.get("VersionId") or head.get("VersionId"),
                }
            )
            copied_objects += 1

        entry.update(
            {
                "batch_id": batch_id,
                "recorded_at_utc": utc_now_iso(),
                "github_run_id": os.getenv("GITHUB_RUN_ID", ""),
                "github_run_attempt": os.getenv("GITHUB_RUN_ATTEMPT", ""),
                "github_sha": os.getenv("GITHUB_SHA", ""),
                "seeded_from_batch_id": source_batch_id,
                "objects": destination_objects,
            }
        )
        _put_json(s3, bucket=bucket, key=batch_entry_key(batch_id, table), payload=entry)
        copied_entries += 1

    manifest = assemble_batch_manifest(
        s3,
        bucket=bucket,
        batch_id=batch_id,
        required_tables=source_manifest.get("required_tables", []),
    )
    if manifest.get("status") != "validated":
        raise RuntimeError(f"Seeded candidate failed validation: {manifest.get('validation')}")

    return {
        "status": "seeded",
        "source_batch_id": source_batch_id,
        "batch_id": batch_id,
        "copied_entries": copied_entries,
        "copied_objects": copied_objects,
        "table_count": manifest.get("table_count"),
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Clone the current complete Oireachtas production batch into a new candidate batch.")
    root.add_argument("--batch-id", required=True)
    root.add_argument("--bucket", default=os.getenv("S3_BUCKET", DEFAULT_BUCKET))
    root.add_argument("--region", default=os.getenv("AWS_REGION", DEFAULT_REGION))
    return root


def main() -> int:
    args = parser().parse_args()
    s3 = make_s3_client(region_name=args.region)
    result = seed_candidate(s3, bucket=args.bucket, batch_id=args.batch_id)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
