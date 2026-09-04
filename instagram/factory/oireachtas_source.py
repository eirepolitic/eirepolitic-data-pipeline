from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date
from typing import Iterable

import boto3
import pandas as pd

from extract.oireachtas.batch import (
    BATCH_POINTER_MODE,
    PRODUCTION_POINTER_KEY,
    batch_manifest_key,
    batch_key_for_production_key,
    read_json_required,
)

DEFAULT_BUCKET = "eirepolitic-data"


@dataclass(frozen=True)
class ResolvedOireachtasBatch:
    bucket: str
    batch_id: str
    pointer: dict
    manifest: dict

    def key_for_table(self, table: str, *, fmt: str = "csv") -> str:
        logical = f"processed/oireachtas_unified/latest/{fmt}/{table}.{fmt}"
        return batch_key_for_production_key(logical, self.batch_id)


def resolve_validated_production_batch(*, bucket: str = DEFAULT_BUCKET, s3=None) -> ResolvedOireachtasBatch:
    s3 = s3 or boto3.client("s3")
    pointer = read_json_required(s3, bucket=bucket, key=PRODUCTION_POINTER_KEY)
    mode = str(pointer.get("mode") or "")
    if mode != BATCH_POINTER_MODE:
        raise RuntimeError(
            f"Instagram generation requires immutable Oireachtas batch mode; production pointer mode={mode!r}"
        )
    batch_id = str(pointer.get("batch_id") or "").strip()
    if not batch_id:
        raise RuntimeError("Oireachtas production pointer has no batch_id")
    manifest = read_json_required(s3, bucket=bucket, key=batch_manifest_key(batch_id))
    if manifest.get("status") != "validated":
        raise RuntimeError(f"Oireachtas batch {batch_id} is not validated: status={manifest.get('status')!r}")
    validation = manifest.get("validation") or {}
    blockers = {
        key: validation.get(key)
        for key in ("missing_tables", "failed_tables", "missing_objects", "duplicate_tables")
        if validation.get(key)
    }
    if blockers:
        raise RuntimeError(f"Oireachtas batch {batch_id} has validation blockers: {blockers}")
    return ResolvedOireachtasBatch(bucket=bucket, batch_id=batch_id, pointer=pointer, manifest=manifest)


def load_csv_tables(
    batch: ResolvedOireachtasBatch,
    tables: Iterable[str],
    *,
    s3=None,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    s3 = s3 or boto3.client("s3")
    frames: dict[str, pd.DataFrame] = {}
    lineage: dict[str, dict] = {}
    for table in tables:
        key = batch.key_for_table(table)
        obj = s3.get_object(Bucket=batch.bucket, Key=key)
        body = obj["Body"].read()
        frame = pd.read_csv(io.BytesIO(body), dtype=str, keep_default_na=False, na_values=[""])
        frames[table] = frame
        lineage[table] = {
            "bucket": batch.bucket,
            "key": key,
            "etag": str(obj.get("ETag") or "").strip('"'),
            "version_id": obj.get("VersionId"),
            "last_modified": obj.get("LastModified").isoformat() if obj.get("LastModified") else None,
            "row_count": int(len(frame)),
            "source_batch_id": batch.batch_id,
        }
    return frames, lineage


def require_completed_calendar_month(period, *, today: date | None = None) -> None:
    current = today or date.today()
    if getattr(period, "kind", None) != "month":
        raise RuntimeError(f"This post requires a calendar month; got kind={getattr(period, 'kind', None)!r}")
    if period.end >= current:
        raise RuntimeError(f"Requested month is not complete: {period.start} to {period.end}; today={current}")
