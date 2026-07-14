"""Immutable batch publication, validation, promotion, and rollback helpers."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from .normalize import stable_json_dumps, utc_now_iso


BATCH_ROOT = "processed/oireachtas_unified/batches"
POINTER_ROOT = "processed/oireachtas_unified/pointers"
PRODUCTION_POINTER_KEY = f"{POINTER_ROOT}/production.json"
PREVIOUS_POINTER_KEY = f"{POINTER_ROOT}/previous.json"
_BATCH_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_LATEST_PATTERN = re.compile(
    r"^processed/oireachtas_unified/latest/(?P<format>csv|parquet)/(?P<table>[^/]+)\.(?P<extension>csv|parquet)$"
)
_REVIEW_LATEST_PATTERN = re.compile(
    r"^processed/oireachtas_unified/review/(?P<table>[^/]+)/latest/(?P<filename>[^/]+)$"
)


def current_batch_id() -> str | None:
    value = os.getenv("OIREACHTAS_BATCH_ID", "").strip()
    return validate_batch_id(value) if value else None


def validate_batch_id(batch_id: str) -> str:
    value = str(batch_id or "").strip()
    if not _BATCH_ID_PATTERN.fullmatch(value):
        raise ValueError(
            "batch_id must begin with an alphanumeric character and contain only "
            "letters, numbers, dots, underscores, or hyphens (maximum 128 characters)"
        )
    return value


def batch_manifest_key(batch_id: str) -> str:
    return f"{BATCH_ROOT}/{validate_batch_id(batch_id)}/manifest.json"


def batch_entry_key(batch_id: str, table: str) -> str:
    safe_table = _safe_component(table, label="table")
    return f"{BATCH_ROOT}/{validate_batch_id(batch_id)}/entries/{safe_table}.json"


def batch_key_for_production_key(key: str, batch_id: str) -> str:
    """Map a mutable production key to its immutable batch location."""
    batch_id = validate_batch_id(batch_id)
    latest = _LATEST_PATTERN.fullmatch(key)
    if latest:
        return (
            f"{BATCH_ROOT}/{batch_id}/tables/{latest.group('table')}/"
            f"{latest.group('format')}/{latest.group('table')}.{latest.group('extension')}"
        )
    compat_prefix = "processed/oireachtas_unified/compat/"
    if key.startswith(compat_prefix):
        relative = key[len(compat_prefix) :]
        if not relative or ".." in relative.split("/"):
            raise ValueError(f"Unsafe compatibility key: {key!r}")
        return f"{BATCH_ROOT}/{batch_id}/compat/{relative}"
    review = _REVIEW_LATEST_PATTERN.fullmatch(key)
    if review:
        return (
            f"{BATCH_ROOT}/{batch_id}/review/{review.group('table')}/"
            f"{review.group('filename')}"
        )
    raise ValueError(f"Not a supported mutable Oireachtas key: {key!r}")


def resolve_production_key(s3: Any, *, bucket: str, production_key: str) -> str:
    """Resolve a logical production key through the current production pointer."""
    pointer = read_json_if_exists(s3, bucket=bucket, key=PRODUCTION_POINTER_KEY)
    if not pointer:
        raise FileNotFoundError(f"Production pointer does not exist: s3://{bucket}/{PRODUCTION_POINTER_KEY}")
    batch_id = validate_batch_id(str(pointer.get("batch_id") or ""))
    return batch_key_for_production_key(production_key, batch_id)


def record_batch_table(
    s3: Any,
    *,
    bucket: str,
    batch_id: str,
    table: str,
    manifest: Mapping[str, Any],
    schema: Mapping[str, Any],
    dq: Mapping[str, Any],
    candidate_keys: Iterable[str],
) -> dict[str, Any]:
    """Write one table entry used to assemble the final batch manifest."""
    batch_id = validate_batch_id(batch_id)
    objects: list[dict[str, Any]] = []
    for key in candidate_keys:
        if not key:
            continue
        batch_key = batch_key_for_production_key(key, batch_id)
        metadata = describe_object_if_exists(s3, bucket=bucket, key=batch_key)
        objects.append({"logical_key": key, "batch_key": batch_key, **metadata})

    entry = {
        "batch_id": batch_id,
        "table": table,
        "recorded_at_utc": utc_now_iso(),
        "status": "validated" if dq.get("dq_status") == "pass" else "failed",
        "dq_status": dq.get("dq_status"),
        "row_count": manifest.get("output_rows"),
        "primary_key": schema.get("primary_key"),
        "schema_columns": schema.get("columns"),
        "source_run_id": manifest.get("run_id"),
        "github_run_id": os.getenv("GITHUB_RUN_ID", ""),
        "github_run_attempt": os.getenv("GITHUB_RUN_ATTEMPT", ""),
        "github_sha": os.getenv("GITHUB_SHA", ""),
        "objects": objects,
        "manifest": dict(manifest),
        "dq": dict(dq),
    }
    _put_json_direct(s3, bucket=bucket, key=batch_entry_key(batch_id, table), payload=entry)
    return entry


def assemble_batch_manifest(
    s3: Any,
    *,
    bucket: str,
    batch_id: str,
    required_tables: Iterable[str] = (),
) -> dict[str, Any]:
    """Assemble and validate a batch manifest from table entries."""
    batch_id = validate_batch_id(batch_id)
    entries = list_batch_entries(s3, bucket=bucket, batch_id=batch_id)
    required = sorted({_safe_component(table, label="table") for table in required_tables if str(table).strip()})
    by_table = {str(entry.get("table")): entry for entry in entries}
    missing_tables = sorted(set(required) - set(by_table))
    failed_tables = sorted(
        table
        for table, entry in by_table.items()
        if entry.get("status") != "validated" or entry.get("dq_status") != "pass"
    )
    missing_objects = sorted(
        str(entry.get("table"))
        for entry in entries
        if not entry.get("objects") or any(not obj.get("exists") for obj in entry.get("objects", []))
    )
    duplicate_tables = sorted(_duplicates(str(entry.get("table")) for entry in entries))
    status = "validated" if entries and not missing_tables and not failed_tables and not missing_objects and not duplicate_tables else "failed"
    manifest = {
        "batch_id": batch_id,
        "status": status,
        "created_at_utc": utc_now_iso(),
        "bucket": bucket,
        "required_tables": required,
        "table_count": len(entries),
        "tables": sorted(entries, key=lambda entry: str(entry.get("table"))),
        "validation": {
            "missing_tables": missing_tables,
            "failed_tables": failed_tables,
            "missing_objects": missing_objects,
            "duplicate_tables": duplicate_tables,
        },
        "github_run_id": os.getenv("GITHUB_RUN_ID", ""),
        "github_run_attempt": os.getenv("GITHUB_RUN_ATTEMPT", ""),
        "github_sha": os.getenv("GITHUB_SHA", ""),
    }
    _put_json_direct(s3, bucket=bucket, key=batch_manifest_key(batch_id), payload=manifest)
    return manifest


def promote_batch(
    s3: Any,
    *,
    bucket: str,
    batch_id: str,
    actor: str = "",
    workflow_run_id: str = "",
) -> dict[str, Any]:
    """Atomically promote one validated batch by updating a single pointer."""
    batch_id = validate_batch_id(batch_id)
    manifest = read_json_required(s3, bucket=bucket, key=batch_manifest_key(batch_id))
    if manifest.get("status") != "validated":
        raise ValueError(f"Batch {batch_id} is not validated; status={manifest.get('status')!r}")
    current = read_json_if_exists(s3, bucket=bucket, key=PRODUCTION_POINTER_KEY)
    if current:
        previous = {
            **current,
            "superseded_at_utc": utc_now_iso(),
            "superseded_by_batch_id": batch_id,
        }
        _put_json_direct(s3, bucket=bucket, key=PREVIOUS_POINTER_KEY, payload=previous)
    pointer = {
        "batch_id": batch_id,
        "manifest_key": batch_manifest_key(batch_id),
        "promoted_at_utc": utc_now_iso(),
        "promoted_by": actor,
        "workflow_run_id": workflow_run_id,
        "previous_batch_id": current.get("batch_id") if current else None,
    }
    _put_json_direct(s3, bucket=bucket, key=PRODUCTION_POINTER_KEY, payload=pointer)
    return pointer


def rollback_batch(
    s3: Any,
    *,
    bucket: str,
    target_batch_id: str,
    actor: str = "",
    workflow_run_id: str = "",
) -> dict[str, Any]:
    """Rollback by promoting an earlier validated immutable batch."""
    current = read_json_if_exists(s3, bucket=bucket, key=PRODUCTION_POINTER_KEY)
    pointer = promote_batch(
        s3,
        bucket=bucket,
        batch_id=target_batch_id,
        actor=actor,
        workflow_run_id=workflow_run_id,
    )
    pointer["operation"] = "rollback"
    pointer["rolled_back_from_batch_id"] = current.get("batch_id") if current else None
    pointer["rolled_back_at_utc"] = utc_now_iso()
    _put_json_direct(s3, bucket=bucket, key=PRODUCTION_POINTER_KEY, payload=pointer)
    return pointer


def list_batch_entries(s3: Any, *, bucket: str, batch_id: str) -> list[dict[str, Any]]:
    prefix = f"{BATCH_ROOT}/{validate_batch_id(batch_id)}/entries/"
    entries: list[dict[str, Any]] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = str(item.get("Key") or "")
            if key.endswith(".json"):
                entries.append(read_json_required(s3, bucket=bucket, key=key))
    return entries


def describe_object_if_exists(s3: Any, *, bucket: str, key: str) -> dict[str, Any]:
    try:
        response = s3.head_object(Bucket=bucket, Key=key)
    except Exception:
        return {"exists": False, "size": None, "etag": None, "version_id": None}
    return {
        "exists": True,
        "size": int(response.get("ContentLength", 0)),
        "etag": str(response.get("ETag", "")).strip('"'),
        "version_id": response.get("VersionId"),
    }


def read_json_if_exists(s3: Any, *, bucket: str, key: str) -> dict[str, Any] | None:
    try:
        return read_json_required(s3, bucket=bucket, key=key)
    except Exception:
        return None


def read_json_required(s3: Any, *, bucket: str, key: str) -> dict[str, Any]:
    obj = s3.get_object(Bucket=bucket, Key=key)
    payload = json.loads(obj["Body"].read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at s3://{bucket}/{key}")
    return payload


def _put_json_direct(s3: Any, *, bucket: str, key: str, payload: Mapping[str, Any]) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=(stable_json_dumps(payload) + "\n").encode("utf-8"),
        ContentType="application/json",
    )


def _safe_component(value: str, *, label: str) -> str:
    text = str(value or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", text):
        raise ValueError(f"Unsafe {label}: {value!r}")
    return text


def _duplicates(values: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates
