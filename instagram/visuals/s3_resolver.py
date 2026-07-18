from __future__ import annotations

import json
import re
from typing import Any

PRODUCTION_POINTER_KEY = "processed/oireachtas_unified/pointers/production.json"
BATCH_ROOT = "processed/oireachtas_unified/batches"
LATEST_PATTERN = re.compile(
    r"^processed/oireachtas_unified/latest/(?P<format>csv|parquet)/(?P<table>[^/]+)\.(?P<extension>csv|parquet)$"
)
COMPAT_PREFIX = "processed/oireachtas_unified/compat/"


def is_unified_logical_key(key: str) -> bool:
    return key.startswith("processed/oireachtas_unified/latest/") or key.startswith(COMPAT_PREFIX)


def batch_key_for_logical_key(key: str, batch_id: str) -> str:
    latest = LATEST_PATTERN.fullmatch(key)
    if latest:
        return (
            f"{BATCH_ROOT}/{batch_id}/tables/{latest.group('table')}/"
            f"{latest.group('format')}/{latest.group('table')}.{latest.group('extension')}"
        )
    if key.startswith(COMPAT_PREFIX):
        relative = key[len(COMPAT_PREFIX) :]
        if not relative or ".." in relative.split("/"):
            raise ValueError(f"Unsafe compatibility key: {key!r}")
        return f"{BATCH_ROOT}/{batch_id}/compat/{relative}"
    raise ValueError(f"Unsupported unified logical key: {key!r}")


def resolve_logical_key(client: Any, *, bucket: str, key: str) -> tuple[str, dict[str, Any]]:
    if not is_unified_logical_key(key):
        return key, {
            "logical_key": key,
            "resolved_key": key,
            "resolution_mode": "direct",
        }

    try:
        response = client.get_object(Bucket=bucket, Key=PRODUCTION_POINTER_KEY)
        pointer = json.loads(response["Body"].read().decode("utf-8"))
    except Exception:
        return key, {
            "logical_key": key,
            "resolved_key": key,
            "resolution_mode": "logical_direct_fallback",
            "pointer_key": PRODUCTION_POINTER_KEY,
        }

    mode = str(pointer.get("mode") or "batch")
    if mode == "legacy_direct":
        return key, {
            "logical_key": key,
            "resolved_key": key,
            "resolution_mode": "legacy_direct",
            "pointer_key": PRODUCTION_POINTER_KEY,
        }
    if mode != "batch":
        raise ValueError(f"Unsupported unified production pointer mode: {mode!r}")

    batch_id = str(pointer.get("batch_id") or "").strip()
    if not batch_id:
        raise ValueError("Unified production pointer is missing batch_id")
    resolved_key = batch_key_for_logical_key(key, batch_id)
    return resolved_key, {
        "logical_key": key,
        "resolved_key": resolved_key,
        "resolution_mode": "batch_pointer",
        "pointer_key": PRODUCTION_POINTER_KEY,
        "batch_id": batch_id,
        "manifest_key": pointer.get("manifest_key"),
        "promoted_at_utc": pointer.get("promoted_at_utc"),
    }


def get_object(client: Any, *, bucket: str, key: str, byte_range: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved_key, resolution = resolve_logical_key(client, bucket=bucket, key=key)
    params: dict[str, Any] = {"Bucket": bucket, "Key": resolved_key}
    if byte_range:
        params["Range"] = byte_range
    try:
        response = client.get_object(**params)
        return response, resolution
    except Exception:
        if resolved_key == key:
            raise
        fallback_params: dict[str, Any] = {"Bucket": bucket, "Key": key}
        if byte_range:
            fallback_params["Range"] = byte_range
        response = client.get_object(**fallback_params)
        return response, {
            **resolution,
            "resolved_key": key,
            "resolution_mode": "logical_direct_after_batch_miss",
        }
